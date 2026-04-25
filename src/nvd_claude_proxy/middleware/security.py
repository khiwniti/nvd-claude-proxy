"""Security middleware stack for production hardening.

This module provides multiple security layers:
1. Security headers (X-Frame-Options, CSP, etc.)
2. SSRF protection (blocks dangerous URL schemes/hosts)
3. Request size limits
4. Suspicious request detection

All security checks are logged for audit purposes.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from fastapi.responses import ORJSONResponse
from urllib.parse import urlparse


_log = structlog.get_logger("nvd_claude_proxy.security")

# ── SSRF Protection ──────────────────────────────────────────────────────────

# URL schemes that are never valid for user-submitted content
BLOCKED_SCHEMES: frozenset[str] = frozenset({"javascript", "file", "ftp", "mailto", "tel", "data"})

# Hosts that are never valid (SSRF targets, internal services)
BLOCKED_HOSTS: frozenset[str] = frozenset(
    {
        "localhost",
        "127.0.0.1",
        "::1",  # Localhost
        "169.254.169.254",  # AWS metadata
        "metadata.google.internal",  # GCP metadata
        "100.100.100.200",  # Alibaba Cloud metadata
    }
)

# Patterns that match blocked hosts (for dynamic blocking)
BLOCKED_HOST_PATTERNS: list[re.Pattern] = [
    re.compile(r"^metadata\.azure\.com$", re.IGNORECASE),
    re.compile(r"^.*\.metadata\.azure\.com$", re.IGNORECASE),
    re.compile(r"^kubernetes\.internal$", re.IGNORECASE),
]

# Maximum URL length
MAX_URL_LENGTH = 8192

# File extensions that indicate dangerous content
BLOCKED_FILE_EXTENSIONS: frozenset[str] = frozenset(
    {".exe", ".dll", ".so", ".dylib", ".bat", ".sh", ".ps1", ".cmd"}
)


def is_url_blocked(url: str) -> tuple[bool, str]:
    """Check if a URL should be blocked for security reasons.

    Returns:
        Tuple of (is_blocked, reason)
    """
    if not url or len(url) > MAX_URL_LENGTH:
        return True, f"URL exceeds maximum length of {MAX_URL_LENGTH}"

    try:
        parsed = urlparse(url)
    except Exception as e:
        return True, f"Invalid URL: {e}"

    # Check scheme
    scheme = parsed.scheme.lower()
    if scheme in BLOCKED_SCHEMES:
        return True, f"Blocked URL scheme: {scheme}"

    if scheme not in ("http", "https", ""):
        return True, f"Unsupported URL scheme: {scheme}"

    # Check hostname
    hostname = parsed.hostname or ""
    if hostname.lower() in BLOCKED_HOSTS:
        return True, f"Blocked hostname: {hostname}"

    # Check against patterns
    for pattern in BLOCKED_HOST_PATTERNS:
        if pattern.match(hostname):
            return True, f"Hostname matches blocked pattern: {hostname}"

    # Check for IP addresses that might be internal
    # This is a simplified check; real implementation would need
    # proper IP range checking (ipaddress module)
    if re.match(r"^10\.\d+\.\d+\.\d+$", hostname):
        return True, "Blocked private IP range: 10.x.x.x"
    if re.match(r"^172\.(1[6-9]|2\d|3[01])\.\d+\.\d+$", hostname):
        return True, "Blocked private IP range: 172.16-31.x.x"
    if re.match(r"^192\.168\.\d+\.\d+$", hostname):
        return True, "Blocked private IP range: 192.168.x.x"

    # Check for @ sign (can be used to bypass scheme checks)
    if "@" in url:
        return True, "URL with credentials (@) not allowed"

    return False, ""


def extract_urls_from_body(body: dict, path: str = "") -> list[tuple[str, str]]:
    """Recursively extract all URLs from a request body.

    Returns:
        List of (url, location_path) tuples
    """
    urls: list[tuple[str, str]] = []

    def walk(obj: Any, current_path: str) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{current_path}.{key}" if current_path else key

                # Check if this value looks like a URL
                if isinstance(value, str) and (
                    value.startswith("http://")
                    or value.startswith("https://")
                    or value.startswith("file://")
                    or value.startswith("data:")
                ):
                    urls.append((value, new_path))
                elif key in ("url", "source", "href", "src", "link"):
                    if isinstance(value, str):
                        urls.append((value, new_path))
                    elif isinstance(value, dict):
                        # Might have nested url
                        if u := value.get("url"):
                            urls.append((u, f"{new_path}.url"))
                        if src := value.get("source"):
                            if isinstance(src, dict):
                                if u := src.get("url"):
                                    urls.append((u, f"{new_path}.source.url"))

                walk(value, new_path)
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                walk(item, f"{current_path}[{idx}]")

    walk(body, path)
    return urls


# ── Security Headers ─────────────────────────────────────────────────────────

# Headers to set on all responses
SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    # Content-Security-Policy can be restrictive since we don't serve HTML
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
}

# Cache-Control for API responses
CACHE_CONTROL_VALUE = "no-store, no-cache, must-revalidate, private"
PRAGMA_VALUE = "no-cache"


# ── Middleware Classes ────────────────────────────────────────────────────────


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add standard security headers to all responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Only add headers to successful responses
        if response.status_code < 400:
            for header, value in SECURITY_HEADERS.items():
                response.headers[header] = value

            # Cache control for API responses
            response.headers["Cache-Control"] = CACHE_CONTROL_VALUE
            response.headers["Pragma"] = PRAGMA_VALUE

        return response


class SSRFProtectionMiddleware(BaseHTTPMiddleware):
    """Block requests containing potentially dangerous URLs.

    This middleware checks all URLs in the request body for:
    - Blocked URL schemes (javascript:, file:, etc.)
    - Blocked hostnames (localhost, cloud metadata endpoints)
    - Private IP ranges
    - Credential-containing URLs
    """

    # Paths that should be checked for URLs
    PROTECTED_PATHS: frozenset[str] = frozenset(
        {
            "/v1/messages",
            "/v1/messages/batches",
            "/v1/files",
        }
    )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Only check POST-like methods with bodies
        if request.method not in ("POST", "PUT", "PATCH"):
            return await call_next(request)

        # Only check protected paths
        if request.url.path not in self.PROTECTED_PATHS:
            return await call_next(request)

        try:
            body = await request.json()
        except Exception:
            # Let the route handle malformed JSON
            return await call_next(request)

        # Extract and check all URLs
        urls = extract_urls_from_body(body)

        blocked_urls: list[dict[str, str]] = []
        for url, location in urls:
            is_blocked, reason = is_url_blocked(url)
            if is_blocked:
                blocked_urls.append(
                    {
                        "url": url[:100] + "..." if len(url) > 100 else url,
                        "location": location,
                        "reason": reason,
                    }
                )

        if blocked_urls:
            _log.warning(
                "security.ssrf_blocked",
                path=request.url.path,
                blocked_count=len(blocked_urls),
                blocked=blocked_urls[:3],  # Log first 3
            )

            return ORJSONResponse(
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": (
                            "Request contains blocked URLs. This may indicate a security issue."
                        ),
                    },
                },
                status_code=400,
            )

        # Store sanitized body for downstream use
        # Note: We re-parse the body since it was consumed
        # In production, you'd want to use a body reader middleware
        request.state._body = body  # type: ignore

        return await call_next(request)


class SuspiciousRequestDetectionMiddleware(BaseHTTPMiddleware):
    """Detect and log potentially suspicious request patterns.

    This is a monitoring/detection layer that flags suspicious activity
    without blocking it (unless explicitly configured).
    """

    # Patterns that might indicate probing/scanning
    SUSPICIOUS_PATHS: list[re.Pattern] = [
        re.compile(r"^/admin"),
        re.compile(r"^/\.env"),
        re.compile(r"^/wp-"),
        re.compile(r"^/api/v1/users"),
        re.compile(r"^/debug"),
        re.compile(r"^/actuator"),
    ]

    # Rate at which to log suspicious activity (1 in N)
    LOG_SAMPLE_RATE = 0.1

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path

        # Check for suspicious paths
        for pattern in self.SUSPICIOUS_PATHS:
            if pattern.match(path):
                client_ip = request.client.host if request.client else "unknown"
                user_agent = request.headers.get("user-agent", "unknown")

                _log.warning(
                    "security.suspicious_path",
                    path=path,
                    client_ip=client_ip,
                    user_agent=user_agent[:100],
                    method=request.method,
                )

        # Check for missing Anthropic headers (might be automated probe)
        if request.method == "POST" and not request.headers.get("anthropic-version"):
            _log.debug(
                "security.missing_anthropic_version",
                path=path,
                client_ip=request.client.host if request.client else None,
            )

        return await call_next(request)


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """Add request timing headers and log slow requests.

    Adds:
    - X-Request-Start: When we started processing
    - X-Response-Time: Total processing time (on response close)

    Logs requests that exceed a threshold (default: 30s).
    """

    def __init__(self, app: Any, slow_request_threshold: float = 30.0) -> None:
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.monotonic()
        request_id = request.headers.get("anthropic-request-id", "unknown")

        response = await call_next(request)

        elapsed = time.monotonic() - start_time
        response.headers["X-Request-Processing-Time"] = f"{elapsed:.3f}"

        # Log slow requests
        if elapsed > self.slow_request_threshold:
            _log.warning(
                "security.slow_request",
                request_id=request_id,
                path=request.url.path,
                method=request.method,
                elapsed_seconds=round(elapsed, 2),
                status_code=response.status_code,
            )

        return response


# ── Audit Logging ─────────────────────────────────────────────────────────────


class AuditLoggerMiddleware(BaseHTTPMiddleware):
    """Log all requests for audit purposes.

    Logs include:
    - Request method, path, headers (sanitized)
    - Response status
    - Timing information
    - Security-relevant metadata
    """

    # Headers to exclude from logs (sensitive)
    EXCLUDED_HEADERS: frozenset[str] = frozenset(
        {
            "authorization",
            "x-api-key",
            "cookie",
            "set-cookie",
        }
    )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        # Capture request metadata
        request_id = request.headers.get("anthropic-request-id", "unknown")
        client_ip = _get_client_ip(request)

        # Extract sanitized headers
        sanitized_headers = {
            k: v[:50] + "..." if len(v) > 50 else v
            for k, v in request.headers.items()
            if k.lower() not in self.EXCLUDED_HEADERS
        }

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            status_code = 500
            raise
        finally:
            elapsed_ms = (time.time() - start_time) * 1000

            _log.info(
                "audit.request",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                client_ip=client_ip,
                status_code=status_code,
                elapsed_ms=round(elapsed_ms, 1),
                user_agent=sanitized_headers.get("user-agent", "")[:100],
            )

        return response


def _get_client_ip(request: Request) -> str:
    """Extract the real client IP, considering proxy headers."""
    # Check X-Forwarded-For first (standard proxy header)
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        # Take the first IP in the chain (original client)
        return forwarded.split(",")[0].strip()

    # Check X-Real-IP (nginx)
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip

    # Fall back to direct connection IP
    if request.client:
        return request.client.host

    return "unknown"


# ── Composite Security Middleware ─────────────────────────────────────────────


def add_security_middleware(app: Any) -> None:
    """Add all security middleware to the FastAPI app.

    Middleware order matters! From outermost to innermost:
    1. SecurityHeadersMiddleware (adds headers to responses)
    2. SSRFProtectionMiddleware (validates request URLs)
    3. SuspiciousRequestDetectionMiddleware (monitors suspicious activity)
    4. RequestTimingMiddleware (timing + slow request detection)
    5. AuditLoggerMiddleware (full audit trail)
    """
    # Get threshold from settings if available, else use default
    # Note: In a real implementation, you'd pass this via app.state
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(SSRFProtectionMiddleware)
    app.add_middleware(SuspiciousRequestDetectionMiddleware)
    app.add_middleware(RequestTimingMiddleware, slow_request_threshold=30.0)
    app.add_middleware(AuditLoggerMiddleware)
