"""Per-client sliding-window rate limiter middleware.

Enabled when ``RATE_LIMIT_RPM > 0`` in settings.  When the limit is
exceeded the middleware returns a 429 with an Anthropic-shaped body and
a ``retry-after`` header.

Uses a sliding-window algorithm (timestamps ring-buffer) rather than a
fixed-window counter.  This prevents the double-quota burst that fixed
windows allow at the boundary (e.g. 60 req at 00:59, 60 more at 01:00).

Client identity (in priority order):
  1. ``metadata.user_id`` from the JSON body  (only for POST /v1/messages)
  2. ``x-forwarded-for`` first IP
  3. ``client.host`` (direct TCP peer)
"""

from __future__ import annotations

import asyncio
import collections
import json
import time

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

_log = structlog.get_logger("nvd_claude_proxy.rate_limiter")

# Endpoints subject to rate limiting.
_RATE_LIMITED_PATHS = {"/v1/messages"}

_WINDOW_SECONDS = 60.0


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """Sliding-window (per-minute) rate limiter.

    Stores a deque of request timestamps per client. On each request,
    expired timestamps (older than 60 s) are evicted before checking the count.
    This gives a true "60 requests in any rolling 60-second window" guarantee.
    """

    def __init__(self, app, rpm_limit: int) -> None:
        super().__init__(app)
        self._rpm = rpm_limit
        # client_key → deque of monotonic timestamps
        self._history: dict[str, collections.deque] = {}
        self._lock = asyncio.Lock()

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path not in _RATE_LIMITED_PATHS:
            return await call_next(request)

        client_key = await self._client_key(request)
        now = time.monotonic()
        cutoff = now - _WINDOW_SECONDS

        async with self._lock:
            dq = self._history.setdefault(client_key, collections.deque())
            # Evict timestamps outside the rolling window.
            while dq and dq[0] <= cutoff:
                dq.popleft()
            count = len(dq)
            if count < self._rpm:
                dq.append(now)
                allowed = True
                retry_after = 0
            else:
                allowed = False
                # Earliest timestamp in window tells us when the window frees a slot.
                retry_after = max(1, int(_WINDOW_SECONDS - (now - dq[0])) + 1)

        if not allowed:
            _log.warning(
                "rate_limit.exceeded",
                client=client_key,
                count=count,
                limit=self._rpm,
                retry_after=retry_after,
            )
            return JSONResponse(
                {
                    "type": "error",
                    "error": {
                        "type": "rate_limit_error",
                        "message": (
                            f"Rate limit exceeded: {self._rpm} requests/minute. "
                            f"Retry after {retry_after}s."
                        ),
                    },
                },
                status_code=429,
                headers={
                    "retry-after": str(retry_after),
                    "anthropic-ratelimit-requests-limit": str(self._rpm),
                    "anthropic-ratelimit-requests-remaining": "0",
                    "anthropic-ratelimit-requests-reset": _iso8601(int(time.time()) + retry_after),
                },
            )

        return await call_next(request)

    async def _client_key(self, request: Request) -> str:
        """Extract client identity for bucket keying."""
        # Try metadata.user_id from JSON body (non-destructive peek).
        if request.method == "POST" and "application/json" in request.headers.get(
            "content-type", ""
        ):
            try:
                body_bytes = await request.body()
                body = json.loads(body_bytes)
                uid = (body.get("metadata") or {}).get("user_id")
                if uid:
                    return f"user:{uid}"
            except Exception:
                pass

        # Forwarded IP (behind a proxy/load-balancer).
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"

        # Direct TCP peer.
        if request.client:
            return f"ip:{request.client.host}"

        return "ip:unknown"


def _iso8601(epoch_s: int) -> str:
    import time as _time

    return _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime(epoch_s))
