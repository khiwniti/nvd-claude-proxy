"""Response headers the Anthropic SDK expects on every reply.

Real-world behavior observed:
  • Python SDK logs `response.headers["request-id"]` at DEBUG.
  • TypeScript SDK surfaces `anthropic-request-id` via `response._request_id`
    and attaches it to exceptions.
  • Claude Code includes the id in its usage telemetry events.
  • SDKs read `anthropic-ratelimit-*` to size exponential backoff; if absent
    they assume an optimistic RPM budget which can thrash the upstream.
"""
from __future__ import annotations

import time
import uuid


def new_request_id() -> str:
    """Mirror Anthropic's `req_…` format (23 chars url-safe base32)."""
    return "req_" + uuid.uuid4().hex[:20]


def standard_response_headers(
    request_id: str,
    *,
    rpm_limit: int = 40,
    tpm_limit: int = 60_000,
) -> dict[str, str]:
    """Anthropic-compatible response headers.

    We fabricate the rate-limit headers from `settings` because NVIDIA NIM does
    not surface per-key counters. The fabricated values are conservative —
    sized to the Build free tier — so SDK backoff is never *too* aggressive.
    """
    now = int(time.time())
    return {
        # Required by ALL official Anthropic SDKs — TypeScript SDK throws if absent.
        "anthropic-version": "2023-06-01",
        "x-anthropic-type": "anthropic-api-response",
        "anthropic-request-id": request_id,
        "request-id": request_id,  # alias some SDKs read
        "anthropic-organization-id": "nvd-proxy-local",
        "anthropic-ratelimit-requests-limit": str(rpm_limit),
        "anthropic-ratelimit-requests-remaining": str(rpm_limit),
        "anthropic-ratelimit-requests-reset": _iso8601(now + 60),
        "anthropic-ratelimit-tokens-limit": str(tpm_limit),
        "anthropic-ratelimit-tokens-remaining": str(tpm_limit),
        "anthropic-ratelimit-tokens-reset": _iso8601(now + 60),
    }


def _iso8601(epoch_s: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch_s))
