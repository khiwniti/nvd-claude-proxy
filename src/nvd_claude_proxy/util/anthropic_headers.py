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
    rpm_limit: int | None = None,
    rpm_remaining: int | None = None,
    tpm_limit: int | None = None,
    tpm_remaining: int | None = None,
) -> dict[str, str]:
    """Anthropic-compatible response headers.

    P1-14: Rate-limit headers are now optional. If unknown, they are omitted
    to avoid confusing SDK backoff logic with fabricated values.
    """
    now = int(time.time())
    headers = {
        "anthropic-version": "2023-06-01",
        "x-anthropic-type": "anthropic-api-response",
        "anthropic-request-id": request_id,
        "request-id": request_id,
        "anthropic-organization-id": "nvd-proxy-local",
    }

    if rpm_limit is not None:
        headers["anthropic-ratelimit-requests-limit"] = str(rpm_limit)
    if rpm_remaining is not None:
        headers["anthropic-ratelimit-requests-remaining"] = str(rpm_remaining)
        headers["anthropic-ratelimit-requests-reset"] = _iso8601(now + 60)

    if tpm_limit is not None:
        headers["anthropic-ratelimit-tokens-limit"] = str(tpm_limit)
    if tpm_remaining is not None:
        headers["anthropic-ratelimit-tokens-remaining"] = str(tpm_remaining)
        headers["anthropic-ratelimit-tokens-reset"] = _iso8601(now + 60)

    return headers


def _iso8601(epoch_s: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch_s))
