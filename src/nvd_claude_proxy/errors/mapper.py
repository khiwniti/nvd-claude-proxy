from __future__ import annotations

from typing import Any

_STATUS_TO_ANTHROPIC_TYPE: dict[int, str] = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    422: "invalid_request_error",
    429: "rate_limit_error",
    500: "api_error",
    502: "api_error",
    503: "overloaded_error",
    504: "api_error",
    529: "overloaded_error",
}


def openai_error_to_anthropic(status: int, body: Any) -> tuple[int, dict]:
    """Map an upstream OpenAI-ish error → (http_status, Anthropic error body)."""
    atype = _STATUS_TO_ANTHROPIC_TYPE.get(status, "api_error")
    msg = ""
    if isinstance(body, dict):
        err = body.get("error") or body
        if isinstance(err, dict):
            msg = err.get("message") or err.get("type") or ""
        else:
            msg = str(err)
    elif isinstance(body, str):
        msg = body
    # NVIDIA appends " None" to some 400 messages when there is no extra detail.
    clean_msg = (msg or f"upstream status {status}").rstrip(". ").removesuffix(" None").strip()
    return status, {
        "type": "error",
        "error": {
            "type": atype,
            "message": clean_msg,
        },
    }
