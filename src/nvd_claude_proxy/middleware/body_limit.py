"""Request body size guard middleware.

Enabled when ``MAX_REQUEST_BODY_MB > 0`` in settings.  Rejects requests
whose ``Content-Length`` exceeds the limit with 413 + Anthropic error body.

Claude Code's largest payloads (273 tools, large system prompts) are typically
well under 2 MB.  Setting ``MAX_REQUEST_BODY_MB=10`` is a safe production default.
"""

from __future__ import annotations

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

_log = structlog.get_logger("nvd_claude_proxy.body_limit")


class BodyLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_bytes: int) -> None:
        super().__init__(app)
        self._max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > self._max_bytes:
                    _log.warning(
                        "body_limit.exceeded",
                        content_length=content_length,
                        limit_bytes=self._max_bytes,
                        path=request.url.path,
                    )
                    return JSONResponse(
                        {
                            "type": "error",
                            "error": {
                                "type": "request_too_large",
                                "message": (
                                    f"Request body too large: "
                                    f"max {self._max_bytes // (1024 * 1024)} MB allowed."
                                ),
                            },
                        },
                        status_code=413,
                    )
            except ValueError:
                pass
        return await call_next(request)
