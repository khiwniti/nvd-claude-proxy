from __future__ import annotations

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

_log = structlog.get_logger("nvd_claude_proxy")


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            _log.exception(
                "unhandled", path=request.url.path, rid=rid, method=request.method
            )
            raise
        dur_ms = (time.perf_counter() - t0) * 1000
        _log.info(
            "request",
            path=request.url.path,
            method=request.method,
            status=response.status_code,
            dur_ms=round(dur_ms, 1),
            rid=rid,
        )
        response.headers["x-request-id"] = rid
        return response
