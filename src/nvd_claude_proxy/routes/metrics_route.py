"""GET /metrics — Prometheus text exposition."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse, Response

from ..util.metrics import is_enabled, prometheus_text

router = APIRouter()


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    if not is_enabled():
        return PlainTextResponse(
            "prometheus-client not installed; install it to enable this endpoint.\n",
            status_code=501,
        )
    body, content_type = prometheus_text()
    return Response(content=body, media_type=content_type)
