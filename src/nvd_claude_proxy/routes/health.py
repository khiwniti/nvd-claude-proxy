from __future__ import annotations

import time

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import ORJSONResponse

from .._version import __version__

router = APIRouter()


@router.get("/healthz")
async def healthz() -> ORJSONResponse:
    return ORJSONResponse({"status": "ok", "version": __version__})


@router.get("/readyz")
async def readyz(request: Request) -> ORJSONResponse:
    """Deep readiness probe — verifies NVIDIA API connectivity.

    Returns 200 if the upstream models endpoint responds, 503 otherwise.
    Load balancers should use this endpoint to gate traffic.
    """
    client: httpx.AsyncClient = request.app.state.nvidia_client._client
    t0 = time.monotonic()
    try:
        resp = await client.get("/models", timeout=5.0)
        elapsed_ms = round((time.monotonic() - t0) * 1000)
        if resp.status_code < 500:
            return ORJSONResponse(
                {"status": "ok", "version": __version__, "upstream_ms": elapsed_ms}
            )
        return ORJSONResponse(
            {"status": "degraded", "upstream_status": resp.status_code, "upstream_ms": elapsed_ms},
            status_code=503,
        )
    except Exception as exc:
        elapsed_ms = round((time.monotonic() - t0) * 1000)
        return ORJSONResponse(
            {"status": "unavailable", "error": str(exc), "upstream_ms": elapsed_ms},
            status_code=503,
        )
