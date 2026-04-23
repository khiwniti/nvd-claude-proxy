from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import ORJSONResponse

from .._version import __version__

router = APIRouter()


@router.get("/healthz")
async def healthz() -> ORJSONResponse:
    return ORJSONResponse({"status": "ok", "version": __version__})
