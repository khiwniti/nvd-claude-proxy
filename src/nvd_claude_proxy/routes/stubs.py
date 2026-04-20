"""Stub endpoints for Anthropic APIs that have no NVIDIA equivalent.

Returns 501 (Not Implemented) with a proper Anthropic-shaped error body so
clients that check the status code (rather than crashing on 404) can surface a
clear message instead of a cryptic failure.

Stubs:
- POST /v1/messages/batches
- GET  /v1/messages/batches/{id}
- POST /v1/messages/batches/{id}/cancel
- GET  /v1/messages/batches/{id}/results
- GET  /v1/messages/batches
- DELETE /v1/messages/batches/{id}
- POST /v1/files
- GET  /v1/files
- GET  /v1/files/{id}
- DELETE /v1/files/{id}
- GET  /v1/files/{id}/content
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import ORJSONResponse

from ..util.anthropic_headers import new_request_id, standard_response_headers

router = APIRouter()

_NOT_IMPLEMENTED = {
    "type": "error",
    "error": {
        "type": "api_error",
        "message": (
            "This endpoint is not implemented by nvd-claude-proxy. "
            "The NVIDIA NIM backend has no equivalent API. "
            "See docs/ANTHROPIC_COMPAT.md for details."
        ),
    },
}


def _stub_response(request: Request) -> ORJSONResponse:
    rid = new_request_id()
    return ORJSONResponse(
        _NOT_IMPLEMENTED,
        status_code=501,
        headers=standard_response_headers(rid),
    )


# ── Message Batches ────────────────────────────────────────────────────────────

@router.post("/v1/messages/batches")
async def create_batch(request: Request) -> ORJSONResponse:
    return _stub_response(request)


@router.get("/v1/messages/batches")
async def list_batches(request: Request) -> ORJSONResponse:
    return _stub_response(request)


@router.get("/v1/messages/batches/{batch_id}")
async def get_batch(batch_id: str, request: Request) -> ORJSONResponse:
    return _stub_response(request)


@router.post("/v1/messages/batches/{batch_id}/cancel")
async def cancel_batch(batch_id: str, request: Request) -> ORJSONResponse:
    return _stub_response(request)


@router.get("/v1/messages/batches/{batch_id}/results")
async def get_batch_results(batch_id: str, request: Request) -> ORJSONResponse:
    return _stub_response(request)


@router.delete("/v1/messages/batches/{batch_id}")
async def delete_batch(batch_id: str, request: Request) -> ORJSONResponse:
    return _stub_response(request)


# ── Files API ──────────────────────────────────────────────────────────────────

@router.post("/v1/files")
async def upload_file(request: Request) -> ORJSONResponse:
    return _stub_response(request)


@router.get("/v1/files")
async def list_files(request: Request) -> ORJSONResponse:
    return _stub_response(request)


@router.get("/v1/files/{file_id}")
async def get_file(file_id: str, request: Request) -> ORJSONResponse:
    return _stub_response(request)


@router.delete("/v1/files/{file_id}")
async def delete_file(file_id: str, request: Request) -> ORJSONResponse:
    return _stub_response(request)


@router.get("/v1/files/{file_id}/content")
async def get_file_content(file_id: str, request: Request) -> ORJSONResponse:
    return _stub_response(request)
