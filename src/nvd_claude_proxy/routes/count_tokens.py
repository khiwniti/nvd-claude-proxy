from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import ORJSONResponse

from ..util.anthropic_headers import new_request_id, standard_response_headers
from ..util.tokens import approximate_tokens

router = APIRouter()


@router.post("/v1/messages/count_tokens")
async def count_tokens(request: Request) -> ORJSONResponse:
    body = await request.json()
    # Include `tools` and `system` in the estimate so the count matches what
    # `/v1/messages` will actually bill as input — matches Anthropic semantics.
    est_body = {
        "messages": body.get("messages") or [],
        "system": body.get("system") or "",
        "tools": body.get("tools") or [],
    }
    n = approximate_tokens(est_body)
    rid = new_request_id()
    return ORJSONResponse({"input_tokens": n}, headers=standard_response_headers(rid))
