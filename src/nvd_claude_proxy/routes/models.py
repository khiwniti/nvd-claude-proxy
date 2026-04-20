from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import ORJSONResponse

from ..util.anthropic_headers import new_request_id, standard_response_headers

router = APIRouter()

# Epoch timestamp used as a stable `created_at` for all proxy-served models.
# Anthropic SDKs use this field for display only; it need not be real.
_PROXY_EPOCH = 1_700_000_000


def _model_dict(alias: str, spec) -> dict:
    """Anthropic-spec model object. Claude Code reads `id`, `type`,
    `display_name`, and capability hints from this."""
    return {
        # Anthropic SDK model object fields (Messages API spec)
        "id": alias,
        "type": "model",
        "display_name": _display_name(alias),
        "created_at": _PROXY_EPOCH,
        # OpenAI-compat aliases (older SDK versions)
        "object": "model",
        "created": _PROXY_EPOCH,
        "owned_by": "nvd-claude-proxy",
        # Proxy-specific metadata
        "nvidia_id": spec.nvidia_id,
        "capabilities": {
            "tools": spec.supports_tools,
            "vision": spec.supports_vision,
            "reasoning": spec.supports_reasoning,
            "reasoning_style": spec.reasoning_style,
            "max_context": spec.max_context,
            "max_output": spec.max_output,
        },
    }


def _display_name(alias: str) -> str:
    """Derive a human-readable name from the alias."""
    return alias.replace("-", " ").replace("_", " ").title()


@router.get("/v1/models")
async def list_models(request: Request) -> ORJSONResponse:
    """List configured Claude→NVIDIA aliases."""
    registry = request.app.state.model_registry
    data = [_model_dict(a, s) for a, s in registry.specs.items()]
    rid = new_request_id()
    return ORJSONResponse(
        {"object": "list", "data": data, "has_more": False},
        headers=standard_response_headers(rid),
    )


@router.get("/v1/models/{model_id:path}")
async def get_model(model_id: str, request: Request) -> ORJSONResponse:
    """Anthropic single-model lookup. Resolves via the same alias+prefix logic
    as /v1/messages so legacy names like `claude-3-5-sonnet-latest` work."""
    registry = request.app.state.model_registry
    rid = new_request_id()
    headers = standard_response_headers(rid)
    if model_id in registry.specs:
        return ORJSONResponse(
            _model_dict(model_id, registry.specs[model_id]), headers=headers
        )
    try:
        spec = registry.resolve(model_id)
    except Exception:
        raise HTTPException(
            404,
            detail={
                "type": "error",
                "error": {
                    "type": "not_found_error",
                    "message": f"unknown model: {model_id}",
                },
            },
        )
    # Echo the *alias* the client asked for to preserve round-trip.
    body = _model_dict(model_id, spec)
    return ORJSONResponse(body, headers=headers)
