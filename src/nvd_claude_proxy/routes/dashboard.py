from __future__ import annotations

import json
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.database import get_db
from ..db.models import ModelMapping, Session, TransformerToggle
from ..clients.nvidia_client import NvidiaClient

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])
_log = structlog.get_logger("nvd_claude_proxy.dashboard")


class FriendlyNameUpdate(BaseModel):
    friendly_name: str


class ModelMappingUpdate(BaseModel):
    anthropic_model: str
    nvd_model: str
    capability_overrides: dict[str, Any] | None = None


class TransformerToggleUpdate(BaseModel):
    session_id: int | None = None
    transformer_name: str
    enabled: bool


@router.get("/sessions")
async def list_sessions(db: Annotated[AsyncSession, Depends(get_db)]):
    """List all sessions with stats."""
    result = await db.execute(select(Session).order_by(Session.last_active.desc()))
    sessions = result.scalars().all()
    return sessions


@router.post("/sessions/{api_key}/friendly_name")
async def update_friendly_name(
    api_key: str, data: FriendlyNameUpdate, db: Annotated[AsyncSession, Depends(get_db)]
):
    """Update friendly name for a session."""
    stmt = (
        update(Session).where(Session.api_key == api_key).values(friendly_name=data.friendly_name)
    )
    await db.execute(stmt)
    await db.commit()
    return {"status": "ok"}


@router.get("/models")
async def list_models(request: Request, db: Annotated[AsyncSession, Depends(get_db)]):
    """List current model mappings and available NVIDIA NIM models."""
    # 1. Get current static mappings from registry.
    registry = request.app.state.model_registry
    static_mappings = {alias: spec.nvidia_id for alias, spec in registry.specs.items()}

    # 2. Get dynamic mappings from DB.
    result = await db.execute(select(ModelMapping))
    dynamic_mappings = result.scalars().all()

    # 3. Get available models from NVIDIA.
    client: NvidiaClient = request.app.state.nvidia_client
    try:
        resp = await client.list_models()
        nvidia_models = resp.json().get("data", []) if resp.status_code == 200 else []
    except Exception as exc:
        _log.error("dashboard.list_nvidia_models_failed", error=str(exc))
        nvidia_models = []

    return {
        "static_mappings": static_mappings,
        "dynamic_mappings": dynamic_mappings,
        "available_nvidia_models": nvidia_models,
    }


@router.post("/models/map")
async def update_model_mapping(
    data: ModelMappingUpdate, db: Annotated[AsyncSession, Depends(get_db)]
):
    """Update model aliases (dynamic mappings)."""
    # Check if mapping already exists.
    stmt = select(ModelMapping).where(ModelMapping.anthropic_model == data.anthropic_model)
    result = await db.execute(stmt)
    existing = result.scalar_one_or_none()

    overrides_json = json.dumps(data.capability_overrides) if data.capability_overrides else None

    if existing:
        existing.nvd_model = data.nvd_model
        existing.capability_overrides_json = overrides_json
    else:
        new_mapping = ModelMapping(
            anthropic_model=data.anthropic_model,
            nvd_model=data.nvd_model,
            capability_overrides_json=overrides_json,
        )
        db.add(new_mapping)

    await db.commit()
    return {"status": "ok"}


@router.get("/transformers")
async def get_transformers(db: Annotated[AsyncSession, Depends(get_db)]):
    """Get current toggle states (global and per-session)."""
    result = await db.execute(select(TransformerToggle))
    toggles = result.scalars().all()
    return toggles


@router.post("/transformers/toggle")
async def toggle_transformer(
    data: TransformerToggleUpdate, db: Annotated[AsyncSession, Depends(get_db)]
):
    """Enable/disable transformers globally or per session."""
    stmt = select(TransformerToggle).where(
        TransformerToggle.session_id == data.session_id,
        TransformerToggle.transformer_name == data.transformer_name,
    )
    result = await db.execute(stmt)
    existing = result.scalar_one_or_none()

    if existing:
        existing.enabled = data.enabled
    else:
        new_toggle = TransformerToggle(
            session_id=data.session_id, transformer_name=data.transformer_name, enabled=data.enabled
        )
        db.add(new_toggle)

    await db.commit()
    return {"status": "ok"}
