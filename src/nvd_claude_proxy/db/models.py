from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


class Session(Base):
    """Persistent session record for clients using sk-ncp-* keys."""

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    api_key: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    friendly_name: Mapped[str | None] = mapped_column(String(255))
    model_alias: Mapped[str | None] = mapped_column(String(255))
    
    # JSON serialized configurations
    transformer_settings_json: Mapped[str | None] = mapped_column(Text)
    tool_id_map_json: Mapped[str | None] = mapped_column(Text)
    
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    
    last_active: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ModelMapping(Base):
    """Custom model mappings defined via the web dashboard."""

    __tablename__ = "model_mappings"

    id: Mapped[int] = mapped_column(primary_key=True)
    anthropic_model: Mapped[str] = mapped_column(String(255), index=True)
    nvd_model: Mapped[str] = mapped_column(String(255))
    capability_overrides_json: Mapped[str | None] = mapped_column(Text)


class TransformerToggle(Base):
    """Per-session or global transformer configuration."""

    __tablename__ = "transformer_toggles"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int | None] = mapped_column(Integer, index=True)
    transformer_name: Mapped[str] = mapped_column(String(255))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
