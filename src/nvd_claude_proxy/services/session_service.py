from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable

import structlog

from ..db.database import async_session_factory
from ..db.models import Session
from ..translators.tool_translator import ToolIdMap
from ..translators.transformers import TransformerChain

if TYPE_CHECKING:
    from ..config.models import CapabilityManifest

_log = structlog.get_logger("nvd_claude_proxy.session_service")


class SessionService:
    @staticmethod
    def get_isolated_tool_id_map(session_obj: Session | None) -> ToolIdMap:
        """Load ToolIdMap from session, or create fresh if missing."""
        if not session_obj or not session_obj.tool_id_map_json:
            return ToolIdMap()

        try:
            data = json.loads(session_obj.tool_id_map_json)
            return ToolIdMap.from_dict(data)
        except Exception:
            _log.exception(
                "session.tool_id_map_load_failed", session_id=getattr(session_obj, "id", "none")
            )
            return ToolIdMap()

    @staticmethod
    def get_isolated_transformer_chain(
        session_obj: Session | None,
        spec: CapabilityManifest,
        build_default_fn: Callable[
            [CapabilityManifest, Callable[[str, Any], None] | None], TransformerChain
        ],
        on_fix: Callable[[str, Any], None] | None = None,
    ) -> TransformerChain:
        """Load TransformerChain from session, or use default factory if missing."""
        if not session_obj or not session_obj.transformer_settings_json:
            return build_default_fn(spec, on_fix)

        try:
            data = json.loads(session_obj.transformer_settings_json)
            return TransformerChain.from_dict(data, on_fix=on_fix)
        except Exception:
            _log.exception(
                "session.transformer_chain_load_failed",
                session_id=getattr(session_obj, "id", "none"),
            )
            return build_default_fn(spec, on_fix)

    @staticmethod
    async def save_session_state(
        session_id: int,
        tool_id_map: ToolIdMap,
        transformer_chain: TransformerChain,
        tokens_inc: int = 0,
    ) -> None:
        """Persist serialized state back to SQLite."""
        async with async_session_factory() as db_session:
            session = await db_session.get(Session, session_id)
            if session:
                session.tool_id_map_json = json.dumps(tool_id_map.to_dict())
                session.transformer_settings_json = json.dumps(transformer_chain.to_dict())
                session.tokens_used += tokens_inc
                await db_session.commit()
                _log.debug("session.state_saved", session_id=session_id, tokens_inc=tokens_inc)
