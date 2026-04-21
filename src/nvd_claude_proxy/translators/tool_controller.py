"""Stateful controller for managing parallel tool invocation and result collection."""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from nvd_claude_proxy.config.models import CapabilityManifest
from nvd_claude_proxy.translators.tool_translator import ToolIdMap

logger = structlog.get_logger(__name__)

# jsonschema is an optional dependency. When absent, schema validation is
# skipped gracefully (arg_validation flag in ToolConfig has no effect).
try:
    from jsonschema import Draft7Validator, ValidationError as _SchemaValidationError

    _HAS_JSONSCHEMA = True
except ImportError:
    _HAS_JSONSCHEMA = False


class ToolInvocationController:
    """Manages tool dispatch and deterministic result collection.

    In proxy mode this controller is used primarily for deterministic schema
    validation of tool arguments returned by the upstream model. The actual
    tool execution happens on the client side (Claude Code); the proxy does
    not call tools directly.

    To enable arg validation pass ``tool_schemas`` at construction time:

        schemas = {t["name"]: t.get("input_schema", {}) for t in body_tools}
        controller = ToolInvocationController(spec, tool_id_map, tool_schemas=schemas)
    """

    def __init__(
        self,
        spec: CapabilityManifest,
        tool_id_map: ToolIdMap,
        tool_schemas: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.spec = spec
        self.tool_id_map = tool_id_map
        # Maps sanitized tool name → Anthropic input_schema dict.
        self._tool_schemas: dict[str, dict[str, Any]] = tool_schemas or {}
        self.active_invocations: dict[str, asyncio.Task] = {}
        self.results: dict[str, Any] = {}
        # Pre-compile validators for every schema when jsonschema is available.
        self._validators: dict[str, Any] = {}
        if _HAS_JSONSCHEMA and self.spec.tools.arg_validation:
            for name, schema in self._tool_schemas.items():
                try:
                    self._validators[name] = Draft7Validator(schema)
                except Exception:  # noqa: BLE001
                    pass  # Malformed schema — skip; model will return what it returns.

    # ── validation ────────────────────────────────────────────────────────

    def validate_schema(self, name: str, args: dict[str, Any]) -> bool:
        """Return True if *args* conform to the tool's declared input_schema.

        Always returns True when:
          • jsonschema is not installed, OR
          • spec.tools.arg_validation is False, OR
          • no schema was registered for *name*.

        On validation failure the error is logged at WARNING level so operators
        can diagnose misbehaving models without crashing the request.
        """
        if not self.spec.tools.arg_validation or not _HAS_JSONSCHEMA:
            return True
        validator = self._validators.get(name)
        if validator is None:
            return True
        try:
            validator.validate(args)
            return True
        except _SchemaValidationError as exc:
            logger.warning(
                "tool.schema_validation_failed",
                tool_name=name,
                error=exc.message,
                path=list(exc.absolute_path),
            )
            return False

    def validate_all(self, tool_uses: list[dict[str, Any]]) -> list[str]:
        """Validate a list of tool_use blocks; return list of failing tool names."""
        failing: list[str] = []
        for tu in tool_uses:
            name = tu.get("name", "")
            # Resolve sanitized → original name for schema lookup.
            original = self.tool_id_map.original_tool_name(name)
            args = tu.get("input") or {}
            if not self.validate_schema(original, args) and not self.validate_schema(name, args):
                failing.append(name)
        return failing

    def has_tool_schema(self, name: str) -> bool:
        """Return True when a tool name exists in the declared request schema map."""
        return name in self._tool_schemas

    def has_registered_schemas(self) -> bool:
        """Return True when request tool schemas were provided."""
        return bool(self._tool_schemas)

    # ── parallel dispatch (proxy-local, future use) ───────────────────────

    async def invoke_parallel(self, calls: list[dict[str, Any]]) -> None:
        """Dispatch multiple tool calls in parallel if the model spec allows it."""
        if not self.spec.tools.parallel:
            for call in calls:
                await self._dispatch(call)
        else:
            tasks = [asyncio.create_task(self._dispatch(call)) for call in calls]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _dispatch(self, call: dict[str, Any]) -> None:
        """Placeholder for local tool execution. Not used in proxy mode."""
        pass
