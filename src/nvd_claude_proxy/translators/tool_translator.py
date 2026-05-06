"""Tool schema + tool_choice + id-map helpers."""

from __future__ import annotations

import re
from typing import Any

import structlog

from ..util.ids import new_tool_use_id
from .schema_sanitizer import (
    sanitize_input_schema,
    sanitize_tool_name,
    truncate_description,
)

from ..config.server_tools import ServerToolRegistry

_log = structlog.get_logger("nvd_claude_proxy.tools")

_PASSTHROUGH_TOOL_TYPES = {None, "custom", "function"}
_DEFAULT_DESC_CAP = 480

def _is_server_tool(tool: dict, registry: ServerToolRegistry | None) -> bool:
    t = tool.get("type")
    if not isinstance(t, str) or not registry:
        return False
    return registry.is_server_tool(t)


def _inject_server_tool_schema(tool: dict, registry: ServerToolRegistry | None) -> dict:
    """Injects the implicit schema for Anthropic's server tools so NIM models can use them."""
    t_type = str(tool.get("type", ""))
    t_name = tool.get("name", "")

    # Base copy
    out = dict(tool)
    out["type"] = "function"

    if registry:
        spec = registry.get_spec(t_type)
        if spec:
            out["name"] = t_name or spec.family
            out["description"] = f"Anthropic server tool: {spec.family}"
            out["input_schema"] = spec.schema

    return out


def anthropic_tools_to_openai(
    tools: list[dict] | None,
    *,
    tool_id_map: "ToolIdMap | None" = None,
    max_tools: int | None = None,
    description_cap: int = _DEFAULT_DESC_CAP,
    server_tool_registry: ServerToolRegistry | None = None,
) -> list[dict]:
    """Anthropic tool defs → OpenAI function-tool defs.

    - Injects schemas for Anthropic server tools (web_search, bash, computer, …) so they work on NIM.
    - Sanitizes names and JSON schemas for NIM compatibility.
    - Optionally caps total tool count and per-tool description length.
    """
    out: list[dict] = []
    renamed: list[tuple[str, str]] = []
    skipped_nameless = 0
    seen_sanitized: dict[str, str] = {}
    collisions: list[tuple[str, str]] = []

    for t in tools or []:
        if _is_server_tool(t, server_tool_registry):
            # Instead of dropping, we now inject the schema if we have one
            t = _inject_server_tool_schema(t, server_tool_registry)

        ttype = t.get("type")
        if ttype is not None and ttype not in _PASSTHROUGH_TOOL_TYPES:
            # If it's still not a passthrough type (e.g. unknown server tool), skip it.
            _log.debug("tools.unknown_type_skipped", type=ttype, name=t.get("name"))
            continue
        raw_name = t.get("name")
        if not raw_name:
            skipped_nameless += 1
            continue

        name = sanitize_tool_name(raw_name)
        if name != raw_name:
            renamed.append((raw_name, name))
            if tool_id_map is not None:
                tool_id_map.register_tool_rename(raw_name, name)
        # Collision detection: two distinct original names mapping to the same
        # sanitized name would make tool_result matching ambiguous.
        if name in seen_sanitized:
            if seen_sanitized[name] != raw_name:
                collisions.append((raw_name, seen_sanitized[name]))
                continue  # Drop the later duplicate to preserve determinism.
        else:
            seen_sanitized[name] = raw_name
        desc = t.get("description", "") or ""
        if description_cap and len(desc) > description_cap:
            desc = truncate_description(desc, description_cap)
        schema = sanitize_input_schema(
            t.get("input_schema") or {"type": "object", "properties": {}}
        )
        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": schema,
                },
            }
        )

    if max_tools is not None and len(out) > max_tools:
        _log.warning(
            "tools.truncated",
            kept=max_tools,
            dropped=len(out) - max_tools,
        )
        out = out[:max_tools]
    if renamed:
        _log.debug("tools.names_sanitized", renames=renamed[:10], total=len(renamed))
    if skipped_nameless:
        _log.warning("tools.nameless_skipped", count=skipped_nameless)
    if collisions:
        _log.warning(
            "tools.name_collision_dropped",
            collisions=[{"dropped": d, "kept": k} for d, k in collisions],
        )
    return out


def anthropic_tool_choice_to_openai(tc: Any) -> Any:
    if tc is None:
        return None
    if isinstance(tc, str):
        return tc  # pass "auto"/"none" unchanged
    t = tc.get("type")
    if t == "auto":
        return "auto"
    if t == "any":
        return "required"
    if t == "none":
        return "none"
    if t == "tool":
        return {
            "type": "function",
            "function": {"name": sanitize_tool_name(tc.get("name", ""))},
        }
    return "auto"


class ToolIdMap:
    """Bidirectional map between Anthropic `toolu_…` and OpenAI `call_…` ids.

    Anthropic ids survive the whole conversation; OpenAI ids are per-response.
    We preserve the Anthropic id whenever possible by using it as the OpenAI id too.
    """

    def __init__(self) -> None:
        self._a_to_o: dict[str, str] = {}
        self._o_to_a: dict[str, str] = {}
        # Map sanitized-name → original-name so tool_use blocks can emit the
        # name Claude Code originally sent (preserves tool_result matching).
        self._sanitized_to_original: dict[str, str] = {}
        # Track the global order of tool_use IDs encountered in assistant messages.
        self._call_order: list[str] = []

    def record_call_order(self, toolu_id: str) -> None:
        """Record that a tool was called, to preserve ordering in results."""
        if toolu_id not in self._call_order:
            self._call_order.append(toolu_id)

    def get_call_index(self, toolu_id: str) -> int:
        """Return the order index of a tool_use ID, or a large number if unknown."""
        try:
            return self._call_order.index(toolu_id)
        except ValueError:
            return 999999

    def register_anthropic(self, toolu_id: str) -> str:
        self._a_to_o[toolu_id] = toolu_id
        self._o_to_a[toolu_id] = toolu_id
        self.record_call_order(toolu_id)
        return toolu_id

    def openai_to_anthropic(self, openai_id: str) -> str:
        if openai_id in self._o_to_a:
            return self._o_to_a[openai_id]
        a = openai_id if openai_id.startswith("toolu_") else new_tool_use_id()
        self._a_to_o[a] = openai_id
        self._o_to_a[openai_id] = a
        return a

    def anthropic_to_openai(self, toolu_id: str) -> str:
        return self._a_to_o.get(toolu_id, toolu_id)

    def register_tool_rename(self, original: str, sanitized: str) -> None:
        self._sanitized_to_original[sanitized] = original

    def original_tool_name(self, sanitized: str) -> str:
        return self._sanitized_to_original.get(sanitized, sanitized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "a_to_o": self._a_to_o,
            "o_to_a": self._o_to_a,
            "sanitized_to_original": self._sanitized_to_original,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolIdMap:
        instance = cls()
        instance._a_to_o = data.get("a_to_o", {})
        instance._o_to_a = data.get("o_to_a", {})
        instance._sanitized_to_original = data.get("sanitized_to_original", {})
        return instance
