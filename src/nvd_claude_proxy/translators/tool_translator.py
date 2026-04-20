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

_log = structlog.get_logger("nvd_claude_proxy.tools")

# Anthropic server-tool `type` strings end in a date. The full catalogue (as of
# early 2026) — kept explicit rather than just matching `_YYYYMMDD` so we can
# log which specific tool was dropped.
_SERVER_TOOL_TYPES = {
    "web_search_20250305",
    "web_search_20250728",
    "bash_20250124",
    "bash_20250728",
    "computer_20250124",
    "computer_20250728",
    "code_execution_20250522",
    "code_execution_20260120",
    "text_editor_20250124",
    "text_editor_20250728",
    "memory_20250818",
}
# Regex catch-all in case Anthropic releases new dated server tools.
_DATED_TOOL_RE = re.compile(r".+_(20\d{6})$")

# MCP client tools (anthropic-beta: mcp-client-*) arrive with `type: "custom"`
# or no type, plus a normal `input_schema`. They behave like regular function
# tools from the model's POV, so we forward them after sanitization.
_PASSTHROUGH_TOOL_TYPES = {None, "custom", "function"}

# Per-tool description cap when a request carries many tools. Keeps the
# per-tool prompt footprint modest without losing intent. Tools are truncated
# only when the aggregate tool-schema budget is exceeded (see
# `anthropic_tools_to_openai`).
_DEFAULT_DESC_CAP = 480
_TIGHT_DESC_CAP = 200


def _is_server_tool(tool: dict) -> bool:
    t = tool.get("type")
    if not isinstance(t, str):
        return False
    if t in _SERVER_TOOL_TYPES:
        return True
    # Any other dated type we haven't enumerated — conservatively treat as
    # server-tool (Anthropic has never used dated suffixes for user tools).
    return bool(_DATED_TOOL_RE.match(t))


def anthropic_tools_to_openai(
    tools: list[dict] | None,
    *,
    tool_id_map: "ToolIdMap | None" = None,
    max_tools: int | None = None,
    description_cap: int = _DEFAULT_DESC_CAP,
) -> list[dict]:
    """Anthropic tool defs → OpenAI function-tool defs.

    - Drops Anthropic server tools (web_search, bash, computer, …).
    - Sanitizes names and JSON schemas for NIM compatibility.
    - Optionally caps total tool count and per-tool description length.
    """
    out: list[dict] = []
    dropped_server: list[str] = []
    renamed: list[tuple[str, str]] = []
    skipped_nameless = 0

    for t in tools or []:
        if _is_server_tool(t):
            dropped_server.append(t.get("name") or t.get("type") or "?")
            continue
        ttype = t.get("type")
        if ttype is not None and ttype not in _PASSTHROUGH_TOOL_TYPES:
            # Unknown tool-type — fall through but log.
            _log.debug("tools.unknown_type", type=ttype, name=t.get("name"))
        raw_name = t.get("name")
        if not raw_name:
            skipped_nameless += 1
            continue
        name = sanitize_tool_name(raw_name)
        if name != raw_name:
            renamed.append((raw_name, name))
            if tool_id_map is not None:
                tool_id_map.register_tool_rename(raw_name, name)
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
    if dropped_server:
        _log.info("tools.server_tools_dropped", names=dropped_server)
    if renamed:
        _log.debug("tools.names_sanitized", renames=renamed[:10], total=len(renamed))
    if skipped_nameless:
        _log.warning("tools.nameless_skipped", count=skipped_nameless)
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

    def register_anthropic(self, toolu_id: str) -> str:
        self._a_to_o[toolu_id] = toolu_id
        self._o_to_a[toolu_id] = toolu_id
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
