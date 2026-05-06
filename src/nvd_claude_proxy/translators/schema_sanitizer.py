"""Sanitize Anthropic/Claude-Code tool schemas for NVIDIA NIM compatibility.

NIM-backed vLLM / TensorRT engines have stricter JSON-Schema support than the
OpenAI gateway. The real-world bugs we've seen when forwarding Claude Code's
tool catalog unchanged:

  • Tool names containing `.` or `::` rejected (OpenAI allows `[a-zA-Z0-9_-]{1,64}`).
  • `$schema`, `$id`, `$defs`, `definitions` top-level keys cause parser errors
    on some NIM containers.
  • `$ref` inside tool parameters is resolved differently than Anthropic expects.
  • `additionalProperties: false` combined with `required: [...]` over-constrains
    Nemotron's tool-calling head and produces empty-arg calls.
  • Extremely long `description` fields (>~2k tokens each × 200 tools) blow the
    context window.

This module provides small, well-tested transforms that normalize schemas
without losing semantic meaning.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

# OpenAI function-tool name regex. Claude Code uses names like `mcp__server__tool`
# and `NotebookEdit` — the first is within limits, but dotted names from some MCP
# servers (`my.server.tool`) need mapping.
_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

# JSON-Schema keywords NIM/vLLM parsers reliably reject or ignore. Dropping them
# is safe because they don't change the validation semantics for flat argument
# objects (which is all NIM's tool parser actually consumes).
_DROP_TOP_KEYS = {"$schema", "$id", "$comment", "definitions", "$defs"}

# Per-field draft-2020 keywords that NIM's tool parser ignores anyway.
_DROP_FIELD_KEYS = {"$comment", "readOnly", "writeOnly", "examples"}


def sanitize_tool_name(name: str) -> str:
    """Coerce a tool name to `^[a-zA-Z0-9_-]{1,64}$`.

    Preserves the original name verbatim when it already complies, so that
    Claude Code's `tool_use` ids keep round-tripping cleanly.
    """
    if _NAME_RE.fullmatch(name):
        return name
    # Replace any invalid char with `_`, then collapse runs of `_`.
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")
    cleaned = re.sub(r"_+", "_", cleaned) or "tool"

    # Truncate to 64 chars. If we truncate, add a hash suffix to avoid collisions.
    if len(cleaned) > 64 or cleaned != name:
        if len(cleaned) > 64:
            import hashlib
            import base64
            digest = base64.urlsafe_b64encode(hashlib.sha256(name.encode()).digest())[:8].decode()
            cleaned = cleaned[:55] + "_" + digest

    return cleaned[:64]


def _sanitize_schema_node(node: Any, depth: int = 0) -> Any:
    """Recursively sanitize a JSON-Schema node. Depth-limited to avoid
    pathological self-referential schemas."""
    if depth > 12 or not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for k, v in node.items():
        if k in _DROP_FIELD_KEYS:
            continue
        if k == "$ref":
            # We can't resolve cross-file refs — drop silently. NIM will treat
            # the property as unconstrained `object`, which is acceptable.
            continue
        if k == "properties" and isinstance(v, dict):
            out[k] = {pk: _sanitize_schema_node(pv, depth + 1) for pk, pv in v.items()}
        elif k == "items":
            out[k] = _sanitize_schema_node(v, depth + 1)
        elif k in ("anyOf", "oneOf", "allOf") and isinstance(v, list):
            # NIM-compatible flattening: merge properties/required from all branches.
            merged_props: dict[str, Any] = {}
            merged_required: set[str] = set()
            for branch in v:
                sanitized_branch = _sanitize_schema_node(branch, depth + 1)
                if isinstance(sanitized_branch, dict):
                    if "properties" in sanitized_branch:
                        merged_props.update(sanitized_branch["properties"])
                    if "required" in sanitized_branch:
                        merged_required.update(sanitized_branch["required"])
            
            if merged_props:
                out["type"] = "object"
                out["properties"] = merged_props
                if merged_required:
                    out["required"] = list(merged_required)
                out["additionalProperties"] = True
            else:
                # Fallback: keep original if no properties found
                out[k] = [_sanitize_schema_node(n, depth + 1) for n in v]
        else:
            out[k] = v
    return out


def sanitize_input_schema(schema: Any) -> dict[str, Any]:
    """Produce a NIM-friendly variant of an Anthropic `input_schema`."""
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    s = deepcopy(schema)
    for k in _DROP_TOP_KEYS:
        s.pop(k, None)

    # Guarantee a root `type: "object"`.
    if s.get("type") != "object":
        # Wrap scalar/array schemas inside an object — NIM's tool parser only
        # calls functions whose parameters form a JSON object.
        return {
            "type": "object",
            "properties": {"value": _sanitize_schema_node(s)},
            "required": ["value"],
        }

    # NIM + Nemotron misbehave when `additionalProperties: false` is combined
    # with a large `required` list; drop the flag — tool parsers ignore extras.
    s.pop("additionalProperties", None)
    if "properties" in s and isinstance(s["properties"], dict):
        s["properties"] = {pk: _sanitize_schema_node(pv) for pk, pv in s["properties"].items()}
    return s


def truncate_description(desc: str, max_chars: int) -> str:
    """Shorten an over-long tool description while keeping the first sentence."""
    if len(desc) <= max_chars:
        return desc
    # Prefer cutting at sentence boundary.
    head = desc[:max_chars]
    cut = head.rfind(". ")
    if cut > max_chars // 2:
        return head[: cut + 1] + " …"
    return head.rstrip() + "…"
