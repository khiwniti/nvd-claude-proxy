"""Non-streaming OpenAI response → Anthropic Messages response."""

from __future__ import annotations

import json
import re
from typing import Any

from ..util.ids import new_message_id, new_thinking_signature
from ..util.tool_args_parser import parse_tool_arguments
from .tool_translator import ToolIdMap
from .transformers import TransformerChain

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_FENCE_RE = re.compile(r"^```[a-z]*\s*", re.MULTILINE)


def _extract_tool_args(raw: str) -> dict:
    """Robustly extract a JSON object from a NIM tool-call arguments string.

    NIM model variants sometimes wrap JSON in markdown fences or prefix with
    prose. We try several cleaning strategies before falling back to an empty
    dict (which lets Claude Code show the tool call without crashing on
    schema validation of ``_raw_arguments``).
    """
    s = raw.strip()
    # Strategy 1: direct parse (fast path — well-behaved models).
    try:
        result = json.loads(s)
        return result if isinstance(result, dict) else {"value": result}
    except json.JSONDecodeError:
        pass
    # Strategy 2: strip markdown code fences.
    if s.startswith("```"):
        s2 = _FENCE_RE.sub("", s, count=1).rstrip("`").strip()
        try:
            result = json.loads(s2)
            return result if isinstance(result, dict) else {"value": result}
        except json.JSONDecodeError:
            s = s2
    # Strategy 3: strip leading prose up to the first opening brace.
    brace = s.find("{")
    if brace > 0:
        s3 = s[brace:]
        try:
            result = json.loads(s3)
            return result if isinstance(result, dict) else {"value": result}
        except json.JSONDecodeError:
            pass
    # Strategy 4: scan for a balanced {...} block (handles trailing garbage).
    for start in range(len(s)):
        if s[start] != "{":
            continue
        depth, in_str, esc = 0, False, False
        for end, ch in enumerate(s[start:], start):
            if esc:
                esc = False
                continue
            if ch == "\\" and in_str:
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if not in_str:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            result = json.loads(s[start : end + 1])
                            return result if isinstance(result, dict) else {"value": result}
                        except json.JSONDecodeError:
                            break
        break
    # Strategy 5 (ported from claude-code-router): multi-strategy JSON repair
    # handles truncated JSON, fence stripping, etc.
    repaired = parse_tool_arguments(raw)
    if repaired and repaired != "{}":
        try:
            result = json.loads(repaired)
            return result if isinstance(result, dict) else {"value": result}
        except json.JSONDecodeError:
            pass
    # All strategies failed — return _raw_arguments so Claude Code gets the
    # unparseable string rather than an empty dict.
    return {"_raw_arguments": raw}


_FINISH_TO_STOP: dict[str | None, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
    None: "end_turn",
}


def _extract_thinking(
    content: str | None, reasoning: str | None, thinking_obj: dict | None = None
) -> tuple[str | None, str, str | None]:
    """Return `(thinking_text, remaining_content, signature)`.

    Accepts both `reasoning_content` and inline `<think>…</think>` since different
    NIM versions surface reasoning differently for the same model.
    """
    signature = thinking_obj.get("signature") if thinking_obj else None
    if reasoning:
        return reasoning, content or "", signature
    if thinking_obj and thinking_obj.get("content"):
        return thinking_obj["content"], content or "", signature
    if content:
        m = _THINK_RE.search(content)
        if m:
            return m.group(1).strip(), _THINK_RE.sub("", content, count=1).lstrip(), signature
    return None, content or "", signature


def translate_response(
    openai_resp: dict,
    anthropic_model: str,
    tool_id_map: ToolIdMap,
    tool_controller: Any | None = None,
    transformer_chain: TransformerChain | None = None,
) -> dict:
    if transformer_chain:
        openai_resp = transformer_chain.transform_response(openai_resp)

    choices = openai_resp.get("choices") or [{}]
    choice = choices[0]
    msg = choice.get("message") or {}
    content_blocks: list[dict] = []

    thinking_text, remaining, signature = _extract_thinking(
        msg.get("content"),
        msg.get("reasoning_content"),
        msg.get("thinking"),
    )
    if thinking_text:
        content_blocks.append(
            {
                "type": "thinking",
                "thinking": thinking_text,
                "signature": signature or new_thinking_signature(),
            }
        )
    if remaining:
        content_blocks.append({"type": "text", "text": remaining})

    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        raw_args = fn.get("arguments") or "{}"
        tool_input = _extract_tool_args(raw_args)
        anth_id = tool_id_map.openai_to_anthropic(tc.get("id", ""))
        sanitized_name = fn.get("name", "")
        # Fuzzy resolve hallucinated tool names
        resolved = (
            tool_controller.resolve_tool_name(sanitized_name) if tool_controller else sanitized_name
        )
        name_to_use = resolved or sanitized_name
        original_name = tool_id_map.original_tool_name(name_to_use)

        # Resolve hallucinated argument keys
        final_input = (
            tool_controller.resolve_tool_arguments(original_name, tool_input)
            if tool_controller
            else tool_input
        )

        is_valid = True
        if tool_controller:
            failing = tool_controller.validate_all(
                [{"type": "tool_use", "name": name_to_use, "input": final_input}]
            )
            if failing:
                is_valid = False

        if is_valid:
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": anth_id,
                    "name": original_name,
                    "input": final_input,
                }
            )
        else:
            content_blocks.append(
                {
                    "type": "text",
                    "text": f"\n[PROXY BLOCKED tool '{original_name}' due to invalid args]\n",
                }
            )

    if not content_blocks:
        # Anthropic requires at least one block; emit an empty text block.
        content_blocks.append({"type": "text", "text": ""})

    usage = openai_resp.get("usage") or {}
    return {
        "id": new_message_id(),
        "type": "message",
        "role": "assistant",
        "model": anthropic_model,
        "content": content_blocks,
        "stop_reason": _FINISH_TO_STOP.get(choice.get("finish_reason"), "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }
