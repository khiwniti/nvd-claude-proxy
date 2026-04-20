"""Non-streaming OpenAI response → Anthropic Messages response."""
from __future__ import annotations

import json
import re

from ..util.ids import new_message_id, new_thinking_signature
from .tool_translator import ToolIdMap

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

_FINISH_TO_STOP: dict[str | None, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
    None: "end_turn",
}


def _extract_thinking(
    content: str | None, reasoning: str | None
) -> tuple[str | None, str]:
    """Return `(thinking_text, remaining_content)`.

    Accepts both `reasoning_content` and inline `<think>…</think>` since different
    NIM versions surface reasoning differently for the same model.
    """
    if reasoning:
        return reasoning, content or ""
    if content and "<think>" in content:
        m = _THINK_RE.search(content)
        if m:
            return m.group(1).strip(), _THINK_RE.sub("", content, count=1).lstrip()
    return None, content or ""


def translate_response(
    openai_resp: dict, anthropic_model: str, tool_id_map: ToolIdMap
) -> dict:
    choices = openai_resp.get("choices") or [{}]
    choice = choices[0]
    msg = choice.get("message") or {}
    content_blocks: list[dict] = []

    thinking_text, remaining = _extract_thinking(
        msg.get("content"),
        msg.get("reasoning_content"),
    )
    if thinking_text:
        content_blocks.append(
            {
                "type": "thinking",
                "thinking": thinking_text,
                "signature": new_thinking_signature(),
            }
        )
    if remaining:
        content_blocks.append({"type": "text", "text": remaining})

    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        raw_args = fn.get("arguments") or "{}"
        try:
            tool_input = json.loads(raw_args)
        except json.JSONDecodeError:
            tool_input = {"_raw_arguments": raw_args}
        anth_id = tool_id_map.openai_to_anthropic(tc.get("id", ""))
        sanitized_name = fn.get("name", "")
        original_name = tool_id_map.original_tool_name(sanitized_name)
        content_blocks.append(
            {
                "type": "tool_use",
                "id": anth_id,
                "name": original_name,
                "input": tool_input,
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
