"""Anthropic Messages request → NVIDIA (OpenAI) chat.completions request."""

from __future__ import annotations

import json as _json
from typing import Any

import structlog

from ..config.models import CapabilityManifest
from ..util.pdf_extractor import document_block_to_text
from ..util.tokens import approximate_tokens

_log = structlog.get_logger("nvd_claude_proxy.translator")
from .thinking_translator import (
    inject_reasoning_toggle,
    strip_prior_thinking_from_history,
)
from .tool_translator import (
    ToolIdMap,
    anthropic_tool_choice_to_openai,
    anthropic_tools_to_openai,
)
from .vision_translator import anthropic_image_to_openai
from .transformers import TransformerChain

# Leave this many tokens of headroom between estimated input+output and the
# model's hard context window. 8k is enough to absorb cl100k vs Nemotron 
# tokenization drift while maximizing usable space in 128k windows.
_CONTEXT_HEADROOM = 8192
# Minimum we will ever allow for output even under heavy clamping so the
# model has room for at least a brief answer.
_MIN_OUTPUT = 256


class ContextOverflowError(ValueError):
    """Raised when estimated input tokens already exceed the model's context window.
    The caller should convert this into a proper Anthropic 400 response rather
    than forwarding the request to NVIDIA (which will always fail with a 400).
    """

    def __init__(self, est_input: int, max_context: int, model: str) -> None:
        self.est_input = est_input
        self.max_context = max_context
        self.model = model
        super().__init__(
            f"Estimated input ({est_input} tokens) exceeds "
            f"{model} context window ({max_context} tokens). "
            "Reduce messages or tool schemas."
        )


def _truncate_messages_to_fit(
    messages: list[dict],
    tools: list[dict],
    max_context: int,
    min_output: int,
) -> tuple[list[dict], int]:
    """Drop oldest non-system turns until the input estimate fits the window.

    Strategy: skip over any leading system messages, then remove one message at
    a time from the oldest end.  Stops as soon as the estimate is below the
    threshold or only one non-system message remains (can't truncate further).

    Returns (truncated_messages, new_est_input).
    """
    threshold = max_context - min_output

    # Index of the first non-system message.
    first_non_system = 0
    while first_non_system < len(messages) and messages[first_non_system].get("role") == "system":
        first_non_system += 1

    msgs = list(messages)
    while len(msgs) > first_non_system + 1:
        est = approximate_tokens({"messages": msgs, "tools": tools})
        if est < threshold:
            return msgs, est
        # Drop the oldest non-system message.
        msgs = msgs[:first_non_system] + msgs[first_non_system + 1:]

    return msgs, approximate_tokens({"messages": msgs, "tools": tools})


_TOOL_DISCIPLINE_ADDENDUM = """\n\n---\nTool use discipline (IMPORTANT):\n- Only call a tool when you are certain it is the correct tool for the task.\n- ALWAYS use the native tool-calling API. NEVER output tags like `command-name>` or `command-arguments>` in your response.\n- When calling the `Skill` tool, use the EXACT skill name shown in the tool description (e.g. \"/vercel:env\", not \"vercel\"). Do not guess skill names.\n- Provide ALL required parameters for every tool call. Check the tool schema before calling.\n- If you are unsure which tool to use, ask the user for clarification instead of guessing.\n- Do not call design or UI tools (e.g. `pencil`) for non-design tasks such as file migration or code editing.\n---"""


def _inject_tool_discipline(messages: list[dict]) -> None:
    """Append tool-discipline guidance to the system message in-place.

    When the tool catalog is large (>50 tools), Nemotron tends to hallucinate
    tool names and call meta-tools with wrong parameters.  A short addendum in
    the system turn is the most token-efficient nudge.
    """
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = (messages[0]["content"] or "") + _TOOL_DISCIPLINE_ADDENDUM
    else:
        messages.insert(0, {"role": "system", "content": _TOOL_DISCIPLINE_ADDENDUM.strip()})


def _flatten_system(system: Any, spec: CapabilityManifest) -> str | list[dict] | None:
    """Anthropic `system` may be a string OR a list of blocks.

    If the model supports vision and the system prompt is a list containing
    images, we preserve them as a multimodal list. Otherwise, we flatten to text.
    """
    if system is None:
        return None
    if isinstance(system, str):
        return system or None

    has_images = any(isinstance(b, dict) and b.get("type") == "image" for b in system)

    if has_images and spec.supports_vision:
        # Preserve as multimodal blocks
        parts: list[dict] = []
        for b in system:
            if not isinstance(b, dict):
                continue
            btype = b.get("type")
            if btype == "text":
                parts.append({"type": "text", "text": b.get("text", "")})
            elif btype == "image":
                parts.append(anthropic_image_to_openai(b))
        return parts if parts else None

    # Fallback to flattening everything to text
    txt_parts: list[str] = []
    for b in system:
        if isinstance(b, dict) and b.get("type") == "text":
            txt = b.get("text")
            if txt:
                txt_parts.append(txt)
    return "\n\n".join(txt_parts) if txt_parts else None


def _anthropic_message_to_openai(
    msg: dict, tool_id_map: ToolIdMap, spec: CapabilityManifest
) -> list[dict]:
    """One Anthropic message may explode into multiple OpenAI messages because
    `tool_result` blocks become separate `role:"tool"` messages."""
    role = msg["role"]
    content = msg.get("content")

    if isinstance(content, str):
        return [{"role": role, "content": content}]

    text_parts: list[dict] = []
    tool_calls: list[dict] = []
    tool_result_messages: list[dict] = []

    for block in content or []:
        btype = block.get("type")
        if btype == "text":
            text_parts.append({"type": "text", "text": block.get("text", "")})
        elif btype == "image":
            text_parts.append(anthropic_image_to_openai(block))
        elif btype in ("thinking", "redacted_thinking"):
            # NVIDIA has no way to consume an opaque Anthropic signature; drop.
            continue
        elif btype == "tool_use":
            tid = block["id"]
            tool_id_map.register_anthropic(tid)
            tool_calls.append(
                {
                    "id": tid,
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": _json.dumps(block.get("input", {}), ensure_ascii=False),
                    },
                }
            )
        elif btype == "tool_result":
            tid = block["tool_use_id"]
            openai_id = tool_id_map.anthropic_to_openai(tid)
            raw = block.get("content", "")

            # Vision passthrough for tool results
            if isinstance(raw, list) and spec.supports_vision:
                content_blocks: list[dict] = []
                for sub in raw:
                    if not isinstance(sub, dict):
                        continue
                    stype = sub.get("type")
                    if stype == "text":
                        content_blocks.append({"type": "text", "text": sub.get("text", "")})
                    elif stype == "image":
                        content_blocks.append(anthropic_image_to_openai(sub))
                raw = content_blocks
            elif isinstance(raw, list):
                # Flatten multi-block tool_result content to text.
                # OpenAI role:"tool" doesn't accept multimodal content, so image
                # blocks are converted to a text placeholder preserving intent.
                flat: list[str] = []
                for sub in raw:
                    if not isinstance(sub, dict):
                        continue
                    stype = sub.get("type")
                    if stype == "text":
                        flat.append(sub.get("text", ""))
                    elif stype == "image":
                        src = sub.get("source") or {}
                        media = src.get("media_type", "image")
                        if src.get("type") == "base64":
                            flat.append(f"[image/{media}: {len(src.get('data', ''))} chars base64]")
                        elif src.get("type") == "url":
                            flat.append(f"[image url: {src.get('url', '')}]")
                        else:
                            flat.append("[image]")
                raw = "\n".join(flat)

            tool_result_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": openai_id,
                    "content": raw,
                }
            )
        elif btype == "document":
            text = document_block_to_text(block)
            if text:
                text_parts.append({"type": "text", "text": text})
        # Other block types (server_tool_*, search_result, container_upload) are dropped.

    out: list[dict] = []
    if role == "assistant":
        m: dict[str, Any] = {"role": "assistant"}
        if text_parts:
            if any(p["type"] != "text" for p in text_parts):
                m["content"] = text_parts
            else:
                m["content"] = "".join(p["text"] for p in text_parts)
        else:
            m["content"] = None
        if tool_calls:
            m["tool_calls"] = tool_calls
        out.append(m)
    else:  # "user" — tool_result blocks live here
        if text_parts:
            if any(p["type"] != "text" for p in text_parts):
                out.append({"role": "user", "content": text_parts})
            else:
                out.append({"role": "user", "content": "".join(p["text"] for p in text_parts)})
        out.extend(tool_result_messages)
    return out


def translate_request(
    anthropic_body: dict,
    spec: CapabilityManifest,
    tool_id_map: ToolIdMap,
    transformer_chain: TransformerChain | None = None,
) -> dict:
    openai_messages: list[dict] = []
    system_content = _flatten_system(anthropic_body.get("system"), spec)
    if system_content:
        openai_messages.append({"role": "system", "content": system_content})
    for m in anthropic_body.get("messages") or []:
        openai_messages.extend(_anthropic_message_to_openai(m, tool_id_map, spec))

    # Prior-turn reasoning cleanup (only affects assistant text already in history).
    openai_messages = strip_prior_thinking_from_history(openai_messages)

    # High-tool-count injection: guide the model to use tools precisely.
    if len(anthropic_body.get("tools") or []) > 50:
        _inject_tool_discipline(openai_messages)

    # Reasoning toggle (Nemotron v1 / v1.5 / Nemotron 3 family).
    thinking = anthropic_body.get("thinking")
    openai_messages = inject_reasoning_toggle(openai_messages, spec, thinking)

    requested_max = int(anthropic_body.get("max_tokens") or spec.max_output)

    # Build the tool payload up front so we can include it in the input-size
    # estimate — Claude Code's `/init` sends ~190 tool schemas that dominate
    # the prompt budget, and missing them caused spurious 400s from NVIDIA.
    # When a lot of tools are present, tighten per-description limits to keep
    # the prompt under the model's context window.
    mapped_tools: list[dict] = []
    _ALLOWED_META_TOOLS = {"bash", "computer", "browser", "memory"}
    if (tools := anthropic_body.get("tools")) and spec.tools.supports:
        tool_count = len(tools)
        if tool_count > 100:
            desc_cap = 160
        elif tool_count > 40:
            desc_cap = 280
        else:
            desc_cap = 480

        filtered_tools = []
        for t in tools:
            # Drop hallucinated meta-tools if they sneak into the input
            if t.get("name") in _ALLOWED_META_TOOLS:
                continue
            filtered_tools.append(t)

        mapped_tools = anthropic_tools_to_openai(
            filtered_tools, tool_id_map=tool_id_map, description_cap=desc_cap
        )

    # NVIDIA rejects the request if `max_tokens + input_tokens > max_context`.
    # Estimate with cl100k_base over *everything* that will be billed as input:
    # messages (system+user+assistant+tool) and tool schemas.
    est_input = approximate_tokens({"messages": openai_messages, "tools": mapped_tools})
    # Pre-flight guard: if the input alone fills the window, try to salvage the
    # request by dropping oldest turns before hard-failing.
    if est_input >= spec.max_context - _MIN_OUTPUT:
        original_est = est_input
        openai_messages, est_input = _truncate_messages_to_fit(
            openai_messages, mapped_tools, spec.max_context, _MIN_OUTPUT
        )
        if est_input >= spec.max_context - _MIN_OUTPUT:
            raise ContextOverflowError(est_input, spec.max_context, spec.alias)
        _log.warning(
            "context.truncated",
            before_tokens=original_est,
            after_tokens=est_input,
            model=spec.alias,
        )

    context_budget = max(
        _MIN_OUTPUT,
        spec.max_context - est_input - _CONTEXT_HEADROOM,
    )
    effective_max = max(
        _MIN_OUTPUT,
        min(requested_max, spec.max_output, context_budget),
    )

    payload: dict[str, Any] = {
        "model": spec.nvidia_id,
        "messages": openai_messages,
        "max_tokens": effective_max,
        "stream": bool(anthropic_body.get("stream", False)),
    }

    # Sampling params.
    temp = anthropic_body.get("temperature")
    if spec.temperature_override is not None:
        payload["temperature"] = spec.temperature_override
    elif temp is not None:
        payload["temperature"] = temp
    if (p := anthropic_body.get("top_p")) is not None:
        payload["top_p"] = p
    if (k := anthropic_body.get("top_k")) is not None:
        # OpenAI REST doesn't standardize top_k; vLLM/NIM accepts it.
        payload["top_k"] = k
    if (ss := anthropic_body.get("stop_sequences")) is not None:
        payload["stop"] = ss

    # Tools.
    if mapped_tools:
        payload["tools"] = mapped_tools
        if (tc := anthropic_body.get("tool_choice")) is not None:
            payload["tool_choice"] = anthropic_tool_choice_to_openai(tc)
            # Anthropic allows callers to disable parallel tool calls; OpenAI/NIM
            # supports this via `parallel_tool_calls: false`.
            if isinstance(tc, dict) and tc.get("disable_parallel_tool_use"):
                payload["parallel_tool_calls"] = False

    # Qwen3-style thinking kwarg (bypasses system-msg toggle).
    if spec.reasoning_style == "qwen-kwargs":
        payload["chat_template_kwargs"] = {
            "enable_thinking": thinking is not None and thinking is not False
        }

    if transformer_chain:
        payload = transformer_chain.transform_request(payload)

    return payload
