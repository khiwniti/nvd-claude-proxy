"""Per-model reasoning-toggle injection and prior-turn reasoning stripping."""
from __future__ import annotations

import re

from ..config.models import ModelSpec

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def inject_reasoning_toggle(
    openai_messages: list[dict],
    spec: ModelSpec,
    thinking_requested: bool,
) -> list[dict]:
    """Prepend/adjust a system message to toggle reasoning per NVIDIA model.

    NVIDIA model cards are explicit: do NOT combine the toggle message with
    additional instructions. We therefore *prepend* a dedicated system message.
    """
    if not spec.supports_reasoning or spec.reasoning_style in ("none", "always-on"):
        return openai_messages

    if spec.reasoning_style == "detailed-thinking-v1":
        directive = "detailed thinking on" if thinking_requested else "detailed thinking off"
        return [{"role": "system", "content": directive}, *openai_messages]

    if spec.reasoning_style == "slash-think":
        directive = "/think" if thinking_requested else "/no_think"
        return [{"role": "system", "content": directive}, *openai_messages]

    # qwen-kwargs is handled via an `extra_body`-style request param in the
    # request translator; no message injection here.
    return openai_messages


def strip_prior_thinking_from_history(openai_messages: list[dict]) -> list[dict]:
    """Remove `<think>…</think>` from all prior assistant text content.

    Qwen3 and Nemotron degrade if prior reasoning is replayed in history. We
    only strip from historical assistant messages.
    """
    out: list[dict] = []
    for m in openai_messages:
        if m.get("role") != "assistant":
            out.append(m)
            continue
        c = m.get("content")
        if isinstance(c, str):
            cleaned = _THINK_RE.sub("", c).lstrip()
            out.append({**m, "content": cleaned})
        else:
            out.append(m)
    return out
