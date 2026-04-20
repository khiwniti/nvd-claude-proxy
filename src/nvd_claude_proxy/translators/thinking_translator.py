"""Per-model reasoning-toggle injection and prior-turn reasoning stripping."""
from __future__ import annotations

import re

from ..config.models import CapabilityManifest

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def inject_reasoning_toggle(
    openai_messages: list[dict],
    spec: CapabilityManifest,
    thinking: dict | bool | None,
) -> list[dict]:
    """Prepend/adjust a system message to toggle reasoning per NVIDIA model.

    NVIDIA model cards are explicit: do NOT combine the toggle message with
    additional instructions. We therefore *prepend* a dedicated system message.
    """
    # Use the canonical nested field (spec.reasoning.style), not the legacy
    # spec.supports_reasoning / spec.reasoning_style which are not populated
    # by the loader and always carry their zero-value defaults.
    style = spec.reasoning.style
    if style in ("none", "always-on"):
        return openai_messages

    is_enabled = thinking is not None and thinking is not False
    effort = "high"
    if isinstance(thinking, dict):
        # Prioritize explicit 'effort' string if present, otherwise look at budget
        effort = thinking.get("effort") or thinking.get("budget_tokens") or "high"

    if style == "detailed-thinking-v1":
        # effort mapping: high -> detailed thinking on, max/xhigh or large budget -> extensive thinking on
        mode = "detailed thinking"
        is_extensive = False
        if isinstance(effort, str) and effort.lower() in ("max", "xhigh"):
            is_extensive = True
        elif isinstance(effort, (int, float)) and effort >= 8000:
            is_extensive = True

        if is_extensive:
            mode = "extensive thinking"

        directive = f"{mode} on" if is_enabled else f"{mode} off"
        return [{"role": "system", "content": directive}, *openai_messages]

    if style == "slash-think":
        directive = "/think" if is_enabled else "/no_think"
        return [{"role": "system", "content": directive}, *openai_messages]

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

        content = m.get("content")
        if isinstance(content, str):
            cleaned = _THINK_RE.sub("", content).lstrip()
            out.append({**m, "content": cleaned})
        elif isinstance(content, list):
            new_content = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    txt = block.get("text", "")
                    cleaned = _THINK_RE.sub("", txt).lstrip()
                    new_content.append({**block, "text": cleaned})
                else:
                    new_content.append(block)
            out.append({**m, "content": new_content})
        else:
            out.append(m)
    return out
