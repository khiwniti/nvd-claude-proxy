from __future__ import annotations

from nvd_claude_proxy.config.models import CapabilityManifest
from nvd_claude_proxy.translators.thinking_translator import (
    inject_reasoning_toggle,
    strip_prior_thinking_from_history,
)


def _spec(style):
    return CapabilityManifest(
        alias="x",
        nvidia_id="x",
        supports_reasoning=True,
        reasoning_style=style,
    )


def test_detailed_thinking_on():
    msgs = [{"role": "user", "content": "hi"}]
    out = inject_reasoning_toggle(msgs, _spec("detailed-thinking-v1"), True)
    assert out[0] == {"role": "system", "content": "detailed thinking on"}


def test_detailed_thinking_off():
    msgs = [{"role": "user", "content": "hi"}]
    out = inject_reasoning_toggle(msgs, _spec("detailed-thinking-v1"), False)
    assert out[0] == {"role": "system", "content": "detailed thinking off"}


def test_slash_think():
    msgs = [{"role": "user", "content": "hi"}]
    out = inject_reasoning_toggle(msgs, _spec("slash-think"), True)
    assert out[0] == {"role": "system", "content": "/think"}


def test_none_style_passthrough():
    spec = CapabilityManifest(alias="x", nvidia_id="x", supports_reasoning=False, reasoning_style="none")
    msgs = [{"role": "user", "content": "hi"}]
    out = inject_reasoning_toggle(msgs, spec, True)
    assert out == msgs


def test_strip_prior_thinking():
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "<think>secret</think>Visible answer."},
    ]
    out = strip_prior_thinking_from_history(msgs)
    assert out[1]["content"] == "Visible answer."
