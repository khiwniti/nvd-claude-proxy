from __future__ import annotations


def test_exact_alias(model_registry):
    spec = model_registry.resolve("claude-opus-4-7")
    assert spec.nvidia_id == "qwen/qwen3-coder-480b-a35b-instruct"
    assert spec.supports_tools is True
    assert spec.reasoning.style == "qwen-kwargs"


def test_sonnet_alias_uses_verified_nemotron_super(model_registry):
    spec = model_registry.resolve("claude-sonnet-4-6")
    assert spec.nvidia_id == "nvidia/llama-3.3-nemotron-super-49b-v1.5"
    assert spec.supports_tools is True
    assert spec.reasoning.style == "detailed-thinking-v1"


def test_haiku_alias_uses_verified_nemotron_nano(model_registry):
    spec = model_registry.resolve("claude-haiku-4-5")
    assert spec.nvidia_id == "nvidia/nemotron-3-nano-30b-a3b"
    assert spec.supports_tools is True
    assert spec.reasoning.style == "slash-think"


def test_qwen3_coder_fallback_alias(model_registry):
    spec = model_registry.resolve("claude-qwen3-coder")
    assert spec.nvidia_id == "qwen/qwen3-coder-480b-a35b-instruct"
    assert spec.supports_tools is True
    assert spec.reasoning.style == "qwen-kwargs"


def test_prefix_fallback(model_registry):
    spec = model_registry.resolve("claude-3-5-sonnet-20240620")
    assert spec.alias == "claude-sonnet-4-6"


def test_unknown_model_falls_back_to_default_big(model_registry):
    spec = model_registry.resolve("totally-unknown-model")
    assert spec.alias == model_registry.default_big


def test_longest_prefix_wins(model_registry):
    # `claude-3-5-sonnet` is a longer, more specific match than any other.
    spec = model_registry.resolve("claude-3-5-sonnet-20250101")
    assert spec.alias == "claude-sonnet-4-6"
