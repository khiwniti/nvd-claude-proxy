from __future__ import annotations


def test_exact_alias(model_registry):
    spec = model_registry.resolve("claude-opus-4-7")
    # Match the actual config value
    assert spec.nvidia_id == "nvidia/nemotron-3-ultra-500b-a50b"


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
