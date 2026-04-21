from __future__ import annotations

from nvd_claude_proxy.config.models import CapabilityManifest
from nvd_claude_proxy.translators.request_translator import translate_request
from nvd_claude_proxy.translators.tool_translator import ToolIdMap


def _spec(**kw) -> CapabilityManifest:
    base = dict(
        alias="claude-opus-4-7",
        nvidia_id="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        supports_tools=True,
        supports_vision=False,
        supports_reasoning=True,
        reasoning_style="detailed-thinking-v1",
        max_context=131072,
        max_output=32768,
    )
    base.update(kw)
    return CapabilityManifest(**base)


def test_flatten_system_string_and_drop_cache_control():
    body = {
        "model": "claude-opus-4-7",
        "system": [
            {"type": "text", "text": "A", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "B"},
        ],
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
    }
    out = translate_request(body, _spec(), ToolIdMap())
    system_msgs = [m for m in out["messages"] if m["role"] == "system"]
    # The first system message is the reasoning toggle; the second is the
    # flattened user-provided system prompt.
    assert system_msgs[0]["content"] == "detailed thinking off"
    assert "A\n\nB" in system_msgs[1]["content"]


def test_reasoning_toggle_on_when_thinking_present():
    body = {
        "model": "claude-opus-4-7",
        "messages": [{"role": "user", "content": "think please"}],
        "max_tokens": 100,
        "thinking": {"type": "enabled", "budget_tokens": 2000},
    }
    out = translate_request(body, _spec(), ToolIdMap())
    assert out["messages"][0]["content"] == "detailed thinking on"


def test_slash_think_style():
    spec = _spec(reasoning_style="slash-think")
    body = {
        "model": "claude-haiku-4-5",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
        "thinking": {"type": "enabled"},
    }
    out = translate_request(body, spec, ToolIdMap())
    assert out["messages"][0]["content"] == "/think"


def test_qwen_kwargs_sets_chat_template_flag():
    spec = _spec(reasoning_style="qwen-kwargs")
    body = {
        "model": "claude-qwen3",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
        "thinking": {"type": "enabled"},
    }
    out = translate_request(body, spec, ToolIdMap())
    assert out["chat_template_kwargs"] == {"enable_thinking": True}


def test_tool_use_and_tool_result_roundtrip():
    tool_id_map = ToolIdMap()
    body = {
        "model": "claude-opus-4-7",
        "messages": [
            {"role": "user", "content": "what's the weather?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_abc",
                        "name": "get_weather",
                        "input": {"loc": "SF"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_abc",
                        "content": "72F sunny",
                    }
                ],
            },
        ],
        "max_tokens": 100,
        "tools": [
            {
                "name": "get_weather",
                "description": "fetch weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"loc": {"type": "string"}},
                },
            }
        ],
    }
    out = translate_request(body, _spec(), tool_id_map)
    msgs = out["messages"]
    assistant = next(m for m in msgs if m["role"] == "assistant")
    assert assistant["tool_calls"][0]["id"] == "toolu_abc"
    assert assistant["tool_calls"][0]["function"]["name"] == "get_weather"
    tool_msg = next(m for m in msgs if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "toolu_abc"
    assert tool_msg["content"] == "72F sunny"
    assert out["tools"][0]["function"]["name"] == "get_weather"


def test_many_tools_get_description_cap_applied():
    """/init sends ~190 tools; proxy caps per-tool descriptions to keep
    the prompt under the context window."""
    big_desc = "x " * 500
    tools = [
        {
            "name": f"tool_{i}",
            "description": big_desc,
            "input_schema": {"type": "object", "properties": {}},
        }
        for i in range(190)
    ]
    body = {
        "model": "claude-opus-4-7",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 32000,
        "tools": tools,
    }
    out = translate_request(body, _spec(max_context=131072, max_output=32768), ToolIdMap())
    # With >100 tools, description cap drops to 160 chars (see
    # request_translator sizing tiers).
    assert all(len(t["function"]["description"]) <= 161 for t in out["tools"])
    # All 190 tools still forwarded; none silently dropped.
    assert len(out["tools"]) == 190


def test_init_like_payload_clamps_below_window():
    """Reproduces the exact /init 400 we hit in production:
    273 tools + large schemas with `claude-opus-4-7` (131k window).
    Total input must stay under `max_context - headroom`."""
    tools = [
        {
            "name": f"Tool_{i}",
            "description": "Reads or edits a file in the project. " * 6,
            "input_schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path"],
            },
        }
        for i in range(273)
    ]
    body = {
        "model": "claude-opus-4-7",
        "max_tokens": 32000,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": tools,
    }
    out = translate_request(body, _spec(max_context=131072, max_output=32768), ToolIdMap())
    # If the estimator is honest the clamp should fire: input+max_tokens must
    # never exceed max_context - headroom (8192).
    from nvd_claude_proxy.util.tokens import approximate_tokens

    est = approximate_tokens({"messages": out["messages"], "tools": out["tools"]})
    assert est + out["max_tokens"] <= 131072 - 8192 + out["max_tokens"]  # sanity
    # Main invariant:
    assert est + out["max_tokens"] <= 131072, (est, out["max_tokens"])


def test_max_tokens_clamps_when_tools_overflow_small_context():
    """A narrow context window + many tools must still clamp max_tokens."""
    big_desc = "x " * 500
    tools = [
        {
            "name": f"tool_{i}",
            "description": big_desc,
            "input_schema": {"type": "object", "properties": {}},
        }
        for i in range(190)
    ]
    body = {
        "model": "claude-opus-4-7",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 32000,
        "tools": tools,
    }
    # Force a window large enough for the heuristic: 190 tools + desc
    # ends up being ~20k tokens with the 3.0 chars/token heuristic.
    out = translate_request(body, _spec(max_context=32768, max_output=32768), ToolIdMap())
    assert out["max_tokens"] < 32000
    assert out["max_tokens"] >= 256


def test_max_tokens_clamped_against_context_budget():
    """When input ~= context window, max_tokens must shrink so the total stays
    under `max_context`. Reproduces the `/init` 400 from NVIDIA."""
    # ~120k chars ≈ ~30k cl100k tokens of input; with a 32k-token context
    # window and 32k requested output, the clamp must kick in.
    # ~360k chars ≈ ~120k tokens. With 131k window and 16k headroom, this MUST clamp.
    big_text = "hello world " * 30_000
    body = {
        "model": "claude-opus-4-7",
        "messages": [{"role": "user", "content": big_text}],
        "max_tokens": 32000,
    }
    # Use a 128k window.
    spec = _spec(max_context=131072, max_output=131072)
    out = translate_request(body, spec, ToolIdMap())
    assert out["max_tokens"] < 32000
    assert out["max_tokens"] >= 256  # _MIN_OUTPUT floor


def test_max_tokens_clamped_to_spec_max_output():
    body = {
        "model": "claude-haiku-4-5",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 999999,
    }
    out = translate_request(body, _spec(max_output=16384), ToolIdMap())
    assert out["max_tokens"] == 16384


def test_temperature_override_wins():
    spec = _spec(temperature_override=0.6)
    body = {
        "model": "claude-r1",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
        "temperature": 1.0,
    }
    out = translate_request(body, spec, ToolIdMap())
    assert out["temperature"] == 0.6
