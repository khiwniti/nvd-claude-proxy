from __future__ import annotations

from nvd_claude_proxy.translators.tool_translator import (
    ToolIdMap,
    anthropic_tool_choice_to_openai,
    anthropic_tools_to_openai,
)


def test_tools_mapped():
    out = anthropic_tools_to_openai(
        [
            {
                "name": "get_weather",
                "description": "d",
                "input_schema": {"type": "object"},
            }
        ]
    )
    assert out == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "d",
                "parameters": {"type": "object"},
            },
        }
    ]


def test_server_tools_dropped():
    out = anthropic_tools_to_openai(
        [
            {"type": "web_search_20250305", "name": "web_search"},
            {"type": "bash_20250124", "name": "bash"},
            {"type": "computer_20250728", "name": "computer"},
            {"type": "memory_20250818", "name": "memory"},
            {"name": "ok", "input_schema": {}},
        ]
    )
    names = [t["function"]["name"] for t in out]
    assert names == ["ok"]


def test_dated_unknown_types_dropped_as_server_tools():
    """Future server-tool release Anthropic hasn't announced yet."""
    out = anthropic_tools_to_openai(
        [
            {"type": "future_thing_20260501", "name": "future"},
            {"name": "keep_me", "input_schema": {}},
        ]
    )
    assert [t["function"]["name"] for t in out] == ["keep_me"]


def test_custom_and_function_types_passthrough():
    """MCP beta tools have `type: "custom"`; function-type tools should also
    pass through unchanged (the converter just forwards them as OpenAI functions)."""
    out = anthropic_tools_to_openai(
        [
            {"type": "custom", "name": "mcp_tool", "input_schema": {}},
            {"type": "function", "name": "plain", "input_schema": {}},
            {"name": "notype", "input_schema": {}},
        ]
    )
    assert [t["function"]["name"] for t in out] == ["mcp_tool", "plain", "notype"]


def test_tool_name_sanitization():
    out = anthropic_tools_to_openai(
        [{"name": "dotted.name.tool", "input_schema": {}}]
    )
    assert out[0]["function"]["name"] == "dotted_name_tool"


def test_description_cap_applied():
    out = anthropic_tools_to_openai(
        [{"name": "x", "description": "x" * 1000, "input_schema": {}}],
        description_cap=200,
    )
    assert len(out[0]["function"]["description"]) <= 201  # + possible ellipsis


def test_max_tools_truncation():
    big = [{"name": f"t{i}", "input_schema": {}} for i in range(10)]
    out = anthropic_tools_to_openai(big, max_tools=4)
    assert len(out) == 4


def test_tool_choice_mapping():
    assert anthropic_tool_choice_to_openai({"type": "auto"}) == "auto"
    assert anthropic_tool_choice_to_openai({"type": "any"}) == "required"
    assert anthropic_tool_choice_to_openai({"type": "none"}) == "none"
    assert anthropic_tool_choice_to_openai({"type": "tool", "name": "x"}) == {
        "type": "function",
        "function": {"name": "x"},
    }
    assert anthropic_tool_choice_to_openai(None) is None


def test_id_map_roundtrip():
    m = ToolIdMap()
    m.register_anthropic("toolu_abc")
    assert m.anthropic_to_openai("toolu_abc") == "toolu_abc"
    # Unknown OpenAI id → invented toolu_
    a = m.openai_to_anthropic("call_zzz")
    assert a.startswith("toolu_")
    assert m.anthropic_to_openai(a) == "call_zzz"
