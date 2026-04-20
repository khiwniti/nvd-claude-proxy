from __future__ import annotations

from nvd_claude_proxy.translators.schema_sanitizer import (
    sanitize_input_schema,
    sanitize_tool_name,
    truncate_description,
)


def test_valid_names_unchanged():
    for ok in ["Read", "Edit", "Bash", "mcp__server__tool", "tool-123", "a"]:
        assert sanitize_tool_name(ok) == ok


def test_invalid_chars_replaced():
    assert sanitize_tool_name("my.server.tool") == "my_server_tool"
    assert sanitize_tool_name("space name") == "space_name"
    assert sanitize_tool_name("weird!@#$name") == "weird_name"


def test_empty_name_fallback():
    assert sanitize_tool_name("") == "tool"
    assert sanitize_tool_name("!!!") == "tool"


def test_long_name_truncated_to_64():
    assert len(sanitize_tool_name("x" * 200)) == 64


def test_input_schema_drops_top_keys():
    raw = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "http://example.com/schema",
        "type": "object",
        "properties": {"a": {"type": "string"}},
    }
    out = sanitize_input_schema(raw)
    assert "$schema" not in out and "$id" not in out
    assert out["type"] == "object"
    assert out["properties"]["a"] == {"type": "string"}


def test_non_object_root_is_wrapped():
    raw = {"type": "string"}
    out = sanitize_input_schema(raw)
    assert out["type"] == "object"
    assert "value" in out["properties"]
    assert out["required"] == ["value"]


def test_additional_properties_false_stripped():
    raw = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"x": {"type": "string"}},
        "required": ["x"],
    }
    out = sanitize_input_schema(raw)
    assert "additionalProperties" not in out
    assert out["required"] == ["x"]


def test_ref_dropped_silently():
    raw = {
        "type": "object",
        "properties": {"a": {"$ref": "#/definitions/Foo"}},
    }
    out = sanitize_input_schema(raw)
    assert out["properties"]["a"] == {}


def test_nested_properties_recursively_cleaned():
    raw = {
        "type": "object",
        "properties": {
            "inner": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"z": {"type": "string", "readOnly": True}},
            }
        },
    }
    out = sanitize_input_schema(raw)
    # readOnly stripped from leaf
    assert out["properties"]["inner"]["properties"]["z"] == {"type": "string"}


def test_truncate_description_preserves_sentence():
    desc = "First sentence. Second sentence. Third sentence."
    out = truncate_description(desc, 25)
    # Should cut at a sentence boundary when reasonable.
    assert out.endswith((" …", "…"))
    assert len(out) <= 30
