from __future__ import annotations

from nvd_claude_proxy.translators.response_translator import translate_response
from nvd_claude_proxy.translators.tool_translator import ToolIdMap


def test_plain_text_response():
    openai_resp = {
        "choices": [
            {"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
    }
    out = translate_response(openai_resp, "claude-opus-4-7", ToolIdMap())
    assert out["type"] == "message"
    assert out["stop_reason"] == "end_turn"
    assert out["content"][0] == {"type": "text", "text": "Hello!"}
    assert out["usage"]["input_tokens"] == 4
    assert out["usage"]["output_tokens"] == 2


def test_reasoning_content_becomes_thinking_block():
    openai_resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "The answer is 42.",
                    "reasoning_content": "First, compute…",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 6},
    }
    out = translate_response(openai_resp, "claude-opus-4-7", ToolIdMap())
    types = [b["type"] for b in out["content"]]
    assert types == ["thinking", "text"]
    assert out["content"][0]["thinking"] == "First, compute…"
    assert out["content"][0]["signature"].startswith("proxy-")


def test_inline_think_tag_extraction():
    openai_resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "<think>hmm</think>Answer.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {},
    }
    out = translate_response(openai_resp, "claude-opus-4-7", ToolIdMap())
    types = [b["type"] for b in out["content"]]
    assert types == ["thinking", "text"]
    assert out["content"][0]["thinking"] == "hmm"
    assert out["content"][1]["text"] == "Answer."


def test_tool_calls_become_tool_use_blocks():
    openai_resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": '{"q": "x"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {},
    }
    tim = ToolIdMap()
    out = translate_response(openai_resp, "claude-opus-4-7", tim)
    assert out["stop_reason"] == "tool_use"
    tu = next(b for b in out["content"] if b["type"] == "tool_use")
    assert tu["name"] == "lookup"
    assert tu["input"] == {"q": "x"}
    # id should have been mapped through tool_id_map
    assert tim.anthropic_to_openai(tu["id"]) == "call_1"


def test_malformed_tool_args_fallback():
    openai_resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "x", "arguments": "not json"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {},
    }
    out = translate_response(openai_resp, "claude-opus-4-7", ToolIdMap())
    tu = next(b for b in out["content"] if b["type"] == "tool_use")
    assert tu["input"] == {"_raw_arguments": "not json"}
