from __future__ import annotations

from nvd_claude_proxy.translators.stream_translator import StreamTranslator
from nvd_claude_proxy.translators.tool_translator import ToolIdMap
from nvd_claude_proxy.translators.tool_controller import ToolInvocationController
from nvd_claude_proxy.config.models import CapabilityManifest


def _collect(chunks, tool_schemas=None):
    spec = CapabilityManifest(alias="claude-opus-4-7", nvidia_id="nvidia/big")
    tool_controller = ToolInvocationController(
        spec,
        ToolIdMap(),
        tool_schemas=tool_schemas or {},
    )
    st = StreamTranslator(
        model_name="claude-opus-4-7", tool_id_map=ToolIdMap(), tool_controller=tool_controller
    )
    events = []
    for c in chunks:
        events.extend(st.feed(c))
    events.extend(st.finalize())
    return events


def test_pure_text_stream():
    chunks = [
        {
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
            ]
        },
        {"choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}},
    ]
    events = _collect(chunks)
    names = [e["event"] for e in events]
    assert names[0] == "message_start"
    assert names[-2:] == ["message_delta", "message_stop"]
    assert any(
        e["event"] == "content_block_delta" and e["data"]["delta"]["type"] == "text_delta"
        for e in events
    )


def test_reasoning_then_text_then_tool_call():
    chunks = [
        {
            "choices": [
                {"index": 0, "delta": {"reasoning_content": "Let me think…"}, "finish_reason": None}
            ]
        },
        {
            "choices": [
                {"index": 0, "delta": {"content": "Sure, I'll check."}, "finish_reason": None}
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": ""},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"loc":'}}]},
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"SF"}'}}]},
                    "finish_reason": None,
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 7}},
    ]
    events = _collect(chunks)
    types_opened = [
        e["data"]["content_block"]["type"] for e in events if e["event"] == "content_block_start"
    ]
    assert types_opened == ["thinking", "text", "tool_use"]
    tool_start = next(
        e
        for e in events
        if e["event"] == "content_block_start" and e["data"]["content_block"]["type"] == "tool_use"
    )
    assert tool_start["data"]["content_block"]["name"] == "get_weather"
    mdelta = next(e for e in events if e["event"] == "message_delta")
    assert mdelta["data"]["delta"]["stop_reason"] == "tool_use"


def test_tool_args_before_id_are_buffered():
    """The first chunk for a tool call may be split — id and name arrive
    before all args — and we MUST NOT emit `content_block_start` until both
    id and name are present."""
    chunks = [
        # id+name arrive with empty args
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_x",
                                "type": "function",
                                "function": {"name": "do_thing", "arguments": ""},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        # args fragment
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"a":1}'}}]},
                    "finish_reason": None,
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}},
    ]
    events = _collect(chunks)
    starts = [e for e in events if e["event"] == "content_block_start"]
    assert len(starts) == 1
    assert starts[0]["data"]["content_block"]["name"] == "do_thing"
    deltas = [
        e
        for e in events
        if e["event"] == "content_block_delta" and e["data"]["delta"]["type"] == "input_json_delta"
    ]
    joined = "".join(d["data"]["delta"]["partial_json"] for d in deltas)
    assert joined == '{"a":1}'


def test_parallel_tool_calls_serialized_into_blocks():
    chunks = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            },
                            {
                                "index": 1,
                                "id": "call_b",
                                "type": "function",
                                "function": {"name": "g", "arguments": "{}"},
                            },
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 4}},
    ]
    events = _collect(chunks)
    names = [
        e["data"]["content_block"]["name"]
        for e in events
        if e["event"] == "content_block_start" and e["data"]["content_block"]["type"] == "tool_use"
    ]
    assert names == ["f", "g"]


def test_thinking_block_emits_signature_before_stop():
    chunks = [
        {
            "choices": [
                {"index": 0, "delta": {"reasoning_content": "reason"}, "finish_reason": None}
            ]
        },
        {"choices": [{"index": 0, "delta": {"content": "answer"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 2}},
    ]
    events = _collect(chunks)
    # Find the signature_delta; it must precede the content_block_stop for the
    # thinking block (index 0).
    sig_idx = next(
        i
        for i, e in enumerate(events)
        if e["event"] == "content_block_delta" and e["data"]["delta"]["type"] == "signature_delta"
    )
    stop_idx = next(
        i
        for i, e in enumerate(events)
        if e["event"] == "content_block_stop" and e["data"]["index"] == 0
    )
    assert sig_idx < stop_idx


def test_declared_skill_tool_is_not_blocked():
    chunks = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_skill",
                                "type": "function",
                                "function": {
                                    "name": "Skill",
                                    "arguments": '{"skill_name":"/vercel:env"}',
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 2}},
    ]
    events = _collect(chunks, tool_schemas={"Skill": {"type": "object", "properties": {}}})
    starts = [
        e
        for e in events
        if e["event"] == "content_block_start" and e["data"]["content_block"]["type"] == "tool_use"
    ]
    assert len(starts) == 1
    assert starts[0]["data"]["content_block"]["name"] == "Skill"
    assert not any(
        e["event"] == "content_block_delta"
        and e["data"]["delta"]["type"] == "text_delta"
        and "PROXY BLOCKED undeclared tool" in e["data"]["delta"]["text"]
        for e in events
    )


def test_undeclared_tool_is_blocked():
    chunks = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_ghost",
                                "type": "function",
                                "function": {"name": "GhostTool", "arguments": '{"x":1}'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 2}},
    ]
    events = _collect(chunks, tool_schemas={"Read": {"type": "object", "properties": {}}})
    assert any(
        e["event"] == "content_block_delta"
        and e["data"]["delta"]["type"] == "text_delta"
        and "PROXY BLOCKED undeclared tool 'GhostTool'" in e["data"]["delta"]["text"]
        for e in events
    )


def test_interleaved_parallel_tool_calls():
    chunks = [
        # Tool A starts
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "type": "function",
                                "function": {"name": "f", "arguments": ""},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        # Tool B starts before Tool A finishes
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 1,
                                "id": "call_b",
                                "type": "function",
                                "function": {"name": "g", "arguments": ""},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        # Tool A gets an arg fragment
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"x": 1}'}}]},
                    "finish_reason": None,
                }
            ]
        },
        # Tool B gets an arg fragment
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 1, "function": {"arguments": '{"y": 2}'}}]},
                    "finish_reason": None,
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 4}},
    ]
    events = _collect(chunks)

    # Assert proper contiguous ordering for Anthropic:
    # message_start -> content_block_start(0) -> delta(0) -> stop(0)
    # -> content_block_start(1) -> delta(1) -> stop(1) -> message_stop

    # Filter to block events
    blocks = [
        e
        for e in events
        if e["event"] in ("content_block_start", "content_block_delta", "content_block_stop")
    ]

    # Assert contiguous delivery per block
    # We expect exact order: start(0), delta(0), stop(0), start(1), delta(1), stop(1)
    types_and_indices = [(b["event"], b["data"]["index"]) for b in blocks]
    expected = [
        ("content_block_start", 0),
        ("content_block_delta", 0),
        ("content_block_stop", 0),
        ("content_block_start", 1),
        ("content_block_delta", 1),
        ("content_block_stop", 1),
    ]
    assert types_and_indices == expected
