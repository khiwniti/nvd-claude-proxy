"""Regression tests for the partial-tag holdback scanner.

Ensures that ``<think>`` / ``</think>`` tags split across chunk boundaries
are reassembled into well-formed thinking blocks instead of leaking the
fragments (e.g. "<th") as visible text_delta events.
"""

from __future__ import annotations

from nvd_claude_proxy.translators.stream_translator import StreamTranslator
from nvd_claude_proxy.translators.tool_controller import ToolInvocationController
from nvd_claude_proxy.translators.tool_translator import ToolIdMap
from nvd_claude_proxy.config.models import CapabilityManifest


def _new_translator() -> StreamTranslator:
    spec = CapabilityManifest(alias="claude-opus-4-7", nvidia_id="nvidia/big")
    return StreamTranslator(
        model_name="claude-opus-4-7",
        tool_id_map=ToolIdMap(),
        tool_controller=ToolInvocationController(spec, ToolIdMap(), tool_schemas={}),
    )


def _content_chunks(*texts: str) -> list[dict]:
    out: list[dict] = []
    for t in texts:
        out.append(
            {"choices": [{"index": 0, "delta": {"content": t}, "finish_reason": None}]}
        )
    out.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    return out


def _emitted_text_deltas(events: list[dict]) -> list[str]:
    return [
        e["data"]["delta"]["text"]
        for e in events
        if e["event"] == "content_block_delta" and e["data"]["delta"]["type"] == "text_delta"
    ]


def _emitted_thinking_deltas(events: list[dict]) -> list[str]:
    return [
        e["data"]["delta"]["thinking"]
        for e in events
        if e["event"] == "content_block_delta"
        and e["data"]["delta"]["type"] == "thinking_delta"
    ]


def _opened_block_types(events: list[dict]) -> list[str]:
    return [
        e["data"]["content_block"]["type"]
        for e in events
        if e["event"] == "content_block_start"
    ]


def _drive(translator: StreamTranslator, chunks: list[dict]) -> list[dict]:
    out: list[dict] = []
    for c in chunks:
        out.extend(translator.feed(c))
    out.extend(translator.finalize())
    return out


def test_open_tag_split_across_two_chunks_no_text_leak():
    """`"<th"` then `"ink>hello"` should produce a thinking block, not text."""
    st = _new_translator()
    events = _drive(st, _content_chunks("<th", "ink>hello", "</think>"))

    text_deltas = _emitted_text_deltas(events)
    thinking_deltas = _emitted_thinking_deltas(events)

    # No fragment of the opening tag may leak as visible text.
    assert all("<th" not in d for d in text_deltas)
    assert all("ink>" not in d for d in text_deltas)
    # The thinking content was reassembled correctly.
    assert "".join(thinking_deltas) == "hello"


def test_open_tag_split_across_seven_one_byte_chunks():
    """The most adversarial split: each char of `<think>` arrives separately."""
    st = _new_translator()
    chunks = _content_chunks("<", "t", "h", "i", "n", "k", ">", "body", "</think>")
    events = _drive(st, chunks)

    text_deltas = _emitted_text_deltas(events)
    thinking_deltas = _emitted_thinking_deltas(events)

    assert text_deltas == [] or all(d == "" for d in text_deltas)
    assert "".join(thinking_deltas) == "body"
    assert "thinking" in _opened_block_types(events)


def test_close_tag_split_across_chunks_no_thinking_leak():
    """`"</thi"` then `"nk>tail"` should close the thinking block cleanly."""
    st = _new_translator()
    events = _drive(st, _content_chunks("<think>secret", "</thi", "nk>tail"))

    thinking_deltas = _emitted_thinking_deltas(events)
    text_deltas = _emitted_text_deltas(events)

    # The closing-tag prefix must never surface as thinking content.
    assert all("</thi" not in d for d in thinking_deltas)
    assert "".join(thinking_deltas) == "secret"
    assert "tail" in "".join(text_deltas)


def test_open_tag_at_eof_flushes_as_plain_text():
    """A stranded `<th` at end-of-stream must flush as text, never a tag."""
    st = _new_translator()
    events = _drive(st, _content_chunks("hello <th"))

    text_deltas = _emitted_text_deltas(events)
    full_text = "".join(text_deltas)

    assert full_text == "hello <th"
    # No thinking block may have been opened.
    assert "thinking" not in _opened_block_types(events)


def test_no_tags_zero_overhead():
    """Plain text with no tags still streams without holdback artefacts."""
    st = _new_translator()
    events = _drive(st, _content_chunks("Hello, ", "world!"))

    assert "".join(_emitted_text_deltas(events)) == "Hello, world!"
    assert _emitted_thinking_deltas(events) == []


def test_reasoning_then_text_via_content_implicitly_closes_thinking():
    """A thinking block opened by ``reasoning_content`` must close when
    ``delta.content`` text starts arriving — they are mutually exclusive
    upstream channels.
    """
    st = _new_translator()
    chunks = [
        {"choices": [{"index": 0, "delta": {"reasoning_content": "musing"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    events = _drive(st, chunks)

    opened = _opened_block_types(events)
    assert opened == ["thinking", "text"]
    assert "".join(_emitted_text_deltas(events)) == "Hello"
