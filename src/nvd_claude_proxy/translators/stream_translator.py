"""The critical streaming state machine.

Anthropic requires a strictly-ordered sequence:

  message_start
    (content_block_start index=0)
      (content_block_delta × N)
    content_block_stop index=0
    (content_block_start index=1)
      …
    content_block_stop index=N-1
  message_delta   (stop_reason, usage)
  message_stop

Each content block is exactly ONE of: text | thinking | tool_use.

OpenAI by contrast emits a flat stream of chunks with any mix of:
  delta.content           (string) → text
  delta.reasoning_content (string) → thinking
  delta.tool_calls[i]     (list)   → tool_use (one block per `index`)

Rules this state machine enforces:
  • At most one text/thinking block open at a time. When the stream switches
    modalities (text→tool / reasoning→text / text→reasoning) we close the
    current block before opening the next.
  • Tool-call blocks are keyed by OpenAI's `tool_calls[].index`. The FIRST
    chunk for a given index carries {id, function.name}; subsequent chunks
    carry only `function.arguments` fragments.
  • We MUST delay emitting content_block_start for a tool_use until we have
    both id and name, otherwise Claude Code cannot match the eventual
    tool_result back to it.
  • Usage from the final empty-choices chunk populates message_delta.
  • Mid-stream OpenAI errors become an Anthropic `error` SSE event.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Literal

import re

from ..util.ids import new_message_id, new_thinking_signature
from .tool_translator import ToolIdMap

_FENCE_RE = re.compile(r"^```[a-z]*\s*", re.MULTILINE)

# Known hallucinated tools from training data mismatch - these do not exist
# in the real Claude Code tool registry and cause infinite loops if allowed
_HALLUCINATED_TOOLS: frozenset[str] = frozenset({"Skill", "Read", "migrate", "status"})


def _clean_tool_args(raw: str) -> str:
    """Strip markdown fences and leading prose from NIM tool-call arguments.

    Some NIM model variants wrap JSON in markdown code fences or prefix it with
    prose like "Here are the parameters:". We normalise to a bare JSON string
    so that the Anthropic SDK's JSON.parse() succeeds.
    """
    s = raw.strip()
    # Strip ```json\n ... ``` or ``` ... ```
    if s.startswith("```"):
        s = _FENCE_RE.sub("", s, count=1).rstrip("`").strip()
    # Strip leading non-JSON prose up to the first '{' or '['
    first = min(
        (s.find("{") if "{" in s else len(s)),
        (s.find("[") if "[" in s else len(s)),
    )
    if first > 0:
        s = s[first:]
    return s

BlockType = Literal["text", "thinking", "tool_use"]

_FINISH_TO_STOP: dict[str, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}


@dataclass(slots=True)
class _ToolBuf:
    """Accumulator for a single streamed OpenAI tool call."""

    openai_id: str | None = None
    name: str | None = None
    anthropic_id: str | None = None
    anth_index: int | None = None
    started: bool = False
    closed: bool = False
    # args received before id+name arrived (flushed into full_args at start)
    pre_start_buffer: str = ""
    # ALL argument fragments accumulated post-start; cleaned and emitted at close
    full_args: str = ""


@dataclass
class StreamTranslator:
    model_name: str
    tool_id_map: ToolIdMap
    # Optional cap on reasoning tokens (from `thinking.budget_tokens`).
    # When set, we track reasoning chars (chars/4 ≈ tokens) and force-close the
    # thinking block once the budget is consumed so the model pivots to text.
    budget_tokens: int | None = None
    # Pre-estimated input tokens emitted in message_start so SDK cost tracking works.
    estimated_input_tokens: int = 0

    _message_id: str = field(default_factory=new_message_id)
    _started: bool = False
    _next_index: int = 0
    _open_block_type: BlockType | None = None
    _open_block_index: int | None = None
    _open_thinking_signature_sent: bool = False
    _tools_by_openai_idx: dict[int, _ToolBuf] = field(default_factory=dict)

    _stop_reason: str = "end_turn"
    _usage_input: int = 0
    _usage_output: int = 0
    _finished: bool = False
    _thinking_chars: int = 0      # accumulated reasoning chars for budget tracking
    _thinking_budget_hit: bool = False  # True once budget exhausted

    # Repetition detection for infinite loop prevention
    _last_tool_names: list[str | None] = field(default_factory=lambda: [None] * 5)
    _repetition_count: int = 0
    _MAX_REPETITIONS: int = 3  # Force stop after seeing same tool pattern N times

    # ── helpers ───────────────────────────────────────────────────────────

    def _emit(self, event: str, data: dict) -> dict:
        return {"event": event, "data": data}

    def _ensure_started(self) -> Iterator[dict]:
        if self._started:
            return
        self._started = True
        yield self._emit(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": self._message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self.model_name,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": self.estimated_input_tokens, "output_tokens": 0},
                },
            },
        )

    def _close_open_text_or_thinking(self) -> Iterator[dict]:
        if (
            self._open_block_type in ("text", "thinking")
            and self._open_block_index is not None
        ):
            if (
                self._open_block_type == "thinking"
                and not self._open_thinking_signature_sent
            ):
                yield self._emit(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self._open_block_index,
                        "delta": {
                            "type": "signature_delta",
                            "signature": new_thinking_signature(),
                        },
                    },
                )
                self._open_thinking_signature_sent = True
            yield self._emit(
                "content_block_stop",
                {"type": "content_block_stop", "index": self._open_block_index},
            )
            self._open_block_type = None
            self._open_block_index = None
            self._open_thinking_signature_sent = False

    def _close_open_tool_use(self) -> Iterator[dict]:
        if (
            self._open_block_type == "tool_use"
            and self._open_block_index is not None
        ):
            # Emit all accumulated arguments as a single clean delta before stop.
            for buf in self._tools_by_openai_idx.values():
                if buf.anth_index == self._open_block_index and not buf.closed:
                    if buf.full_args:
                        clean = _clean_tool_args(buf.full_args)
                        if clean:
                            yield self._emit(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": buf.anth_index,
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": clean,
                                    },
                                },
                            )
                    break
            yield self._emit(
                "content_block_stop",
                {"type": "content_block_stop", "index": self._open_block_index},
            )
            # Mark matching tool buf as closed.
            for buf in self._tools_by_openai_idx.values():
                if buf.anth_index == self._open_block_index:
                    buf.closed = True
                    break
            self._open_block_type = None
            self._open_block_index = None

    def _close_any_open(self) -> Iterator[dict]:
        yield from self._close_open_text_or_thinking()
        yield from self._close_open_tool_use()

    def _open_text_block(self) -> Iterator[dict]:
        idx = self._next_index
        self._next_index += 1
        self._open_block_type = "text"
        self._open_block_index = idx
        yield self._emit(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": idx,
                "content_block": {"type": "text", "text": ""},
            },
        )

    def _open_thinking_block(self) -> Iterator[dict]:
        idx = self._next_index
        self._next_index += 1
        self._open_block_type = "thinking"
        self._open_block_index = idx
        self._open_thinking_signature_sent = False
        yield self._emit(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": idx,
                "content_block": {
                    "type": "thinking",
                    "thinking": "",
                    "signature": "",
                },
            },
        )

    # ── public API ────────────────────────────────────────────────────────

    def feed(self, openai_chunk: dict) -> Iterator[dict]:
        """Consume one OpenAI chunk and yield zero or more Anthropic events."""
        yield from self._ensure_started()

        # Trailing usage-only chunk (`choices == []`).
        if not openai_chunk.get("choices"):
            if (usage := openai_chunk.get("usage")):
                self._usage_input = usage.get("prompt_tokens", self._usage_input)
                self._usage_output = usage.get(
                    "completion_tokens", self._usage_output
                )
            return

        choice = openai_chunk["choices"][0]
        delta = choice.get("delta") or {}
        finish = choice.get("finish_reason")

        reasoning = delta.get("reasoning_content")
        text = delta.get("content")
        tool_calls = delta.get("tool_calls") or []

        # 1) Reasoning chunk.
        if reasoning and not self._thinking_budget_hit:
            if self._open_block_type != "thinking":
                yield from self._close_any_open()
                yield from self._open_thinking_block()
            # Budget enforcement: chars / 4 ≈ tokens (conservative).
            if self.budget_tokens is not None:
                remaining_chars = max(0, self.budget_tokens * 4 - self._thinking_chars)
                if remaining_chars <= 0:
                    # Budget exhausted — close block and suppress future reasoning.
                    self._thinking_budget_hit = True
                    yield from self._close_open_text_or_thinking()
                    reasoning = ""
                elif len(reasoning) > remaining_chars:
                    reasoning = reasoning[:remaining_chars]
                    self._thinking_chars += len(reasoning)
                    yield self._emit(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": self._open_block_index,
                            "delta": {"type": "thinking_delta", "thinking": reasoning},
                        },
                    )
                    self._thinking_budget_hit = True
                    yield from self._close_open_text_or_thinking()
                    reasoning = ""
                else:
                    self._thinking_chars += len(reasoning)
            if reasoning:
                yield self._emit(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self._open_block_index,
                        "delta": {"type": "thinking_delta", "thinking": reasoning},
                    },
                )

        # 2) Text chunk.
        if text:
            if self._open_block_type != "text":
                yield from self._close_any_open()
                yield from self._open_text_block()
            yield self._emit(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": self._open_block_index,
                    "delta": {"type": "text_delta", "text": text},
                },
            )

        # 3) Tool-call chunks.
        for tc in tool_calls:
            o_idx = tc.get("index", 0)
            buf = self._tools_by_openai_idx.setdefault(o_idx, _ToolBuf())
            fn = tc.get("function") or {}
            if (_id := tc.get("id")):
                buf.openai_id = _id
            if (nm := fn.get("name")):
                buf.name = nm
            new_args = fn.get("arguments") or ""

            # DEFENSIVE: Detect hallucinated tools that don't exist in registry.
            # These occur when the model confuses training data with actual tool
            # schemas, leading to infinite loops with fake "Skill"/"Read" calls.
            if buf.name in _HALLUCINATED_TOOLS:
                # Convert hallucinated tool to a text warning and skip processing
                if self._open_block_type != "text":
                    yield from self._close_any_open()
                    yield from self._open_text_block()
                warning = (
                    f"\n[PROXY BLOCKED hallucinated tool '{buf.name}' "
                    f"with args: {new_args or buf.pre_start_buffer}]\n"
                )
                yield self._emit(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self._open_block_index,
                        "delta": {"type": "text_delta", "text": warning},
                    },
                )
                # Mark as processed to prevent re-emission
                buf.started = True
                buf.closed = True
                continue

            # Can we open this block yet?
            if not buf.started and buf.openai_id and buf.name:
                # Close any currently-open block (including a different tool_use).
                if (
                    self._open_block_type == "tool_use"
                    and self._open_block_index is not None
                    and self._open_block_index != buf.anth_index
                ):
                    yield from self._close_open_tool_use()
                else:
                    yield from self._close_open_text_or_thinking()

                buf.anthropic_id = self.tool_id_map.openai_to_anthropic(buf.openai_id)
                buf.anth_index = self._next_index
                self._next_index += 1
                self._open_block_type = "tool_use"
                self._open_block_index = buf.anth_index
                buf.started = True
                # Restore the original name so Claude Code can match it.
                emit_name = self.tool_id_map.original_tool_name(buf.name or "")
                yield self._emit(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": buf.anth_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": buf.anthropic_id,
                            "name": emit_name,
                            "input": {},
                        },
                    },
                )
                # Flush pre-start buffer + new fragment into full_args.
                # (All args are emitted as a single clean delta at close time.)
                buf.full_args = buf.pre_start_buffer + new_args
                buf.pre_start_buffer = ""
            elif buf.started and new_args:
                # If a different tool is currently open, switch to this one.
                if (
                    self._open_block_type == "tool_use"
                    and self._open_block_index != buf.anth_index
                ):
                    yield from self._close_open_tool_use()
                    self._open_block_type = "tool_use"
                    self._open_block_index = buf.anth_index
                # Accumulate — do not emit yet; cleaned + emitted at close.
                buf.full_args += new_args
            elif not buf.started and new_args:
                # id/name haven't arrived yet — keep buffering.
                buf.pre_start_buffer += new_args

            # Track tool names for repetition detection
            if buf.name and buf.started:
                self._last_tool_names.pop(0)
                self._last_tool_names.append(buf.name)
                # Check for repeating pattern (e.g., Skill -> Read -> Skill -> Read)
                unique_tools = set(self._last_tool_names)
                if len(unique_tools) <= 2 and self._last_tool_names.count(buf.name) >= 4:
                    self._repetition_count += 1
                    if self._repetition_count >= self._MAX_REPETITIONS:
                        # Force stop the stream - we're in a loop
                        yield from self._close_any_open()
                        yield self._emit(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": self._open_block_index if self._open_block_index is not None else 0,
                                "delta": {
                                    "type": "text_delta",
                                    "text": "\n[PROXY ERROR: Detected infinite tool loop - forcing termination]\n",
                                },
                            },
                        )
                        self._stop_reason = "refusal"
                        self._finished = True
                        return

        # 4) Finish reason.
        if finish:
            yield from self._close_any_open()
            self._stop_reason = _FINISH_TO_STOP.get(finish, "end_turn")

    def finalize(self) -> Iterator[dict]:
        """Emit terminal `message_delta` + `message_stop`. Call once."""
        if self._finished:
            return
        self._finished = True
        # Defensive: close anything still open.
        yield from self._close_any_open()
        yield self._emit(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": self._stop_reason,
                    "stop_sequence": None,
                },
                "usage": {"output_tokens": self._usage_output},
            },
        )
        yield self._emit("message_stop", {"type": "message_stop"})
