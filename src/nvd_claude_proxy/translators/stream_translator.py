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
  • Tool arguments are emitted progressively (input_json_delta) as fragments
    arrive once the JSON object start character '{' or '[' is found. Leading
    prose and markdown fences are discarded.
  • Usage from the final empty-choices chunk populates message_delta.
  • Mid-stream OpenAI errors become an Anthropic `error` SSE event.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator, Literal

from ..util.ids import new_message_id, new_thinking_signature
from .tool_controller import ToolInvocationController
from .tool_translator import ToolIdMap
from .transformers import TransformerChain

_FENCE_RE = re.compile(r"^```[a-z]*\s*", re.MULTILINE)
# Matches first JSON object/array start character.
_JSON_START_RE = re.compile(r"[{\[]")

# Detect hallucinated tag-based tool calling in text deltas.
_TAG_HALLUCINATION_RE = re.compile(r"(command-name>|command-arguments>)", re.IGNORECASE)

# Claude Code internal tools that are always allowed.
# These are NOT in the user's tool schemas but are legitimate Claude Code
# tools that the model may call. We allow them through without validation.
_CLAUDE_CODE_INTERNAL_TOOLS = frozenset({
    # Claude Code core commands
    "Read", "Write", "Edit", "Bash", "NotebookEdit",
    # Skill tool (for invoking agent skills)
    "Skill", "skill",
    # MCP tools
    "mcp__tool",
    # AskUserQuestion is sometimes generated
    "AskUserQuestion",
    # Claude Code internal commands that may appear in responses
    "WebSearch", "WebFetch", "WebRead",
    # Container/Browser tools
    "computer", "bash", "browser",
    # Legacy/alternative names
    "AskUser", "Ask", "ask_user_question",
    # Task/plan tools
    "Task", "Plan", "plan",
    # Any tool starting with common prefixes that might be Claude Code internals
})

def _is_claude_code_internal_tool(name: str) -> bool:
    """Check if a tool name is a known Claude Code internal tool."""
    if not name:
        return False
    # Direct match
    if name in _CLAUDE_CODE_INTERNAL_TOOLS:
        return True
    # Case-insensitive match for common tools
    name_lower = name.lower()
    if name_lower in _CLAUDE_CODE_INTERNAL_TOOLS:
        return True
    # Match by prefix for MCP-style tools
    if name_lower.startswith("mcp__"):
        return True
    # Match by prefix for Skill invocations
    if name_lower.startswith("skill:"):
        return True
    return False

BlockType = Literal["text", "thinking", "tool_use"]

_FINISH_TO_STOP: dict[str, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    # content_filter maps to end_turn — "refusal" is not a valid Anthropic
    # stop_reason and causes SDK deserialization errors.
    "content_filter": "end_turn",
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
    # args received before id+name arrived (flushed into streaming once started)
    pre_start_buffer: str = ""
    # Once the tool block is open, tracks whether we've seen the JSON start
    # character so we can strip leading prose/fences before emitting.
    _json_mode: bool = False
    # Accumulates raw text before the first JSON start char is found.
    _pre_json_acc: str = ""


def _strip_leading_prose(raw: str) -> str:
    """Strip markdown code fences and leading non-JSON text from a tool-args string.

    Returns the substring starting from the first '{' or '[', or the original
    string if it already starts with a JSON character.
    """
    s = raw
    if s.startswith("```"):
        s = _FENCE_RE.sub("", s, count=1).rstrip("`").strip()
    m = _JSON_START_RE.search(s)
    return s[m.start() :] if m else ""


@dataclass
class StreamTranslator:
    model_name: str
    tool_id_map: ToolIdMap
    tool_controller: ToolInvocationController
    # Optional cap on reasoning tokens (from `thinking.budget_tokens`).
    # When set, we track reasoning chars (chars/4 ≈ tokens) and force-close the
    # thinking block once the budget is consumed so the model pivots to text.
    budget_tokens: int | None = None
    # Pre-estimated input tokens emitted in message_start so SDK cost tracking works.
    estimated_input_tokens: int = 0
    transformer_chain: TransformerChain | None = None

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
    _thinking_chars: int = 0  # accumulated reasoning chars for budget tracking
    _thinking_budget_hit: bool = False  # True once budget exhausted

    # Repetition detection for infinite loop prevention — tracks the last N
    # tool names seen. Uses a plain list (not None-padded) so the count
    # check is accurate from the very first tool call.
    _last_tool_names: list[str] = field(default_factory=list)
    _repetition_count: int = 0
    _MAX_REPETITIONS: int = 3  # Force stop after seeing same tool pattern N times

    # ── helpers ───────────────────────────────────────────────────────────

    def _emit(self, event: str, data: dict) -> Iterator[dict]:
        # Pings should always be emitted to keep the connection alive
        if event == "ping":
            yield {"event": event, "data": data}
            return

        if self.transformer_chain:
            data = self.transformer_chain.transform_stream_chunk(data)
            if data is None:
                return
        yield {"event": event, "data": data}

    def _ensure_started(self) -> Iterator[dict]:
        if self._started:
            return
        self._started = True
        yield from self._emit(
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
        if self._open_block_type in ("text", "thinking") and self._open_block_index is not None:
            if self._open_block_type == "thinking" and not self._open_thinking_signature_sent:
                yield from self._emit(
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
            yield from self._emit(
                "content_block_stop",
                {"type": "content_block_stop", "index": self._open_block_index},
            )
            self._open_block_type = None
            self._open_block_index = None
            self._open_thinking_signature_sent = False

    def _close_open_tool_use(self) -> Iterator[dict]:
        """Close the currently open tool_use block.

        With progressive streaming, argument fragments have already been emitted
        as input_json_delta events; we only need to emit content_block_stop.
        """
        if self._open_block_type == "tool_use" and self._open_block_index is not None:
            yield from self._emit(
                "content_block_stop",
                {"type": "content_block_stop", "index": self._open_block_index},
            )
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
        yield from self._emit(
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
        yield from self._emit(
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

    def _emit_tool_args_fragment(self, buf: _ToolBuf, fragment: str) -> Iterator[dict]:
        """Progressively emit one tool-argument fragment as input_json_delta.

        Leading prose and markdown fences are discarded before the first JSON
        start character '{' or '[' is encountered. Once JSON mode is active,
        all subsequent fragments bypass the scanner and are emitted directly.
        """
        if not fragment:
            return
        if buf._json_mode:
            yield from self._emit(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": buf.anth_index,
                    "delta": {"type": "input_json_delta", "partial_json": fragment},
                },
            )
            return
        # Not yet in JSON mode — accumulate and scan for the JSON start.
        combined = buf._pre_json_acc + fragment
        if combined.startswith("```"):
            combined = _FENCE_RE.sub("", combined, count=1).rstrip("`").strip()
        m = _JSON_START_RE.search(combined)
        if m:
            buf._json_mode = True
            buf._pre_json_acc = ""
            json_fragment = combined[m.start() :]
            if json_fragment:
                yield from self._emit(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": buf.anth_index,
                        "delta": {"type": "input_json_delta", "partial_json": json_fragment},
                    },
                )
        else:
            buf._pre_json_acc = combined

    def _is_declared_tool_name(self, streamed_name: str | None) -> bool:
        """Whether a streamed tool name exists in the request's declared tool list.

        If no tool schemas were provided by the caller, return True so we do not
        block valid tool calls in permissive/non-validating deployments.
        """
        if not streamed_name:
            return False
        # Allow Claude Code internal tools unconditionally
        if _is_claude_code_internal_tool(streamed_name):
            return True
        if not self.tool_controller or not self.tool_controller.has_registered_schemas():
            return True
        
        # Check if the name matches directly or through fuzzy resolution.
        if self.tool_controller.resolve_tool_name(streamed_name) is not None:
            return True
            
        original = self.tool_id_map.original_tool_name(streamed_name)
        return self.tool_controller.resolve_tool_name(original) is not None

    # ── public API ────────────────────────────────────────────────────────

    def feed(self, openai_chunk: dict) -> Iterator[dict]:
        """Consume one OpenAI chunk and yield zero or more Anthropic events."""
        if self._finished:
            return
        yield from self._ensure_started()

        # Trailing usage-only chunk (`choices == []`).
        if not openai_chunk.get("choices"):
            if usage := openai_chunk.get("usage"):
                self._usage_input = usage.get("prompt_tokens", self._usage_input)
                self._usage_output = usage.get("completion_tokens", self._usage_output)
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
                    self._thinking_budget_hit = True
                    yield from self._close_open_text_or_thinking()
                    reasoning = ""
                elif len(reasoning) > remaining_chars:
                    reasoning = reasoning[:remaining_chars]
                    self._thinking_chars += len(reasoning)
                    yield from self._emit(
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
                yield from self._emit(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self._open_block_index,
                        "delta": {"type": "thinking_delta", "thinking": reasoning},
                    },
                )

        # 2) Text chunk.
        if text:
            # Check for <think> tag start
            if "<think>" in text:
                parts = text.split("<think>", 1)
                pre_think = parts[0]
                post_think = parts[1]
                
                if pre_think:
                    if self._open_block_type != "text":
                        yield from self._close_any_open()
                        yield from self._open_text_block()
                    yield from self._emit(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": self._open_block_index,
                            "delta": {"type": "text_delta", "text": pre_think},
                        },
                    )
                
                yield from self._close_any_open()
                yield from self._open_thinking_block()
                
                if post_think:
                    # Check for closing tag in the same chunk
                    if "</think>" in post_think:
                        think_parts = post_think.split("</think>", 1)
                        inner_think = think_parts[0]
                        after_think = think_parts[1]
                        
                        if inner_think:
                            yield from self._emit(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": self._open_block_index,
                                    "delta": {"type": "thinking_delta", "thinking": inner_think},
                                },
                            )
                        
                        yield from self._close_any_open()
                        if after_think:
                            yield from self._open_text_block()
                            yield from self._emit(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": self._open_block_index,
                                    "delta": {"type": "text_delta", "text": after_think},
                                },
                            )
                    else:
                        yield from self._emit(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": self._open_block_index,
                                "delta": {"type": "thinking_delta", "thinking": post_think},
                            },
                        )
                return

            # Check for </think> tag end (if currently in thinking block)
            if self._open_block_type == "thinking" and "</think>" in text:
                parts = text.split("</think>", 1)
                inner_think = parts[0]
                post_think = parts[1]
                
                if inner_think:
                    yield from self._emit(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": self._open_block_index,
                            "delta": {"type": "thinking_delta", "thinking": inner_think},
                        },
                    )
                
                yield from self._close_any_open()
                if post_think:
                    yield from self._open_text_block()
                    yield from self._emit(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": self._open_block_index,
                            "delta": {"type": "text_delta", "text": post_think},
                        },
                    )
                return

            if _TAG_HALLUCINATION_RE.search(text):
                yield from self._close_any_open()
                yield from self._open_text_block()
                yield from self._emit(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self._open_block_index,
                        "delta": {
                            "type": "text_delta",
                            "text": (
                                "\n[PROXY: Detected malformed tag-based "
                                "tool call — stopping generation]\n"
                            ),
                        },
                    },
                )
                yield from self._close_open_text_or_thinking()
                self._stop_reason = "end_turn"
                self._finished = True
                return

            if self._open_block_type != "text":
                yield from self._close_any_open()
                yield from self._open_text_block()
            yield from self._emit(
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
            if _id := tc.get("id"):
                buf.openai_id = _id
            if nm := fn.get("name"):
                # Fuzzy resolve hallucinated tool names (e.g. Read -> read_file)
                resolved = (
                    self.tool_controller.resolve_tool_name(nm)
                    if self.tool_controller
                    else nm
                )
                buf.name = resolved or nm
            new_args = fn.get("arguments") or ""

            # DEFENSIVE: Block only truly undeclared tools. Do not block by
            # hard-coded names (e.g. "Skill", "Read") because those may be
            # legitimate tools in Claude Code sessions.
            if buf.name and not self._is_declared_tool_name(buf.name):
                if self._open_block_type != "text":
                    yield from self._close_any_open()
                    yield from self._open_text_block()
                warning = (
                    f"\n[PROXY BLOCKED undeclared tool '{buf.name}' "
                    f"with args: {new_args or buf.pre_start_buffer}]\n"
                )
                yield from self._emit(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self._open_block_index,
                        "delta": {"type": "text_delta", "text": warning},
                    },
                )
                buf.started = True
                buf.closed = True
                continue

            # Can we open this block yet? Need both id and name.
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

                # REPETITION DETECTION: ONLY check when tool call FIRST starts.
                self._last_tool_names.append(buf.name)
                if len(self._last_tool_names) > 10:
                    self._last_tool_names.pop(0)

                if len(self._last_tool_names) >= 4:
                    unique = set(self._last_tool_names[-6:])
                    recent_count = self._last_tool_names[-6:].count(buf.name)
                    if len(unique) <= 2 and recent_count >= 4:
                        self._repetition_count += 1
                        if self._repetition_count >= self._MAX_REPETITIONS:
                            yield from self._close_any_open()
                            yield from self._open_text_block()
                            yield from self._emit(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": self._open_block_index,
                                    "delta": {
                                        "type": "text_delta",
                                        "text": (
                                            "\n[PROXY: Detected infinite tool-use "
                                            "loop — stopping generation]\n"
                                        ),
                                    },
                                },
                            )
                            yield from self._close_open_text_or_thinking()
                            self._stop_reason = "end_turn"
                            self._finished = True
                            return

                emit_name = self.tool_id_map.original_tool_name(buf.name or "")
                yield from self._emit(
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
                # Flush pre-start buffer + new fragment progressively.
                initial = buf.pre_start_buffer + new_args
                buf.pre_start_buffer = ""
                yield from self._emit_tool_args_fragment(buf, initial)

            elif buf.started and not buf.closed and new_args:
                # If a different tool is currently open, switch to this one.
                if self._open_block_type == "tool_use" and self._open_block_index != buf.anth_index:
                    yield from self._close_open_tool_use()
                    self._open_block_type = "tool_use"
                    self._open_block_index = buf.anth_index
                # Emit this fragment progressively.
                yield from self._emit_tool_args_fragment(buf, new_args)

            elif not buf.started and new_args:
                # id/name haven't arrived yet — keep buffering.
                buf.pre_start_buffer += new_args

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
        yield from self._emit(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": self._stop_reason,
                    "stop_sequence": None,
                },
                "usage": {
                    "input_tokens": self._usage_input or self.estimated_input_tokens,
                    "output_tokens": self._usage_output,
                },
            },
        )
        yield from self._emit("message_stop", {"type": "message_stop"})
