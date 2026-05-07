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
from .vision_translator import openai_image_url_to_anthropic

_FENCE_RE = re.compile(r"^```[a-z]*\s*", re.MULTILINE)
# Matches first JSON object/array start character.
_JSON_START_RE = re.compile(r"[{\[]")

# Detect hallucinated tag-based tool calling in text deltas.
_TAG_HALLUCINATION_RE = re.compile(r"(command-name>|command-arguments>)", re.IGNORECASE)

# Tags that gate text↔thinking transitions when reasoning is exposed inline.
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _max_tag_prefix_suffix(s: str, tag: str) -> int:
    """Return the largest k>0 such that s endswith tag[:k]; 0 if none.

    Bounded by len(tag)-1 — we never hold back a complete tag (it would have
    been matched by find()) nor more characters than could form one.
    """
    limit = min(len(s), len(tag) - 1)
    for k in range(limit, 0, -1):
        if s.endswith(tag[:k]):
            return k
    return 0

# Claude Code internal tools that are always allowed.
# These are NOT in the user's tool schemas but are legitimate Claude Code
# tools that the model may call. We allow them through without validation.
_CLAUDE_CODE_INTERNAL_TOOLS = frozenset(
    {
        # Claude Code core commands
        "Read",
        "Write",
        "Edit",
        "Bash",
        "NotebookEdit",
        # Skill tool (for invoking agent skills)
        "Skill",
        "skill",
        # MCP tools
        "mcp__tool",
        # AskUserQuestion is sometimes generated
        "AskUserQuestion",
        # Claude Code internal commands that may appear in responses
        "WebSearch",
        "WebFetch",
        "WebRead",
        # Container/Browser tools
        "computer",
        "bash",
        "browser",
        # Legacy/alternative names
        "AskUser",
        "Ask",
        "ask_user_question",
        # Task/plan tools
        "Task",
        "Plan",
        "plan",
        # Any tool starting with common prefixes that might be Claude Code internals
    }
)


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
    # Match by prefix for MCP-style tools. Claude Code may emit canonical
    # MCP names (mcp__server__tool) or plugin-adapter sanitized names
    # (mcp_plugin_<server>_<tool>).
    if name_lower.startswith(("mcp__", "mcp_plugin_")):
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
    "content_filter": "refusal",
    "async-yield": "pause_turn",
}


# Moved to module level for better separation of concerns
@dataclass(slots=True)
class ToolBuf:
    """Accumulator for a single streamed OpenAI tool call."""

    _MAX_ARGS_BYTES: int = 10 * 1024 * 1024  # 10 MB hard cap

    openai_id: str | None = None
    name: str | None = None
    anthropic_id: str | None = None
    anth_index: int | None = None
    started: bool = False
    closed: bool = False
    # Accumulated raw text
    args: str = ""


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
    _tools_by_openai_idx: dict[int, ToolBuf] = field(default_factory=dict)

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
    _hallucination_count: int = 0
    _MAX_REPETITIONS: int = 3  # Force stop after seeing same tool pattern N times
    _MAX_HALLUCINATIONS: int = 3  # Force stop after seeing N malformed tags

    # The OpenAI tool index currently being streamed progressively.
    # To ensure contiguous Anthropic blocks, we only stream one tool at a time.
    _streaming_tool_openai_idx: int | None = None

    # Carryover bytes from the previous text chunk that may form a partial
    # <think>/</think> tag once the next chunk arrives. Bounded by
    # len(tag)-1, so at most 7 chars are ever held back.
    _text_holdback: str = ""
    # True when the currently open thinking block was started by an upstream
    # `reasoning_content` chunk (separate field) rather than an inline
    # ``<think>`` tag inside ``delta.content``. Reasoning-content thinking
    # is implicitly terminated by the arrival of any ``delta.content`` text.
    _thinking_opened_by_reasoning: bool = False

    # Cached verdict: True when the transformer chain is absent or has zero
    # registered transformers, so per-emit dispatch can be skipped entirely.
    _chain_inert: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        chain = self.transformer_chain
        self._chain_inert = chain is None or not getattr(chain, "transformers", None)

    # ── helpers ───────────────────────────────────────────────────────────

    def _emit(self, event: str, data: dict) -> Iterator[dict]:
        # Pings should always be emitted to keep the connection alive
        if event == "ping":
            yield {"event": event, "data": data}
            return

        if self._chain_inert:
            yield {"event": event, "data": data}
            return

        transformed_data = self.transformer_chain.transform_stream_chunk(data)  # type: ignore[union-attr]
        if transformed_data is None:
            return
        yield {"event": event, "data": transformed_data}

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

    # ── text/thinking emit helpers ────────────────────────────────────────

    def _emit_text_delta(self, text: str) -> Iterator[dict]:
        """Emit a text_delta, opening a text block lazily and stripping any
        hallucinated tag-based tool call markers. May terminate the stream
        on repeated hallucination — the caller MUST check self._finished.
        """
        if not text:
            return

        if _TAG_HALLUCINATION_RE.search(text):
            self._hallucination_count += 1
            text = _TAG_HALLUCINATION_RE.sub("[TAG STRIPPED]", text)
            if self._hallucination_count >= self._MAX_HALLUCINATIONS:
                if self._open_block_type != "text":
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
                                f"\n{text}\n[PROXY: Detected repeated malformed "
                                "tag-based tool calls — stopping generation]\n"
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

    def _emit_thinking_delta(self, text: str) -> Iterator[dict]:
        """Emit a thinking_delta into the currently open thinking block."""
        if not text:
            return
        yield from self._emit(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": self._open_block_index,
                "delta": {"type": "thinking_delta", "thinking": text},
            },
        )

    def _scan_text(self, incoming: str) -> Iterator[dict]:
        """Stream text through a partial-tag-aware scanner.

        Holds back the smallest tail of bytes that could be the prefix of
        ``<think>`` or ``</think>`` until the next chunk resolves whether
        it is a real tag or just literal text. This prevents fragments
        like "<th" from rendering as visible text in the client UI.

        If a thinking block is currently open *and* it was opened by an
        upstream ``reasoning_content`` chunk (a separate field), the
        arrival of ``delta.content`` text implicitly terminates it — the
        OpenAI delta protocol does not interleave the two fields.
        """
        if self._open_block_type == "thinking" and self._thinking_opened_by_reasoning:
            yield from self._close_any_open()
            self._thinking_opened_by_reasoning = False

        buf = self._text_holdback + incoming
        self._text_holdback = ""
        while buf:
            if self._open_block_type == "thinking":
                idx = buf.find(_THINK_CLOSE)
                if idx >= 0:
                    yield from self._emit_thinking_delta(buf[:idx])
                    yield from self._close_any_open()
                    buf = buf[idx + len(_THINK_CLOSE):]
                    continue
                keep = _max_tag_prefix_suffix(buf, _THINK_CLOSE)
                if keep:
                    if len(buf) > keep:
                        yield from self._emit_thinking_delta(buf[:-keep])
                    self._text_holdback = buf[-keep:]
                    return
                yield from self._emit_thinking_delta(buf)
                return
            else:
                idx = buf.find(_THINK_OPEN)
                if idx >= 0:
                    if idx:
                        yield from self._emit_text_delta(buf[:idx])
                        if self._finished:
                            return
                    yield from self._close_any_open()
                    yield from self._open_thinking_block()
                    self._thinking_opened_by_reasoning = False
                    buf = buf[idx + len(_THINK_OPEN):]
                    continue
                keep = _max_tag_prefix_suffix(buf, _THINK_OPEN)
                if keep:
                    if len(buf) > keep:
                        yield from self._emit_text_delta(buf[:-keep])
                    self._text_holdback = buf[-keep:]
                    return
                yield from self._emit_text_delta(buf)
                return

    def _flush_tool(self, buf: ToolBuf, o_idx: int) -> Iterator[dict]:
        if not buf.openai_id or not buf.name:
            return

        if buf.started:
            # Already streamed. Just ensure it's closed.
            if self._open_block_type == "tool_use" and self._open_block_index == buf.anth_index:
                yield from self._close_open_tool_use()
            return

        emit_name = self.tool_id_map.original_tool_name(buf.name or "")

        if not self._is_declared_tool_name(buf.name):
            if self._open_block_type != "text":
                yield from self._close_any_open()
                yield from self._open_text_block()
            warning = f"\n[PROXY BLOCKED undeclared tool '{buf.name}' with args: {buf.args}]\n"
            yield from self._emit(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": self._open_block_index,
                    "delta": {"type": "text_delta", "text": warning},
                },
            )
            return

        # Strip prose
        cleaned_args = _strip_leading_prose(buf.args)

        # Validation
        import json

        try:
            if cleaned_args:
                parsed_args = json.loads(cleaned_args)
            else:
                parsed_args = {}

            if self.tool_controller:
                failing = self.tool_controller.validate_all(
                    [
                        {
                            "type": "tool_use",
                            "id": self.tool_id_map.openai_to_anthropic(buf.openai_id),
                            "name": buf.name,
                            "input": parsed_args,
                        }
                    ]
                )
                if failing:
                    raise ValueError(f"Schema validation failed for tool '{failing[0]}'")
        except Exception as e:
            if self._open_block_type != "text":
                yield from self._close_any_open()
                yield from self._open_text_block()
            warning = f"\n[PROXY BLOCKED tool '{buf.name}' due to invalid args: {str(e)}]\n"
            yield from self._emit(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": self._open_block_index,
                    "delta": {"type": "text_delta", "text": warning},
                },
            )
            return

        yield from self._close_any_open()
        buf.anthropic_id = self.tool_id_map.openai_to_anthropic(buf.openai_id)
        buf.anth_index = self._next_index
        self._next_index += 1
        self._open_block_type = "tool_use"
        self._open_block_index = buf.anth_index

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
        if cleaned_args:
            yield from self._emit(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": buf.anth_index,
                    "delta": {"type": "input_json_delta", "partial_json": cleaned_args},
                },
            )
        yield from self._close_open_tool_use()

    def _flush_tools(self) -> Iterator[dict]:
        for o_idx in sorted(self._tools_by_openai_idx.keys()):
            buf = self._tools_by_openai_idx[o_idx]
            yield from self._flush_tool(buf, o_idx)

    def _is_declared_tool_name(self, streamed_name: str | None) -> bool:
        """Whether a streamed tool name exists in the request's declared tool list.

        If no tool schemas were provided by the caller, return True so we do not
        block valid tool calls in permissive/non-validating deployments.
        """
        if not streamed_name:
            return False
        if _is_claude_code_internal_tool(streamed_name):
            return True
        if streamed_name.lower().startswith("mcp_"):
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
        image_url = delta.get("image_url")
        tool_calls = delta.get("tool_calls") or []

        # 1.1) Image chunk (multimodal response).
        if image_url:
            yield from self._close_any_open()
            idx = self._next_index
            self._next_index += 1
            yield from self._emit(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": openai_image_url_to_anthropic(image_url),
                },
            )
            yield from self._emit(
                "content_block_stop", {"type": "content_block_stop", "index": idx}
            )

        # 1) Reasoning chunk.
        if reasoning and not self._thinking_budget_hit:
            if self._open_block_type != "thinking":
                yield from self._close_any_open()
                yield from self._open_thinking_block()
                self._thinking_opened_by_reasoning = True
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

        # 2) Text chunk — route through the holdback scanner so partial
        # <think>/</think> tags split across chunks never leak as text.
        if text:
            yield from self._scan_text(text)
            if self._finished:
                return

        # 3) Tool-call chunks.
        for tc in tool_calls:
            o_idx = tc.get("index", 0)
            buf = self._tools_by_openai_idx.setdefault(o_idx, ToolBuf())
            fn = tc.get("function") or {}
            if _id := tc.get("id"):
                buf.openai_id = _id
            if nm := fn.get("name"):
                # Fuzzy resolve hallucinated tool names (e.g. Read -> read_file)
                resolved = (
                    self.tool_controller.resolve_tool_name(nm) if self.tool_controller else nm
                )
                buf.name = resolved or nm

            new_args = fn.get("arguments") or ""

            # REPETITION DETECTION: Check when tool name FIRST arrives.
            if buf.name and buf.args == "" and not new_args:
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

            # PROGRESSIVE STREAMING logic:
            # If this is the FIRST tool that appeared, or it's the one we are already
            # streaming, emit it now.
            if self._streaming_tool_openai_idx is None:
                self._streaming_tool_openai_idx = o_idx

            if self._streaming_tool_openai_idx == o_idx:
                # We can stream this tool.
                if not buf.started and buf.openai_id and buf.name:
                    # Validate name BEFORE starting.
                    if not self._is_declared_tool_name(buf.name):
                        # Treat as text (blocked).
                        # We MUST NOT set buf.started=True here because we're not
                        # starting a tool_use block. We'll let it buffer and
                        # flush as text later, OR we can emit text now.
                        # For now, we'll stop streaming THIS tool and let it
                        # be handled by the finish_reason flusher (which blocks it).
                        self._streaming_tool_openai_idx = None
                    else:
                        yield from self._close_any_open()
                        buf.anth_index = self._next_index
                        self._next_index += 1
                        buf.anthropic_id = self.tool_id_map.openai_to_anthropic(buf.openai_id)
                        self._open_block_type = "tool_use"
                        self._open_block_index = buf.anth_index
                        buf.started = True

                        emit_name = self.tool_id_map.original_tool_name(buf.name)
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
                        # Emit any buffered args (e.g. from previous chunks)
                        if buf.args:
                            yield from self._emit(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": buf.anth_index,
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": buf.args,
                                    },
                                },
                            )

                if buf.started and not buf.closed and new_args:
                    yield from self._emit(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": buf.anth_index,
                            "delta": {"type": "input_json_delta", "partial_json": new_args},
                        },
                    )

            buf.args += new_args

        # 4) Finish reason.
        if finish:
            yield from self._close_any_open()
            if self._tools_by_openai_idx:
                yield from self._flush_tools()
            self._stop_reason = _FINISH_TO_STOP.get(finish, "end_turn")

    def finalize(self) -> Iterator[dict]:
        """Emit terminal `message_delta` + `message_stop`. Call once."""
        if self._finished:
            return
        # Flush any residual text-holdback as plain text BEFORE marking
        # finished, so _emit_text_delta can still open a text block. A
        # stranded "<th" at EOF is just text, never a thinking block.
        if self._text_holdback:
            stranded = self._text_holdback
            self._text_holdback = ""
            if self._open_block_type == "thinking":
                yield from self._emit_thinking_delta(stranded)
            else:
                yield from self._emit_text_delta(stranded)
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
