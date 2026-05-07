from __future__ import annotations

import re
from typing import Any, Iterable, TYPE_CHECKING, Literal

from .events import RawOpenAIChunk, TranslatedEvent, StreamState, StreamProcessor
from .tool_accumulator import ToolAccumulator
from ..translators.vision_translator import openai_image_url_to_anthropic

if TYPE_CHECKING:
    from ..translators.tool_translator import ToolIdMap
    from ..translators.tool_controller import ToolInvocationController

BlockType = Literal["text", "thinking", "tool_use"]


class BaseProcessor(StreamProcessor):
    """Shared helpers for all processors."""

    def _emit(self, event: str, data: dict[str, Any]) -> TranslatedEvent:
        return TranslatedEvent(event, data)

    def _ensure_started(self, state: StreamState) -> Iterable[TranslatedEvent]:
        if state.started:
            return
        state.started = True

        usage_data = {"input_tokens": state.estimated_input_tokens, "output_tokens": 0}
        if state.cache_creation_input_tokens > 0 or state.cache_read_input_tokens > 0:
            usage_data["cache_creation_input_tokens"] = state.cache_creation_input_tokens
            usage_data["cache_read_input_tokens"] = state.cache_read_input_tokens

        yield self._emit(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": state.message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": state.model_name,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": usage_data,
                },
            },
        )

    def _close_open_block(self, state: StreamState) -> Iterable[TranslatedEvent]:
        if state.open_block_type is not None and state.open_block_index is not None:
            if state.open_block_type == "thinking":
                from ..util.ids import new_thinking_signature
                yield self._emit(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": state.open_block_index,
                        "delta": {"type": "signature_delta", "signature": new_thinking_signature()},
                    },
                )
            yield self._emit(
                "content_block_stop",
                {"type": "content_block_stop", "index": state.open_block_index},
            )
            # Per Anthropic spec, message_delta is emitted exactly once at end
            # (in FinalizerProcessor.finalize). Per-block usage snapshots cause
            # SDKs to over-render and cost trackers to double-count.
            state.open_block_type = None
            state.open_block_index = None


class MetadataProcessor(BaseProcessor):
    """Handles stream start and usage accumulation."""

    def process(self, chunk: RawOpenAIChunk, state: StreamState) -> Iterable[TranslatedEvent]:
        if state.finished:
            return []

        # We must return an iterable, so we use a generator pattern with yield from.
        def _gen():
            yield from self._ensure_started(state)

            if not chunk.data.get("choices"):
                usage = chunk.data.get("usage")
                if usage:
                    state.usage_input = usage.get("prompt_tokens", state.usage_input)
                    state.usage_output = usage.get("completion_tokens", state.usage_output)

        return _gen()


class TextProcessor(BaseProcessor):
    """Handles content and reasoning_content from OpenAI."""

    def process(self, chunk: RawOpenAIChunk, state: StreamState) -> Iterable[TranslatedEvent]:
        if state.finished:
            return []

        choices = chunk.data.get("choices") or []
        if not choices:
            return []

        def _gen():
            delta = choices[0].get("delta") or {}
            reasoning = delta.get("reasoning_content")
            text = delta.get("content")
            image_url = delta.get("image_url")

            if image_url:
                yield from self._handle_image(image_url, state)
            if reasoning:
                yield from self._handle_reasoning(reasoning, state)
            if text:
                yield from self._handle_text(text, state)

        return _gen()

    def _handle_image(self, image_url: dict, state: StreamState) -> Iterable[TranslatedEvent]:
        yield from self._close_open_block(state)
        idx = state.next_index
        state.next_index += 1
        yield self._emit(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": idx,
                "content_block": openai_image_url_to_anthropic(image_url),
            },
        )
        yield self._emit("content_block_stop", {"type": "content_block_stop", "index": idx})

    def _handle_reasoning(self, reasoning: str, state: StreamState) -> Iterable[TranslatedEvent]:
        if state.thinking_budget_hit:
            return

        if state.open_block_type != "thinking":
            yield from self._close_open_block(state)
            yield from self._open_block("thinking", state)

        yield self._emit(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": state.open_block_index,
                "delta": {"type": "thinking_delta", "thinking": reasoning},
            },
        )

    def _handle_text(self, text: str, state: StreamState) -> Iterable[TranslatedEvent]:
        state.accumulated_text += text
        if state.open_block_type != "text":
            yield from self._close_open_block(state)
            yield from self._open_block("text", state)

        yield self._emit(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": state.open_block_index,
                "delta": {"type": "text_delta", "text": text},
            },
        )

    def _open_block(self, block_type: BlockType, state: StreamState) -> Iterable[TranslatedEvent]:
        idx = state.next_index
        state.next_index += 1
        state.open_block_type = block_type
        state.open_block_index = idx

        content_block: dict[str, Any] = {"type": block_type}
        if block_type == "text":
            content_block["text"] = ""
        elif block_type == "thinking":
            content_block["thinking"] = ""
            content_block["signature"] = ""

        yield self._emit(
            "content_block_start",
            {"type": "content_block_start", "index": idx, "content_block": content_block},
        )


class ToolProcessor(BaseProcessor):
    """Buffers, validates, and emits tool_use blocks progressively."""

    def __init__(self, tool_id_map: ToolIdMap, tool_controller: ToolInvocationController) -> None:
        self.tool_id_map = tool_id_map
        self.tool_controller = tool_controller
        self._accumulators: dict[int, ToolAccumulator] = {}
        self._streaming_tool_openai_idx: int | None = None

    def process(self, chunk: RawOpenAIChunk, state: StreamState) -> Iterable[TranslatedEvent]:
        if state.finished:
            return []

        choices = chunk.data.get("choices") or []
        if not choices:
            return []

        def _gen():
            delta = choices[0].get("delta") or {}
            tool_calls = delta.get("tool_calls") or []

            for tc in tool_calls:
                idx = tc.get("index", 0)
                acc = self._accumulators.setdefault(idx, ToolAccumulator())

                if "id" in tc:
                    acc.openai_id = tc["id"]
                fn = tc.get("function") or {}
                if "name" in fn:
                    nm = fn["name"]
                    resolved = (
                        self.tool_controller.resolve_tool_name(nm) if self.tool_controller else nm
                    )
                    acc.name = resolved or nm

                new_args = fn.get("arguments") or ""

                if self._streaming_tool_openai_idx is None:
                    self._streaming_tool_openai_idx = idx

                if self._streaming_tool_openai_idx == idx:
                    # Stream this tool
                    if not acc.started and acc.openai_id and acc.name:
                        # Before starting, check if name is declared.
                        # This matches stream_translator progressive logic.
                        if self.tool_controller and not self.tool_controller.is_declared(acc.name):
                            self._streaming_tool_openai_idx = None
                        else:
                            yield from self._close_open_block(state)
                            acc.anth_index = state.next_index
                            state.next_index += 1
                            acc.anthropic_id = self.tool_id_map.openai_to_anthropic(acc.openai_id)
                            state.open_block_type = "tool_use"
                            state.open_block_index = acc.anth_index
                            acc.started = True

                            original_name = self.tool_id_map.original_tool_name(acc.name)
                            yield self._emit(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": acc.anth_index,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": acc.anthropic_id,
                                        "name": original_name,
                                        "input": {},
                                    },
                                },
                            )
                            if acc.arguments:
                                yield self._emit(
                                    "content_block_delta",
                                    {
                                        "type": "content_block_delta",
                                        "index": acc.anth_index,
                                        "delta": {
                                            "type": "input_json_delta",
                                            "partial_json": acc.arguments,
                                        },
                                    },
                                )

                    if acc.started and not acc.closed and new_args:
                        state.accumulated_tool_json += new_args
                        yield self._emit(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": acc.anth_index,
                                "delta": {"type": "input_json_delta", "partial_json": new_args},
                            },
                        )

                acc.arguments += new_args
            
            # Simplified: just accumulate all generated tool args for reconciliation
            for tc in tool_calls:
                args = (tc.get("function") or {}).get("arguments") or ""
                state.accumulated_tool_json += args

            finish_reason = choices[0].get("finish_reason")
            if finish_reason in ("tool_calls", "function_call") or (
                finish_reason == "stop" and self._accumulators
            ):
                for idx in sorted(self._accumulators.keys()):
                    acc = self._accumulators[idx]
                    if not acc.closed:
                        yield from self._flush_tool(acc, state)

        return _gen()

    def _flush_tool(self, acc: ToolAccumulator, state: StreamState) -> Iterable[TranslatedEvent]:
        if not acc.openai_id or not acc.name:
            return

        if acc.started:
            if state.open_block_type == "tool_use" and state.open_block_index == acc.anth_index:
                yield from self._close_open_block(state)
            acc.closed = True
            return

        # It wasn't progressively streamed, so flush it fully now
        if self.tool_controller and not self.tool_controller.is_declared(acc.name):
            # Not declared, don't flush as tool_use
            return

        original_name = self.tool_id_map.original_tool_name(acc.name)
        yield from self._close_open_block(state)

        acc.anthropic_id = self.tool_id_map.openai_to_anthropic(acc.openai_id)
        acc.anth_index = state.next_index
        state.next_index += 1
        state.open_block_type = "tool_use"
        state.open_block_index = acc.anth_index
        acc.started = True

        yield self._emit(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": acc.anth_index,
                "content_block": {
                    "type": "tool_use",
                    "id": acc.anthropic_id,
                    "name": original_name,
                    "input": {},
                },
            },
        )

        # Clean markdown fences from arguments
        args = acc.arguments.strip()
        args = re.sub(r"^```[a-z]*\s*", "", args, flags=re.MULTILINE)
        if args.endswith("```"):
            args = args[:-3].strip()

        yield self._emit(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": acc.anth_index,
                "delta": {"type": "input_json_delta", "partial_json": args},
            },
        )

        yield from self._close_open_block(state)
        acc.closed = True


class SafetyProcessor(BaseProcessor):
    """Detects and prevents infinite loops."""

    def __init__(self, max_repetitions: int = 3, max_hallucinations: int = 3) -> None:
        self.max_repetitions = max_repetitions
        self.max_hallucinations = max_hallucinations
        self._last_tool_names: list[str] = []
        self._repetition_count = 0
        self._hallucination_count = 0
        self._tag_hallucination_re = re.compile(
            r"(command-name>|command-arguments>)", re.IGNORECASE
        )

    def process(self, chunk: RawOpenAIChunk, state: StreamState) -> Iterable[TranslatedEvent]:
        if state.finished:
            return []

        choices = chunk.data.get("choices") or []
        if not choices:
            return []

        delta = choices[0].get("delta") or {}
        text = delta.get("content")
        tool_calls = delta.get("tool_calls") or []

        if text and self._tag_hallucination_re.search(text):
            self._hallucination_count += 1
            if self._hallucination_count >= self.max_hallucinations:
                state.finished = True
                state.stop_reason = "end_turn"

        for tc in tool_calls:
            fn = tc.get("function") or {}
            name = fn.get("name")
            if name and not fn.get("arguments"):
                self._last_tool_names.append(name)
                if len(self._last_tool_names) > 10:
                    self._last_tool_names.pop(0)

                if len(self._last_tool_names) >= 4:
                    unique = set(self._last_tool_names[-6:])
                    recent_count = self._last_tool_names[-6:].count(name)
                    if len(unique) <= 2 and recent_count >= 4:
                        self._repetition_count += 1
                        if self._repetition_count >= self.max_repetitions:
                            state.finished = True
                            state.stop_reason = "end_turn"

        return []


class FinalizerProcessor(BaseProcessor):
    """Handles finish reason and stream closure. Should run LAST."""

    def process(self, chunk: RawOpenAIChunk, state: StreamState) -> Iterable[TranslatedEvent]:
        if state.finished:
            return []

        choices = chunk.data.get("choices") or []
        if not choices:
            return []

        def _gen():
            choice = choices[0]
            finish = choice.get("finish_reason")
            if finish:
                yield from self._close_open_block(state)
                state.stop_reason = self._map_finish_reason(finish)
                state.finished = True
                yield from self.finalize(state)

        return _gen()

    def _map_finish_reason(self, finish: str) -> str:
        mapping = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "function_call": "tool_use",
            "content_filter": "refusal",
            "async-yield": "pause_turn",
        }
        return mapping.get(finish, "end_turn")

    def finalize(self, state: StreamState) -> Iterable[TranslatedEvent]:
        if not state.started:
            yield from self._ensure_started(state)

        yield from self._close_open_block(state)

        # P1-15: Output token reconciliation
        from ..util.tokens import approximate_tokens
        reconciled_output = approximate_tokens(state.accumulated_text + state.accumulated_tool_json)
        # NIM usage often misses thinking tokens or has minor drift.
        # We use upstream as primary but use local as floor/reconciler if drift is high.
        final_output = state.usage_output
        drift = abs(final_output - reconciled_output)
        if final_output > 0 and drift / final_output > 0.10: # >10% drift
             import structlog
             structlog.get_logger("nvd_claude_proxy.stream").warning(
                 "usage_drift_warning", 
                 upstream=final_output, 
                 reconciled=reconciled_output,
                 drift_pct=round(drift / final_output * 100, 1)
             )
             # If upstream is way too low (e.g. 0 on some SGLang NIMs), use reconciled.
             if final_output < reconciled_output * 0.5:
                 final_output = reconciled_output

        usage_data = {
            "input_tokens": state.usage_input or state.estimated_input_tokens,
            "output_tokens": final_output,
        }
        if state.cache_creation_input_tokens > 0 or state.cache_read_input_tokens > 0:
            usage_data["cache_creation_input_tokens"] = state.cache_creation_input_tokens
            usage_data["cache_read_input_tokens"] = state.cache_read_input_tokens
            
            # P1-6: Breakdown by TTL
            usage_data["cache_creation_input_tokens_breakdown"] = {
                "ephemeral_5m_input_tokens": getattr(state, "ephemeral_5m_input_tokens", 0),
                "ephemeral_1h_input_tokens": getattr(state, "ephemeral_1h_input_tokens", 0),
            }

        yield self._emit(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": state.stop_reason, "stop_sequence": None},
                "usage": usage_data,
            },
        )
        yield self._emit("message_stop", {"type": "message_stop"})
        state.finished = True
