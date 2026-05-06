from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Iterable


@dataclass(slots=True)
class RawOpenAIChunk:
    """The raw dictionary received from the upstream NVIDIA/OpenAI API."""

    data: dict[str, Any]


@dataclass(slots=True)
class TranslatedEvent:
    """An Anthropic-compatible SSE event ready for the wire."""

    event: str
    data: dict[str, Any]
    id: str | None = None


@dataclass(slots=True)
class StreamState:
    """Shared state for the duration of a single stream."""

    message_id: str
    model_name: str
    next_index: int = 0
    next_event_id: int = 1
    open_block_type: Literal["text", "thinking", "tool_use"] | None = None
    open_block_index: int | None = None

    # Lifecycle flags
    started: bool = False
    finished: bool = False
    stop_reason: str = "end_turn"

    # Usage and Budget
    usage_input: int = 0
    usage_output: int = 0
    accumulated_text: str = ""
    accumulated_tool_json: str = ""
    estimated_input_tokens: int = 0
    thinking_chars: int = 0
    thinking_budget_hit: bool = False
    budget_tokens: int | None = None

    # Cache Accounting
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    ephemeral_5m_input_tokens: int = 0
    ephemeral_1h_input_tokens: int = 0


class StreamProcessor(ABC):
    """Base class for components in the streaming pipeline."""

    @abstractmethod
    def process(self, chunk: RawOpenAIChunk, state: StreamState) -> Iterable[TranslatedEvent]:
        """Process a raw chunk and yield translated events."""
        pass
