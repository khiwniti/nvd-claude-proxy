"""Internal canonical types used across the proxy translation pipeline.

These plain-Python dataclasses serve as the single source of truth for the
intermediate representation between Anthropic wire format and NVIDIA NIM.
They are NOT serialised over the wire — use schemas/anthropic.py for inbound
Anthropic shapes and schemas/openai.py for NIM wire types.

The Pydantic ProxyX models are kept for test/validation convenience; the
lightweight dataclass variants are used on hot paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Union

from pydantic import BaseModel, Field


# ── Pydantic helpers (for tests / IDE type hints) ─────────────────────────────


class ProxyTextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ProxyImageBlock(BaseModel):
    type: Literal["image"] = "image"
    source: dict[str, Any]


class ProxyToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ProxyToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Any = ""
    is_error: bool | None = None


class ProxyThinkingBlock(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str = ""


class ProxyRedactedThinkingBlock(BaseModel):
    type: Literal["redacted_thinking"] = "redacted_thinking"
    data: str


ProxyContentBlock = Union[
    ProxyTextBlock,
    ProxyImageBlock,
    ProxyToolUseBlock,
    ProxyToolResultBlock,
    ProxyThinkingBlock,
    ProxyRedactedThinkingBlock,
]


class ProxyMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[ProxyContentBlock]


class CanonicalRequest(BaseModel):
    """Full normalised form of an inbound Anthropic Messages request."""

    model: str
    messages: list[ProxyMessage]
    system: str | list[dict[str, Any]] | None = None
    max_tokens: int = 1024
    stream: bool = False
    thinking: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    metadata: dict[str, Any] | None = None
    # Accepted and ignored by the proxy (no NIM equivalent).
    service_tier: str | None = None


# ── Lightweight dataclasses for hot-path use ──────────────────────────────────


@dataclass(slots=True)
class CanonicalText:
    text: str
    kind: Literal["text"] = "text"


@dataclass(slots=True)
class CanonicalThinking:
    thinking: str
    signature: str
    kind: Literal["thinking"] = "thinking"


@dataclass(slots=True)
class CanonicalToolUse:
    id: str
    name: str
    input: dict[str, Any]
    kind: Literal["tool_use"] = "tool_use"


@dataclass(slots=True)
class CanonicalToolResult:
    tool_use_id: str
    content: Any
    is_error: bool = False
    kind: Literal["tool_result"] = "tool_result"


CanonicalBlock = Union[CanonicalText, CanonicalThinking, CanonicalToolUse, CanonicalToolResult]


@dataclass(slots=True)
class CanonicalUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class CanonicalResponse:
    """Normalised completed (non-streaming) response."""

    id: str
    model: str
    content: list[CanonicalBlock]
    # Valid values: end_turn | max_tokens | stop_sequence | tool_use
    stop_reason: str
    stop_sequence: str | None
    usage: CanonicalUsage = field(default_factory=CanonicalUsage)


@dataclass(slots=True)
class SSEEvent:
    """A single server-sent event ready for wire encoding."""

    event: str
    data: dict[str, Any]
