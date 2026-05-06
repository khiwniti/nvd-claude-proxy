"""Internal canonical types used across the proxy translation pipeline.

These Pydantic v2 models serve as the single source of truth for the
intermediate representation (IR) between Anthropic and NVIDIA NIM.
They are 'frozen' to ensure immutability across pipeline stages.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, model_validator


# ── Content blocks ─────────────────────────────────────────────────────────────


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageBlock(BaseModel):
    type: Literal["image"] = "image"
    source: dict[str, Any]


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Any = ""
    is_error: bool | None = None


class ThinkingBlock(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str = ""


class RedactedThinkingBlock(BaseModel):
    type: Literal["redacted_thinking"] = "redacted_thinking"
    data: str


# Server tool and MCP block types
class SearchResultBlock(BaseModel):
    type: Literal["search_result"] = "search_result"
    content: str
    title: str | None = None
    url: str | None = None


class WebSearchToolResultBlock(BaseModel):
    type: Literal["web_search_tool_result"] = "web_search_tool_result"
    search_results: list[SearchResultBlock] = Field(default_factory=list)
    is_error: bool | None = None


class CodeExecutionToolResultBlock(BaseModel):
    type: Literal["code_execution_tool_result"] = "code_execution_tool_result"
    output: str | None = None
    error: str | None = None
    is_error: bool | None = None


class MCPToolUseBlock(BaseModel):
    type: Literal["mcp_tool_use"] = "mcp_tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    server_name: str


class MCPToolResultBlock(BaseModel):
    type: Literal["mcp_tool_result"] = "mcp_tool_result"
    tool_use_id: str
    content: Any = ""
    is_error: bool | None = None


ContentBlock = Annotated[
    Union[
        TextBlock,
        ImageBlock,
        ToolUseBlock,
        ToolResultBlock,
        ThinkingBlock,
        RedactedThinkingBlock,
        SearchResultBlock,
        WebSearchToolResultBlock,
        CodeExecutionToolResultBlock,
        MCPToolUseBlock,
        MCPToolResultBlock,
    ],
    Field(discriminator="type"),
]


# ── Messages & Request ─────────────────────────────────────────────────────────


class CanonicalMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: list[ContentBlock]


class CanonicalRequest(BaseModel, frozen=True):
    """Immutable IR for an Anthropic Messages request."""

    model: str
    messages: tuple[CanonicalMessage, ...]
    system: tuple[TextBlock, ...] | None = None
    tools: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    tool_choice: Any = None
    thinking: dict[str, Any] | None = None
    betas: frozenset[str] = Field(default_factory=frozenset)
    service_tier: str | None = None
    container: dict[str, Any] | None = None
    mcp_servers: tuple[dict[str, Any], ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_protocol_invariants(self) -> CanonicalRequest:
        # TODO: Implement strict tool_use -> tool_result correlation
        # and block-ordering invariant checks.
        return self


# ── Usage & Response ──────────────────────────────────────────────────────────


class CanonicalUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    # Breaking down creation by TTL
    ephemeral_5m_input_tokens: int = 0
    ephemeral_1h_input_tokens: int = 0


class CanonicalResponse(BaseModel):
    """Normalised completed response."""

    id: str
    model: str
    content: list[ContentBlock]
    stop_reason: Literal[
        "end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal"
    ]
    stop_sequence: str | None = None
    usage: CanonicalUsage = Field(default_factory=CanonicalUsage)
