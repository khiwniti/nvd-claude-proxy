"""Pydantic schemas for the Anthropic Messages API (2023-06-01).

These are *informational*: routes accept raw dicts for maximum forward-compat
with new Anthropic fields. Use these for tests, IDE hints, and validation.

Coverage targets the full surface area exercised by Claude Code and the
official Python/TypeScript SDKs, including extended-thinking blocks, document
blocks (PDF/URL/text sources), cache-control ephemeral blocks, and all
tool-choice variants.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field


# ── Cache control ──────────────────────────────────────────────────────────────


class CacheControlEphemeral(BaseModel):
    type: Literal["ephemeral"] = "ephemeral"
    ttl: Literal["5m", "1h"] | None = None


# ── Content block types ────────────────────────────────────────────────────────


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str
    cache_control: CacheControlEphemeral | None = None


class ImageSourceBase64(BaseModel):
    type: Literal["base64"] = "base64"
    media_type: str
    data: str


class ImageSourceURL(BaseModel):
    type: Literal["url"] = "url"
    url: str


class FileSource(BaseModel):
    type: Literal["file"] = "file"
    file_id: str


class ImageBlock(BaseModel):
    type: Literal["image"] = "image"
    source: ImageSourceBase64 | ImageSourceURL | FileSource
    cache_control: CacheControlEphemeral | None = None


# Document sources


class DocumentSourceBase64(BaseModel):
    type: Literal["base64"] = "base64"
    media_type: str
    data: str


class DocumentSourceText(BaseModel):
    type: Literal["text"] = "text"
    media_type: Literal["text/plain"] = "text/plain"
    data: str


class DocumentSourceURL(BaseModel):
    type: Literal["url"] = "url"
    url: str


DocumentSource = Union[DocumentSourceBase64, DocumentSourceText, DocumentSourceURL, FileSource]


class DocumentBlock(BaseModel):
    type: Literal["document"] = "document"
    source: DocumentSource
    title: str | None = None
    context: str | None = None
    cache_control: CacheControlEphemeral | None = None


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
    cache_control: CacheControlEphemeral | None = None


class ThinkingBlock(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str = ""


class RedactedThinkingBlock(BaseModel):
    """Emitted by Anthropic when extended thinking content is policy-redacted."""

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
        DocumentBlock,
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


# ── Thinking config ────────────────────────────────────────────────────────────


class ThinkingConfigEnabled(BaseModel):
    type: Literal["enabled"] = "enabled"
    budget_tokens: int = Field(ge=1024)


class ThinkingConfigDisabled(BaseModel):
    type: Literal["disabled"] = "disabled"


ThinkingConfig = Union[ThinkingConfigEnabled, ThinkingConfigDisabled]


# ── Tool definitions ───────────────────────────────────────────────────────────


class Tool(BaseModel):
    name: str
    description: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=dict)
    cache_control: CacheControlEphemeral | None = None


# ── Tool choice variants ───────────────────────────────────────────────────────


class ToolChoiceAuto(BaseModel):
    type: Literal["auto"] = "auto"
    disable_parallel_tool_use: bool | None = None


class ToolChoiceAny(BaseModel):
    type: Literal["any"] = "any"
    disable_parallel_tool_use: bool | None = None


class ToolChoiceTool(BaseModel):
    type: Literal["tool"] = "tool"
    name: str
    disable_parallel_tool_use: bool | None = None


class ToolChoiceNone(BaseModel):
    type: Literal["none"] = "none"


ToolChoice = Annotated[
    Union[ToolChoiceAuto, ToolChoiceAny, ToolChoiceTool, ToolChoiceNone],
    Field(discriminator="type"),
]


# ── Message types ──────────────────────────────────────────────────────────────


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[dict[str, Any]]


# ── Request / Response ─────────────────────────────────────────────────────────


class MessagesRequest(BaseModel):
    model: str
    messages: list[Message]
    system: str | list[dict[str, Any]] | None = None
    max_tokens: int = 1024
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    thinking: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    mcp_servers: list[dict[str, Any]] | None = None
    container: dict[str, Any] | None = None
    # Prompt caching (accepted, not forwarded to NIM)
    # service_tier accepted and silently ignored
    service_tier: str | None = None


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: list[dict[str, Any]]
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: Usage = Field(default_factory=Usage)
