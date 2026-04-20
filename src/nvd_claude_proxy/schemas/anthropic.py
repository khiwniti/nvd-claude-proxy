"""Minimal pydantic schemas for the Anthropic Messages request/response.

These are *informational*: routes accept raw dicts for maximum forward-compat
with new Anthropic fields. Use these for tests and IDE hints.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageSourceBase64(BaseModel):
    type: Literal["base64"]
    media_type: str
    data: str


class ImageSourceURL(BaseModel):
    type: Literal["url"]
    url: str


class ImageBlock(BaseModel):
    type: Literal["image"] = "image"
    source: ImageSourceBase64 | ImageSourceURL


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


ContentBlock = TextBlock | ImageBlock | ToolUseBlock | ToolResultBlock | ThinkingBlock


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[dict[str, Any]]


class Tool(BaseModel):
    name: str
    description: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=dict)


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
