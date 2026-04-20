"""Canonical request schemas for Anthropic Messages API parity."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, RootModel


class ProxyTextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ProxyImageBlock(BaseModel):
    type: Literal["image"] = "image"
    source: dict[str, Any]  # Simplified; rely on downstream conversion


class ProxyToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ProxyThinkingBlock(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str = ""


ProxyContentBlock = ProxyTextBlock | ProxyImageBlock | ProxyToolUseBlock | ProxyThinkingBlock


class ProxyMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[ProxyContentBlock]


class CanonicalRequest(BaseModel):
    model: str
    messages: list[ProxyMessage]
    system: str | list[dict[str, Any]] | None = None
    max_tokens: int = 1024
    stream: bool = False
    thinking: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
