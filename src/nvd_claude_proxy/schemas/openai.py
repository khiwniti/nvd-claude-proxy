"""Minimal OpenAI chunk/response schemas for reference/testing."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class FunctionCall(BaseModel):
    name: str | None = None
    arguments: str | None = None


class ToolCall(BaseModel):
    index: int = 0
    id: str | None = None
    type: Literal["function"] = "function"
    function: FunctionCall = Field(default_factory=FunctionCall)


class ChoiceDelta(BaseModel):
    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: ChoiceDelta = Field(default_factory=ChoiceDelta)
    finish_reason: str | None = None


class UsageBlock(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class StreamChunk(BaseModel):
    id: str | None = None
    object: str | None = None
    choices: list[StreamChoice] = Field(default_factory=list)
    usage: UsageBlock | None = None


class CompletionMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class Choice(BaseModel):
    index: int = 0
    message: CompletionMessage = Field(default_factory=CompletionMessage)
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str | None = None
    object: str | None = None
    model: str | None = None
    choices: list[Choice] = Field(default_factory=list)
    usage: UsageBlock | None = None
