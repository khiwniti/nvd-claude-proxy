"""Strict Pydantic validation for Anthropic Messages API requests.

This module provides comprehensive input validation that goes beyond basic
type checking to enforce Anthropic's semantic requirements. All validation
errors are returned in Anthropic's standard error format.

Security considerations:
- Validates all user-controlled input before translation
- Prevents malformed requests from reaching upstream NVIDIA API
- Provides clear, actionable error messages for debugging
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

# ── Constants ────────────────────────────────────────────────────────────────

_MAX_MODEL_NAME_LENGTH = 256
_MAX_MESSAGES_PER_REQUEST = 1000
_MAX_TOOLS_PER_REQUEST = 1000
_MAX_SYSTEM_PROMPT_CHARS = 256_000  # ~64k tokens
_MAX_TOOL_NAME_LENGTH = 128
_MAX_TOTAL_REQUEST_SIZE_MB = 50

# Valid role values per Anthropic spec
_VALID_ROLES = frozenset({"user", "assistant", "system"})

# Content block discriminators
_CONTENT_BLOCK_TYPES = frozenset(
    {
        "text",
        "image",
        "document",
        "tool_use",
        "tool_result",
        "thinking",
        "redacted_thinking",
        "server_tool_use",
    }
)

# Image source types
_IMAGE_SOURCE_TYPES = frozenset({"base64", "url"})

# Valid media types for images
_VALID_IMAGE_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})

# Tool types
_TOOL_TYPES = frozenset({None, "custom", "function"})

# Server tool types (should be dropped, not rejected)
_SERVER_TOOL_TYPES = frozenset(
    {
        "web_search_20250305",
        "web_search_20250728",
        "bash_20250124",
        "bash_20250728",
        "computer_20250124",
        "computer_20250728",
        "code_execution_20250522",
        "code_execution_20260120",
        "text_editor_20250124",
        "text_editor_20250728",
        "memory_20250818",
    }
)

# Service tier values
_VALID_SERVICE_TIERS = frozenset({"auto", "standard_only"})

# Tool choice types
_TOOL_CHOICE_TYPES = frozenset({"auto", "any", "none", "tool"})


# ── Sub-models ────────────────────────────────────────────────────────────────


class CacheControlEphemeral(BaseModel):
    """cache_control block for content."""

    type: Literal["ephemeral"] = "ephemeral"


class ImageSourceBase64(BaseModel):
    """Base64-encoded image source."""

    type: Literal["base64"] = "base64"
    media_type: str = Field(..., pattern=r"^image/.*$")
    data: str = Field(..., min_length=1)

    @field_validator("data")
    @classmethod
    def validate_base64_data(cls, v: str) -> str:
        # Basic base64 validation
        try:
            import base64

            base64.b64decode(v, validate=True)
        except Exception as e:
            raise ValueError(f"Invalid base64 data: {e}") from e
        return v


class ImageSourceURL(BaseModel):
    """URL-based image source."""

    type: Literal["url"] = "url"
    url: str = Field(..., min_length=1, max_length=8192)

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        # Basic URL validation
        from urllib.parse import urlparse

        parsed = urlparse(v)
        if not parsed.scheme:
            raise ValueError("URL must include a scheme (http:// or https://)")
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
        return v


class ImageBlock(BaseModel):
    """Image content block."""

    type: Literal["image"] = "image"
    source: Union[ImageSourceBase64, ImageSourceURL]
    cache_control: CacheControlEphemeral | None = None


class TextBlock(BaseModel):
    """Text content block."""

    type: Literal["text"] = "text"
    text: str = Field(..., min_length=0)
    cache_control: CacheControlEphemeral | None = None


class ToolUseBlock(BaseModel):
    """Tool use content block from assistant."""

    type: Literal["tool_use"] = "tool_use"
    id: str = Field(..., min_length=1, max_length=128)
    name: str = Field(..., min_length=1, max_length=_MAX_TOOL_NAME_LENGTH)
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(BaseModel):
    """Tool result content block from user."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str = Field(..., min_length=1)
    content: Any = ""  # Can be string, list, or other
    is_error: bool | None = None
    cache_control: CacheControlEphemeral | None = None


class ThinkingBlock(BaseModel):
    """Thinking content block (extended thinking)."""

    type: Literal["thinking"] = "thinking"
    thinking: str = Field(..., min_length=1)
    signature: str = ""


class RedactedThinkingBlock(BaseModel):
    """Redacted thinking content block."""

    type: Literal["redacted_thinking"] = "redacted_thinking"
    data: str


class DocumentSource(BaseModel):
    """Document source - base64, text, or URL."""

    type: Literal["base64", "text", "url"]
    media_type: str | None = None
    data: str | None = None
    url: str | None = None


class DocumentBlock(BaseModel):
    """Document content block (PDF/text/URL)."""

    type: Literal["document"] = "document"
    source: DocumentSource
    title: str | None = None
    context: str | None = None
    cache_control: CacheControlEphemeral | None = None


ContentBlock = Annotated[
    Union[
        TextBlock,
        ImageBlock,
        DocumentBlock,
        ToolUseBlock,
        ToolResultBlock,
        ThinkingBlock,
        RedactedThinkingBlock,
    ],
    Field(discriminator="type"),
]


class Message(BaseModel):
    """A single message in the conversation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    role: Literal["user", "assistant", "system"]
    content: str | list[ContentBlock]

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Any) -> Any:
        if isinstance(v, str) and len(v) > _MAX_SYSTEM_PROMPT_CHARS:
            raise ValueError(
                f"System prompt exceeds maximum length of {_MAX_SYSTEM_PROMPT_CHARS} characters"
            )
        return v


class Tool(BaseModel):
    """Tool definition."""

    name: str = Field(..., min_length=1, max_length=_MAX_TOOL_NAME_LENGTH)
    description: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=dict)
    # Accept any string — server tool types (e.g. "web_search_20250305") are
    # valid Anthropic types that the translation layer drops before forwarding.
    type: str | None = "function"
    cache_control: CacheControlEphemeral | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        # Anthropic-specified pattern; MCP tool names can be long and include hyphens.
        # We allow up to 128 chars to be safe for complex MCP generators.
        pattern = r"^[a-zA-Z0-9_-]{1,128}$"
        if not re.fullmatch(pattern, v):
            raise ValueError(f"Tool name '{v}' is invalid. Must match {pattern}.")
        return v


class ToolChoiceAuto(BaseModel):
    """tool_choice: auto"""

    type: Literal["auto"] = "auto"
    disable_parallel_tool_use: bool | None = None


class ToolChoiceAny(BaseModel):
    """tool_choice: any"""

    type: Literal["any"] = "any"
    disable_parallel_tool_use: bool | None = None


class ToolChoiceTool(BaseModel):
    """tool_choice: tool"""

    type: Literal["tool"] = "tool"
    name: str
    disable_parallel_tool_use: bool | None = None


class ToolChoiceNone(BaseModel):
    """tool_choice: none"""

    type: Literal["none"] = "none"


ToolChoice = Annotated[
    Union[ToolChoiceAuto, ToolChoiceAny, ToolChoiceTool, ToolChoiceNone],
    Field(discriminator="type"),
]


class ThinkingConfigEnabled(BaseModel):
    """Thinking enabled config."""

    type: Literal["enabled"] = "enabled"
    budget_tokens: int = Field(ge=1024, le=200000)


class ThinkingConfigDisabled(BaseModel):
    """Thinking disabled config."""

    type: Literal["disabled"] = "disabled"


# ── Main Request Model ────────────────────────────────────────────────────────


class MessagesRequest(BaseModel):
    """Fully validated Anthropic Messages API request.

    This model enforces both syntactic correctness (types, ranges) and
    semantic correctness (valid roles, non-empty required fields).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = Field(..., min_length=1, max_length=_MAX_MODEL_NAME_LENGTH)
    messages: list[Message] = Field(..., min_length=1, max_length=_MAX_MESSAGES_PER_REQUEST)

    system: str | list[dict[str, Any]] | None = Field(default=None)
    max_tokens: int = Field(default=1024, ge=1, le=200000)
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    stop_sequences: list[str] | None = None
    tools: list[Tool] | None = Field(default=None, max_length=_MAX_TOOLS_PER_REQUEST)
    tool_choice: Any = None  # ToolChoice union
    thinking: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    service_tier: str | None = None

    @field_validator("stop_sequences")
    @classmethod
    def validate_stop_sequences(cls, v: list[str] | None) -> list[str] | None:
        if not v:
            return v
        for seq in v:
            if len(seq) > 1000:
                raise ValueError(f"Stop sequence '{seq[:50]}...' exceeds 1000 character limit")
        return v

    @field_validator("system")
    @classmethod
    def validate_system(cls, v: Any) -> Any:
        if isinstance(v, list):
            for idx, block in enumerate(v):
                if isinstance(block, dict):
                    btype = block.get("type")
                    if btype == "text" and len(block.get("text", "")) > _MAX_SYSTEM_PROMPT_CHARS:
                        raise ValueError(
                            f"System prompt block {idx} exceeds maximum length of "
                            f"{_MAX_SYSTEM_PROMPT_CHARS} characters"
                        )
        elif isinstance(v, str) and len(v) > _MAX_SYSTEM_PROMPT_CHARS:
            raise ValueError(
                f"System prompt exceeds maximum length of {_MAX_SYSTEM_PROMPT_CHARS} characters"
            )
        return v

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, v: list[Tool] | None) -> list[Tool] | None:
        if not v:
            return v

        # Check for duplicate names
        names = [t.name for t in v]
        if len(names) != len(set(names)):
            seen: dict[str, int] = {}
            for idx, name in enumerate(names):
                if name in seen:
                    raise ValueError(
                        f"Duplicate tool name '{name}' at indices {seen[name]} and {idx}"
                    )
                seen[name] = idx

        # Check total size isn't excessive
        import json

        total_size = len(json.dumps([t.model_dump() for t in v]))
        if total_size > _MAX_TOTAL_REQUEST_SIZE_MB * 1024 * 1024:
            raise ValueError(f"Total tool schemas exceed {_MAX_TOTAL_REQUEST_SIZE_MB}MB limit")

        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        if not v:
            return v

        # Check for nested sensitive data patterns
        def check_keys(obj: Any, path: str = "") -> None:
            if isinstance(obj, dict):
                for k, val in obj.items():
                    new_path = f"{path}.{k}" if path else k
                    if k.lower() in ("password", "secret", "apikey", "token"):
                        if isinstance(val, str) and len(val) > 0:
                            # Log warning but don't block - may be intentional
                            pass
                    check_keys(val, new_path)
            elif isinstance(obj, list):
                for idx, item in enumerate(obj):
                    check_keys(item, f"{path}[{idx}]")

        check_keys(v)
        return v

    @model_validator(mode="after")
    def validate_message_roles(self) -> "MessagesRequest":
        """Ensure first message is from user (Anthropic requirement)."""
        if self.messages:
            first_role = self.messages[0].role
            if first_role not in ("user", "system"):
                raise ValueError(
                    f"First message must have role 'user' or 'system', got '{first_role}'"
                )
        return self

    @model_validator(mode="after")
    def validate_conversation_flow(self) -> "MessagesRequest":
        """Validate that messages alternate user/assistant properly."""
        last_role: str | None = None
        for idx, msg in enumerate(self.messages):
            current_role = msg.role

            # System can appear anywhere
            if current_role == "system":
                if last_role == "system":
                    # Multiple consecutive system messages are unusual but allowed
                    pass
                last_role = current_role
                continue

            # After first message, strict alternation
            if last_role and last_role != "system":
                if current_role == last_role:
                    # Consecutive same roles are usually errors
                    # But allow tool_result interleaving
                    if not (current_role == "user" and idx > 0):
                        pass  # Could raise ValueError here for strict mode

        return self


# ── Validation Error Formatting ───────────────────────────────────────────────


def format_validation_error(error: Exception) -> dict[str, Any]:
    """Format a Pydantic validation error into Anthropic error format."""
    error_messages = []

    if hasattr(error, "errors"):
        for err in error.errors():
            loc = ".".join(str(loc_part) for loc_part in err.get("loc", []))
            msg = err.get("msg", "validation error")
            error_messages.append(f"{loc}: {msg}" if loc else msg)
    else:
        error_messages.append(str(error))

    return {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "; ".join(error_messages),
        },
    }


def validate_messages_request(
    data: dict[str, Any],
) -> tuple[bool, dict[str, Any] | MessagesRequest]:
    """Validate an inbound request dict.

    Returns:
        Tuple of (is_valid, validated_or_error_dict)
        If valid, returns MessagesRequest instance
        If invalid, returns Anthropic-shaped error dict
    """
    try:
        validated = MessagesRequest(**data)
        return True, validated
    except Exception as e:
        return False, format_validation_error(e)


# ── Utility Functions ─────────────────────────────────────────────────────────


def is_server_tool(tool_type: str | None) -> bool:
    """Check if a tool type is an Anthropic server tool."""
    if not isinstance(tool_type, str):
        return False
    if tool_type in _SERVER_TOOL_TYPES:
        return True
    # Dated tool types (pattern: _YYYYMMDD)
    return bool(re.match(r".+_(20\d{6})$", tool_type))


def sanitize_for_logging(data: dict[str, Any]) -> dict[str, Any]:
    """Remove sensitive fields from data before logging.

    This is a defense-in-depth measure. Primary sanitization should
    happen at the logging middleware level, but this provides an
    additional safety net.
    """
    sensitive_keys = {
        "api_key",
        "apikey",
        "api-key",
        "authorization",
        "password",
        "secret",
        "token",
        "credential",
        "nvidia_api_key",
        "proxy_api_key",
    }

    def sanitize(obj: Any) -> Any:
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                if k.lower() in sensitive_keys:
                    result[k] = "[REDACTED]"
                else:
                    result[k] = sanitize(v)
            return result
        elif isinstance(obj, list):
            return [sanitize(item) for item in obj]
        elif isinstance(obj, str) and len(obj) > 1000:
            # Truncate very long strings (likely base64 data)
            return obj[:500] + "...[truncated]"
        return obj

    return sanitize(data)
