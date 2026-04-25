"""Anthropic-shaped OpenAPI specification endpoint.

This module provides an OpenAPI 3.0 spec that matches Anthropic's
Messages API documentation format. This is useful for:
- SDK generators that produce client libraries
- API documentation tools
- Developer onboarding

The spec is available at GET /v1/openapi.json
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import ORJSONResponse

from ..util.anthropic_headers import standard_response_headers, new_request_id

router = APIRouter()

# ── Anthropic API OpenAPI Schema ──────────────────────────────────────────────

ANTHROPIC_OPENAPI_SCHEMA: dict[str, Any] = {
    "openapi": "3.0.3",
    "info": {
        "title": "Anthropic Messages API",
        "description": """The Anthropic Messages API allows you to create conversational
        and multi-turn AI interactions with Claude models.

        ## Authentication
        Include your API key in the `Authorization` header:
        ```
        Authorization: Bearer YOUR_API_KEY
        ```

        ## Rate Limits
        Rate limits vary by service tier. The response will include
        `anthropic-ratelimit-*` headers indicating your current limits.

        ## Streaming
        Set `stream: true` in your request to receive Server-Sent Events (SSE)
        with incremental response updates.
        """,
        "version": "2023-06-01",
        "contact": {
            "name": "Anthropic API Support",
            "url": "https://anthropic.com/support",
        },
        "license": {
            "name": "Anthropic API Terms of Service",
            "url": "https://anthropic.com/terms",
        },
    },
    "servers": [
        {
            "url": "https://api.anthropic.com/v1",
            "description": "Production",
        },
    ],
    "paths": {
        "/messages": {
            "post": {
                "operationId": "createMessage",
                "summary": "Create a Message",
                "description": """Create a model response for a given prompt.

                The request includes conversation history and the model
                generates the next assistant message.

                **Streaming Response**: When `stream: true`, the response
                will be a series of SSE events. See the streaming section
                for event format details.

                **Tool Use**: Include a `tools` array to enable function
                calling. Claude will return `tool_use` content blocks
                which the client should execute and submit as `tool_result`
                blocks in subsequent requests.
                """,
                "tags": ["Messages"],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/MessageRequest",
                            },
                            "examples": {
                                "simple": {
                                    "summary": "Simple text request",
                                    "value": {
                                        "model": "claude-opus-4-7",
                                        "messages": [{"role": "user", "content": "Hello, Claude"}],
                                        "max_tokens": 1024,
                                    },
                                },
                                "with_tools": {
                                    "summary": "Request with tools",
                                    "value": {
                                        "model": "claude-opus-4-7",
                                        "messages": [
                                            {"role": "user", "content": "What's the weather?"}
                                        ],
                                        "max_tokens": 1024,
                                        "tools": [
                                            {
                                                "name": "get_weather",
                                                "description": "Get current weather",
                                                "input_schema": {
                                                    "type": "object",
                                                    "properties": {
                                                        "location": {
                                                            "type": "string",
                                                            "description": "City name",
                                                        }
                                                    },
                                                    "required": ["location"],
                                                },
                                            }
                                        ],
                                    },
                                },
                                "streaming": {
                                    "summary": "Streaming response",
                                    "value": {
                                        "model": "claude-opus-4-7",
                                        "messages": [{"role": "user", "content": "Write a story"}],
                                        "max_tokens": 2048,
                                        "stream": True,
                                    },
                                },
                            },
                        },
                    },
                },
                "responses": {
                    "201": {
                        "description": "Message created successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Message",
                                },
                            },
                        },
                        "headers": {
                            "anthropic-request-id": {
                                "schema": {"type": "string"},
                                "description": "Unique request identifier",
                            },
                            "anthropic-ratelimit-requests-limit": {
                                "schema": {"type": "string"},
                                "description": "Requests per minute limit",
                            },
                            "anthropic-ratelimit-requests-remaining": {
                                "schema": {"type": "string"},
                                "description": "Remaining requests in window",
                            },
                            "anthropic-ratelimit-tokens-limit": {
                                "schema": {"type": "string"},
                                "description": "Tokens per minute limit",
                            },
                            "anthropic-ratelimit-tokens-remaining": {
                                "schema": {"type": "string"},
                                "description": "Remaining tokens in window",
                            },
                        },
                    },
                    "400": {
                        "description": "Invalid request",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                            },
                        },
                    },
                    "401": {
                        "description": "Authentication failed",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                            },
                        },
                    },
                    "429": {
                        "description": "Rate limit exceeded",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                            },
                        },
                        "headers": {
                            "retry-after": {
                                "schema": {"type": "string"},
                                "description": "Seconds to wait before retrying",
                            },
                        },
                    },
                    "500": {
                        "description": "Internal server error",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                            },
                        },
                    },
                },
            },
        },
        "/messages/count_tokens": {
            "post": {
                "operationId": "countTokens",
                "summary": "Count Tokens",
                "description": "Estimate the token count for a request without executing it.",
                "tags": ["Messages"],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "messages": {
                                        "type": "array",
                                        "items": {"$ref": "#/components/schemas/Message"},
                                    },
                                    "system": {
                                        "oneOf": [
                                            {"type": "string"},
                                            {"type": "array"},
                                        ],
                                    },
                                    "tools": {
                                        "type": "array",
                                        "items": {"$ref": "#/components/schemas/Tool"},
                                    },
                                },
                                "required": ["messages"],
                            },
                        },
                    },
                },
                "responses": {
                    "200": {
                        "description": "Token count",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "input_tokens": {
                                            "type": "integer",
                                            "description": "Estimated input token count",
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
        "/models": {
            "get": {
                "operationId": "listModels",
                "summary": "List Available Models",
                "description": "Returns a list of available models.",
                "tags": ["Models"],
                "responses": {
                    "200": {
                        "description": "Model list",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "data": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/Model"},
                                        },
                                        "has_more": {
                                            "type": "boolean",
                                            "default": False,
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
        "/models/{model_id}": {
            "get": {
                "operationId": "getModel",
                "summary": "Get Model",
                "description": "Returns a specific model by ID.",
                "tags": ["Models"],
                "parameters": [
                    {
                        "name": "model_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Model identifier",
                    },
                ],
                "responses": {
                    "200": {
                        "description": "Model details",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Model"},
                            },
                        },
                    },
                    "404": {
                        "description": "Model not found",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                            },
                        },
                    },
                },
            },
        },
    },
    "components": {
        "schemas": {
            "MessageRequest": {
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Model to use (e.g., 'claude-opus-4-7')",
                    },
                    "messages": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/Message"},
                        "description": "Conversation messages",
                    },
                    "system": {
                        "oneOf": [
                            {"type": "string"},
                            {
                                "type": "array",
                                "items": {"$ref": "#/components/schemas/ContentBlock"},
                            },
                        ],
                        "description": "System prompt (string or content blocks)",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Maximum tokens to generate",
                    },
                    "stream": {
                        "type": "boolean",
                        "default": False,
                        "description": "Stream response as SSE events",
                    },
                    "temperature": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 2,
                        "description": "Sampling temperature",
                    },
                    "top_p": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Nucleus sampling threshold",
                    },
                    "tools": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/Tool"},
                        "description": "Available tools for function calling",
                    },
                    "tool_choice": {
                        "oneOf": [
                            {"type": "string", "enum": ["auto", "any", "none"]},
                            {"$ref": "#/components/schemas/ToolChoice"},
                        ],
                        "description": "How to choose which tool to use",
                    },
                    "thinking": {
                        "type": "object",
                        "description": "Extended thinking configuration",
                        "properties": {
                            "type": {"type": "string", "enum": ["enabled", "disabled"]},
                            "budget_tokens": {
                                "type": "integer",
                                "minimum": 1024,
                                "description": "Token budget for thinking",
                            },
                        },
                    },
                    "metadata": {
                        "type": "object",
                        "description": "User-defined metadata (e.g., user_id)",
                    },
                    "stop_sequences": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Custom stop sequences",
                    },
                },
                "required": ["model", "messages", "max_tokens"],
            },
            "Message": {
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": ["user", "assistant", "system"],
                    },
                    "content": {
                        "oneOf": [
                            {"type": "string"},
                            {
                                "type": "array",
                                "items": {"$ref": "#/components/schemas/ContentBlock"},
                            },
                        ],
                    },
                },
                "required": ["role", "content"],
            },
            "ContentBlock": {
                "oneOf": [
                    {"$ref": "#/components/schemas/TextBlock"},
                    {"$ref": "#/components/schemas/ToolUseBlock"},
                    {"$ref": "#/components/schemas/ToolResultBlock"},
                    {"$ref": "#/components/schemas/ThinkingBlock"},
                ],
                "discriminator": {
                    "propertyName": "type",
                    "mapping": {
                        "text": "#/components/schemas/TextBlock",
                        "tool_use": "#/components/schemas/ToolUseBlock",
                        "tool_result": "#/components/schemas/ToolResultBlock",
                        "thinking": "#/components/schemas/ThinkingBlock",
                    },
                },
            },
            "TextBlock": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "const": "text"},
                    "text": {"type": "string"},
                },
                "required": ["type", "text"],
            },
            "ToolUseBlock": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "const": "tool_use"},
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "input": {"type": "object"},
                },
                "required": ["type", "id", "name", "input"],
            },
            "ToolResultBlock": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "const": "tool_result"},
                    "tool_use_id": {"type": "string"},
                    "content": {},
                    "is_error": {"type": "boolean"},
                },
                "required": ["type", "tool_use_id", "content"],
            },
            "ThinkingBlock": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "const": "thinking"},
                    "thinking": {"type": "string"},
                    "signature": {"type": "string"},
                },
                "required": ["type", "thinking"],
            },
            "Tool": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "input_schema": {"type": "object"},
                },
                "required": ["name", "input_schema"],
            },
            "ToolChoice": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["auto", "any", "tool"]},
                    "name": {"type": "string"},
                },
                "required": ["type"],
            },
            "Model": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "display_name": {"type": "string"},
                    "description": {"type": "string"},
                    "max_tokens": {"type": "integer"},
                    "supported_features": {
                        "type": "object",
                        "properties": {
                            "vision": {"type": "boolean"},
                            "tool_use": {"type": "boolean"},
                            "thinking": {"type": "boolean"},
                        },
                    },
                },
                "required": ["id"],
            },
            "ErrorResponse": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "error": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": [
                                    "invalid_request_error",
                                    "authentication_error",
                                    "permission_error",
                                    "not_found_error",
                                    "rate_limit_error",
                                    "api_error",
                                    "overloaded_error",
                                ],
                            },
                            "message": {"type": "string"},
                            "retry_after": {"type": "number"},
                        },
                        "required": ["type", "message"],
                    },
                },
                "required": ["type", "error"],
            },
        },
        "securitySchemes": {
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "x-api-key",
                "description": "API key for authentication",
            },
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "API-Key",
                "description": "Bearer token authentication",
            },
        },
    },
    "tags": [
        {
            "name": "Messages",
            "description": "Create and manage conversations with Claude",
        },
        {
            "name": "Models",
            "description": "List and get information about available models",
        },
    ],
    "security": [
        {"ApiKeyAuth": []},
        {"BearerAuth": []},
    ],
}


@router.get("/v1/openapi.json", include_in_schema=False)
async def get_openapi_spec(request: Request) -> ORJSONResponse:
    """Return the Anthropic-shaped OpenAPI specification.

    This endpoint provides an OpenAPI 3.0 spec matching Anthropic's
    Messages API documentation format.
    """
    request_id = new_request_id()
    return ORJSONResponse(
        ANTHROPIC_OPENAPI_SCHEMA,
        headers=standard_response_headers(request_id),
    )


@router.get("/v1/messages/schema", include_in_schema=False)
async def get_messages_schema(request: Request) -> ORJSONResponse:
    """Return JSON Schema for the Messages API request/response.

    Useful for SDK code generation and validation.
    """
    request_id = new_request_id()

    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "MessageRequest": ANTHROPIC_OPENAPI_SCHEMA["components"]["schemas"]["MessageRequest"],
        "Message": ANTHROPIC_OPENAPI_SCHEMA["components"]["schemas"]["Message"],
        "ContentBlock": ANTHROPIC_OPENAPI_SCHEMA["components"]["schemas"]["ContentBlock"],
        "Tool": ANTHROPIC_OPENAPI_SCHEMA["components"]["schemas"]["Tool"],
        "ErrorResponse": ANTHROPIC_OPENAPI_SCHEMA["components"]["schemas"]["ErrorResponse"],
    }

    return ORJSONResponse(
        schema,
        headers=standard_response_headers(request_id),
    )
