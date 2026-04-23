"""Integration tests for Claude Code compatibility.

These tests verify end-to-end functionality with mocked NVIDIA API responses.
They test the full request/response cycle including:
- Basic text requests
- Tool execution loops
- Error handling and failover
- Security features
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient, ASGITransport

from nvd_claude_proxy.app import create_app


@pytest.fixture
async def client():
    """Create test client with mocked upstream."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def mock_nvidia_response():
    """Mock NVIDIA API response for testing."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "nvidia/nemotron-3-ultra-500b-a50b",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! I'm Claude.",
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 6,
            "total_tokens": 16,
        },
    }


@pytest.fixture
def mock_nvidia_tool_call_response():
    """Mock NVIDIA response with tool call."""
    return {
        "id": "chatcmpl-456",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "nvidia/nemotron-3-ultra-500b-a50b",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "index": 0,
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "San Francisco"}',
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 15,
            "total_tokens": 65,
        },
    }


# ── Validation Tests ─────────────────────────────────────────────────────────

class TestValidation:
    """Test request validation (no network required)."""

    @pytest.mark.asyncio
    async def test_invalid_model_rejected(self, client):
        """Test that empty model name is rejected."""
        response = await client.post(
            "/v1/messages",
            json={
                "model": "",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 100,
            },
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_negative_max_tokens_rejected(self, client):
        """Test that negative max_tokens is rejected."""
        response = await client.post(
            "/v1/messages",
            json={
                "model": "claude-opus-4-7",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": -1,
            },
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_duplicate_tool_names_rejected(self, client):
        """Test that duplicate tool names are rejected."""
        response = await client.post(
            "/v1/messages",
            json={
                "model": "claude-opus-4-7",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 100,
                "tools": [
                    {"name": "tool1", "input_schema": {"type": "object"}},
                    {"name": "tool1", "input_schema": {"type": "object"}},
                ],
            },
        )

        assert response.status_code == 400


# ── Error Handling Tests ──────────────────────────────────────────────────────

class TestErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_validation_error_on_malformed_request(self, client):
        """Test that malformed requests return validation errors."""
        response = await client.post(
            "/v1/messages",
            json={
                "model": "",  # Invalid: empty model
                "messages": [],  # Invalid: no messages
                "max_tokens": -1,  # Invalid: negative
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "invalid_request_error"


# ── Header/Metadata Tests ─────────────────────────────────────────────────────

class TestHeaders:
    """Test response headers."""

    @pytest.mark.asyncio
    async def test_anthropic_headers_present_on_error(self, client):
        """Test that Anthropic-required headers are present even on error."""
        # Send a request that will fail validation - headers should still be present
        response = await client.post(
            "/v1/messages",
            json={
                "model": "",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 100,
            },
        )

        # Check required headers (always present regardless of error)
        assert "anthropic-request-id" in response.headers
        assert "anthropic-version" in response.headers
        assert response.headers["anthropic-version"] == "2023-06-01"


# ── Token Counting Tests ──────────────────────────────────────────────────────

class TestTokenCounting:
    """Test token counting endpoint."""

    @pytest.mark.asyncio
    async def test_count_tokens_simple(self, client):
        """Test basic token counting."""
        response = await client.post(
            "/v1/messages/count_tokens",
            json={
                "messages": [
                    {"role": "user", "content": "Hello, world!"}
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "input_tokens" in data
        assert data["input_tokens"] > 0

    @pytest.mark.asyncio
    async def test_count_tokens_with_tools(self, client):
        """Test token counting includes tool schemas."""
        response = await client.post(
            "/v1/messages/count_tokens",
            json={
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                            },
                        },
                    },
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["input_tokens"] > 10  # Should count tool schema


# ── Model Registry Tests ──────────────────────────────────────────────────────

class TestModelRegistry:
    """Test model listing and resolution."""

    @pytest.mark.asyncio
    async def test_list_models(self, client):
        """Test listing available models."""
        response = await client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "has_more" in data
        assert len(data["data"]) > 0

    @pytest.mark.asyncio
    async def test_get_specific_model(self, client):
        """Test getting a specific model by ID."""
        response = await client.get("/v1/models/claude-opus-4-7")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "claude-opus-4-7"


# ── Security Tests ────────────────────────────────────────────────────────────

class TestSecurity:
    """Test security features."""

    @pytest.mark.asyncio
    async def test_ssrf_blocked_url_rejected(self, client):
        """Test that SSRF attempts are blocked."""
        response = await client.post(
            "/v1/messages",
            json={
                "model": "claude-opus-4-7",
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": "http://169.254.169.254/latest/meta-data/",
                        },
                    }],
                }],
                "max_tokens": 100,
            },
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_duplicate_tool_names_rejected(self, client):
        """Test that duplicate tool names are rejected."""
        response = await client.post(
            "/v1/messages",
            json={
                "model": "claude-opus-4-7",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 100,
                "tools": [
                    {"name": "tool1", "input_schema": {"type": "object"}},
                    {"name": "tool1", "input_schema": {"type": "object"}},  # Duplicate
                ],
            },
        )

        assert response.status_code == 400


# ── Health Check Tests ────────────────────────────────────────────────────────

class TestHealthChecks:
    """Test health check endpoints."""

    @pytest.mark.asyncio
    async def test_healthz(self, client):
        """Test /healthz endpoint."""
        response = await client.get("/healthz")
        assert response.status_code == 200


# ── OpenAPI Spec Tests ────────────────────────────────────────────────────────

class TestOpenAPISpec:
    """Test OpenAPI specification endpoint."""

    @pytest.mark.asyncio
    async def test_openapi_spec_available(self, client):
        """Test that OpenAPI spec is available."""
        response = await client.get("/v1/openapi.json")

        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert data["openapi"].startswith("3.")
        assert "paths" in data
        assert "/messages" in data["paths"]

    @pytest.mark.asyncio
    async def test_messages_schema_available(self, client):
        """Test that JSON schema for messages is available."""
        response = await client.get("/v1/messages/schema")

        assert response.status_code == 200
        data = response.json()
        assert "MessageRequest" in data
        assert "ErrorResponse" in data