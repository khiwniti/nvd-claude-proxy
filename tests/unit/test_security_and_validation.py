"""Tests for security middleware and request validation.

These tests verify:
1. SSRF protection blocks dangerous URLs
2. Security headers are added to responses
3. Request validation catches malformed requests
4. Circuit breaker patterns work correctly
"""

from __future__ import annotations

import pytest
import time
import asyncio

from starlette.requests import Request
from starlette.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock

from nvd_claude_proxy.middleware.security import (
    is_url_blocked,
    extract_urls_from_body,
    SecurityHeadersMiddleware,
    SSRFProtectionMiddleware,
)
from nvd_claude_proxy.schemas.validators import (
    MessagesRequest,
    validate_messages_request,
    sanitize_for_logging,
    is_server_tool,
    format_validation_error,
)
from nvd_claude_proxy.util.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitState,
    CircuitBreakerRegistry,
)


# ── SSRF Protection Tests ─────────────────────────────────────────────────────

class TestSSRFProtection:
    def test_block_localhost_http(self):
        """Localhost URLs should be blocked."""
        blocked, reason = is_url_blocked("http://localhost/index.html")
        assert blocked, f"localhost should be blocked: {reason}"
        
        blocked, reason = is_url_blocked("http://127.0.0.1/admin")
        assert blocked, f"127.0.0.1 should be blocked: {reason}"
        
        blocked, reason = is_url_blocked("http://[::1]/secret")
        assert blocked, f"[::1] should be blocked: {reason}"

    def test_block_cloud_metadata(self):
        """AWS/GCP/Azure metadata endpoints should be blocked."""
        blocked, reason = is_url_blocked("http://169.254.169.254/latest/meta-data/")
        assert blocked, f"AWS metadata should be blocked: {reason}"
        
        blocked, reason = is_url_blocked("https://metadata.google.internal/computeMetadata/v1/")
        assert blocked, f"GCP metadata should be blocked: {reason}"

    def test_block_file_scheme(self):
        """File:// URLs should be blocked."""
        blocked, reason = is_url_blocked("file:///etc/passwd")
        assert blocked, f"file:// should be blocked: {reason}"

    def test_block_javascript_scheme(self):
        """javascript: URLs should be blocked."""
        blocked, reason = is_url_blocked("javascript:alert('xss')")
        assert blocked, f"javascript: should be blocked: {reason}"

    def test_block_private_ip_ranges(self):
        """Private IP ranges should be blocked."""
        blocked, reason = is_url_blocked("http://10.0.0.1/admin")
        assert blocked, f"10.x.x.x should be blocked: {reason}"
        
        blocked, reason = is_url_blocked("http://172.16.0.1/admin")
        assert blocked, f"172.16.x.x should be blocked: {reason}"
        
        blocked, reason = is_url_blocked("http://192.168.1.1/admin")
        assert blocked, f"192.168.x.x should be blocked: {reason}"

    def test_allow_public_urls(self):
        """Public URLs should be allowed."""
        allowed, _ = is_url_blocked("https://api.anthropic.com/v1/messages")
        assert not allowed, "Public URLs should be allowed"
        
        allowed, _ = is_url_blocked("https://images.unsplash.com/photo.jpg")
        assert not allowed, "Image URLs should be allowed"

    def test_block_urls_with_credentials(self):
        """URLs with @ sign (credentials) should be blocked."""
        blocked, reason = is_url_blocked("https://example.com@evil.com/")
        assert blocked, f"URLs with credentials should be blocked: {reason}"


class TestExtractURLsFromBody:
    def test_extract_simple_url(self):
        """Should extract URL from image source."""
        body = {
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": "https://example.com/image.png"
                    }
                }]
            }]
        }
        urls = extract_urls_from_body(body)
        assert ("https://example.com/image.png", "messages[0].content[0].source.url") in urls

    def test_extract_nested_urls(self):
        """Should extract URLs from nested structures."""
        body = {
            "system": [{
                "type": "image",
                "source": {"type": "url", "url": "https://example.com/logo.png"}
            }],
            "messages": [{
                "role": "user",
                "content": "Check this: https://example.com/doc.pdf"
            }]
        }
        urls = extract_urls_from_body(body)
        assert any("example.com" in url for url, _ in urls)


# ── Validation Tests ──────────────────────────────────────────────────────────

class TestMessagesRequestValidation:
    def test_valid_minimal_request(self):
        """Minimal valid request should pass."""
        body = {
            "model": "claude-opus-4-7",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100
        }
        is_valid, result = validate_messages_request(body)
        assert is_valid, f"Valid request should pass: {result}"
        assert isinstance(result, MessagesRequest)

    def test_reject_empty_model(self):
        """Empty model name should be rejected."""
        body = {
            "model": "",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100
        }
        is_valid, result = validate_messages_request(body)
        assert not is_valid, "Empty model should be rejected"
        assert "model" in str(result).lower()

    def test_reject_missing_messages(self):
        """Missing messages should be rejected."""
        body = {
            "model": "claude-opus-4-7",
            "max_tokens": 100
        }
        is_valid, result = validate_messages_request(body)
        assert not is_valid, "Missing messages should be rejected"

    def test_reject_invalid_role(self):
        """Invalid message roles should be rejected."""
        body = {
            "model": "claude-opus-4-7",
            "messages": [{"role": "bot", "content": "Hello"}],
            "max_tokens": 100
        }
        is_valid, result = validate_messages_request(body)
        assert not is_valid, "Invalid role should be rejected"

    def test_reject_temperature_out_of_range(self):
        """Temperature outside 0-2 should be rejected."""
        body = {
            "model": "claude-opus-4-7",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "temperature": 3.0
        }
        is_valid, result = validate_messages_request(body)
        assert not is_valid, "Temperature > 2 should be rejected"

    def test_reject_max_tokens_zero(self):
        """max_tokens of 0 should be rejected."""
        body = {
            "model": "claude-opus-4-7",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 0
        }
        is_valid, result = validate_messages_request(body)
        assert not is_valid, "max_tokens=0 should be rejected"

    def test_reject_duplicate_tool_names(self):
        """Duplicate tool names should be rejected."""
        body = {
            "model": "claude-opus-4-7",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "tools": [
                {"name": "read_file", "input_schema": {"type": "object"}},
                {"name": "read_file", "input_schema": {"type": "object"}},  # Duplicate
            ]
        }
        is_valid, result = validate_messages_request(body)
        assert not is_valid, "Duplicate tool names should be rejected"
        assert "duplicate" in str(result).lower()

    def test_reject_invalid_tool_name(self):
        """Invalid tool names should be rejected."""
        body = {
            "model": "claude-opus-4-7",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "tools": [
                {"name": "invalid name", "input_schema": {"type": "object"}},  # Contains space
            ]
        }
        is_valid, result = validate_messages_request(body)
        assert not is_valid, "Invalid tool name should be rejected"

    def test_accept_valid_tools(self):
        """Valid tools should be accepted."""
        body = {
            "model": "claude-opus-4-7",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "tools": [
                {"name": "read_file", "description": "Read a file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}},
                {"name": "WriteFile", "description": "Write a file", "input_schema": {"type": "object"}},
                {"name": "my-tool_v2", "description": "Test tool", "input_schema": {"type": "object"}},
            ]
        }
        is_valid, result = validate_messages_request(body)
        assert is_valid, f"Valid tools should be accepted: {result}"

    def test_first_message_must_be_user_or_system(self):
        """First message must be from user or system."""
        body = {
            "model": "claude-opus-4-7",
            "messages": [{"role": "assistant", "content": "Hello"}],
            "max_tokens": 100
        }
        is_valid, result = validate_messages_request(body)
        assert not is_valid, "First message cannot be from assistant"


class TestServerToolDetection:
    def test_detect_server_tools(self):
        """Should detect Anthropic server tools."""
        assert is_server_tool("web_search_20250305")
        assert is_server_tool("bash_20250124")
        assert is_server_tool("computer_20250728")
        assert is_server_tool("memory_20250818")

    def test_allow_regular_tools(self):
        """Should allow regular function tools."""
        assert not is_server_tool("function")
        assert not is_server_tool("custom")
        assert not is_server_tool(None)
        assert not is_server_tool("read_file")
        assert not is_server_tool("my_mcp__tool")


class TestSanitizeForLogging:
    def test_redacts_api_keys(self):
        """Should redact API keys."""
        data = {
            "api_key": "secret123",
            "NVIDIA_API_KEY": "key456",
            "request": {"data": "normal"}
        }
        sanitized = sanitize_for_logging(data)
        assert sanitized["api_key"] == "[REDACTED]"
        assert sanitized["NVIDIA_API_KEY"] == "[REDACTED]"
        assert sanitized["request"]["data"] == "normal"

    def test_truncates_long_strings(self):
        """Should truncate very long strings."""
        long_string = "x" * 2000
        data = {"content": long_string}
        sanitized = sanitize_for_logging(data)
        assert len(sanitized["content"]) < len(long_string)
        assert "[truncated]" in sanitized["content"]


# ── Circuit Breaker Tests ─────────────────────────────────────────────────────

class TestCircuitBreaker:
    @pytest.fixture
    def breaker(self):
        """Create a fresh circuit breaker for each test."""
        return CircuitBreaker(
            "test_service",
            CircuitBreakerConfig(
                failure_threshold=3,
                success_threshold=2,
                timeout=1.0,
                half_open_max_calls=2
            )
        )

    @pytest.mark.asyncio
    async def test_initial_state_closed(self, breaker):
        """Circuit should start in closed state."""
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_success_increments_count(self, breaker):
        """Successful calls should not affect circuit."""
        async def success():
            return "ok"
        
        await breaker.call(success)
        assert breaker.state == CircuitState.CLOSED
        assert breaker._success_calls == 1

    @pytest.mark.asyncio
    async def test_failures_open_circuit(self, breaker):
        """Consecutive failures should open the circuit."""
        async def fail():
            raise ValueError("test error")
        
        for _ in range(breaker.config.failure_threshold):
            with pytest.raises(ValueError):
                await breaker.call(fail)
        
        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_open_circuit_rejects_requests(self, breaker):
        """Open circuit should reject requests immediately when within timeout."""
        # Force open the circuit AND set a recent failure time
        await breaker.force_open()
        breaker._last_failure_time = time.time()  # Set recent failure to prevent immediate half-open
        
        async def dummy():
            return "test"
        
        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            await breaker.call(dummy)
        
        assert exc_info.value.upstream == "test_service"

    @pytest.mark.asyncio
    async def test_half_open_after_timeout(self, breaker):
        """After timeout, circuit should go to half-open."""
        await breaker.force_open()
        breaker._last_failure_time = time.time() - breaker.config.timeout - 1  # Past timeout
        
        # Next call should transition to half-open
        async def success():
            return "ok"
        
        # The call will succeed and transition to half-open
        result = await breaker.call(success)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_success_threshold_closes_circuit(self, breaker):
        """Enough successes in half-open should close circuit."""
        # Open the circuit with past timeout
        await breaker.force_open()
        breaker._last_failure_time = time.time() - breaker.config.timeout - 1  # Past timeout
        
        # Make success_threshold successful calls
        async def success():
            return "ok"
        
        for _ in range(breaker.config.success_threshold):
            await breaker.call(success)
        
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_failure_in_half_open_reopens(self, breaker):
        """Failure in half-open should reopen circuit."""
        await breaker.force_open()
        breaker._last_failure_time = time.time() - breaker.config.timeout - 1  # Past timeout
        
        # First call succeeds (transitions to half-open)
        async def success():
            return "ok"
        await breaker.call(success)
        assert breaker.state == CircuitState.HALF_OPEN
        
        # Second call fails (reopens)
        async def fail():
            raise ValueError("fail")
        try:
            await breaker.call(fail)
        except ValueError:
            pass
        
        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, breaker):
        """Should track call metrics."""
        metrics = breaker.metrics
        assert "total_calls" in metrics
        assert "success_calls" in metrics
        assert "failure_calls" in metrics
        assert "state" in metrics

    @pytest.mark.asyncio
    async def test_force_close_resets_state(self, breaker):
        """force_close should reset circuit to closed."""
        await breaker.force_open()
        await breaker.force_close()
        assert breaker.state == CircuitState.CLOSED


class TestCircuitBreakerRegistry:
    @pytest.mark.asyncio
    async def test_get_or_create(self):
        """Should create new breaker or return existing."""
        registry = CircuitBreakerRegistry()
        
        breaker1 = await registry.get_or_create("service1")
        breaker2 = await registry.get_or_create("service1")
        
        assert breaker1 is breaker2, "Should return same instance"
        
        breaker3 = await registry.get_or_create("service2")
        assert breaker3 is not breaker1, "Different services get different breakers"

    @pytest.mark.asyncio
    async def test_get_all_metrics(self):
        """Should return metrics for all breakers."""
        registry = CircuitBreakerRegistry()
        
        await registry.get_or_create("service1")
        await registry.get_or_create("service2")
        
        all_metrics = await registry.get_all_metrics()
        assert "service1" in all_metrics
        assert "service2" in all_metrics


# ── Integration Tests ─────────────────────────────────────────────────────────

class TestSecurityMiddlewareIntegration:
    """Test security middleware with the full app."""
    
    def test_security_headers_present(self):
        """Security headers should be present in responses."""
        from nvd_claude_proxy.app import create_app
        
        with TestClient(create_app()) as client:
            response = client.get("/healthz")
            
            assert response.headers.get("X-Content-Type-Options") == "nosniff"
            assert response.headers.get("X-Frame-Options") == "DENY"
            assert response.headers.get("X-XSS-Protection") == "1; mode=block"
            assert response.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
            assert "Cache-Control" in response.headers

    def test_ssrf_protection_on_messages(self):
        """SSRF protection should block dangerous URLs."""
        from nvd_claude_proxy.app import create_app
        
        with TestClient(create_app()) as client:
            response = client.post(
                "/v1/messages",
                json={
                    "model": "claude-opus-4-7",
                    "messages": [{
                        "role": "user",
                        "content": [{
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": "file:///etc/passwd"
                            }
                        }]
                    }],
                    "max_tokens": 100
                }
            )
            
            # Should be rejected with 400
            assert response.status_code == 400
            data = response.json()
            assert data["error"]["type"] == "invalid_request_error"