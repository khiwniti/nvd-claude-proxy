from __future__ import annotations

from nvd_claude_proxy.errors.mapper import openai_error_to_anthropic


def test_429_is_rate_limit():
    status, body = openai_error_to_anthropic(429, {"error": {"message": "slow down"}})
    assert status == 429
    assert body["error"]["type"] == "rate_limit_error"
    assert body["error"]["message"] == "slow down"


def test_unknown_status_defaults_to_api_error():
    status, body = openai_error_to_anthropic(418, {})
    assert status == 418
    assert body["error"]["type"] == "api_error"


def test_401_is_auth_error():
    status, body = openai_error_to_anthropic(401, {"error": {"message": "nope"}})
    assert body["error"]["type"] == "authentication_error"


def test_413_is_request_too_large():
    _, body = openai_error_to_anthropic(413, {"error": {"message": "big"}})
    assert body["error"]["type"] == "request_too_large"


def test_529_is_overloaded():
    _, body = openai_error_to_anthropic(529, {})
    assert body["error"]["type"] == "overloaded_error"


def test_503_is_overloaded():
    _, body = openai_error_to_anthropic(503, {})
    assert body["error"]["type"] == "overloaded_error"
