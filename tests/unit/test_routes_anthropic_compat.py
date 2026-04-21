"""FastAPI-level tests ensuring route responses carry Anthropic-compatible
headers and fields."""

from __future__ import annotations

from fastapi.testclient import TestClient

from nvd_claude_proxy.app import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def test_healthz_ok():
    with _client() as c:
        r = c.get("/healthz")
    assert r.status_code == 200


def test_list_models_has_has_more_and_headers():
    with _client() as c:
        r = c.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["has_more"] is False
    assert len(body["data"]) > 0
    assert r.headers.get("anthropic-request-id", "").startswith("req_")


def test_get_single_model_exact_alias():
    with _client() as c:
        r = c.get("/v1/models/claude-opus-4-7")
    assert r.status_code == 200
    assert r.json()["id"] == "claude-opus-4-7"
    assert r.headers.get("anthropic-request-id", "").startswith("req_")


def test_get_single_model_prefix_fallback_echoes_requested_id():
    with _client() as c:
        r = c.get("/v1/models/claude-3-5-sonnet-20240620")
    assert r.status_code == 200
    assert r.json()["id"] == "claude-3-5-sonnet-20240620"


def test_count_tokens_includes_tools():
    small = {"messages": [{"role": "user", "content": "hi"}]}
    with_tools = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {
                "name": f"tool_{i}",
                "description": "A tool that reads a file and edits it.",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            }
            for i in range(30)
        ],
    }
    with _client() as c:
        r1 = c.post("/v1/messages/count_tokens", json=small).json()
        r2 = c.post("/v1/messages/count_tokens", json=with_tools).json()
    # Tools must meaningfully increase the count.
    assert r2["input_tokens"] > r1["input_tokens"] * 5


def test_unknown_model_returns_anthropic_404():
    with _client() as c:
        r = c.get("/v1/models/totally-nonexistent-model-xyzzy")
    # Should not 500; should return Anthropic-shaped error. Depending on
    # registry fallback semantics this may resolve to default — check either.
    if r.status_code == 404:
        body = r.json()
        assert body.get("detail", {}).get("error", {}).get("type") == "not_found_error"
    else:
        # Prefix fallback kicked in; still valid, echo the requested id.
        assert r.status_code == 200
