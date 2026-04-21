from __future__ import annotations

import httpx

from nvd_claude_proxy.clients.nvidia_client import NvidiaClient
from nvd_claude_proxy.config.settings import Settings


def _settings() -> Settings:
    return Settings(
        NVIDIA_API_KEY="nvapi-test",
        NVIDIA_BASE_URL="https://integrate.api.nvidia.com/v1",
        PROXY_HOST="127.0.0.1",
        PROXY_PORT=8787,
        PROXY_API_KEY=None,
        LOG_LEVEL="INFO",
        MODEL_CONFIG_PATH="config/models.yaml",
        REQUEST_TIMEOUT_SECONDS=30.0,
        MAX_RETRIES=2,
    )


async def test_retries_on_502(monkeypatch):
    """502 → 502 → 200. Client should retry twice and return the 200."""
    s = _settings()
    # Avoid any real sleeping.
    import nvd_claude_proxy.clients.nvidia_client as mod

    async def _no_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr(mod.asyncio, "sleep", _no_sleep)

    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if call_count["n"] < 3:
            return httpx.Response(502, text="bad gateway")
        return httpx.Response(
            200,
            json={
                "id": "cmpl-1",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {},
            },
        )

    transport = httpx.MockTransport(handler)
    client = NvidiaClient(s)
    client._client = httpx.AsyncClient(
        base_url=s.nvidia_base_url, transport=transport
    )
    try:
        resp = await client.chat_completions({"model": "x", "messages": []})
    finally:
        await client.aclose()
    assert resp.status_code == 200
    assert call_count["n"] == 3


async def test_gives_up_after_max_retries(monkeypatch):
    s = _settings()
    import nvd_claude_proxy.clients.nvidia_client as mod

    async def _no_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr(mod.asyncio, "sleep", _no_sleep)

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, text="unavailable")

    transport = httpx.MockTransport(handler)
    client = NvidiaClient(s)
    client._client = httpx.AsyncClient(
        base_url=s.nvidia_base_url, transport=transport
    )
    try:
        resp = await client.chat_completions({"model": "x", "messages": []})
    finally:
        await client.aclose()
    # 1 initial + max_retries (2) retries = 3 attempts total
    assert calls["n"] == 3
    assert resp.status_code == 503


async def test_400_not_retried(monkeypatch):
    """User errors (400) must NOT be retried — wastes tokens + user time."""
    s = _settings()
    import nvd_claude_proxy.clients.nvidia_client as mod

    async def _no_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr(mod.asyncio, "sleep", _no_sleep)

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="invalid request")

    transport = httpx.MockTransport(handler)
    client = NvidiaClient(s)
    client._client = httpx.AsyncClient(
        base_url=s.nvidia_base_url, transport=transport
    )
    try:
        resp = await client.chat_completions({"model": "x", "messages": []})
    finally:
        await client.aclose()
    assert calls["n"] == 1
    assert resp.status_code == 400
