from __future__ import annotations

import asyncio
import json
import random
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
import structlog

from .._version import __version__ as _version
from ..config.settings import Settings

_log = structlog.get_logger("nvd_claude_proxy.nvidia")

# 5xx responses and transient network errors are candidates for retry. 429
# means we're hitting the 40 RPM free-tier cap — retrying with backoff respects
# it but won't help once we're truly rate-limited; still worth one gentle try.
_RETRY_STATUSES = {429, 500, 502, 503, 504}


class NvidiaClient:
    """Async httpx wrapper for NVIDIA NIM `/v1/chat/completions`."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.nvidia_base_url,
            timeout=httpx.Timeout(settings.request_timeout_seconds, connect=10.0),
            headers={
                "Authorization": f"Bearer {settings.nvidia_api_key}",
                "Accept": "application/json",
                "User-Agent": f"nvd-claude-proxy/{_version}",
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ── internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _backoff_seconds(attempt: int) -> float:
        """Exponential backoff with jitter. attempt is 1-based."""
        base = min(2.0 ** (attempt - 1), 8.0)
        return base + random.random() * 0.5

    async def chat_completions(self, payload: dict) -> httpx.Response:
        """POST with retry on transient errors. Returns the final response
        (possibly an error — caller maps it)."""
        last_exc: Exception | None = None
        for attempt in range(1, self._settings.max_retries + 2):
            try:
                resp = await self._client.post("/chat/completions", json=payload)
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as exc:
                last_exc = exc
                _log.warning("nvidia.network_error", attempt=attempt, err=str(exc))
                if attempt > self._settings.max_retries:
                    raise
                await asyncio.sleep(self._backoff_seconds(attempt))
                continue
            if resp.status_code in _RETRY_STATUSES and attempt <= self._settings.max_retries:
                _log.warning(
                    "nvidia.transient_error",
                    attempt=attempt,
                    status=resp.status_code,
                )
                await resp.aclose()
                await asyncio.sleep(self._backoff_seconds(attempt))
                continue
            return resp
        # Shouldn't reach here; loop either returned or raised.
        if last_exc:
            raise last_exc
        raise RuntimeError("nvidia retry loop exited unexpectedly")

    async def list_models(self) -> httpx.Response:
        return await self._client.get("/models")

    @asynccontextmanager
    async def _stream(self, payload: dict):
        async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
            yield resp

    async def astream_chat_completions(self, payload: dict) -> AsyncIterator[dict]:
        """Yield parsed OpenAI SSE JSON chunks. Swallows the `[DONE]` sentinel.

        Raises `httpx.HTTPStatusError` if the upstream returns >=400 before the
        stream begins.
        """
        from nvd_claude_proxy.util.sse import SSEDecoder

        payload = {**payload}
        payload.setdefault("stream_options", {"include_usage": True})
        async with self._stream(payload) as resp:
            if resp.status_code >= 400:
                body = await resp.aread()
                raise httpx.HTTPStatusError(
                    f"NVIDIA {resp.status_code}: {body.decode(errors='replace')}",
                    request=resp.request,
                    response=resp,
                )

            decoder = SSEDecoder()
            async for chunk in resp.aiter_bytes():
                events = decoder.decode(chunk)
                for event in events:
                    if event.data == "[DONE]":
                        return
                    try:
                        yield json.loads(event.data)
                    except json.JSONDecodeError:
                        continue
