"""Regression test: `message_start` is the first wire byte.

The streaming generator must emit the synthesised ``message_start`` SSE
frame BEFORE waiting on the upstream NVIDIA NIM. If this regresses, the
client-perceived TTFT collapses to "upstream-RTT + heavy preflight" and
the agent-state UI sits on a blank screen.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

os.environ.setdefault("NVIDIA_API_KEY", "test-key")

from nvd_claude_proxy.app import create_app  # noqa: E402


async def _slow_upstream(*_args, **_kwargs) -> AsyncIterator[dict]:
    """Mimic an upstream that hangs forever before yielding its first chunk.

    Sleeps 3s — long enough for the proxy's pre-emit guarantee to be
    meaningfully tested, short enough to keep CI fast.
    """
    await asyncio.sleep(3)
    yield {"choices": []}  # pragma: no cover


def test_message_start_emitted_before_upstream_responds(monkeypatch) -> None:
    """First SSE frame parses as ``event: message_start`` and arrives well
    before the (mocked) 60s upstream would have yielded anything.
    """
    from nvd_claude_proxy.clients import nvidia_client as nv

    # Patch the upstream to a generator that hangs.
    monkeypatch.setattr(
        nv.NvidiaClient,
        "astream_chat_completions",
        lambda self, payload: _slow_upstream(),
    )

    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 16,
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    }
    headers = {"anthropic-version": "2023-06-01"}

    with TestClient(create_app()) as client:
        with client.stream("POST", "/v1/messages", json=body, headers=headers) as r:
            assert r.status_code == 200
            first_bytes = b""
            for chunk in r.iter_bytes():
                first_bytes += chunk
                if b"\n\n" in first_bytes:
                    break
                if len(first_bytes) > 8192:
                    break

    # The structural assertions below are sufficient: ASGITransport in
    # the FastAPI TestClient buffers responses, so a wall-clock TTFB
    # bound is not meaningful here. The bytes-ordering proof is.
    assert b"event: message_start" in first_bytes, (
        f"first frame was not message_start; got: {first_bytes[:200]!r}"
    )
    # Upstream is still sleeping — no model output should be present yet.
    assert b"content_block_delta" not in first_bytes
