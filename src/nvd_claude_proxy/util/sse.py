from __future__ import annotations

import orjson


def encode_sse(event: str, data: dict) -> bytes:
    """Anthropic SSE frame: `event: X\\ndata: {json}\\n\\n`."""
    return b"event: " + event.encode() + b"\ndata: " + orjson.dumps(data) + b"\n\n"
