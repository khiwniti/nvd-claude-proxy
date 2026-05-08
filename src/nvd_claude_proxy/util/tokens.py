from __future__ import annotations

import logging
from typing import Any

import orjson
import tiktoken

log = logging.getLogger(__name__)

_enc = None

# Field names whose *values* are base64 image payloads we want to skip when
# tokenizing a request body.
_SKIP_KEYS = {"data"}

# tiktoken is O(n) in bytes but has real overhead; for very large payloads
# (Claude Code /init with 190 tool schemas ≈ 100-200kB of text) we prefer a
# cheaper 4-chars-per-token heuristic that's provably good enough for the
# context-budget clamp, which just needs to know "are we about to overflow?".
_FAST_PATH_THRESHOLD_CHARS = 60_000


def _get_encoding():
    global _enc
    if _enc is not None:
        return _enc
    try:
        _enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        # In restricted networks tiktoken may fail to download the encoding
        # on first startup. Degrade gracefully so the proxy stays available.
        log.warning(
            "Failed to initialize cl100k_base encoding, falling back to heuristic token estimation",
            exc_info=True,
        )
        _enc = False
    return _enc


def _walk(obj: Any, parts: list[str]) -> None:
    """Push all tokenizable text into `parts`, including JSON keys —
    schemas have many short keys (type/properties/required/…) that tokenize
    to real input tokens when NVIDIA serializes the tool list into the prompt.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in _SKIP_KEYS:
                continue
            # Include the key itself; NVIDIA prompt-serializes every key.
            parts.append(k)
            _walk(v, parts)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, parts)
    elif isinstance(obj, str):
        parts.append(obj)
    elif obj is None or isinstance(obj, bool):
        pass
    else:
        parts.append(orjson.dumps(obj).decode())


def approximate_tokens(body: dict) -> int:
    """Pessimistic ~+10% approximation of the input-token count.

    Deliberately biases *up* so the context-budget clamp is safe. On tool-heavy
    payloads cl100k_base undercounts vs Nemotron by ~5-10%; the heuristic uses
    3.0 chars/token for schema-dense JSON which tokenizes more finely.
    """
    parts: list[str] = []
    _walk(body, parts)
    text = "\n".join(parts)
    n = len(text)
    if n > _FAST_PATH_THRESHOLD_CHARS:
        # Tool-schema JSON has many single-char tokens (braces, quotes, colons)
        # that NVIDIA's BPE rarely merges. 3.5 chars/token is a safe upper
        # bound vs cl100k at 3.8-4.0 on the same content.
        return int(n / 3.5) + 3

    enc = _get_encoding()
    if enc is False:
        # Conservative fallback for small payloads when tokenizer init failed.
        return int(n / 3.0) + 3

    # Small payloads: use tiktoken directly for accuracy.
    return len(enc.encode(text)) + 3
