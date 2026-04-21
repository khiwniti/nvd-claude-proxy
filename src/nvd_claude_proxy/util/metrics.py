"""Prometheus metrics for the proxy.

All metrics are guarded by a try/except so the proxy starts fine even when
``prometheus-client`` is not installed (it is an optional dependency).

Usage::

    from .util.metrics import inc_requests, observe_tokens

Noop stubs are provided when the library is absent.
"""

from __future__ import annotations

import logging

_log = logging.getLogger("nvd_claude_proxy.metrics")

try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

    _requests_total = Counter(
        "nvd_proxy_requests_total",
        "Total number of /v1/messages requests",
        ["model", "stream", "status"],
    )
    _tokens_input = Counter(
        "nvd_proxy_input_tokens_total",
        "Total input tokens billed by upstream",
        ["model"],
    )
    _tokens_output = Counter(
        "nvd_proxy_output_tokens_total",
        "Total output tokens billed by upstream",
        ["model"],
    )
    _request_duration = Histogram(
        "nvd_proxy_request_duration_seconds",
        "End-to-end request duration (non-streaming only)",
        ["model"],
        buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120],
    )
    _ENABLED = True

    def inc_requests(model: str, stream: bool, status: int) -> None:
        _requests_total.labels(model=model, stream=str(stream).lower(), status=str(status)).inc()

    def inc_tokens(model: str, input_tokens: int, output_tokens: int) -> None:
        if input_tokens:
            _tokens_input.labels(model=model).inc(input_tokens)
        if output_tokens:
            _tokens_output.labels(model=model).inc(output_tokens)

    def observe_duration(model: str, seconds: float) -> None:
        _request_duration.labels(model=model).observe(seconds)

    def prometheus_text() -> tuple[bytes, str]:
        """Return (body_bytes, content_type) for the /metrics endpoint."""
        return generate_latest(), CONTENT_TYPE_LATEST

except ImportError:
    _ENABLED = False
    _log.info(
        "prometheus-client not installed; /metrics endpoint will return 501. "
        "`pip install prometheus-client` to enable."
    )

    def inc_requests(model: str, stream: bool, status: int) -> None:  # type: ignore[misc]
        pass

    def inc_tokens(model: str, input_tokens: int, output_tokens: int) -> None:  # type: ignore[misc]
        pass

    def observe_duration(model: str, seconds: float) -> None:  # type: ignore[misc]
        pass

    def prometheus_text() -> tuple[bytes, str]:  # type: ignore[misc]
        raise RuntimeError("prometheus-client not installed")


def is_enabled() -> bool:
    return _ENABLED
