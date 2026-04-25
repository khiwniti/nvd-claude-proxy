"""Enhanced Prometheus metrics for production monitoring.

This module adds metrics beyond the basic request/response tracking:
- Cache token accounting (estimated)
- Circuit breaker state tracking
- Load shedding rejection tracking
- Upstream vs proxy duration breakdown
- Stream chunk translation latency
- Cost estimation tracking

Import from this module for enhanced metrics, or use metrics.py for basic metrics.

Usage::

    from .util.metrics_enhanced import (
        inc_cache_tokens,
        set_circuit_breaker_state,
        observe_chunk_translation_latency,
    )
"""

from __future__ import annotations

import logging

_log = logging.getLogger("nvd_claude_proxy.metrics_enhanced")

try:
    from prometheus_client import Counter, Histogram, Gauge

    # ── Cache token metrics (estimated) ───────────────────────────────────
    _tokens_cache_creation = Counter(
        "nvd_proxy_cache_creation_tokens_total",
        "Estimated cache creation tokens",
        ["model"],
    )
    _tokens_cache_read = Counter(
        "nvd_proxy_cache_read_tokens_total",
        "Estimated cache read tokens",
        ["model"],
    )
    _cache_savings_usd = Counter(
        "nvd_proxy_cache_savings_usd",
        "Estimated USD savings from prompt caching",
        ["model"],
    )

    # ── Circuit breaker metrics ───────────────────────────────────────────
    _circuit_breaker_state = Gauge(
        "nvd_proxy_circuit_breaker_state",
        "Circuit breaker state (0=closed, 1=half-open, 2=open)",
        ["upstream"],
    )
    _circuit_breaker_rejected = Counter(
        "nvd_proxy_circuit_breaker_rejected_total",
        "Requests rejected due to open circuit breaker",
        ["upstream"],
    )
    _circuit_breaker_transitions = Counter(
        "nvd_proxy_circuit_breaker_transitions_total",
        "Circuit breaker state transitions",
        ["upstream", "from_state", "to_state"],
    )

    # ── Load shedding metrics ─────────────────────────────────────────────
    _load_shedding_rejected = Counter(
        "nvd_proxy_load_shedding_rejected_total",
        "Requests rejected due to load shedding",
        ["reason"],
    )
    _active_requests = Gauge(
        "nvd_proxy_active_requests",
        "Currently active requests being processed",
    )
    _system_load = Gauge(
        "nvd_proxy_system_load",
        "System load metrics",
        ["metric"],  # cpu_percent, memory_percent, queue_depth
    )

    # ── Upstream metrics ──────────────────────────────────────────────────
    _upstream_duration = Histogram(
        "nvd_proxy_upstream_duration_seconds",
        "Upstream NVIDIA API duration",
        ["model", "status_code"],
        buckets=[0.5, 1, 3, 5, 10, 30, 60, 120, 300],
    )
    _upstream_retries = Counter(
        "nvd_proxy_upstream_retries_total",
        "Total upstream retries",
        ["model", "reason"],
    )

    # ── Stream translation metrics ────────────────────────────────────────
    _stream_chunk_latency = Histogram(
        "nvd_proxy_stream_chunk_latency_seconds",
        "Time to translate each stream chunk",
        ["model", "chunk_type"],
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
    )
    _stream_chunks_total = Counter(
        "nvd_proxy_stream_chunks_total",
        "Total stream chunks processed",
        ["model", "chunk_type"],
    )

    # ── Error metrics by type ─────────────────────────────────────────────
    _errors_total = Counter(
        "nvd_proxy_errors_total",
        "Total errors by type",
        ["error_type", "model", "is_upstream"],
    )

    # ── Cost estimation ───────────────────────────────────────────────────
    _cost_estimated_usd = Counter(
        "nvd_proxy_cost_estimated_usd",
        "Estimated cost in USD",
        ["model"],
    )
    _cost_input_usd = Counter(
        "nvd_proxy_cost_input_usd",
        "Estimated input cost in USD",
        ["model", "is_cached"],
    )
    _cost_output_usd = Counter(
        "nvd_proxy_cost_output_usd",
        "Estimated output cost in USD",
        ["model"],
    )

    # ── Validation metrics ────────────────────────────────────────────────
    _validation_errors = Counter(
        "nvd_proxy_validation_errors_total",
        "Request validation errors by type",
        ["validation_type"],
    )
    _security_blocks = Counter(
        "nvd_proxy_security_blocks_total",
        "Security checks that blocked requests",
        ["check_type"],
    )

    _ENABLED = True

    def inc_cache_tokens(
        model: str,
        cache_creation: int = 0,
        cache_read: int = 0,
    ) -> None:
        """Record estimated cache tokens."""
        if cache_creation:
            _tokens_cache_creation.labels(model=model).inc(cache_creation)
        if cache_read:
            _tokens_cache_read.labels(model=model).inc(cache_read)

    def inc_cache_savings(model: str, usd: float) -> None:
        """Record estimated cache cost savings."""
        if usd > 0:
            _cache_savings_usd.labels(model=model).inc(usd)

    def set_circuit_breaker_state(upstream: str, state: int) -> None:
        """Set circuit breaker state (0=closed, 1=half-open, 2=open)."""
        _circuit_breaker_state.labels(upstream=upstream).set(state)

    def inc_circuit_breaker_rejected(upstream: str) -> None:
        """Increment rejected count when circuit is open."""
        _circuit_breaker_rejected.labels(upstream=upstream).inc()

    def inc_circuit_breaker_transition(
        upstream: str,
        from_state: str,
        to_state: str,
    ) -> None:
        """Record circuit breaker state transition."""
        _circuit_breaker_transitions.labels(
            upstream=upstream,
            from_state=from_state,
            to_state=to_state,
        ).inc()

    def inc_load_shedding_rejected(reason: str) -> None:
        """Increment count of requests rejected due to load shedding."""
        _load_shedding_rejected.labels(reason=reason).inc()

    def set_active_requests(count: int) -> None:
        """Set current active request count."""
        _active_requests.set(count)

    def set_system_load(metric: str, value: float) -> None:
        """Set system load metric (cpu_percent, memory_percent, etc)."""
        _system_load.labels(metric=metric).set(value)

    def observe_upstream_duration(model: str, status_code: int, seconds: float) -> None:
        """Record upstream API duration."""
        _upstream_duration.labels(model=model, status_code=str(status_code)).observe(seconds)

    def inc_upstream_retries(model: str, reason: str) -> None:
        """Record an upstream retry."""
        _upstream_retries.labels(model=model, reason=reason).inc()

    def observe_chunk_translation_latency(
        model: str,
        chunk_type: str,
        seconds: float,
    ) -> None:
        """Record stream chunk translation latency."""
        _stream_chunk_latency.labels(model=model, chunk_type=chunk_type).observe(seconds)

    def inc_stream_chunk(model: str, chunk_type: str) -> None:
        """Count a processed stream chunk."""
        _stream_chunks_total.labels(model=model, chunk_type=chunk_type).inc()

    def inc_error(error_type: str, model: str, is_upstream: bool = False) -> None:
        """Increment error counter."""
        _errors_total.labels(
            error_type=error_type,
            model=model,
            is_upstream="1" if is_upstream else "0",
        ).inc()

    def inc_cost_estimate(model: str, input_cost: float, output_cost: float) -> None:
        """Record estimated cost in USD."""
        _cost_estimated_usd.labels(model=model).inc(input_cost + output_cost)
        if input_cost > 0:
            _cost_input_usd.labels(model=model, is_cached="0").inc(input_cost)
        if output_cost > 0:
            _cost_output_usd.labels(model=model).inc(output_cost)

    def inc_validation_error(validation_type: str) -> None:
        """Record a request validation error."""
        _validation_errors.labels(validation_type=validation_type).inc()

    def inc_security_block(check_type: str) -> None:
        """Record a security check that blocked a request."""
        _security_blocks.labels(check_type=check_type).inc()

except ImportError:
    _ENABLED = False
    _log.info(
        "Enhanced metrics unavailable - prometheus-client not installed. "
        "`pip install prometheus-client` to enable."
    )

    def inc_cache_tokens(model: str, cache_creation: int = 0, cache_read: int = 0) -> None:
        pass

    def inc_cache_savings(model: str, usd: float) -> None:
        pass

    def set_circuit_breaker_state(upstream: str, state: int) -> None:
        pass

    def inc_circuit_breaker_rejected(upstream: str) -> None:
        pass

    def inc_circuit_breaker_transition(upstream: str, from_state: str, to_state: str) -> None:
        pass

    def inc_load_shedding_rejected(reason: str) -> None:
        pass

    def set_active_requests(count: int) -> None:
        pass

    def set_system_load(metric: str, value: float) -> None:
        pass

    def observe_upstream_duration(model: str, status_code: int, seconds: float) -> None:
        pass

    def inc_upstream_retries(model: str, reason: str) -> None:
        pass

    def observe_chunk_translation_latency(model: str, chunk_type: str, seconds: float) -> None:
        pass

    def inc_stream_chunk(model: str, chunk_type: str) -> None:
        pass

    def inc_error(error_type: str, model: str, is_upstream: bool = False) -> None:
        pass

    def inc_cost_estimate(model: str, input_cost: float, output_cost: float) -> None:
        pass

    def inc_validation_error(validation_type: str) -> None:
        pass

    def inc_security_block(check_type: str) -> None:
        pass


def is_enabled() -> bool:
    """Return True if enhanced metrics are available."""
    return _ENABLED
