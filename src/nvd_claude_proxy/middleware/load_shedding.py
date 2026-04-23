"""Load shedding middleware for protecting the proxy under heavy load.

This middleware implements queue-based load shedding to protect the proxy
from being overwhelmed during traffic spikes. When the system is under
heavy load, requests are rejected with a 529 (overloaded) status rather
than queuing indefinitely.

This is part of a defense-in-depth strategy alongside:
- Circuit breaker (prevents upstream cascade)
- Rate limiter (enforces per-client limits)
- Request timeouts (prevents hung requests)
"""

from __future__ import annotations

import asyncio
import psutil
import time
from dataclasses import dataclass, field
from typing import Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import ORJSONResponse, Response

from ..util.anthropic_headers import new_request_id, standard_response_headers

_log = structlog.get_logger("nvd_claude_proxy.load_shedding")

# Default thresholds
DEFAULT_MAX_QUEUE_DEPTH = 100
DEFAULT_MAX_CPU_PERCENT = 85.0
DEFAULT_MAX_MEMORY_PERCENT = 90.0
DEFAULT_MIN_RESPONSE_TIME_MS = 100  # Requests faster than this increase load estimate


@dataclass
class LoadSheddingConfig:
    """Configuration for load shedding behavior."""
    max_queue_depth: int = DEFAULT_MAX_QUEUE_DEPTH
    max_cpu_percent: float = DEFAULT_MAX_CPU_PERCENT
    max_memory_percent: float = DEFAULT_MAX_MEMORY_PERCENT
    check_interval_seconds: float = 1.0
    # How many recent requests to track for response time estimation
    rolling_window_size: int = 100
    # If avg response time < this, assume system is fast and allow higher queue
    fast_response_threshold_ms: float = DEFAULT_MIN_RESPONSE_TIME_MS


@dataclass
class LoadStats:
    """Current system load statistics."""
    active_requests: int = 0
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    avg_response_time_ms: float = 0.0
    requests_accepted: int = 0
    requests_rejected: int = 0
    last_check: float = field(default_factory=time.time)
    
    @property
    def is_overloaded(self) -> bool:
        """Return True if system is under heavy load."""
        return (
            self.cpu_percent > 85.0 or
            self.memory_percent > 90.0 or
            self.active_requests > 100
        )


class LoadSheddingMiddleware(BaseHTTPMiddleware):
    """Middleware that sheds load when system is overwhelmed.
    
    This middleware monitors:
    - CPU usage (via psutil)
    - Memory usage (via psutil)
    - Active request count (tracked internally)
    - Response times (rolling average)
    
    When thresholds are exceeded, new requests receive a 529 response
    with a retry-after header indicating when to retry.
    """
    
    def __init__(
        self,
        app,
        config: LoadSheddingConfig | None = None,
    ) -> None:
        super().__init__(app)
        self.config = config or LoadSheddingConfig()
        self._stats = LoadStats()
        self._active_requests: int = 0
        self._response_times: list[float] = []
        self._lock = asyncio.Lock()
    
    @property
    def stats(self) -> LoadStats:
        """Return current load statistics."""
        return self._stats
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = new_request_id()
        
        # Check if we should shed this request
        should_shed, reason = await self._should_shed_load()
        
        if should_shed:
            self._stats.requests_rejected += 1
            _log.warning(
                "load_shedding.rejected",
                request_id=request_id,
                path=request.url.path,
                reason=reason,
                active_requests=self._active_requests,
                cpu_percent=round(self._stats.cpu_percent, 1),
            )
            
            headers = standard_response_headers(request_id)
            # Suggest retry after 1-5 seconds based on load
            retry_after = self._calculate_retry_after()
            headers["retry-after"] = str(retry_after)
            
            return ORJSONResponse(
                {
                    "type": "error",
                    "error": {
                        "type": "overloaded_error",
                        "message": (
                            f"Server is under heavy load ({reason}). "
                            f"Please retry after {retry_after} seconds."
                        ),
                        "retry_after": retry_after,
                    },
                },
                status_code=529,
                headers=headers,
            )
        
        # Track active request
        async with self._lock:
            self._active_requests += 1
            self._stats.active_requests = self._active_requests
        
        start_time = time.monotonic()
        
        try:
            response = await call_next(request)
            
            # Track response time
            elapsed_ms = (time.monotonic() - start_time) * 1000
            await self._record_response_time(elapsed_ms)
            
            return response
            
        except Exception as e:
            # Record failure but don't shed - let errors propagate
            raise
        finally:
            async with self._lock:
                self._active_requests = max(0, self._active_requests - 1)
                self._stats.active_requests = self._active_requests
    
    async def _should_shed_load(self) -> tuple[bool, str]:
        """Check if we should shed the current request.
        
        Returns:
            Tuple of (should_shed, reason)
        """
        # Always allow health checks
        if hasattr(self, '_is_health_check'):
            return False, ""
        
        # Update system metrics periodically
        await self._update_system_metrics()
        
        # Check CPU
        if self._stats.cpu_percent >= self.config.max_cpu_percent:
            return True, f"CPU at {self._stats.cpu_percent:.1f}%"
        
        # Check memory
        if self._stats.memory_percent >= self.config.max_memory_percent:
            return True, f"Memory at {self._stats.memory_percent:.1f}%"
        
        # Check queue depth with dynamic adjustment
        queue_threshold = self._get_dynamic_queue_threshold()
        if self._active_requests >= queue_threshold:
            return True, f"Queue depth {self._active_requests} >= {queue_threshold}"
        
        return False, ""
    
    def _get_dynamic_queue_threshold(self) -> int:
        """Calculate dynamic queue threshold based on system health.
        
        If system is fast (low avg response time), allow higher queue.
        If system is slow, lower the threshold.
        """
        base_threshold = self.config.max_queue_depth
        
        # If average response is fast, allow more headroom
        if self._stats.avg_response_time_ms < self.config.fast_response_threshold_ms:
            return int(base_threshold * 1.5)
        
        # If average response is slow, reduce headroom
        if self._stats.avg_response_time_ms > 500:
            return int(base_threshold * 0.5)
        
        return base_threshold
    
    async def _update_system_metrics(self) -> None:
        """Update CPU and memory metrics."""
        try:
            self._stats.cpu_percent = psutil.cpu_percent(interval=0.1)
            self._stats.memory_percent = psutil.virtual_memory().percent
        except Exception as e:
            _log.debug("load_shedding.metric_error", error=str(e))
    
    async def _record_response_time(self, elapsed_ms: float) -> None:
        """Record response time for rolling average."""
        async with self._lock:
            self._response_times.append(elapsed_ms)
            
            # Keep rolling window bounded
            if len(self._response_times) > self.config.rolling_window_size:
                self._response_times = self._response_times[-self.config.rolling_window_size:]
            
            # Update rolling average
            if self._response_times:
                self._stats.avg_response_time_ms = sum(self._response_times) / len(self._response_times)
            
            self._stats.requests_accepted += 1
    
    def _calculate_retry_after(self) -> int:
        """Calculate appropriate retry-after based on current load."""
        base = 1
        
        # Increase delay based on queue depth
        if self._active_requests > 50:
            base = 2
        if self._active_requests > 100:
            base = 5
        
        # Increase based on CPU
        if self._stats.cpu_percent > 80:
            base = max(base, 3)
        if self._stats.cpu_percent > 90:
            base = max(base, 5)
        
        # Cap at 30 seconds
        return min(base, 30)


# ── Integration with app.py ───────────────────────────────────────────────────

def add_load_shedding_middleware(app, config: LoadSheddingConfig | None = None) -> None:
    """Add load shedding middleware to the app."""
    app.add_middleware(LoadSheddingMiddleware, config=config or LoadSheddingConfig())
    structlog.get_logger("nvd_claude_proxy.app").info("load_shedding.enabled")