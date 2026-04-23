"""Circuit breaker pattern for resilient upstream API calls.

The circuit breaker pattern prevents cascading failures by:
1. Tracking failures to the upstream API
2. Opening the circuit when failures exceed a threshold
3. Testing recovery periodically (half-open state)
4. Closing the circuit when recovery succeeds

This is critical for the NVIDIA NIM backend, which may experience:
- Transient network errors
- Rate limiting (429)
- Server overload (5xx)
- Planned maintenance windows
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, TypeVar, Generic

import structlog

_log = structlog.get_logger("nvd_claude_proxy.circuit_breaker")

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, requests flow through
    OPEN = "open"          # Failing, requests are rejected immediately
    HALF_OPEN = "half_open"  # Testing recovery, limited requests allowed


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5       # Number of failures before opening
    success_threshold: int = 2       # Consecutive successes to close from half-open
    timeout: float = 30.0            # Seconds before transitioning open → half-open
    half_open_max_calls: int = 3     # Max concurrent calls in half-open state
    excluded_status_codes: frozenset[int] = frozenset()  # Don't count these as failures


class CircuitBreakerOpenError(Exception):
    """Raised when a request is rejected due to open circuit."""
    def __init__(self, upstream: str, retry_after: float | None = None):
        self.upstream = upstream
        self.retry_after = retry_after
        super().__init__(f"Circuit breaker is OPEN for {upstream}")


class CircuitBreaker(Generic[T]):
    """Async circuit breaker for protecting upstream API calls.
    
    Usage:
        cb = CircuitBreaker("nvidia_api", CircuitBreakerConfig(failure_threshold=5))
        
        async def call_upstream():
            response = await nvidia_client.chat_completions(payload)
            return response
        
        try:
            result = await cb.call(call_upstream)
        except CircuitBreakerOpenError:
            return ORJSONResponse({"error": "service temporarily unavailable"}, 503)
    """
    
    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # State
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()
        
        # Metrics (exposed for Prometheus)
        self._total_calls = 0
        self._rejected_calls = 0
        self._success_calls = 0
        self._failure_calls = 0
    
    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state
    
    @property
    def metrics(self) -> dict[str, int | float]:
        """Return circuit breaker metrics."""
        return {
            "state": self._state.value,
            "total_calls": self._total_calls,
            "rejected_calls": self._rejected_calls,
            "success_calls": self._success_calls,
            "failure_calls": self._failure_calls,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
        }
    
    async def call(self, func: Callable[..., T]) -> T:
        """Execute a function with circuit breaker protection.
        
        Args:
            func: Async function to call. Should raise an exception on failure.
            
        Returns:
            Result of the function call.
            
        Raises:
            CircuitBreakerOpenError: If the circuit is open and timeout hasn't elapsed.
        """
        async with self._lock:
            self._total_calls += 1
            
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    self._rejected_calls += 1
                    elapsed = time.time() - (self._last_failure_time or 0)
                    retry_after = max(0, self.config.timeout - elapsed)
                    _log.debug(
                        "circuit_breaker.rejected",
                        name=self.name,
                        elapsed_since_failure=round(elapsed, 1),
                        retry_after=round(retry_after, 1),
                    )
                    raise CircuitBreakerOpenError(self.name, retry_after)
            
            # Check half-open concurrency limit
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    self._rejected_calls += 1
                    raise CircuitBreakerOpenError(self.name)
                self._half_open_calls += 1
        
        try:
            # Execute the function
            result = await func()
            
            # Record success
            await self._record_success()
            return result
            
        except Exception as e:
            # Record failure
            await self._record_failure(e)
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return True
        return time.time() - self._last_failure_time >= self.config.timeout
    
    def _transition_to_half_open(self) -> None:
        """Transition from OPEN to HALF_OPEN."""
        self._state = CircuitState.HALF_OPEN
        self._half_open_calls = 0
        self._success_count = 0
        _log.info(
            "circuit_breaker.half_open",
            name=self.name,
            timeout=self.config.timeout,
        )
    
    async def _record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            self._success_calls += 1
            
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                self._half_open_calls = max(0, self._half_open_calls - 1)
                
                if self._success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success in closed state
                self._failure_count = 0
    
    async def _record_failure(self, error: Exception) -> None:
        """Record a failed call."""
        async with self._lock:
            self._failure_calls += 1
            
            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._transition_to_open(error)
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                self._last_failure_time = time.time()
                
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to_open(error)
    
    def _transition_to_open(self, error: Exception | None = None) -> None:
        """Transition to OPEN state."""
        self._state = CircuitState.OPEN
        self._half_open_calls = 0
        _log.warning(
            "circuit_breaker.opened",
            name=self.name,
            failure_count=self._failure_count,
            threshold=self.config.failure_threshold,
            error_type=type(error).__name__ if error else None,
        )
    
    def _transition_to_closed(self) -> None:
        """Transition to CLOSED state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        _log.info(
            "circuit_breaker.closed",
            name=self.name,
            total_calls=self._total_calls,
        )
    
    async def force_open(self) -> None:
        """Manually open the circuit (for administrative purposes)."""
        async with self._lock:
            self._transition_to_open(None)
    
    async def force_close(self) -> None:
        """Manually close the circuit (for administrative purposes)."""
        async with self._lock:
            self._transition_to_closed()
    
    def reset_stats(self) -> None:
        """Reset all statistics (but keep current state)."""
        self._total_calls = 0
        self._rejected_calls = 0
        self._success_calls = 0
        self._failure_calls = 0


# ── Circuit Breaker Registry ──────────────────────────────────────────────────

class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers.
    
    This allows different upstream services to have independent
    circuit breakers with different configurations.
    """
    
    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()
    
    async def get_or_create(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        """Get an existing circuit breaker or create a new one."""
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]
    
    async def get_all_metrics(self) -> dict[str, dict[str, int | float]]:
        """Get metrics for all registered circuit breakers."""
        return {
            name: cb.metrics
            for name, cb in self._breakers.items()
        }
    
    async def get_state(self) -> dict[str, str]:
        """Get current state of all circuit breakers."""
        return {
            name: cb.state.value
            for name, cb in self._breakers.items()
        }


# ── Global Registry ───────────────────────────────────────────────────────────

# Global registry instance (can be accessed from routes)
_registry: CircuitBreakerRegistry | None = None


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry."""
    global _registry
    if _registry is None:
        _registry = CircuitBreakerRegistry()
    return _registry


# ── Decorator ─────────────────────────────────────────────────────────────────

from functools import wraps
from typing import ParamSpec, Callable

P = ParamSpec("P")


def circuit_breaker(
    name: str,
    config: CircuitBreakerConfig | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to add circuit breaker protection to a function.
    
    Usage:
        @circuit_breaker("nvidia_api")
        async def call_nvidia(payload):
            return await nvidia_client.chat_completions(payload)
    """
    _breaker: CircuitBreaker | None = None
    
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            nonlocal _breaker
            if _breaker is None:
                _breaker = await get_circuit_breaker_registry().get_or_create(name, config)
            
            return await _breaker.call(lambda: func(*args, **kwargs))
        
        return wrapper
    
    return decorator