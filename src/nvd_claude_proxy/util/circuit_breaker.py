from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, TypeVar

import structlog

_log = structlog.get_logger("nvd_claude_proxy.resilience")

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 30.0
    recovery_timeout: float | None = None
    half_open_max_calls: int = 3

    def __post_init__(self) -> None:
        if self.recovery_timeout is None:
            self.recovery_timeout = self.timeout
        else:
            self.timeout = self.recovery_timeout


class CircuitBreakerOpenError(Exception):
    """Raised when the circuit is open and cannot accept requests."""

    def __init__(self, message: str, retry_after: float, upstream: str | None = None):
        self.retry_after = retry_after
        self.upstream = upstream
        super().__init__(message)


class CircuitBreaker:
    """Circuit breaker for protecting upstream calls."""

    def __init__(self, name: str, config: CircuitBreakerConfig = CircuitBreakerConfig()):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.last_failure_time: float | None = None
        self.half_open_calls = 0
        self._success_calls = 0
        self._failure_calls = 0
        self._total_calls = 0
        self._lock = asyncio.Lock()

    @property
    def _last_failure_time(self) -> float | None:
        return self.last_failure_time

    @_last_failure_time.setter
    def _last_failure_time(self, value: float | None) -> None:
        self.last_failure_time = value

    @property
    def metrics(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "total_calls": self._total_calls,
            "success_calls": self._success_calls,
            "failure_calls": self._failure_calls,
            "consecutive_failures": self.failures,
            "half_open_calls": self.half_open_calls,
            "last_failure_time": self.last_failure_time,
        }

    async def call(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        async with self._lock:
            await self._before_call()
            self._total_calls += 1

        try:
            result = await func(*args, **kwargs)
            if hasattr(result, "status_code") and result.status_code >= 500:
                await self._on_failure()
            else:
                await self._on_success()
            return result
        except Exception:
            await self._on_failure()
            raise

    async def _before_call(self) -> None:
        if self.state == CircuitState.OPEN:
            elapsed = time.time() - (self.last_failure_time or 0)
            recovery_timeout = self.config.recovery_timeout or self.config.timeout
            if elapsed > recovery_timeout:
                _log.info("circuit.half_opening", name=self.name)
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                self._success_calls = 0
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit '{self.name}' is OPEN",
                    max(0.0, recovery_timeout - elapsed),
                    upstream=self.name,
                )

        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.config.half_open_max_calls:
                raise CircuitBreakerOpenError(
                    f"Circuit '{self.name}' is HALF_OPEN and saturated",
                    self.config.recovery_timeout or self.config.timeout,
                    upstream=self.name,
                )
            self.half_open_calls += 1

    async def _on_success(self) -> None:
        async with self._lock:
            self._success_calls += 1
            if self.state == CircuitState.HALF_OPEN:
                self.failures = 0
                if self._success_calls >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.half_open_calls = 0
                    _log.info("circuit.closed", name=self.name)
            elif self.state == CircuitState.CLOSED:
                self.failures = 0

    async def _on_failure(self) -> None:
        async with self._lock:
            self._failure_calls += 1
            self.failures += 1
            self.last_failure_time = time.time()
            if (
                self.state == CircuitState.HALF_OPEN
                or self.failures >= self.config.failure_threshold
            ):
                if self.state != CircuitState.OPEN:
                    _log.error("circuit.opened", name=self.name, failures=self.failures)
                self.state = CircuitState.OPEN
                self.half_open_calls = 0
                self._success_calls = 0

    async def record_failure(self) -> None:
        """Manually record a failure (useful for streaming connections where call() doesn't wrap the full lifecycle)."""
        await self._on_failure()

    async def force_open(self) -> None:
        async with self._lock:
            self.state = CircuitState.OPEN
            self.last_failure_time = time.time()
            self.half_open_calls = 0
            self._success_calls = 0

    async def force_close(self) -> None:
        async with self._lock:
            self.state = CircuitState.CLOSED
            self.failures = 0
            self.last_failure_time = None
            self.half_open_calls = 0
            self._success_calls = 0


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self, name: str, config: CircuitBreakerConfig = CircuitBreakerConfig()
    ) -> CircuitBreaker:
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]

    async def get_all_metrics(self) -> dict[str, dict[str, Any]]:
        async with self._lock:
            return {name: breaker.metrics for name, breaker in self._breakers.items()}


_registry = CircuitBreakerRegistry()


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Entry point for accessing the global registry."""
    return _registry
