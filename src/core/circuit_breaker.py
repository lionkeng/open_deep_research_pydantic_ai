"""
Circuit Breaker implementation for resilient async operations.

This module provides a thread-safe, async-compatible circuit breaker pattern
implementation with three states: CLOSED, OPEN, and HALF_OPEN. Supports generic
key types including enums.
"""

import asyncio
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Hashable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Generic, Protocol, TypeVar

import logfire

logger = logfire

K = TypeVar("K", bound=Hashable)
T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = auto()  # Normal operation, requests pass through
    OPEN = auto()  # Circuit is broken, requests fail immediately
    HALF_OPEN = auto()  # Testing if service has recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, message: str, last_failure_time: float | None = None):
        super().__init__(message)
        self.last_failure_time = last_failure_time


class MetricsCollector(Protocol):
    """Protocol for metrics collection."""

    async def record_state_change(
        self, key: Any, old_state: CircuitState, new_state: CircuitState
    ) -> None:
        """Record a state change event."""
        ...

    async def record_attempt(self, key: Any, success: bool) -> None:
        """Record an attempt (success or failure)."""
        ...


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5
    """Number of failures before opening circuit."""

    success_threshold: int = 2
    """Number of successes in HALF_OPEN before closing circuit."""

    timeout_seconds: float = 60.0
    """Time to wait before attempting recovery (moving to HALF_OPEN)."""

    half_open_max_attempts: int = 3
    """Maximum concurrent attempts allowed in HALF_OPEN state."""

    excluded_exceptions: tuple[type[Exception], ...] = field(default_factory=tuple)
    """Exceptions that should not trigger circuit breaker."""

    name: str = "circuit_breaker"
    """Name for logging and monitoring."""

    def __post_init__(self):
        """Validate configuration."""
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if self.success_threshold < 1:
            raise ValueError("success_threshold must be at least 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.half_open_max_attempts < 1:
            raise ValueError("half_open_max_attempts must be at least 1")


@dataclass
class CircuitBreakerMetrics:
    """Metrics snapshot for monitoring."""

    total_attempts: int = 0
    total_failures: int = 0
    total_successes: int = 0
    rejected_calls: int = 0
    current_state: CircuitState = CircuitState.CLOSED
    consecutive_errors: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    last_state_change: float | None = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_attempts == 0:
            return 1.0
        return self.total_successes / self.total_attempts


@dataclass
class CircuitStateData:
    """Thread-safe circuit state for a single key."""

    state: CircuitState = CircuitState.CLOSED
    consecutive_errors: int = 0
    consecutive_successes: int = 0
    last_error_time: float = 0.0
    last_success_time: float = 0.0
    half_open_attempts: int = 0
    metrics: CircuitBreakerMetrics = field(default_factory=CircuitBreakerMetrics)


class CircuitBreaker(Generic[K]):
    """
    Generic circuit breaker supporting any hashable key type.

    The circuit breaker prevents cascading failures by failing fast when
    a service is unavailable, then periodically testing if it has recovered.

    State transitions:
    - CLOSED -> OPEN: After failure_threshold failures
    - OPEN -> HALF_OPEN: After timeout_seconds
    - HALF_OPEN -> CLOSED: After success_threshold successes
    - HALF_OPEN -> OPEN: After any failure

    Examples:
        # With enums (recommended)
        breaker: CircuitBreaker[AgentType] = CircuitBreaker()

        # With strings
        breaker: CircuitBreaker[str] = CircuitBreaker()
    """

    def __init__(
        self,
        config: CircuitBreakerConfig | None = None,
        metrics_collector: MetricsCollector | None = None,
        fallback_factory: Callable[[K], Awaitable[Any]] | None = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            config: Configuration for circuit breaker behavior
            metrics_collector: Optional metrics collector for observability
            fallback_factory: Optional factory for creating fallback responses
        """
        self.config = config or CircuitBreakerConfig()
        self.metrics_collector = metrics_collector
        self.fallback_factory = fallback_factory

        # Thread-safe state management
        self._states: dict[K, CircuitStateData] = {}
        self._locks: defaultdict[K, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._global_lock = asyncio.Lock()

        logger.info(
            f"Circuit breaker '{self.config.name}' initialized",
            failure_threshold=self.config.failure_threshold,
            timeout=self.config.timeout_seconds,
        )

    async def _get_or_create_state(self, key: K) -> CircuitStateData:
        """Get or initialize state for a key (thread-safe)."""
        if key not in self._states:
            async with self._global_lock:
                # Double-check pattern for async
                if key not in self._states:
                    self._states[key] = CircuitStateData()
                    logger.debug(f"Initialized circuit state for key: {key}")
        return self._states[key]

    def get_state(self, key: K) -> CircuitState | None:
        """Get current state for a key (non-blocking)."""
        state_data = self._states.get(key)
        return state_data.state if state_data else None

    def is_open(self, key: K) -> bool:
        """Check if circuit is open for a key."""
        state = self.get_state(key)
        return state == CircuitState.OPEN if state else False

    def is_closed(self, key: K) -> bool:
        """Check if circuit is closed for a key."""
        state = self.get_state(key)
        return state == CircuitState.CLOSED if state else True

    def is_half_open(self, key: K) -> bool:
        """Check if circuit is half-open for a key."""
        state = self.get_state(key)
        return state == CircuitState.HALF_OPEN if state else False

    async def _should_attempt_reset(self, state_data: CircuitStateData) -> bool:
        """Check if enough time has passed to attempt reset."""
        if state_data.last_error_time == 0:
            return False
        elapsed = time.time() - state_data.last_error_time
        return elapsed >= self.config.timeout_seconds

    async def _transition_to(self, key: K, state_data: CircuitStateData, new_state: CircuitState):
        """Transition to a new state."""
        if state_data.state != new_state:
            old_state = state_data.state
            state_data.state = new_state
            state_data.metrics.current_state = new_state
            state_data.metrics.last_state_change = time.time()

            logger.info(
                f"Circuit breaker '{self.config.name}' state transition",
                key=str(key),
                old_state=old_state.name,
                new_state=new_state.name,
            )

            # Notify metrics collector
            if self.metrics_collector:
                await self.metrics_collector.record_state_change(key, old_state, new_state)

            # Reset counters based on new state
            if new_state == CircuitState.CLOSED:
                state_data.consecutive_errors = 0
                state_data.consecutive_successes = 0
                state_data.half_open_attempts = 0
            elif new_state == CircuitState.HALF_OPEN:
                state_data.consecutive_successes = 0
                state_data.half_open_attempts = 0

    async def _record_success(self, key: K):
        """Record a successful call and update state."""
        async with self._locks[key]:
            state_data = await self._get_or_create_state(key)
            state_data.last_success_time = time.time()
            state_data.metrics.total_attempts += 1
            state_data.metrics.total_successes += 1
            state_data.metrics.last_success_time = state_data.last_success_time

            if self.metrics_collector:
                await self.metrics_collector.record_attempt(key, success=True)

            if state_data.state == CircuitState.HALF_OPEN:
                state_data.consecutive_successes += 1
                state_data.half_open_attempts -= 1

                if state_data.consecutive_successes >= self.config.success_threshold:
                    await self._transition_to(key, state_data, CircuitState.CLOSED)
                    logger.info(
                        f"Circuit breaker '{self.config.name}' recovered",
                        key=str(key),
                        successes=state_data.consecutive_successes,
                    )
            elif state_data.state == CircuitState.CLOSED:
                # Reset failure count on success in closed state
                state_data.consecutive_errors = 0
                state_data.metrics.consecutive_errors = 0

    async def _record_failure(self, key: K, exc: Exception):
        """Record a failed call and update state."""
        # Check if exception should be ignored
        if isinstance(exc, self.config.excluded_exceptions):
            logger.debug(
                f"Circuit breaker '{self.config.name}' ignoring excluded exception",
                key=str(key),
                exception_type=type(exc).__name__,
            )
            return

        async with self._locks[key]:
            state_data = await self._get_or_create_state(key)
            state_data.last_error_time = time.time()
            state_data.metrics.total_attempts += 1
            state_data.metrics.total_failures += 1
            state_data.metrics.last_failure_time = state_data.last_error_time

            if self.metrics_collector:
                await self.metrics_collector.record_attempt(key, success=False)

            if state_data.state == CircuitState.CLOSED:
                state_data.consecutive_errors += 1
                state_data.metrics.consecutive_errors = state_data.consecutive_errors

                if state_data.consecutive_errors >= self.config.failure_threshold:
                    await self._transition_to(key, state_data, CircuitState.OPEN)
                    logger.warning(
                        f"Circuit breaker '{self.config.name}' opened",
                        key=str(key),
                        failures=state_data.consecutive_errors,
                        error=str(exc),
                    )
            elif state_data.state == CircuitState.HALF_OPEN:
                state_data.half_open_attempts -= 1
                await self._transition_to(key, state_data, CircuitState.OPEN)
                logger.warning(
                    f"Circuit breaker '{self.config.name}' reopened after half-open failure",
                    key=str(key),
                    error=str(exc),
                )

    async def _check_state(self, key: K) -> None:
        """
        Check current state and potentially transition.

        Raises:
            CircuitBreakerError: If circuit is open and should reject calls
        """
        async with self._locks[key]:
            state_data = await self._get_or_create_state(key)

            if state_data.state == CircuitState.OPEN:
                if await self._should_attempt_reset(state_data):
                    await self._transition_to(key, state_data, CircuitState.HALF_OPEN)
                    logger.info(
                        f"Circuit breaker '{self.config.name}' attempting recovery",
                        key=str(key),
                    )
                else:
                    state_data.metrics.rejected_calls += 1
                    time_remaining = self.config.timeout_seconds - (
                        time.time() - state_data.last_error_time
                        if state_data.last_error_time
                        else 0
                    )
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.config.name}' is OPEN for {key}. "
                        f"Retry in {time_remaining:.1f} seconds.",
                        last_failure_time=state_data.last_error_time,
                    )

            elif state_data.state == CircuitState.HALF_OPEN:
                if state_data.half_open_attempts >= self.config.half_open_max_attempts:
                    state_data.metrics.rejected_calls += 1
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.config.name}' is HALF_OPEN for {key} but "
                        f"max concurrent attempts ({self.config.half_open_max_attempts}) reached"
                    )
                state_data.half_open_attempts += 1

    async def call(
        self, key: K, func: Callable[..., T | Awaitable[T]], *args: Any, **kwargs: Any
    ) -> T:
        """
        Execute a function through the circuit breaker.

        Args:
            key: Key to identify the circuit (e.g., AgentType.RESEARCH_EXECUTOR)
            func: Function to execute (can be async or sync)
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Any exception raised by func
        """
        await self._check_state(key)

        try:
            # Handle both async and sync functions
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            await self._record_success(key)
            return result

        except Exception as exc:
            await self._record_failure(key, exc)
            raise

    async def call_with_fallback(
        self, key: K, func: Callable[..., T | Awaitable[T]], *args: Any, **kwargs: Any
    ) -> T:
        """
        Execute function with circuit breaker and fallback.

        Args:
            key: Key to identify the circuit
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from func or fallback
        """
        try:
            return await self.call(key, func, *args, **kwargs)
        except CircuitBreakerError:
            if self.fallback_factory:
                logger.info(f"Using fallback for {key}")
                return await self.fallback_factory(key)
            raise

    @asynccontextmanager
    async def protect(self, key: K):
        """
        Context manager for circuit breaker protection.

        Usage:
            async with circuit_breaker.protect(AgentType.RESEARCH_EXECUTOR):
                # Protected code here
                await some_operation()
        """
        await self._check_state(key)

        try:
            yield self
            await self._record_success(key)
        except Exception as exc:
            await self._record_failure(key, exc)
            raise

    def decorator(self, key: K):
        """
        Decorator for protecting functions with circuit breaker.

        Usage:
            @circuit_breaker.decorator(AgentType.RESEARCH_EXECUTOR)
            async def protected_function():
                # Function code here
                pass
        """

        def wrapper(func: Callable[..., T | Awaitable[T]]) -> Callable[..., Awaitable[T]]:
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                return await self.call(key, func, *args, **kwargs)

            return async_wrapper

        return wrapper

    async def reset(self, key: K | None = None):
        """
        Manually reset the circuit breaker.

        Args:
            key: Specific key to reset, or None to reset all
        """
        if key is not None:
            async with self._locks[key]:
                if key in self._states:
                    state_data = self._states[key]
                    await self._transition_to(key, state_data, CircuitState.CLOSED)
                    state_data.consecutive_errors = 0
                    state_data.consecutive_successes = 0
                    state_data.half_open_attempts = 0
                    logger.info(f"Circuit breaker '{self.config.name}' manually reset for {key}")
        else:
            # Reset all circuits
            async with self._global_lock:
                for key, state_data in self._states.items():
                    async with self._locks[key]:
                        await self._transition_to(key, state_data, CircuitState.CLOSED)
                        state_data.consecutive_errors = 0
                        state_data.consecutive_successes = 0
                        state_data.half_open_attempts = 0
                logger.info(f"Circuit breaker '{self.config.name}' manually reset all circuits")

    def get_metrics(self, key: K) -> CircuitBreakerMetrics | None:
        """Get metrics for a specific key."""
        state_data = self._states.get(key)
        return state_data.metrics if state_data else None

    def get_all_metrics(self) -> dict[K, CircuitBreakerMetrics]:
        """Get metrics for all keys."""
        return {key: state_data.metrics for key, state_data in self._states.items()}


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    Useful for managing circuit breakers for different services or endpoints.
    """

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker[Any]] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self, name: str, config: CircuitBreakerConfig | None = None
    ) -> CircuitBreaker[Any]:
        """
        Get existing circuit breaker or create new one.

        Args:
            name: Unique identifier for circuit breaker
            config: Configuration (used only for new breakers)

        Returns:
            Circuit breaker instance
        """
        async with self._lock:
            if name not in self._breakers:
                breaker_config = config or CircuitBreakerConfig(name=name)
                self._breakers[name] = CircuitBreaker(breaker_config)
            return self._breakers[name]

    async def reset_all(self):
        """Reset all circuit breakers."""
        async with self._lock:
            for breaker in self._breakers.values():
                await breaker.reset()

    def get_all_metrics(self) -> dict[str, dict[Any, CircuitBreakerMetrics]]:
        """Get metrics of all circuit breakers."""
        return {name: breaker.get_all_metrics() for name, breaker in self._breakers.items()}


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()
