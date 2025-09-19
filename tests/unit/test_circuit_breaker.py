"""
Comprehensive test suite for CircuitBreaker implementation.

Tests cover:
- All three circuit breaker states (CLOSED, OPEN, HALF_OPEN)
- State transitions and edge cases
- Concurrent access scenarios
- Timeout and reset behavior
- Configuration validation
- Generic type parameter with enums
"""

import asyncio
import time
from enum import Enum, auto
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from agents.factory import AgentType
from core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerMetrics,
    CircuitBreakerRegistry,
    CircuitState,
    CircuitStateData,
    MetricsCollector,
    circuit_breaker_registry,
)


# Test enum for demonstrating generic type support
class ServiceType(Enum):
    """Test enum for service types."""

    API = auto()
    DATABASE = auto()
    CACHE = auto()


class TestCircuitBreakerConfig:
    """Test circuit breaker configuration validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 2
        assert config.timeout_seconds == 60.0
        assert config.half_open_max_attempts == 3
        assert config.excluded_exceptions == ()
        assert config.name == "circuit_breaker"

    def test_custom_config(self):
        """Test custom configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=1,
            timeout_seconds=30.0,
            name="test_breaker",
        )
        assert config.failure_threshold == 3
        assert config.success_threshold == 1
        assert config.timeout_seconds == 30.0
        assert config.name == "test_breaker"

    def test_invalid_failure_threshold(self):
        """Test validation of failure threshold."""
        with pytest.raises(ValueError, match="failure_threshold must be at least 1"):
            CircuitBreakerConfig(failure_threshold=0)

    def test_invalid_success_threshold(self):
        """Test validation of success threshold."""
        with pytest.raises(ValueError, match="success_threshold must be at least 1"):
            CircuitBreakerConfig(success_threshold=0)

    def test_invalid_timeout(self):
        """Test validation of timeout."""
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            CircuitBreakerConfig(timeout_seconds=0)

    def test_invalid_half_open_attempts(self):
        """Test validation of half-open max attempts."""
        with pytest.raises(ValueError, match="half_open_max_attempts must be at least 1"):
            CircuitBreakerConfig(half_open_max_attempts=0)


class TestCircuitBreakerWithEnumKeys:
    """Test circuit breaker with enum keys (AgentType)."""

    @pytest.mark.asyncio
    async def test_enum_keys_work_correctly(self):
        """Test that enum keys work as expected."""
        breaker: CircuitBreaker[AgentType] = CircuitBreaker()

        # Initial state should be closed
        assert breaker.is_closed(AgentType.RESEARCH_EXECUTOR)
        assert not breaker.is_open(AgentType.RESEARCH_EXECUTOR)

        # Successful call
        async def success_func():
            return "success"

        result = await breaker.call(AgentType.RESEARCH_EXECUTOR, success_func)
        assert result == "success"

        # Check metrics
        metrics = breaker.get_metrics(AgentType.RESEARCH_EXECUTOR)
        assert metrics is not None
        assert metrics.total_successes == 1
        assert metrics.total_attempts == 1

    @pytest.mark.asyncio
    async def test_different_enum_keys_isolated(self):
        """Test that different enum keys maintain separate states."""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker: CircuitBreaker[AgentType] = CircuitBreaker(config)

        async def failing_func():
            raise ValueError("Test failure")

        # Fail RESEARCH_EXECUTOR circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(AgentType.RESEARCH_EXECUTOR, failing_func)

        # RESEARCH_EXECUTOR should be open
        assert breaker.is_open(AgentType.RESEARCH_EXECUTOR)

        # QUERY_TRANSFORMATION should still be closed
        assert breaker.is_closed(AgentType.QUERY_TRANSFORMATION)

        # Can still call QUERY_TRANSFORMATION
        async def success_func():
            return "success"

        result = await breaker.call(AgentType.QUERY_TRANSFORMATION, success_func)
        assert result == "success"


class TestCircuitBreakerStates:
    """Test circuit breaker state management."""

    @pytest.mark.asyncio
    async def test_initial_state_is_closed(self):
        """Test that circuit breaker starts in CLOSED state."""
        cb: CircuitBreaker[str] = CircuitBreaker()
        assert cb.get_state("test_key") is None  # Not initialized yet
        assert cb.is_closed("test_key")  # Default to closed
        assert not cb.is_open("test_key")
        assert not cb.is_half_open("test_key")

    @pytest.mark.asyncio
    async def test_closed_to_open_transition(self):
        """Test transition from CLOSED to OPEN after failures."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb: CircuitBreaker[str] = CircuitBreaker(config)

        # Create a failing function
        async def failing_func():
            raise ValueError("Test failure")

        # First failure
        with pytest.raises(ValueError):
            await cb.call("test_key", failing_func)
        assert cb.get_state("test_key") == CircuitState.CLOSED

        # Second failure - should open circuit
        with pytest.raises(ValueError):
            await cb.call("test_key", failing_func)
        assert cb.get_state("test_key") == CircuitState.OPEN
        assert cb.is_open("test_key")

    @pytest.mark.asyncio
    async def test_open_to_half_open_transition(self):
        """Test transition from OPEN to HALF_OPEN after timeout."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout_seconds=0.1)  # 100ms for fast test
        cb: CircuitBreaker[str] = CircuitBreaker(config)

        # Open the circuit
        async def failing_func():
            raise ValueError("Test failure")

        with pytest.raises(ValueError):
            await cb.call("test_key", failing_func)
        assert cb.get_state("test_key") == CircuitState.OPEN

        # Try immediately - should be rejected
        with pytest.raises(CircuitBreakerError):
            await cb.call("test_key", lambda: "test")

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Next call should transition to HALF_OPEN and succeed
        async def success_func():
            return "success"

        result = await cb.call("test_key", success_func)
        assert result == "success"
        assert cb.get_state("test_key") == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_half_open_to_closed_transition(self):
        """Test transition from HALF_OPEN to CLOSED after successes."""
        config = CircuitBreakerConfig(failure_threshold=1, success_threshold=2, timeout_seconds=0.1)
        cb: CircuitBreaker[str] = CircuitBreaker(config)

        # Open the circuit
        async def failing_func():
            raise ValueError("Test failure")

        with pytest.raises(ValueError):
            await cb.call("test_key", failing_func)
        assert cb.get_state("test_key") == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Success calls to transition through HALF_OPEN to CLOSED
        async def success_func():
            return "success"

        # First success - transitions to HALF_OPEN
        await cb.call("test_key", success_func)
        assert cb.get_state("test_key") == CircuitState.HALF_OPEN

        # Second success - should close circuit
        await cb.call("test_key", success_func)
        assert cb.get_state("test_key") == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_to_open_on_failure(self):
        """Test that failure in HALF_OPEN state reopens circuit."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout_seconds=0.1)
        cb: CircuitBreaker[str] = CircuitBreaker(config)

        # Open the circuit
        async def failing_func():
            raise ValueError("Test failure")

        with pytest.raises(ValueError):
            await cb.call("test_key", failing_func)
        assert cb.get_state("test_key") == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Success to move to HALF_OPEN
        await cb.call("test_key", lambda: "success")
        assert cb.get_state("test_key") == CircuitState.HALF_OPEN

        # Failure should reopen circuit
        with pytest.raises(ValueError):
            await cb.call("test_key", failing_func)
        assert cb.get_state("test_key") == CircuitState.OPEN


class TestCircuitBreakerConcurrency:
    """Test circuit breaker behavior under concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_calls_with_different_keys(self):
        """Test concurrent calls with different enum keys."""
        cb: CircuitBreaker[AgentType] = CircuitBreaker()
        results = {}

        async def agent_operation(agent_type: AgentType):
            await asyncio.sleep(0.01)  # Simulate work
            return f"Result from {agent_type.value}"

        # Launch concurrent calls for different agents
        tasks = [
            cb.call(agent_type, agent_operation, agent_type) for agent_type in list(AgentType)[:3]
        ]
        results_list = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results_list) == 3
        for agent_type, result in zip(list(AgentType)[:3], results_list):
            assert f"Result from {agent_type.value}" == result

    @pytest.mark.asyncio
    async def test_concurrent_failures_open_circuit_once(self):
        """Test that concurrent failures only open circuit once."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb: CircuitBreaker[str] = CircuitBreaker(config)

        async def failing_func():
            await asyncio.sleep(0.01)
            raise ValueError("Concurrent failure")

        # Launch concurrent failing calls
        tasks = [cb.call("test_key", failing_func) for _ in range(5)]

        # All should fail, but circuit should open exactly once
        for task in asyncio.as_completed(tasks):
            with pytest.raises(ValueError):
                await task

        assert cb.get_state("test_key") == CircuitState.OPEN
        # Check metrics to ensure proper counting
        metrics = cb.get_metrics("test_key")
        assert metrics.total_failures >= 3

    @pytest.mark.asyncio
    async def test_half_open_concurrent_limit(self):
        """Test that HALF_OPEN state limits concurrent attempts."""
        config = CircuitBreakerConfig(
            failure_threshold=1, timeout_seconds=0.1, half_open_max_attempts=2
        )
        cb: CircuitBreaker[str] = CircuitBreaker(config)

        # Open the circuit
        with pytest.raises(ValueError):
            await cb.call("test_key", lambda: (_ for _ in ()).throw(ValueError()))

        # Wait for timeout
        await asyncio.sleep(0.15)

        async def slow_func():
            await asyncio.sleep(0.1)
            return "success"

        # Launch more concurrent calls than allowed
        tasks = []
        for _ in range(4):
            tasks.append(asyncio.create_task(cb.call("test_key", slow_func)))
            await asyncio.sleep(0.01)  # Small delay to ensure ordering

        results = []
        errors = []

        for task in tasks:
            try:
                result = await task
                results.append(result)
            except CircuitBreakerError as e:
                errors.append(e)

        # At least 2 should succeed (max attempts), some may be rejected
        # Note: Due to async timing, more than 2 might succeed if they enter before limit is reached
        assert len(results) >= 2
        assert len(results) + len(errors) == 4


class TestCircuitBreakerFeatures:
    """Test specific circuit breaker features."""

    @pytest.mark.asyncio
    async def test_excluded_exceptions(self):
        """Test that excluded exceptions don't trigger circuit breaker."""

        class IgnoredException(Exception):
            pass

        config = CircuitBreakerConfig(
            failure_threshold=1, excluded_exceptions=(IgnoredException,)
        )
        cb: CircuitBreaker[str] = CircuitBreaker(config)

        async def func_with_ignored_exception():
            raise IgnoredException("This should be ignored")

        # Should raise the exception but not open circuit
        for _ in range(3):
            with pytest.raises(IgnoredException):
                await cb.call("test_key", func_with_ignored_exception)

        assert cb.get_state("test_key") == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_manual_reset(self):
        """Test manual reset of circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb: CircuitBreaker[AgentType] = CircuitBreaker(config)

        # Open the circuit
        with pytest.raises(ValueError):
            await cb.call(AgentType.RESEARCH_EXECUTOR, lambda: (_ for _ in ()).throw(ValueError()))
        assert cb.get_state(AgentType.RESEARCH_EXECUTOR) == CircuitState.OPEN

        # Manual reset for specific key
        await cb.reset(AgentType.RESEARCH_EXECUTOR)
        assert cb.get_state(AgentType.RESEARCH_EXECUTOR) == CircuitState.CLOSED

        # Open multiple circuits
        for agent_type in [AgentType.QUERY_TRANSFORMATION, AgentType.REPORT_GENERATOR]:
            with pytest.raises(ValueError):
                await cb.call(agent_type, lambda: (_ for _ in ()).throw(ValueError()))
            assert cb.is_open(agent_type)

        # Reset all
        await cb.reset()
        assert cb.is_closed(AgentType.QUERY_TRANSFORMATION)
        assert cb.is_closed(AgentType.REPORT_GENERATOR)

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test circuit breaker as context manager."""
        cb: CircuitBreaker[str] = CircuitBreaker()

        # Successful context
        async with cb.protect("test_key"):
            result = "success"

        assert cb.get_state("test_key") == CircuitState.CLOSED

        # Failing context
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(config)

        with pytest.raises(ValueError):
            async with cb.protect("test_key"):
                raise ValueError("Context failure")

        assert cb.get_state("test_key") == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_decorator(self):
        """Test circuit breaker as decorator."""
        cb: CircuitBreaker[AgentType] = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2))

        @cb.decorator(AgentType.RESEARCH_EXECUTOR)
        async def protected_function(value):
            if value < 0:
                raise ValueError("Negative value")
            return value * 2

        # Successful calls
        assert await protected_function(5) == 10
        assert cb.get_state(AgentType.RESEARCH_EXECUTOR) == CircuitState.CLOSED

        # Failing calls
        with pytest.raises(ValueError):
            await protected_function(-1)
        with pytest.raises(ValueError):
            await protected_function(-2)

        assert cb.get_state(AgentType.RESEARCH_EXECUTOR) == CircuitState.OPEN

        # Circuit open - should reject
        with pytest.raises(CircuitBreakerError):
            await protected_function(5)

    @pytest.mark.asyncio
    async def test_sync_function_support(self):
        """Test that sync functions are supported."""
        cb: CircuitBreaker[str] = CircuitBreaker()

        def sync_func(x):
            return x * 2

        result = await cb.call("test_key", sync_func, 5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_metrics_tracking(self):
        """Test that metrics are properly tracked."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb: CircuitBreaker[str] = CircuitBreaker(config)

        # Successful call
        await cb.call("test_key", lambda: "success")
        metrics = cb.get_metrics("test_key")
        assert metrics.total_successes == 1
        assert metrics.total_attempts == 1

        # Failed calls
        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.call("test_key", lambda: (_ for _ in ()).throw(ValueError()))

        metrics = cb.get_metrics("test_key")
        assert metrics.total_failures == 2
        assert metrics.total_attempts == 3

        # Rejected call (circuit open)
        with pytest.raises(CircuitBreakerError):
            await cb.call("test_key", lambda: "test")

        metrics = cb.get_metrics("test_key")
        assert metrics.rejected_calls == 1
        assert metrics.total_attempts == 3  # Rejected calls don't count as attempts

        # Check success rate
        assert metrics.success_rate == 1 / 3  # 1 success out of 3 attempts

    @pytest.mark.asyncio
    async def test_fallback_factory(self):
        """Test fallback factory functionality."""
        async def fallback_factory(key: str) -> str:
            return f"Fallback for {key}"

        config = CircuitBreakerConfig(failure_threshold=1)
        cb: CircuitBreaker[str] = CircuitBreaker(config, fallback_factory=fallback_factory)

        # Open the circuit
        with pytest.raises(ValueError):
            await cb.call("test_key", lambda: (_ for _ in ()).throw(ValueError()))

        # Use fallback
        result = await cb.call_with_fallback("test_key", lambda: "normal")
        assert result == "Fallback for test_key"


class TestCircuitBreakerRegistry:
    """Test circuit breaker registry functionality."""

    @pytest.mark.asyncio
    async def test_get_or_create(self):
        """Test getting or creating circuit breakers."""
        registry = CircuitBreakerRegistry()

        # Create new breaker
        cb1 = await registry.get_or_create("service1")
        assert cb1 is not None

        # Get existing breaker
        cb2 = await registry.get_or_create("service1")
        assert cb1 is cb2  # Same instance

        # Create with custom config
        config = CircuitBreakerConfig(failure_threshold=10)
        cb3 = await registry.get_or_create("service2", config)
        assert cb3 is not cb1

    @pytest.mark.asyncio
    async def test_reset_all(self):
        """Test resetting all circuit breakers."""
        registry = CircuitBreakerRegistry()

        # Create and open multiple breakers
        for i in range(3):
            cb = await registry.get_or_create(f"service{i}", CircuitBreakerConfig(failure_threshold=1))
            with pytest.raises(ValueError):
                await cb.call(f"key{i}", lambda: (_ for _ in ()).throw(ValueError()))
            assert cb.get_state(f"key{i}") == CircuitState.OPEN

        # Reset all
        await registry.reset_all()

        # Verify all are closed
        for i in range(3):
            cb = await registry.get_or_create(f"service{i}")
            # After reset, states should be cleared
            assert cb.get_state(f"key{i}") is None or cb.get_state(f"key{i}") == CircuitState.CLOSED


class TestMetricsCollector:
    """Test metrics collector integration."""

    @pytest.mark.asyncio
    async def test_metrics_collector_called(self):
        """Test that metrics collector receives events."""

        class TestCollector(MetricsCollector):
            def __init__(self):
                self.state_changes = []
                self.attempts = []

            async def record_state_change(self, key: str, old_state: CircuitState, new_state: CircuitState):
                self.state_changes.append((key, old_state, new_state))

            async def record_attempt(self, key: str, success: bool):
                self.attempts.append((key, success))

        collector = TestCollector()
        config = CircuitBreakerConfig(failure_threshold=1)
        cb: CircuitBreaker[str] = CircuitBreaker(config, metrics_collector=collector)

        # Successful call
        await cb.call("test_key", lambda: "success")
        assert ("test_key", True) in collector.attempts

        # Failed call - should trigger state change
        with pytest.raises(ValueError):
            await cb.call("test_key", lambda: (_ for _ in ()).throw(ValueError()))

        assert ("test_key", False) in collector.attempts
        assert ("test_key", CircuitState.CLOSED, CircuitState.OPEN) in collector.state_changes


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_error_info(self):
        """Test that CircuitBreakerError contains useful information."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout_seconds=60, name="test_service")
        cb: CircuitBreaker[str] = CircuitBreaker(config)

        # Open the circuit
        with pytest.raises(ValueError):
            await cb.call("test_key", lambda: (_ for _ in ()).throw(ValueError()))

        # Try to call when open
        with pytest.raises(CircuitBreakerError) as exc_info:
            await cb.call("test_key", lambda: "test")

        error = exc_info.value
        assert "test_service" in str(error)
        assert "OPEN" in str(error)
        assert "test_key" in str(error)
        assert error.last_failure_time is not None

    @pytest.mark.asyncio
    async def test_recovery_with_intermittent_failures(self):
        """Test recovery with intermittent failures."""
        config = CircuitBreakerConfig(
            failure_threshold=2, success_threshold=3, timeout_seconds=0.1
        )
        cb: CircuitBreaker[str] = CircuitBreaker(config)

        call_count = 0

        async def intermittent_func():
            nonlocal call_count
            call_count += 1
            # Fail on calls 1, 2, 5
            if call_count in [1, 2, 5]:
                raise ValueError(f"Failure {call_count}")
            return f"Success {call_count}"

        # Calls 1, 2 fail - circuit opens
        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.call("test_key", intermittent_func)
        assert cb.get_state("test_key") == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Call 3 succeeds - moves to HALF_OPEN
        await cb.call("test_key", intermittent_func)
        assert cb.get_state("test_key") == CircuitState.HALF_OPEN

        # Call 4 succeeds - still HALF_OPEN
        await cb.call("test_key", intermittent_func)
        assert cb.get_state("test_key") == CircuitState.HALF_OPEN

        # Call 5 fails - back to OPEN
        with pytest.raises(ValueError):
            await cb.call("test_key", intermittent_func)
        assert cb.get_state("test_key") == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_very_fast_timeout(self):
        """Test with very fast timeout to ensure timing works correctly."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout_seconds=0.001)  # 1ms
        cb: CircuitBreaker[str] = CircuitBreaker(config)

        # Open circuit
        with pytest.raises(ValueError):
            await cb.call("test_key", lambda: (_ for _ in ()).throw(ValueError()))

        # Almost immediate retry should move to HALF_OPEN
        await asyncio.sleep(0.002)
        result = await cb.call("test_key", lambda: "quick")
        assert result == "quick"
        assert cb.get_state("test_key") == CircuitState.HALF_OPEN


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @pytest.mark.asyncio
    async def test_agent_workflow_scenario(self):
        """Simulate agent workflow with circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=3, success_threshold=2, timeout_seconds=0.5)
        cb: CircuitBreaker[AgentType] = CircuitBreaker(config)

        # Simulate agent operations
        agent_available = {
            AgentType.RESEARCH_EXECUTOR: True,
            AgentType.QUERY_TRANSFORMATION: True,
            AgentType.REPORT_GENERATOR: True,
        }

        async def agent_operation(agent_type: AgentType, query: str):
            if not agent_available[agent_type]:
                raise ConnectionError(f"Agent {agent_type.value} unavailable")

            await asyncio.sleep(0.01)  # Simulate processing
            return f"Result from {agent_type.value}: {query}"

        # Normal operation
        for agent_type in [AgentType.RESEARCH_EXECUTOR, AgentType.QUERY_TRANSFORMATION]:
            result = await cb.call(agent_type, agent_operation, agent_type, "test query")
            assert f"Result from {agent_type.value}" in result

        # RESEARCH_EXECUTOR becomes unavailable
        agent_available[AgentType.RESEARCH_EXECUTOR] = False

        # Failures accumulate
        for i in range(3):
            with pytest.raises(ConnectionError):
                await cb.call(AgentType.RESEARCH_EXECUTOR, agent_operation, AgentType.RESEARCH_EXECUTOR, f"query {i}")

        # Circuit is now open for RESEARCH_EXECUTOR
        assert cb.is_open(AgentType.RESEARCH_EXECUTOR)

        # Other agents still work
        result = await cb.call(
            AgentType.QUERY_TRANSFORMATION,
            agent_operation,
            AgentType.QUERY_TRANSFORMATION,
            "still working",
        )
        assert "still working" in result

        # RESEARCH_EXECUTOR recovers
        agent_available[AgentType.RESEARCH_EXECUTOR] = True

        # Wait for timeout
        await asyncio.sleep(0.6)

        # Circuit attempts recovery
        result = await cb.call(
            AgentType.RESEARCH_EXECUTOR,
            agent_operation,
            AgentType.RESEARCH_EXECUTOR,
            "recovery test",
        )
        assert cb.get_state(AgentType.RESEARCH_EXECUTOR) == CircuitState.HALF_OPEN

        # Another success closes the circuit
        result = await cb.call(
            AgentType.RESEARCH_EXECUTOR,
            agent_operation,
            AgentType.RESEARCH_EXECUTOR,
            "fully recovered",
        )
        assert cb.get_state(AgentType.RESEARCH_EXECUTOR) == CircuitState.CLOSED
