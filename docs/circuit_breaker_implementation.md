# Circuit Breaker Implementation

## Overview

This document describes the production-ready circuit breaker implementation for the Open Deep Research project. The circuit breaker pattern prevents cascading failures by failing fast when a service is unavailable, then periodically testing if it has recovered.

## Key Features

### 1. Generic Type-Safe Implementation

The implementation uses Python generics to support any hashable key type, with AgentType enums as the primary use case:

```python
# Type-safe with enums (recommended)
breaker: CircuitBreaker[AgentType] = CircuitBreaker()

# Also supports strings if needed
breaker: CircuitBreaker[str] = CircuitBreaker()
```

### 2. Three-State Circuit Breaker

The implementation follows the standard circuit breaker pattern with three states:

- **CLOSED**: Normal operation, requests pass through
- **OPEN**: Circuit is broken, requests fail immediately
- **HALF_OPEN**: Testing if service has recovered

### State Transitions

```
CLOSED -> OPEN: After failure_threshold failures
OPEN -> HALF_OPEN: After timeout_seconds
HALF_OPEN -> CLOSED: After success_threshold successes
HALF_OPEN -> OPEN: After any failure
```

### 3. Thread-Safe Async Implementation

- Uses `asyncio.Lock` for thread-safe state management
- Per-key locks with `defaultdict` to prevent contention
- Global lock for safe state initialization
- Supports both async and sync functions
- Handles concurrent access correctly

### 4. Configuration Options

```python
@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 2          # Successes to close
    timeout_seconds: float = 60.0       # Recovery timeout
    half_open_max_attempts: int = 3     # Concurrent attempts in HALF_OPEN
    excluded_exceptions: tuple = ()     # Exceptions to ignore
    name: str = "circuit_breaker"       # For logging/monitoring
```

### 5. Multiple Usage Patterns

#### As a Function Wrapper
```python
circuit_breaker = CircuitBreaker[AgentType]()
result = await circuit_breaker.call(
    AgentType.RESEARCH_EXECUTOR,
    async_function,
    arg1,
    arg2
)
```

#### As a Decorator
```python
@circuit_breaker.decorator(AgentType.RESEARCH_EXECUTOR)
async def protected_function():
    # Function code
    pass
```

#### As a Context Manager
```python
async with circuit_breaker.protect(AgentType.RESEARCH_EXECUTOR):
    # Protected code
    await operation()
```

### 6. Comprehensive Metrics and Observability

The implementation includes:
- Metrics tracking (successes, failures, rejections)
- Success rate calculation
- State change history
- MetricsCollector protocol for pluggable backends
- Built-in logging with logfire

### 7. Fallback Support

```python
async def fallback_factory(key: AgentType) -> Any:
    return {"fallback": True, "agent": key.value}

breaker = CircuitBreaker(
    config=config,
    fallback_factory=fallback_factory
)

# Use with fallback
result = await breaker.call_with_fallback(
    AgentType.RESEARCH_EXECUTOR,
    potentially_failing_function
)
```

## Integration with ResearchWorkflow

### Problem Solved

The original implementation had critical issues:
1. **Type mismatch**: AgentType enum was passed but string keys were expected
2. **Race conditions**: No thread safety in async operations
3. **Missing HALF_OPEN state**: Only had CLOSED and OPEN states
4. **Incomplete reset logic**: Error count wasn't reset on timeout

### Solution: Using AgentType Enums Directly

The fix uses AgentType enums directly as dictionary keys, which is the Pythonic approach:

```python
class ResearchWorkflow:
    def __init__(self):
        # Use AgentType enum as keys directly
        self._consecutive_errors: dict[AgentType, int] = {}
        self._last_error_time: dict[AgentType, float] = {}
        self._circuit_open: dict[AgentType, bool] = {}

    def _check_circuit_breaker(self, agent_type: AgentType) -> bool:
        # Now accepts enum directly, no conversion needed
        return not self._circuit_open.get(agent_type, False)
```

### Benefits of Using Enums as Keys

1. **Type Safety**: IDE and type checkers catch errors at development time
2. **Performance**: No conversion overhead
3. **Cleaner Code**: No `.value` or `.name` conversions needed
4. **Bug Prevention**: Can't accidentally pass wrong string or typo
5. **Pythonic**: Enums are designed to be hashable dictionary keys

### Integration Example

The `ResearchWorkflowWithCircuitBreaker` class shows proper integration:

```python
class ResearchWorkflowWithCircuitBreaker:
    def __init__(self):
        # Type-safe circuit breaker with AgentType keys
        self.circuit_breaker: CircuitBreaker[AgentType] = CircuitBreaker()

    async def execute_agent_with_circuit_breaker(
        self,
        agent_type: AgentType,  # Use enum directly
        deps: ResearchDependencies,
        **kwargs
    ) -> Any:
        # No conversion needed - use enum as key
        return await self.circuit_breaker.call(
            agent_type,
            agent.run,
            deps,
            **kwargs
        )
```

### Adaptive Configuration

Different agents get different circuit breaker settings based on criticality:

```python
# Critical agents: More lenient
AgentType.RESEARCH_EXECUTOR: CircuitBreakerConfig(
    failure_threshold=5,    # More attempts
    timeout_seconds=60.0,   # Longer recovery
    half_open_max_attempts=3
)

# Optional agents: Fail fast
AgentType.QUERY_TRANSFORMATION: CircuitBreakerConfig(
    failure_threshold=2,    # Fewer attempts
    timeout_seconds=30.0,   # Shorter recovery
    half_open_max_attempts=1
)
```

## Testing Strategy

### Comprehensive Test Coverage

The test suite (`tests/unit/test_circuit_breaker.py`) covers:

1. **Enum Key Support**:
   - Tests with actual AgentType enum values
   - Verifies different enum keys maintain separate states
   - Confirms type safety

2. **State Management**:
   - Initial state verification
   - All state transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
   - Edge cases in state changes

3. **Concurrent Access**:
   - Multiple concurrent calls with different enum keys
   - Thread-safe state transitions
   - Per-key lock contention
   - HALF_OPEN concurrent limits

4. **Configuration**:
   - Validation of all config parameters
   - Custom vs default configurations
   - Invalid configuration handling

5. **Features**:
   - Excluded exceptions
   - Manual reset
   - Context manager usage
   - Decorator pattern
   - Metrics tracking
   - Fallback factory

### Running Tests

```bash
# Run circuit breaker tests
uv run pytest tests/unit/test_circuit_breaker.py -xvs

# Run with coverage
uv run pytest tests/unit/test_circuit_breaker.py --cov=src.core.circuit_breaker

# Type checking
uv run pyright src/core/circuit_breaker.py
```

## Performance Considerations

1. **Minimal Overhead**: Circuit breaker adds minimal latency in CLOSED state
2. **Fast Failure**: OPEN state fails immediately without calling the service
3. **Controlled Recovery**: HALF_OPEN state limits concurrent recovery attempts
4. **Per-Key Locks**: Prevents contention between different agents
5. **Async-First**: Built for async operations, no blocking calls

## Monitoring and Observability

### Built-in Metrics

```python
# Get metrics for specific agent
metrics = breaker.get_metrics(AgentType.RESEARCH_EXECUTOR)
print(f"Success rate: {metrics.success_rate:.2%}")
print(f"Total attempts: {metrics.total_attempts}")
print(f"Rejected calls: {metrics.rejected_calls}")

# Get all metrics
all_metrics = breaker.get_all_metrics()
```

### Custom Metrics Collector

```python
class CustomCollector(MetricsCollector[AgentType]):
    async def record_state_change(
        self,
        key: AgentType,
        old_state: CircuitState,
        new_state: CircuitState
    ):
        # Send to monitoring system
        await send_to_prometheus(key.value, old_state, new_state)

    async def record_attempt(self, key: AgentType, success: bool):
        # Track in analytics
        await track_event(f"agent.{key.value}.attempt", {"success": success})
```

### Health Check Endpoint

```python
async def health_check() -> dict[str, Any]:
    workflow = ResearchWorkflowWithCircuitBreaker()
    health = await workflow.health_check()

    return {
        "overall_health": health["overall_health"],
        "agents": health["agents"],
        "timestamp": health["timestamp"]
    }
```

## Best Practices

### 1. Use Enums for Type Safety

```python
# Good - Type-safe with IDE support
breaker: CircuitBreaker[AgentType] = CircuitBreaker()
await breaker.call(AgentType.RESEARCH_EXECUTOR, func)

# Avoid - Less type safety
breaker: CircuitBreaker[str] = CircuitBreaker()
await breaker.call("research_executor", func)
```

### 2. Configure Based on Service Criticality

- **Critical services**: Higher thresholds, longer timeouts
- **Optional services**: Lower thresholds, fail fast
- **External APIs**: Consider rate limits in configuration

### 3. Implement Fallbacks

```python
# Always have a degraded mode
async def fallback_factory(agent_type: AgentType):
    if agent_type == AgentType.RESEARCH_EXECUTOR:
        return cached_results()
    return default_response()
```

### 4. Monitor Circuit States

- Set up alerts for circuits that frequently open
- Track success rates over time
- Log state transitions for debugging

### 5. Test Recovery Scenarios

- Regularly test that services can recover from failures
- Verify fallback responses are acceptable
- Test circuit breaker behavior under load

## Migration Guide

### From Broken Implementation to Fixed

1. **Update dictionary type hints**:
```python
# Before (broken)
self._consecutive_errors: dict[str, int] = {}

# After (fixed)
self._consecutive_errors: dict[AgentType, int] = {}
```

2. **Update method signatures**:
```python
# Before (broken)
def _check_circuit_breaker(self, agent_type: str) -> bool:

# After (fixed)
def _check_circuit_breaker(self, agent_type: AgentType) -> bool:
```

3. **Remove string conversions**:
```python
# Before (would be needed)
agent_key = agent_type.value

# After (not needed)
# Use agent_type directly
```

4. **Update logging for readability**:
```python
# Use .value only for display
logger.info(f"Circuit opened for {agent_type.value}")
```

## Files

- **Main Implementation**: `/src/core/circuit_breaker.py`
- **Workflow Integration**: `/src/core/workflow_with_circuit_breaker.py`
- **Original Workflow (Fixed)**: `/src/core/workflow.py`
- **Test Suite**: `/tests/unit/test_circuit_breaker.py`
- **This Documentation**: `/docs/circuit_breaker_implementation.md`

## Summary

The circuit breaker implementation provides a robust, type-safe solution for preventing cascading failures in the research workflow. By using AgentType enums directly as dictionary keys, the solution is:

- **Type-safe**: Compile-time checking prevents errors
- **Performant**: No conversion overhead
- **Pythonic**: Uses language features as intended
- **Production-ready**: Thread-safe with comprehensive testing
- **Observable**: Built-in metrics and monitoring support

The implementation fixes all issues in the original code while providing a clean, maintainable solution that follows Python best practices.
