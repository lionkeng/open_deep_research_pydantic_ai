"""Parallel execution service for research executor."""

import asyncio
import time
from collections import defaultdict
from collections.abc import Callable, Coroutine
from enum import Enum
from typing import Any, TypeVar

import logfire

from models.research_executor import OptimizationConfig

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for external service calls."""

    def __init__(self, failure_threshold: int, timeout: int):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening
            timeout: Timeout in seconds before attempting half-open
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.state = CircuitState.CLOSED

    def call_succeeded(self) -> None:
        """Record a successful call."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def call_failed(self) -> None:
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def can_execute(self) -> bool:
        """Check if execution is allowed.

        Returns:
            True if execution is allowed
        """
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.timeout:
                self.state = CircuitState.HALF_OPEN
                return True
            return False

        # HALF_OPEN state
        return True


class ParallelExecutor:
    """Service for parallel execution of research tasks."""

    def __init__(self, config: OptimizationConfig):
        """Initialize parallel executor.

        Args:
            config: Optimization configuration
        """
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrent_tasks)
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self.connection_pools: dict[str, list[Any]] = defaultdict(list)
        self.metrics = {
            "parallel_tasks_executed": 0,
            "circuit_breaker_trips": 0,
            "task_failures": 0,
            "total_execution_time": 0.0,
        }
        self.logger = logfire

    def _get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for a service.

        Args:
            service_name: Name of the service

        Returns:
            Circuit breaker instance
        """
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker(
                self.config.circuit_breaker_threshold, self.config.circuit_breaker_timeout
            )
        return self.circuit_breakers[service_name]

    async def execute_parallel(
        self,
        tasks: list[Coroutine[Any, Any, T]],
        batch_size: int | None = None,
        service_name: str | None = None,
    ) -> list[tuple[bool, T | None]]:
        """Execute tasks in parallel with batching.

        Args:
            tasks: List of async tasks to execute
            batch_size: Optional batch size override
            service_name: Optional service name for circuit breaker

        Returns:
            List of (success, result) tuples
        """
        if not self.config.enable_parallel_execution:
            # Execute sequentially if parallel execution is disabled
            results = []
            for task in tasks:
                try:
                    result = await task
                    results.append((True, result))
                except Exception as e:
                    self.logger.error(f"Task failed: {e}")
                    results.append((False, None))
            return results

        batch_size = batch_size or self.config.batch_size
        results: list[tuple[bool, T | None]] = []

        # Process in batches
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            batch_results = await self._execute_batch(batch, service_name)
            results.extend(batch_results)

        self.metrics["parallel_tasks_executed"] += len(tasks)
        return results

    async def _execute_batch(
        self, batch: list[Coroutine[Any, Any, T]], service_name: str | None = None
    ) -> list[tuple[bool, T | None]]:
        """Execute a batch of tasks.

        Args:
            batch: Batch of tasks
            service_name: Optional service name for circuit breaker

        Returns:
            List of (success, result) tuples
        """

        async def execute_with_semaphore(task: Coroutine[Any, Any, T]) -> tuple[bool, T | None]:
            """Execute a single task with semaphore control."""
            async with self.semaphore:
                # Check circuit breaker if service name provided
                if service_name:
                    breaker = self._get_circuit_breaker(service_name)
                    if not breaker.can_execute():
                        self.metrics["circuit_breaker_trips"] += 1
                        self.logger.warning(f"Circuit breaker open for {service_name}")
                        return (False, None)

                try:
                    start_time = time.time()
                    result = await asyncio.wait_for(
                        task, timeout=self.config.request_timeout_seconds
                    )
                    execution_time = time.time() - start_time
                    self.metrics["total_execution_time"] += execution_time

                    if service_name:
                        breaker.call_succeeded()

                    return (True, result)

                except TimeoutError:
                    self.logger.error(
                        f"Task timed out after {self.config.request_timeout_seconds}s"
                    )
                    if service_name:
                        breaker = self._get_circuit_breaker(service_name)
                        breaker.call_failed()
                    self.metrics["task_failures"] += 1
                    return (False, None)

                except Exception as e:
                    self.logger.error(f"Task failed: {e}")
                    if service_name:
                        breaker = self._get_circuit_breaker(service_name)
                        breaker.call_failed()
                    self.metrics["task_failures"] += 1
                    return (False, None)

        # Execute all tasks in the batch concurrently
        results = await asyncio.gather(
            *[execute_with_semaphore(task) for task in batch], return_exceptions=False
        )

        return results

    async def map_parallel(
        self,
        func: Callable[[Any], Coroutine[Any, Any, T]],
        items: list[Any],
        batch_size: int | None = None,
    ) -> list[tuple[bool, T | None]]:
        """Map a function over items in parallel.

        Args:
            func: Async function to apply
            items: Items to process
            batch_size: Optional batch size override

        Returns:
            List of (success, result) tuples
        """
        tasks = [func(item) for item in items]
        return await self.execute_parallel(tasks, batch_size)

    def get_connection(self, pool_name: str) -> Any:
        """Get a connection from the pool.

        Args:
            pool_name: Name of the connection pool

        Returns:
            Connection object or None
        """
        pool = self.connection_pools[pool_name]
        if pool:
            return pool.pop()
        return None

    def return_connection(self, pool_name: str, connection: Any) -> None:
        """Return a connection to the pool.

        Args:
            pool_name: Name of the connection pool
            connection: Connection to return
        """
        pool = self.connection_pools[pool_name]
        if len(pool) < self.config.connection_pool_size:
            pool.append(connection)

    def get_metrics(self) -> dict[str, Any]:
        """Get execution metrics.

        Returns:
            Dictionary of metrics
        """
        return {
            "parallel_tasks_executed": self.metrics["parallel_tasks_executed"],
            "circuit_breaker_trips": self.metrics["circuit_breaker_trips"],
            "task_failures": self.metrics["task_failures"],
            "total_execution_time": self.metrics["total_execution_time"],
            "avg_execution_time": (
                self.metrics["total_execution_time"]
                / max(self.metrics["parallel_tasks_executed"], 1)
            ),
            "circuit_breaker_states": {
                name: breaker.state.value for name, breaker in self.circuit_breakers.items()
            },
        }
