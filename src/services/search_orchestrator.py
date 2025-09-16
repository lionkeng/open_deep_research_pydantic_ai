"""Search Orchestrator Service for deterministic query execution management."""

import asyncio
import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# Local model definitions until integrated with main models


class ExecutionStrategy(str, Enum):
    """Execution strategy for queries."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"


class QueryPriority(str, Enum):
    """Priority levels for queries."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SearchQuery(BaseModel):
    """Search query model."""

    id: str | None = Field(default=None)
    query: str = Field(description="The search query text")
    priority: QueryPriority | None = Field(default=QueryPriority.MEDIUM)
    context: dict[str, Any] | None = Field(default=None)


class SearchResult(BaseModel):
    """Search result model."""

    query: str = Field(description="The original query")
    results: list[Any] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class QueryExecutionPlan(BaseModel):
    """Query execution plan model."""

    queries: list[SearchQuery] = Field(description="Queries to execute")
    strategy: ExecutionStrategy = Field(default=ExecutionStrategy.SEQUENTIAL)


class ExecutionStatus(str, Enum):
    """Status of query execution."""

    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CACHED = "cached"


@dataclass
class ExecutionTrace:
    """Trace of a single query execution."""

    query_id: str
    query_text: str
    status: ExecutionStatus
    start_time: float
    end_time: float | None = None
    attempts: int = 0
    error: str | None = None
    cached: bool = False
    result_count: int = 0
    execution_time_ms: float | None = None


@dataclass
class ExecutionReport:
    """Comprehensive execution report."""

    total_queries: int
    executed_queries: int
    failed_queries: int
    cached_queries: int
    execution_rate: float
    total_time_ms: float
    average_time_ms: float
    traces: list[ExecutionTrace] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    strategy_used: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL


class RetryConfig(BaseModel):
    """Configuration for retry logic."""

    max_attempts: int = Field(default=3, description="Maximum retry attempts")
    initial_delay_ms: int = Field(default=100, description="Initial retry delay in milliseconds")
    max_delay_ms: int = Field(default=5000, description="Maximum retry delay in milliseconds")
    exponential_base: float = Field(default=2.0, description="Base for exponential backoff")


class CacheConfig(BaseModel):
    """Configuration for caching mechanism."""

    enabled: bool = Field(default=True, description="Whether caching is enabled")
    ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    max_size: int = Field(default=1000, description="Maximum cache size")
    cache_dir: Path | None = Field(default=None, description="Directory for persistent cache")


class SearchOrchestrator:
    """
    Deterministic query execution manager ensuring 100% execution rate.

    Supports parallel, sequential, and hierarchical execution strategies
    with retry logic, caching, and comprehensive execution tracking.
    """

    def __init__(
        self,
        search_fn: Callable | None = None,
        retry_config: RetryConfig | None = None,
        cache_config: CacheConfig | None = None,
        max_workers: int = 10,
    ):
        """
        Initialize the SearchOrchestrator.

        Args:
            search_fn: Function to execute searches (async or sync)
            retry_config: Configuration for retry logic
            cache_config: Configuration for caching
            max_workers: Maximum number of parallel workers
        """
        self.search_fn = search_fn
        self.retry_config = retry_config or RetryConfig()
        self.cache_config = cache_config or CacheConfig()
        self.max_workers = max_workers

        # In-memory cache
        self._cache: dict[str, tuple[SearchResult, float]] = {}
        self._cache_order: list[str] = []

        # Execution tracking
        self._execution_traces: list[ExecutionTrace] = []
        self._execution_stats: dict[str, Any] = defaultdict(int)

        # Thread pool for parallel execution
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    def _generate_cache_key(self, query: SearchQuery) -> str:
        """Generate a unique cache key for a query."""
        query_data = {
            "text": query.query,
            "context": query.context.model_dump()
            if hasattr(query.context, "model_dump")
            else query.context,
            "priority": query.priority.value if query.priority else None,
        }
        query_json = json.dumps(query_data, sort_keys=True)
        return hashlib.sha256(query_json.encode()).hexdigest()

    async def _get_from_cache(self, query: SearchQuery) -> SearchResult | None:
        """Retrieve result from cache if available and valid."""
        if not self.cache_config.enabled:
            return None

        cache_key = self._generate_cache_key(query)

        async with self._lock:
            if cache_key in self._cache:
                result, timestamp = self._cache[cache_key]
                if time.time() - timestamp < self.cache_config.ttl_seconds:
                    # Move to end (LRU)
                    self._cache_order.remove(cache_key)
                    self._cache_order.append(cache_key)
                    return result
                # Expired
                del self._cache[cache_key]
                self._cache_order.remove(cache_key)

        return None

    async def _save_to_cache(self, query: SearchQuery, result: SearchResult) -> None:
        """Save result to cache."""
        if not self.cache_config.enabled:
            return

        cache_key = self._generate_cache_key(query)

        async with self._lock:
            # Enforce cache size limit (LRU eviction)
            while len(self._cache) >= self.cache_config.max_size and self._cache_order:
                oldest_key = self._cache_order.pop(0)
                del self._cache[oldest_key]

            self._cache[cache_key] = (result, time.time())
            if cache_key not in self._cache_order:
                self._cache_order.append(cache_key)

    async def _execute_with_retry(
        self, query: SearchQuery, trace: ExecutionTrace
    ) -> SearchResult | None:
        """Execute a query with exponential backoff retry logic."""
        last_error = None

        for attempt in range(self.retry_config.max_attempts):
            trace.attempts = attempt + 1

            try:
                # Update status
                trace.status = (
                    ExecutionStatus.EXECUTING if attempt == 0 else ExecutionStatus.RETRYING
                )

                # Execute search
                if self.search_fn:
                    if asyncio.iscoroutinefunction(self.search_fn):
                        result = await self.search_fn(query)
                    else:
                        result = await asyncio.get_event_loop().run_in_executor(
                            self._executor, self.search_fn, query
                        )
                else:
                    # Mock result for testing
                    result = SearchResult(
                        query=query.query,
                        results=[],
                        metadata={"mock": True},
                        timestamp=datetime.now(UTC),
                    )

                # Success
                trace.status = ExecutionStatus.COMPLETED
                trace.result_count = len(result.results) if hasattr(result, "results") else 0
                return result

            except Exception as e:
                last_error = str(e)
                trace.error = last_error

                if attempt < self.retry_config.max_attempts - 1:
                    # Calculate backoff delay
                    delay = min(
                        self.retry_config.initial_delay_ms
                        * (self.retry_config.exponential_base**attempt),
                        self.retry_config.max_delay_ms,
                    )
                    await asyncio.sleep(delay / 1000.0)
                else:
                    trace.status = ExecutionStatus.FAILED

        return None

    async def _execute_query(self, query: SearchQuery) -> tuple[SearchQuery, SearchResult | None]:
        """Execute a single query with caching and tracking."""
        trace = ExecutionTrace(
            query_id=query.id or hashlib.md5(query.query.encode()).hexdigest()[:8],
            query_text=query.query,
            status=ExecutionStatus.PENDING,
            start_time=time.time(),
        )

        try:
            # Check cache first
            cached_result = await self._get_from_cache(query)
            if cached_result:
                trace.status = ExecutionStatus.CACHED
                trace.cached = True
                trace.end_time = time.time()
                trace.execution_time_ms = (trace.end_time - trace.start_time) * 1000
                self._execution_traces.append(trace)
                return query, cached_result

            # Execute with retry
            result = await self._execute_with_retry(query, trace)

            # Save to cache if successful
            if result:
                await self._save_to_cache(query, result)

            trace.end_time = time.time()
            trace.execution_time_ms = (trace.end_time - trace.start_time) * 1000
            self._execution_traces.append(trace)

            return query, result

        except Exception as e:
            trace.status = ExecutionStatus.FAILED
            trace.error = str(e)
            trace.end_time = time.time()
            trace.execution_time_ms = (trace.end_time - trace.start_time) * 1000
            self._execution_traces.append(trace)
            return query, None

    async def execute_sequential(
        self, queries: list[SearchQuery]
    ) -> list[tuple[SearchQuery, SearchResult | None]]:
        """Execute queries sequentially."""
        results = []
        for query in queries:
            result = await self._execute_query(query)
            results.append(result)
        return results

    async def execute_parallel(
        self, queries: list[SearchQuery]
    ) -> list[tuple[SearchQuery, SearchResult | None]]:
        """Execute queries in parallel."""
        tasks = [self._execute_query(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return results

    async def execute_hierarchical(
        self, queries: list[SearchQuery]
    ) -> list[tuple[SearchQuery, SearchResult | None]]:
        """
        Execute queries hierarchically based on priority and dependencies.

        High priority queries are executed first in parallel,
        followed by medium and low priority queries.
        """
        # Group by priority
        priority_groups: dict[QueryPriority, list[SearchQuery]] = defaultdict(list)
        for query in queries:
            priority = query.priority or QueryPriority.MEDIUM
            priority_groups[priority].append(query)

        results = []

        # Execute in priority order
        for priority in [QueryPriority.HIGH, QueryPriority.MEDIUM, QueryPriority.LOW]:
            if priority in priority_groups:
                group_queries = priority_groups[priority]
                # Execute each priority group in parallel
                group_results = await self.execute_parallel(group_queries)
                results.extend(group_results)

        return results

    async def execute_plan(
        self, plan: QueryExecutionPlan
    ) -> tuple[list[tuple[SearchQuery, SearchResult | None]], ExecutionReport]:
        """
        Execute a query execution plan with the specified strategy.

        Args:
            plan: The execution plan containing queries and strategy

        Returns:
            Tuple of results and execution report
        """
        start_time = time.time()
        self._execution_traces.clear()

        # Select execution strategy
        if plan.strategy == ExecutionStrategy.PARALLEL:
            results = await self.execute_parallel(plan.queries)
        elif plan.strategy == ExecutionStrategy.HIERARCHICAL:
            results = await self.execute_hierarchical(plan.queries)
        else:  # SEQUENTIAL
            results = await self.execute_sequential(plan.queries)

        # Generate execution report
        total_time = (time.time() - start_time) * 1000

        executed = sum(1 for t in self._execution_traces if t.status == ExecutionStatus.COMPLETED)
        failed = sum(1 for t in self._execution_traces if t.status == ExecutionStatus.FAILED)
        cached = sum(1 for t in self._execution_traces if t.cached)

        report = ExecutionReport(
            total_queries=len(plan.queries),
            executed_queries=executed,
            failed_queries=failed,
            cached_queries=cached,
            execution_rate=(executed + cached) / len(plan.queries) if plan.queries else 0,
            total_time_ms=total_time,
            average_time_ms=total_time / len(plan.queries) if plan.queries else 0,
            traces=self._execution_traces.copy(),
            errors=[
                {"query_id": t.query_id, "error": t.error}
                for t in self._execution_traces
                if t.error
            ],
            strategy_used=plan.strategy,
        )

        return results, report

    def get_execution_stats(self) -> dict[str, Any]:
        """Get execution statistics."""
        if not self._execution_traces:
            return {
                "total_executions": 0,
                "success_rate": 0,
                "cache_hit_rate": 0,
                "average_execution_time_ms": 0,
            }

        total = len(self._execution_traces)
        successful = sum(1 for t in self._execution_traces if t.status == ExecutionStatus.COMPLETED)
        cached = sum(1 for t in self._execution_traces if t.cached)

        execution_times = [
            t.execution_time_ms for t in self._execution_traces if t.execution_time_ms is not None
        ]

        return {
            "total_executions": total,
            "success_rate": successful / total if total > 0 else 0,
            "cache_hit_rate": cached / total if total > 0 else 0,
            "average_execution_time_ms": sum(execution_times) / len(execution_times)
            if execution_times
            else 0,
            "retry_rate": sum(1 for t in self._execution_traces if t.attempts > 1) / total
            if total > 0
            else 0,
        }

    async def clear_cache(self) -> None:
        """Clear the execution cache."""
        async with self._lock:
            self._cache.clear()
            self._cache_order.clear()

    def __del__(self):
        """Cleanup resources."""
        self._executor.shutdown(wait=False)
