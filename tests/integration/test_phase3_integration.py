"""Integration tests for Phase 3 optimization features."""

import asyncio
import json
import time

import pytest

from src.models.research_executor import (
    ConfidenceLevel,
    HierarchicalFinding,
    ImportanceLevel,
    OptimizationConfig,
)
from src.services.cache_manager import CacheManager
from src.services.metrics_collector import MetricsCollector
from src.services.optimization_manager import OptimizationManager
from src.services.parallel_executor import CircuitBreaker, ParallelExecutor


class TestCacheManager:
    """Test cache management functionality."""

    def test_cache_basic_operations(self):
        """Test basic cache get/set operations."""
        config = OptimizationConfig(enable_caching=True, cache_ttl_seconds=60)
        cache = CacheManager(config)

        # Test set and get
        test_data = {"key": "value", "number": 42}
        cache_key = cache.set("test", "content1", test_data)

        assert cache_key is not None
        retrieved = cache.get("test", "content1")
        assert retrieved == test_data

        # Test cache hit metrics
        metrics = cache.get_metrics()
        assert metrics["hits"] == 1
        assert metrics["misses"] == 0

    def test_cache_expiration(self):
        """Test cache TTL and expiration."""
        config = OptimizationConfig(enable_caching=True, cache_ttl_seconds=1)
        cache = CacheManager(config)

        # Set with short TTL
        cache.set("test", "expires", "data", ttl_override=1)

        # Should be available immediately
        assert cache.get("test", "expires") == "data"

        # Wait for expiration
        time.sleep(1.5)

        # Should be expired
        assert cache.get("test", "expires") is None

        metrics = cache.get_metrics()
        assert metrics["misses"] == 1

    def test_cache_size_limit(self):
        """Test cache size limit enforcement."""
        config = OptimizationConfig(
            enable_caching=True,
            max_cache_size_mb=0.001  # Very small limit for testing
        )
        cache = CacheManager(config)

        # Add multiple items that exceed size limit
        for i in range(100):
            large_data = "x" * 1000  # 1KB each
            cache.set("test", f"item_{i}", large_data)

        # Check that evictions occurred
        metrics = cache.get_metrics()
        assert metrics["evictions"] > 0
        assert metrics["current_size_mb"] <= config.max_cache_size_mb

    def test_cache_invalidation(self):
        """Test cache invalidation."""
        config = OptimizationConfig(enable_caching=True)
        cache = CacheManager(config)

        # Add multiple items
        cache.set("synthesis", "item1", "data1")
        cache.set("synthesis", "item2", "data2")
        cache.set("vectorization", "item3", "data3")

        # Invalidate by type
        invalidated = cache.invalidate(cache_type="synthesis")
        assert invalidated == 2

        # Verify items are gone
        assert cache.get("synthesis", "item1") is None
        assert cache.get("synthesis", "item2") is None
        assert cache.get("vectorization", "item3") == "data3"

        # Clear all
        cache.clear()
        assert cache.get("vectorization", "item3") is None


class TestParallelExecutor:
    """Test parallel execution functionality."""

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Test basic parallel execution."""
        config = OptimizationConfig(
            enable_parallel_execution=True,
            max_concurrent_tasks=4
        )
        executor = ParallelExecutor(config)

        # Create test tasks
        async def test_task(value: int) -> int:
            await asyncio.sleep(0.1)
            return value * 2

        tasks = [test_task(i) for i in range(10)]
        results = await executor.execute_parallel(tasks)

        # Check all succeeded
        assert all(success for success, _ in results)

        # Check results
        values = [result for success, result in results if success]
        assert values == [i * 2 for i in range(10)]

        # Check metrics
        metrics = executor.get_metrics()
        assert metrics["parallel_tasks_executed"] == 10
        assert metrics["task_failures"] == 0

    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch processing."""
        config = OptimizationConfig(
            enable_parallel_execution=True,
            max_concurrent_tasks=2,
            batch_size=3
        )
        executor = ParallelExecutor(config)

        executed_batches = []

        async def track_batch(value: int) -> int:
            executed_batches.append(value)
            return value

        tasks = [track_batch(i) for i in range(9)]
        await executor.execute_parallel(tasks, batch_size=3)

        # Should process in batches of 3
        assert len(executed_batches) == 9

    @pytest.mark.asyncio
    async def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        breaker = CircuitBreaker(failure_threshold=3, timeout=1)

        # Should start closed
        assert breaker.can_execute()

        # Record failures
        for _ in range(3):
            breaker.call_failed()

        # Should be open after threshold
        assert not breaker.can_execute()

        # Wait for timeout
        time.sleep(1.1)

        # Should allow retry (half-open)
        assert breaker.can_execute()

        # Success should close it
        breaker.call_succeeded()
        assert breaker.can_execute()

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling in parallel execution."""
        config = OptimizationConfig(
            enable_parallel_execution=True,
            request_timeout_seconds=1
        )
        executor = ParallelExecutor(config)

        async def slow_task():
            await asyncio.sleep(2)
            return "completed"

        results = await executor.execute_parallel([slow_task()])

        # Should timeout
        success, result = results[0]
        assert not success
        assert result is None

        metrics = executor.get_metrics()
        assert metrics["task_failures"] == 1


class TestMetricsCollector:
    """Test metrics collection functionality."""

    def test_metrics_collection(self):
        """Test basic metrics collection."""
        config = OptimizationConfig(enable_metrics_collection=True)
        collector = MetricsCollector(config)

        # Start collection
        collector.start_collection()

        # Record various metrics
        collector.record_timing("synthesis_time", 1.5)
        collector.record_count("findings_processed", 10)
        collector.record_quality_metric("accuracy", 0.95)
        collector.record_api_usage(5, 1000)

        # End collection
        collector.end_collection()

        # Check snapshot was saved
        assert len(collector.snapshots) == 1
        snapshot = collector.snapshots[0]

        assert snapshot.performance.synthesis_time == 1.5
        assert snapshot.performance.findings_processed == 10
        assert snapshot.quality_metrics["accuracy"] == 0.95
        assert snapshot.cost_metrics["api_calls"] == 5

    def test_confidence_distribution(self):
        """Test confidence distribution recording."""
        config = OptimizationConfig(enable_metrics_collection=True)
        collector = MetricsCollector(config)

        collector.start_collection()

        confidences = [0.3, 0.5, 0.7, 0.9, 0.95]
        collector.record_confidence_distribution(confidences)

        collector.end_collection()

        dist = collector.snapshots[0].quality_metrics["confidence_distribution"]
        assert dist["min"] == 0.3
        assert dist["max"] == 0.95
        assert dist["mean"] == pytest.approx(0.67, 0.01)
        assert dist["median"] == 0.7

    def test_export_json(self):
        """Test JSON export of metrics."""
        config = OptimizationConfig(
            enable_metrics_collection=True,
            metrics_export_format="json"
        )
        collector = MetricsCollector(config)

        # Create a snapshot
        collector.start_collection()
        collector.record_timing("synthesis_time", 1.0)
        collector.end_collection()

        # Export as JSON
        json_str = collector.export_metrics("json")
        data = json.loads(json_str)

        assert len(data) == 1
        assert "performance" in data[0]
        assert data[0]["performance"]["synthesis_time"] == 1.0

    def test_export_csv(self):
        """Test CSV export of metrics."""
        config = OptimizationConfig(
            enable_metrics_collection=True,
            metrics_export_format="csv"
        )
        collector = MetricsCollector(config)

        # Create snapshots
        for i in range(3):
            collector.start_collection()
            collector.record_count("findings_processed", i * 10)
            collector.end_collection()

        # Export as CSV
        csv_str = collector.export_metrics("csv")

        # Should have header and 3 data rows
        lines = csv_str.strip().split("\n")
        assert len(lines) == 4
        assert "perf_findings_processed" in lines[0]

    def test_metrics_summary(self):
        """Test metrics summary generation."""
        config = OptimizationConfig(enable_metrics_collection=True)
        collector = MetricsCollector(config)

        # Create multiple sessions
        for i in range(3):
            collector.start_collection()
            collector.record_count("findings_processed", 10)
            collector.record_count("patterns_detected", 2)
            collector.record_synthesis_quality(0.8 + i * 0.05, 0.9)
            collector.end_collection()

        summary = collector.get_summary()

        assert summary["sessions"] == 3
        assert summary["total_findings_processed"] == 30
        assert summary["total_patterns_detected"] == 6
        assert summary["avg_synthesis_quality"] == pytest.approx(0.85, 0.01)


class TestOptimizationManager:
    """Test optimization management functionality."""

    def test_batch_findings(self):
        """Test findings batching."""
        config = OptimizationConfig(batch_size=3)
        manager = OptimizationManager(config)

        # Create test findings
        findings = [
            HierarchicalFinding(
                finding=f"Finding {i}",
                confidence=ConfidenceLevel.HIGH,
                importance=ImportanceLevel.MEDIUM
            )
            for i in range(10)
        ]

        batches = manager.batch_findings(findings)

        assert len(batches) == 4  # 10 items in batches of 3
        assert len(batches[0]) == 3
        assert len(batches[-1]) == 1  # Last batch has remainder

    def test_adaptive_thresholds(self):
        """Test adaptive quality threshold adjustment."""
        config = OptimizationConfig(enable_adaptive_thresholds=True)
        manager = OptimizationManager(config)

        # Large dataset should increase threshold
        threshold = manager.adapt_quality_thresholds(1500, 10.0)
        assert threshold > 0.5

        # Reset for next test
        manager.current_confidence_threshold = 0.5

        # Small dataset with fast processing should decrease threshold
        threshold = manager.adapt_quality_thresholds(50, 2.0)
        assert threshold < 0.5

    def test_graceful_degradation(self):
        """Test graceful degradation under load."""
        config = OptimizationConfig(enable_graceful_degradation=True)
        manager = OptimizationManager(config)

        # Normal load - all features enabled
        features = manager.apply_graceful_degradation("normal")
        assert all(features.values())

        # High load - reduced features
        features = manager.apply_graceful_degradation("high")
        assert not features["enable_clustering"]
        assert features["enable_pattern_recognition"]

        # Critical load - minimal features
        features = manager.apply_graceful_degradation("critical")
        assert not features["enable_clustering"]
        assert not features["enable_pattern_recognition"]
        assert features["enable_contradiction_detection"]  # Critical feature retained

    def test_resource_monitoring(self):
        """Test resource usage monitoring."""
        config = OptimizationConfig(memory_limit_mb=1000)
        manager = OptimizationManager(config)

        usage = manager.check_resource_usage()

        assert "memory_mb" in usage
        assert "memory_percent" in usage
        assert "cpu_percent" in usage
        assert usage["memory_limit_mb"] == 1000

    @pytest.mark.asyncio
    async def test_optimize_async_operations(self):
        """Test async operation optimization."""
        config = OptimizationConfig(
            enable_graceful_degradation=True,
            memory_limit_mb=10000  # High limit to avoid triggering
        )
        manager = OptimizationManager(config)

        async def test_op(value: int) -> int:
            await asyncio.sleep(0.01)
            return value * 2

        operations = [asyncio.create_task(test_op(i)) for i in range(5)]
        results = await manager.optimize_async_operations(operations)

        assert len(results) == 5
        assert results == [0, 2, 4, 6, 8]

    def test_optimization_status(self):
        """Test optimization status reporting."""
        config = OptimizationConfig()
        manager = OptimizationManager(config)

        status = manager.get_optimization_status()

        assert "load_level" in status
        assert "resource_usage" in status
        assert "current_confidence_threshold" in status
        assert "current_batch_size" in status
        assert "features" in status

        # Load level should be normal in test environment
        assert status["load_level"] in ["normal", "high", "critical"]


class TestPhase3Integration:
    """Integration tests for all Phase 3 components."""

    @pytest.mark.asyncio
    async def test_integrated_optimization_flow(self):
        """Test integrated optimization flow with all components."""
        # Configure all optimization features
        config = OptimizationConfig(
            enable_caching=True,
            cache_ttl_seconds=60,
            enable_parallel_execution=True,
            max_concurrent_tasks=4,
            batch_size=5,
            enable_adaptive_thresholds=True,
            enable_metrics_collection=True,
            enable_graceful_degradation=True,
        )

        # Initialize all services
        cache = CacheManager(config)
        executor = ParallelExecutor(config)
        collector = MetricsCollector(config)
        optimizer = OptimizationManager(config)

        # Start metrics collection
        collector.start_collection()

        # Simulate processing with caching
        test_data = [f"data_{i}" for i in range(10)]

        # First pass - cache misses
        for item in test_data:
            cached = cache.get("processing", item)
            if cached is None:
                # Simulate processing
                result = f"processed_{item}"
                cache.set("processing", item, result)

        # Second pass - cache hits
        for item in test_data:
            cached = cache.get("processing", item)
            assert cached is not None

        # Parallel processing with batching
        findings = [
            HierarchicalFinding(
                finding=f"Finding {i}",
                confidence=ConfidenceLevel.MEDIUM,
                importance=ImportanceLevel.MEDIUM,
                confidence_score=min(0.5 + i * 0.03, 1.0),  # Cap at 1.0 to avoid validation error
            )
            for i in range(15)
        ]

        batches = optimizer.batch_findings(findings, batch_size=5)

        async def process_batch(batch: list[HierarchicalFinding]) -> int:
            await asyncio.sleep(0.01)
            return len(batch)

        # Process batches in parallel
        tasks = [process_batch(batch) for batch in batches]
        await executor.execute_parallel(tasks)

        # Record metrics
        collector.record_count("findings_processed", len(findings))
        collector.record_confidence_distribution(
            [f.confidence_score for f in findings]
        )

        # Check optimization status
        status = optimizer.get_optimization_status()

        # End metrics collection
        collector.end_collection()

        # Verify integration
        cache_metrics = cache.get_metrics()
        assert cache_metrics["hits"] > 0

        exec_metrics = executor.get_metrics()
        assert exec_metrics["parallel_tasks_executed"] > 0

        summary = collector.get_summary()
        assert summary["total_findings_processed"] == 15

        assert status["load_level"] in ["normal", "high", "critical"]

    @pytest.mark.asyncio
    async def test_performance_improvement(self):
        """Test that optimizations actually improve performance."""
        # Test without optimizations
        config_no_opt = OptimizationConfig(
            enable_caching=False,
            enable_parallel_execution=False,
            enable_adaptive_thresholds=False,
        )

        # Test with optimizations
        config_opt = OptimizationConfig(
            enable_caching=True,
            enable_parallel_execution=True,
            enable_adaptive_thresholds=True,
            max_concurrent_tasks=4,
        )

        # Create a persistent cache for warm run test
        cache_instance = CacheManager(config_opt)

        async def simulate_processing(
            config: OptimizationConfig, cache: CacheManager | None = None
        ) -> float:
            """Simulate processing with given configuration."""
            executor = ParallelExecutor(config)

            start_time = time.time()

            async def process_item(item: int) -> int:
                # Check cache if available
                if cache:
                    cached = cache.get("test", str(item))
                    if cached is not None:
                        return cached

                # Simulate work
                await asyncio.sleep(0.01)
                result = item * 2

                # Store in cache if available
                if cache:
                    cache.set("test", str(item), result)

                return result

            # Process items
            tasks = [process_item(i) for i in range(20)]

            if config.enable_parallel_execution:
                await executor.execute_parallel(tasks)
            else:
                for task in tasks:
                    await task

            return time.time() - start_time

        # First run without optimizations
        time_no_opt = await simulate_processing(config_no_opt)

        # First run with optimizations (cache cold)
        time_opt_cold = await simulate_processing(config_opt, cache=cache_instance)

        # Second run with optimizations (cache warm - same cache instance)
        time_opt_warm = await simulate_processing(config_opt, cache=cache_instance)

        # Parallel execution should be faster than sequential
        assert time_opt_cold < time_no_opt

        # Cached run should be significantly faster
        assert time_opt_warm < time_opt_cold * 0.5  # At least 50% faster with warm cache
