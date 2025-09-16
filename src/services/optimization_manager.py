"""Optimization management service for research executor."""

import asyncio
import gc
from typing import Any

import logfire
import psutil

from models.research_executor import HierarchicalFinding, OptimizationConfig


class OptimizationManager:
    """Manages optimization strategies for the research executor."""

    def __init__(self, config: OptimizationConfig):
        """Initialize optimization manager.

        Args:
            config: Optimization configuration
        """
        self.config = config
        self.process = psutil.Process()
        self.logger = logfire

        # Adaptive thresholds
        self.current_confidence_threshold = 0.5
        self.current_batch_size = config.batch_size

        # Resource monitoring
        self.high_load_detected = False
        self.memory_pressure_detected = False

    def check_resource_usage(self) -> dict[str, Any]:
        """Check current resource usage.

        Returns:
            Dictionary with resource usage metrics
        """
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        memory_percent = self.process.memory_percent()

        try:
            cpu_percent = self.process.cpu_percent()  # Non-blocking after first call
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            cpu_percent = 0

        return {
            "memory_mb": memory_mb,
            "memory_percent": memory_percent,
            "cpu_percent": cpu_percent,
            "memory_limit_mb": self.config.memory_limit_mb,
        }

    def apply_memory_optimization(self) -> None:
        """Apply memory optimization strategies."""
        # Force garbage collection
        gc.collect()

        # Check if we're approaching memory limit
        usage = self.check_resource_usage()
        if usage["memory_mb"] > self.config.memory_limit_mb * 0.8:
            self.memory_pressure_detected = True
            self.logger.warning(f"Memory pressure detected: {usage['memory_mb']:.1f}MB used")

            # Reduce batch size
            if self.config.enable_adaptive_thresholds:
                self.current_batch_size = max(1, self.current_batch_size // 2)
                self.logger.info(f"Reduced batch size to {self.current_batch_size}")
        else:
            self.memory_pressure_detected = False

    def batch_findings(
        self, findings: list[HierarchicalFinding], batch_size: int | None = None
    ) -> list[list[HierarchicalFinding]]:
        """Batch findings for efficient processing.

        Args:
            findings: List of findings to batch
            batch_size: Optional batch size override

        Returns:
            List of batches
        """
        batch_size = batch_size or self.current_batch_size
        batches = []

        for i in range(0, len(findings), batch_size):
            batch = findings[i : i + batch_size]
            batches.append(batch)

        self.logger.debug(f"Created {len(batches)} batches of size {batch_size}")
        return batches

    def adapt_quality_thresholds(self, data_volume: int, processing_time: float) -> float:
        """Adapt quality thresholds based on data volume and performance.

        Args:
            data_volume: Number of items to process
            processing_time: Current processing time

        Returns:
            Adjusted confidence threshold
        """
        if not self.config.enable_adaptive_thresholds:
            return self.current_confidence_threshold

        # Increase threshold for large datasets to reduce processing
        if data_volume > 1000:
            self.current_confidence_threshold = min(0.7, self.current_confidence_threshold + 0.1)
        elif data_volume > 500:
            self.current_confidence_threshold = min(0.6, self.current_confidence_threshold + 0.05)
        elif data_volume < 100 and processing_time < 5.0:
            # Can afford to be less strict with small datasets
            self.current_confidence_threshold = max(0.3, self.current_confidence_threshold - 0.05)

        self.logger.debug(f"Adjusted confidence threshold to {self.current_confidence_threshold}")
        return self.current_confidence_threshold

    def apply_graceful_degradation(self, load_level: str = "normal") -> dict[str, bool]:
        """Apply graceful degradation under high load.

        Args:
            load_level: Current load level (normal, high, critical)

        Returns:
            Dictionary of feature flags
        """
        if not self.config.enable_graceful_degradation:
            return {
                "enable_clustering": True,
                "enable_pattern_recognition": True,
                "enable_confidence_analysis": True,
                "enable_contradiction_detection": True,
            }

        features = {}

        if load_level == "critical":
            # Disable non-essential features
            features = {
                "enable_clustering": False,
                "enable_pattern_recognition": False,
                "enable_confidence_analysis": False,
                "enable_contradiction_detection": True,  # Keep critical feature
            }
            self.logger.warning("Critical load: disabled non-essential features")

        elif load_level == "high":
            # Reduce feature set
            features = {
                "enable_clustering": False,
                "enable_pattern_recognition": True,
                "enable_confidence_analysis": True,
                "enable_contradiction_detection": True,
            }
            self.logger.info("High load: reduced feature set")

        else:
            # Normal operation
            features = {
                "enable_clustering": True,
                "enable_pattern_recognition": True,
                "enable_confidence_analysis": True,
                "enable_contradiction_detection": True,
            }

        return features

    def detect_load_level(self) -> str:
        """Detect current system load level.

        Returns:
            Load level (normal, high, critical)
        """
        usage = self.check_resource_usage()

        # Check memory
        memory_ratio = usage["memory_mb"] / self.config.memory_limit_mb

        # Check CPU
        cpu_high = usage["cpu_percent"] > 80

        if memory_ratio > 0.9 or usage["cpu_percent"] > 90:
            return "critical"
        if memory_ratio > 0.7 or cpu_high:
            return "high"
        return "normal"

    async def optimize_async_operations(self, operations: list[asyncio.Task]) -> list[Any]:
        """Optimize async operations execution.

        Args:
            operations: List of async operations

        Returns:
            Results from operations
        """
        # Apply memory optimization before heavy operations
        self.apply_memory_optimization()

        # Detect load and apply degradation if needed
        load_level = self.detect_load_level()
        if load_level != "normal":
            self.high_load_detected = True

        # Execute with appropriate strategy
        if load_level == "critical":
            # Sequential execution under critical load
            results = []
            for op in operations:
                try:
                    result = await op
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Operation failed: {e}")
                    results.append(None)
            return results
        # Normal parallel execution
        return await asyncio.gather(*operations, return_exceptions=True)

    def get_optimization_status(self) -> dict[str, Any]:
        """Get current optimization status.

        Returns:
            Status dictionary
        """
        usage = self.check_resource_usage()
        load_level = self.detect_load_level()

        return {
            "load_level": load_level,
            "resource_usage": usage,
            "high_load_detected": self.high_load_detected,
            "memory_pressure_detected": self.memory_pressure_detected,
            "current_confidence_threshold": self.current_confidence_threshold,
            "current_batch_size": self.current_batch_size,
            "features": self.apply_graceful_degradation(load_level),
        }

    def reset_adaptations(self) -> None:
        """Reset adaptive parameters to defaults."""
        self.current_confidence_threshold = 0.5
        self.current_batch_size = self.config.batch_size
        self.high_load_detected = False
        self.memory_pressure_detected = False
        self.logger.info("Reset optimization adaptations to defaults")
