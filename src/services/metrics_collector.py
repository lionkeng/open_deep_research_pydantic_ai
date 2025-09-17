"""Metrics collection service for research executor."""

import csv
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

import logfire
import psutil

from models.research_executor import (
    OptimizationConfig,
    PatternAnalysis,
    PerformanceMetrics,
)


def _normalize_patterns(
    patterns: Sequence[PatternAnalysis | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convert pattern payloads to JSON-compatible dictionaries."""

    normalized: list[dict[str, Any]] = []
    for item in patterns:
        if isinstance(item, PatternAnalysis):
            normalized.append(item.model_dump(mode="json", exclude_none=True))
        elif isinstance(item, Mapping):
            normalized.append(dict(item))
        else:
            raise TypeError(f"Unsupported pattern payload type: {type(item)!r}")
    return normalized


@dataclass
class MetricsSnapshot:
    """Snapshot of metrics at a point in time."""

    timestamp: datetime = field(default_factory=datetime.now)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    quality_metrics: dict[str, Any] = field(default_factory=dict)
    cost_metrics: dict[str, Any] = field(default_factory=dict)
    system_metrics: dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and exports metrics for the research executor."""

    def __init__(self, config: OptimizationConfig):
        """Initialize metrics collector.

        Args:
            config: Optimization configuration
        """
        self.config = config
        self.snapshots: list[MetricsSnapshot] = []
        self.current_snapshot: MetricsSnapshot | None = None
        self.start_time: float | None = None
        self.logger = logfire

        # Initialize system metrics
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / (1024 * 1024)  # MB

    def start_collection(self) -> None:
        """Start a new metrics collection session."""
        if not self.config.enable_metrics_collection:
            return

        self.start_time = time.time()
        self.current_snapshot = MetricsSnapshot()
        self.current_snapshot.system_metrics["start_memory_mb"] = self.initial_memory
        self.logger.debug("Started metrics collection")

    def end_collection(self) -> None:
        """End the current metrics collection session."""
        if not self.config.enable_metrics_collection or not self.current_snapshot:
            return

        if self.start_time:
            total_time = time.time() - self.start_time
            self.current_snapshot.performance.total_execution_time = total_time

        # Collect final system metrics
        self._collect_system_metrics()

        # Save snapshot
        self.snapshots.append(self.current_snapshot)
        self.current_snapshot = None
        self.start_time = None
        self.logger.debug("Ended metrics collection")

    def record_timing(self, metric_name: str, duration: float) -> None:
        """Record a timing metric.

        Args:
            metric_name: Name of the timing metric
            duration: Duration in seconds
        """
        if not self.current_snapshot:
            return

        if hasattr(self.current_snapshot.performance, metric_name):
            setattr(self.current_snapshot.performance, metric_name, duration)
        self.logger.debug(f"Recorded timing {metric_name}: {duration:.3f}s")

    def record_count(self, metric_name: str, count: int) -> None:
        """Record a count metric.

        Args:
            metric_name: Name of the count metric
            count: Count value
        """
        if not self.current_snapshot:
            return

        if hasattr(self.current_snapshot.performance, metric_name):
            current = getattr(self.current_snapshot.performance, metric_name)
            setattr(self.current_snapshot.performance, metric_name, current + count)
        self.logger.debug(f"Recorded count {metric_name}: {count}")

    def record_quality_metric(self, name: str, value: Any) -> None:
        """Record a quality metric.

        Args:
            name: Metric name
            value: Metric value
        """
        if not self.current_snapshot:
            return

        self.current_snapshot.quality_metrics[name] = value
        self.logger.debug(f"Recorded quality metric {name}: {value}")

    def record_confidence_distribution(self, confidences: list[float]) -> None:
        """Record confidence score distribution.

        Args:
            confidences: List of confidence scores
        """
        if not self.current_snapshot or not confidences:
            return

        self.current_snapshot.quality_metrics["confidence_distribution"] = {
            "min": min(confidences),
            "max": max(confidences),
            "mean": sum(confidences) / len(confidences),
            "median": sorted(confidences)[len(confidences) // 2],
            "count": len(confidences),
        }

    def record_pattern_strength(
        self, patterns: Sequence[PatternAnalysis | Mapping[str, Any]]
    ) -> None:
        """Record pattern strength metrics.

        Args:
            patterns: Sequence of pattern payloads with confidence data
        """
        if not self.current_snapshot or not patterns:
            return

        normalized_patterns = _normalize_patterns(patterns)
        strengths = [pattern.get("confidence", 0.0) for pattern in normalized_patterns]
        self.current_snapshot.quality_metrics["pattern_strength"] = {
            "mean": sum(strengths) / len(strengths),
            "max": max(strengths),
            "count": len(normalized_patterns),
        }

    def record_synthesis_quality(self, synthesis_score: float, completeness: float) -> None:
        """Record synthesis quality metrics.

        Args:
            synthesis_score: Overall synthesis quality score
            completeness: Research completeness score
        """
        if not self.current_snapshot:
            return

        self.current_snapshot.quality_metrics["synthesis_quality"] = synthesis_score
        self.current_snapshot.quality_metrics["research_completeness"] = completeness

    def record_api_usage(self, api_calls: int, estimated_tokens: int) -> None:
        """Record API usage metrics.

        Args:
            api_calls: Number of API calls made
            estimated_tokens: Estimated tokens used
        """
        if not self.current_snapshot:
            return

        self.current_snapshot.performance.api_calls_made = api_calls
        self.current_snapshot.performance.estimated_tokens_used = estimated_tokens

        # Estimate costs (example rates, adjust as needed)
        token_cost_per_1k = 0.01  # Example rate
        estimated_cost = (estimated_tokens / 1000) * token_cost_per_1k

        self.current_snapshot.cost_metrics = {
            "api_calls": api_calls,
            "estimated_tokens": estimated_tokens,
            "estimated_cost_usd": estimated_cost,
        }

    async def record_synthesis_metrics(self, metrics: Mapping[str, Any]) -> None:
        """Asynchronously record synthesis metrics.

        Args:
            metrics: Mapping of synthesis metric name to value
        """
        if not self.config.enable_metrics_collection:
            return

        if self.current_snapshot is None:
            self.start_collection()

        if self.current_snapshot is None:  # start_collection may be disabled
            return

        self.current_snapshot.quality_metrics.update(dict(metrics))

    def _collect_system_metrics(self) -> None:
        """Collect system metrics."""
        if not self.current_snapshot:
            return

        # Memory metrics
        memory_info = self.process.memory_info()
        current_memory_mb = memory_info.rss / (1024 * 1024)
        peak_memory_mb = max(current_memory_mb, self.initial_memory)

        self.current_snapshot.performance.memory_usage_mb = peak_memory_mb
        self.current_snapshot.system_metrics["current_memory_mb"] = current_memory_mb
        self.current_snapshot.system_metrics["memory_increase_mb"] = (
            current_memory_mb - self.initial_memory
        )

        # CPU metrics
        try:
            cpu_percent = self.process.cpu_percent()
            self.current_snapshot.system_metrics["cpu_percent"] = cpu_percent
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            self.logger.warning(f"Could not get CPU percentage: {e}")
            self.current_snapshot.system_metrics["cpu_percent"] = 0.0

    def export_metrics(self, export_format: str | None = None) -> str:
        """Export collected metrics.

        Args:
            format: Export format (json or csv), defaults to config setting

        Returns:
            Exported metrics as string
        """
        export_format = export_format or self.config.metrics_export_format

        if export_format == "csv":
            return self._export_csv()
        return self._export_json()

    def _export_json(self) -> str:
        """Export metrics as JSON.

        Returns:
            JSON string
        """
        data = []
        for snapshot in self.snapshots:
            snapshot_dict = {
                "timestamp": snapshot.timestamp.isoformat(),
                "performance": snapshot.performance.model_dump(),
                "quality": snapshot.quality_metrics,
                "cost": snapshot.cost_metrics,
                "system": snapshot.system_metrics,
            }
            data.append(snapshot_dict)

        return json.dumps(data, indent=2)

    def _export_csv(self) -> str:
        """Export metrics as CSV.

        Returns:
            CSV string
        """
        if not self.snapshots:
            return ""

        output = StringIO()

        # Flatten metrics for CSV
        rows = []
        for snapshot in self.snapshots:
            row = {
                "timestamp": snapshot.timestamp.isoformat(),
                **{f"perf_{k}": v for k, v in snapshot.performance.model_dump().items()},
                **{
                    f"quality_{k}": v
                    for k, v in snapshot.quality_metrics.items()
                    if not isinstance(v, dict | list)
                },
                **{f"cost_{k}": v for k, v in snapshot.cost_metrics.items()},
                **{
                    f"system_{k}": v
                    for k, v in snapshot.system_metrics.items()
                    if not isinstance(v, dict | list)
                },
            }
            rows.append(row)

        if rows:
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        return output.getvalue()

    def save_metrics(self, filepath: Path) -> None:
        """Save metrics to a file.

        Args:
            filepath: Path to save metrics to
        """
        export_format = "csv" if filepath.suffix == ".csv" else "json"
        content = self.export_metrics(export_format)

        filepath.write_text(content)
        self.logger.info(f"Saved metrics to {filepath}")

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all collected metrics.

        Returns:
            Summary dictionary
        """
        if not self.snapshots:
            return {}

        total_time = sum(s.performance.total_execution_time for s in self.snapshots)
        total_findings = sum(s.performance.findings_processed for s in self.snapshots)
        total_patterns = sum(s.performance.patterns_detected for s in self.snapshots)
        total_contradictions = sum(s.performance.contradictions_found for s in self.snapshots)

        quality_scores = [
            s.quality_metrics.get("synthesis_quality", 0)
            for s in self.snapshots
            if "synthesis_quality" in s.quality_metrics
        ]

        return {
            "sessions": len(self.snapshots),
            "total_execution_time": total_time,
            "total_findings_processed": total_findings,
            "total_patterns_detected": total_patterns,
            "total_contradictions_found": total_contradictions,
            "avg_synthesis_quality": sum(quality_scores) / max(len(quality_scores), 1)
            if quality_scores
            else 0,
            "total_api_calls": sum(s.performance.api_calls_made for s in self.snapshots),
            "total_estimated_tokens": sum(
                s.performance.estimated_tokens_used for s in self.snapshots
            ),
        }
