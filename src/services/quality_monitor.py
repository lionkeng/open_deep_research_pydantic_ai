"""Quality Monitor Service for real-time synthesis quality assessment."""

import math
import statistics
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# Local model definitions until integrated with main models


class PatternType(str, Enum):
    """Types of patterns that can be detected."""

    TEMPORAL = "temporal"
    CAUSAL = "causal"
    CORRELATIVE = "correlative"
    COMPARATIVE = "comparative"


class ContradictionType(str, Enum):
    """Types of contradictions."""

    FACTUAL = "factual"
    TEMPORAL = "temporal"
    QUANTITATIVE = "quantitative"


class InformationHierarchy(str, Enum):
    """Information hierarchy levels."""

    PRIMARY = "primary"
    SUPPORTING = "supporting"
    CONTEXTUAL = "contextual"
    TANGENTIAL = "tangential"


class SynthesisResult(BaseModel):
    """Synthesis result model."""

    key_findings: list[str] = Field(default_factory=list)
    synthesis: str = Field(default="")
    confidence_score: float = Field(default=0.5)
    metadata: dict[str, Any] | None = Field(default=None)


class SearchResult(BaseModel):
    """Search result model."""

    query: str = Field(description="The original query")
    results: list[Any] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class QualityMetric(str, Enum):
    """Types of quality metrics tracked."""

    EXECUTION_RATE = "execution_rate"
    SOURCE_DIVERSITY = "source_diversity"
    PATTERN_ACCURACY = "pattern_accuracy"
    SYNTHESIS_COHERENCE = "synthesis_coherence"
    HIERARCHY_DISTRIBUTION = "hierarchy_distribution"
    CONTRADICTION_RATE = "contradiction_rate"
    CONFIDENCE_CALIBRATION = "confidence_calibration"


class AlertSeverity(str, Enum):
    """Severity levels for quality alerts."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class QualityAlert:
    """Alert for quality issues."""

    metric: QualityMetric
    severity: AlertSeverity
    message: str
    value: float
    threshold: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSnapshot:
    """Snapshot of a single metric at a point in time."""

    metric: QualityMetric
    value: float
    timestamp: datetime
    details: dict[str, Any] = field(default_factory=dict)


class QualityThresholds(BaseModel):
    """Configurable thresholds for quality metrics."""

    min_execution_rate: float = Field(default=0.95, description="Minimum execution rate")
    min_source_diversity: float = Field(
        default=0.3, description="Minimum source diversity (entropy)"
    )
    min_pattern_accuracy: float = Field(default=0.7, description="Minimum pattern accuracy")
    min_synthesis_coherence: float = Field(default=0.6, description="Minimum synthesis coherence")
    max_contradiction_rate: float = Field(default=0.1, description="Maximum contradiction rate")
    min_confidence_calibration: float = Field(
        default=0.8, description="Minimum confidence calibration"
    )
    ideal_hierarchy_distribution: dict[str, float] = Field(
        default={
            "primary": 0.2,
            "supporting": 0.3,
            "contextual": 0.3,
            "tangential": 0.2,
        },
        description="Ideal distribution of information hierarchy",
    )


class QualityReport(BaseModel):
    """Comprehensive quality assessment report."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metrics: dict[QualityMetric, float] = Field(default_factory=dict)
    alerts: list[QualityAlert] = Field(default_factory=list)
    overall_score: float = Field(default=0.0, description="Overall quality score (0-1)")
    recommendations: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class QualityMonitor:
    """
    Real-time synthesis quality assessment and monitoring.

    Tracks 7 key quality metrics, generates alerts, and maintains
    historical metrics for trend analysis.
    """

    def __init__(
        self,
        thresholds: QualityThresholds | None = None,
        history_size: int = 100,
        alert_cooldown_seconds: int = 300,
    ):
        """
        Initialize the QualityMonitor.

        Args:
            thresholds: Quality thresholds for alert generation
            history_size: Number of historical snapshots to maintain
            alert_cooldown_seconds: Cooldown period between similar alerts
        """
        self.thresholds = thresholds or QualityThresholds()
        self.history_size = history_size
        self.alert_cooldown_seconds = alert_cooldown_seconds

        # Historical metrics tracking
        self._metric_history: dict[QualityMetric, deque] = {
            metric: deque(maxlen=history_size) for metric in QualityMetric
        }

        # Alert tracking
        self._active_alerts: list[QualityAlert] = []
        self._alert_history: deque = deque(maxlen=history_size * 2)
        self._last_alert_time: dict[tuple[QualityMetric, AlertSeverity], datetime] = {}

        # Current state
        self._current_metrics: dict[QualityMetric, float] = {}
        self._synthesis_count = 0

    def calculate_shannon_entropy(self, sources: list[str]) -> float:
        """
        Calculate Shannon entropy for source diversity.

        Higher entropy indicates more diverse sources.
        """
        if not sources:
            return 0.0

        # Count occurrences of each source
        source_counts = Counter(sources)
        total = len(sources)

        # Calculate entropy
        entropy = 0.0
        for count in source_counts.values():
            if count > 0:
                probability = count / total
                entropy -= probability * math.log2(probability)

        # Normalize by maximum possible entropy
        max_entropy = math.log2(len(source_counts)) if len(source_counts) > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        return normalized_entropy

    def assess_execution_rate(
        self, executed_queries: int, total_queries: int
    ) -> tuple[float, dict[str, Any]]:
        """Assess the execution rate metric."""
        if total_queries == 0:
            return 1.0, {"executed": 0, "total": 0}

        rate = executed_queries / total_queries
        details = {
            "executed": executed_queries,
            "total": total_queries,
            "failed": total_queries - executed_queries,
        }

        return rate, details

    def assess_source_diversity(
        self, search_results: list[SearchResult]
    ) -> tuple[float, dict[str, Any]]:
        """Assess source diversity using Shannon entropy."""
        all_sources = []
        for result in search_results:
            if hasattr(result, "results"):
                for item in result.results:
                    if hasattr(item, "source"):
                        all_sources.append(item.source)

        entropy = self.calculate_shannon_entropy(all_sources)
        unique_sources = len(set(all_sources))

        details = {
            "entropy": entropy,
            "unique_sources": unique_sources,
            "total_references": len(all_sources),
            "source_distribution": dict(Counter(all_sources)),
        }

        return entropy, details

    def assess_pattern_accuracy(
        self, synthesis: SynthesisResult, patterns_found: list[PatternType]
    ) -> tuple[float, dict[str, Any]]:
        """Assess the accuracy of identified patterns."""
        if not patterns_found:
            return 1.0, {"patterns": [], "confidence": 1.0}

        # Calculate average confidence for patterns
        pattern_confidence = []
        pattern_distribution = Counter(patterns_found)

        # Simulate confidence scores (would come from pattern detection)
        for pattern in patterns_found:
            if pattern == PatternType.TEMPORAL:
                confidence = 0.85
            elif pattern == PatternType.CAUSAL:
                confidence = 0.75
            elif pattern == PatternType.CORRELATIVE:
                confidence = 0.70
            else:
                confidence = 0.65
            pattern_confidence.append(confidence)

        avg_confidence = statistics.mean(pattern_confidence) if pattern_confidence else 0

        details = {
            "patterns": list(pattern_distribution.keys()),
            "pattern_counts": dict(pattern_distribution),
            "average_confidence": avg_confidence,
            "total_patterns": len(patterns_found),
        }

        return avg_confidence, details

    def assess_synthesis_coherence(
        self, synthesis: SynthesisResult
    ) -> tuple[float, dict[str, Any]]:
        """Assess the coherence of the synthesis."""
        coherence_score = 1.0
        issues = []

        # Check for key indicators of coherence
        if synthesis.key_findings:
            # Check if findings are well-structured
            avg_finding_length = statistics.mean(
                len(finding.split()) for finding in synthesis.key_findings
            )
            if avg_finding_length < 5:
                coherence_score -= 0.2
                issues.append("Findings too brief")
            elif avg_finding_length > 100:
                coherence_score -= 0.1
                issues.append("Findings too verbose")
        else:
            coherence_score -= 0.3
            issues.append("No key findings")

        # Check synthesis text
        if synthesis.synthesis:
            word_count = len(synthesis.synthesis.split())
            if word_count < 50:
                coherence_score -= 0.2
                issues.append("Synthesis too brief")

            # Check for structure (paragraphs)
            paragraphs = synthesis.synthesis.split("\n\n")
            if len(paragraphs) < 2:
                coherence_score -= 0.1
                issues.append("Lack of structure")
        else:
            coherence_score -= 0.3
            issues.append("No synthesis text")

        # Check confidence score validity
        if not (0 <= synthesis.confidence_score <= 1):
            coherence_score -= 0.2
            issues.append("Invalid confidence score")

        coherence_score = max(0, coherence_score)

        details = {
            "coherence_score": coherence_score,
            "issues": issues,
            "has_findings": bool(synthesis.key_findings),
            "has_synthesis": bool(synthesis.synthesis),
            "confidence": synthesis.confidence_score,
        }

        return coherence_score, details

    def assess_hierarchy_distribution(
        self, hierarchy_items: list[InformationHierarchy]
    ) -> tuple[float, dict[str, Any]]:
        """Assess the distribution of information hierarchy."""
        if not hierarchy_items:
            return 0.0, {"distribution": {}, "score": 0.0}

        # Count distribution
        distribution = Counter(item.value for item in hierarchy_items)
        total = sum(distribution.values())

        # Calculate distribution percentages
        actual_distribution = {level: count / total for level, count in distribution.items()}

        # Compare with ideal distribution
        ideal = self.thresholds.ideal_hierarchy_distribution
        divergence = 0.0

        for level in ["primary", "supporting", "contextual", "tangential"]:
            actual = actual_distribution.get(level, 0)
            expected = ideal.get(level, 0.25)
            divergence += abs(actual - expected)

        # Convert divergence to score (0-1, where 1 is perfect match)
        score = max(0, 1 - (divergence / 2))  # Max divergence is 2

        details = {
            "actual_distribution": actual_distribution,
            "ideal_distribution": ideal,
            "divergence": divergence,
            "total_items": total,
        }

        return score, details

    def assess_contradiction_rate(
        self, contradictions: list[ContradictionType]
    ) -> tuple[float, dict[str, Any]]:
        """Assess the rate and types of contradictions."""
        if not contradictions:
            return 0.0, {"count": 0, "types": {}}

        contradiction_types = Counter(contradictions)
        total = len(contradictions)

        # Weight different types of contradictions
        weights = {
            ContradictionType.FACTUAL: 1.0,
            ContradictionType.TEMPORAL: 0.8,
            ContradictionType.QUANTITATIVE: 0.9,
        }

        weighted_count = sum(
            count * weights.get(ctype, 1.0) for ctype, count in contradiction_types.items()
        )

        # Normalize (assuming max acceptable is 10 contradictions)
        rate = min(1.0, weighted_count / 10)

        details = {
            "count": total,
            "types": dict(contradiction_types),
            "weighted_count": weighted_count,
        }

        return rate, details

    def assess_confidence_calibration(
        self, predicted_confidence: float, actual_accuracy: float
    ) -> tuple[float, dict[str, Any]]:
        """Assess how well confidence scores align with actual accuracy."""
        # Calculate calibration error
        calibration_error = abs(predicted_confidence - actual_accuracy)

        # Convert to score (0-1, where 1 is perfect calibration)
        calibration_score = 1 - calibration_error

        details = {
            "predicted_confidence": predicted_confidence,
            "actual_accuracy": actual_accuracy,
            "calibration_error": calibration_error,
        }

        return calibration_score, details

    def _generate_alert(
        self, metric: QualityMetric, value: float, threshold: float, message: str
    ) -> QualityAlert | None:
        """Generate an alert if not in cooldown period."""
        # Determine severity
        deviation = abs(value - threshold) / threshold if threshold != 0 else 1.0

        if deviation < 0.1:
            severity = AlertSeverity.INFO
        elif deviation < 0.25:
            severity = AlertSeverity.WARNING
        elif deviation < 0.5:
            severity = AlertSeverity.ERROR
        else:
            severity = AlertSeverity.CRITICAL

        # Check cooldown
        alert_key = (metric, severity)
        now = datetime.now(UTC)

        if alert_key in self._last_alert_time:
            time_since_last = (now - self._last_alert_time[alert_key]).total_seconds()
            if time_since_last < self.alert_cooldown_seconds:
                return None

        # Create alert
        alert = QualityAlert(
            metric=metric,
            severity=severity,
            message=message,
            value=value,
            threshold=threshold,
            timestamp=now,
            context={"deviation": deviation},
        )

        self._last_alert_time[alert_key] = now
        self._active_alerts.append(alert)
        self._alert_history.append(alert)

        return alert

    def assess_synthesis_quality(
        self,
        synthesis: SynthesisResult,
        search_results: list[SearchResult],
        execution_stats: dict[str, Any],
    ) -> QualityReport:
        """
        Perform comprehensive quality assessment of a synthesis.

        Args:
            synthesis: The synthesis result to assess
            search_results: Search results used in synthesis
            execution_stats: Execution statistics

        Returns:
            Comprehensive quality report
        """
        self._synthesis_count += 1

        metrics = {}
        alerts = []
        details = {}

        # 1. Execution Rate
        executed = execution_stats.get("executed_queries", 0)
        total = execution_stats.get("total_queries", 1)
        exec_rate, exec_details = self.assess_execution_rate(executed, total)
        metrics[QualityMetric.EXECUTION_RATE] = exec_rate
        details["execution"] = exec_details

        if exec_rate < self.thresholds.min_execution_rate:
            alert = self._generate_alert(
                QualityMetric.EXECUTION_RATE,
                exec_rate,
                self.thresholds.min_execution_rate,
                f"Execution rate {exec_rate:.2%} below threshold",
            )
            if alert:
                alerts.append(alert)

        # 2. Source Diversity
        diversity, diversity_details = self.assess_source_diversity(search_results)
        metrics[QualityMetric.SOURCE_DIVERSITY] = diversity
        details["diversity"] = diversity_details

        if diversity < self.thresholds.min_source_diversity:
            alert = self._generate_alert(
                QualityMetric.SOURCE_DIVERSITY,
                diversity,
                self.thresholds.min_source_diversity,
                f"Source diversity {diversity:.2f} below threshold",
            )
            if alert:
                alerts.append(alert)

        # 3. Pattern Accuracy
        patterns = synthesis.metadata.get("patterns", []) if synthesis.metadata else []
        pattern_acc, pattern_details = self.assess_pattern_accuracy(synthesis, patterns)
        metrics[QualityMetric.PATTERN_ACCURACY] = pattern_acc
        details["patterns"] = pattern_details

        # 4. Synthesis Coherence
        coherence, coherence_details = self.assess_synthesis_coherence(synthesis)
        metrics[QualityMetric.SYNTHESIS_COHERENCE] = coherence
        details["coherence"] = coherence_details

        if coherence < self.thresholds.min_synthesis_coherence:
            alert = self._generate_alert(
                QualityMetric.SYNTHESIS_COHERENCE,
                coherence,
                self.thresholds.min_synthesis_coherence,
                f"Synthesis coherence {coherence:.2f} below threshold",
            )
            if alert:
                alerts.append(alert)

        # 5. Hierarchy Distribution
        hierarchy_items = synthesis.metadata.get("hierarchy", []) if synthesis.metadata else []
        hierarchy_score, hierarchy_details = self.assess_hierarchy_distribution(hierarchy_items)
        metrics[QualityMetric.HIERARCHY_DISTRIBUTION] = hierarchy_score
        details["hierarchy"] = hierarchy_details

        # 6. Contradiction Rate
        contradictions = synthesis.metadata.get("contradictions", []) if synthesis.metadata else []
        contradiction_rate, contradiction_details = self.assess_contradiction_rate(contradictions)
        metrics[QualityMetric.CONTRADICTION_RATE] = contradiction_rate
        details["contradictions"] = contradiction_details

        if contradiction_rate > self.thresholds.max_contradiction_rate:
            alert = self._generate_alert(
                QualityMetric.CONTRADICTION_RATE,
                contradiction_rate,
                self.thresholds.max_contradiction_rate,
                f"Contradiction rate {contradiction_rate:.2f} above threshold",
            )
            if alert:
                alerts.append(alert)

        # 7. Confidence Calibration
        actual_accuracy = statistics.mean([exec_rate, 1 - contradiction_rate, coherence])
        calibration, calibration_details = self.assess_confidence_calibration(
            synthesis.confidence_score, actual_accuracy
        )
        metrics[QualityMetric.CONFIDENCE_CALIBRATION] = calibration
        details["calibration"] = calibration_details

        # Calculate overall score
        weights = {
            QualityMetric.EXECUTION_RATE: 0.2,
            QualityMetric.SOURCE_DIVERSITY: 0.15,
            QualityMetric.PATTERN_ACCURACY: 0.1,
            QualityMetric.SYNTHESIS_COHERENCE: 0.2,
            QualityMetric.HIERARCHY_DISTRIBUTION: 0.1,
            QualityMetric.CONTRADICTION_RATE: 0.15,
            QualityMetric.CONFIDENCE_CALIBRATION: 0.1,
        }

        overall_score = sum(metrics[metric] * weight for metric, weight in weights.items())

        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, details)

        # Update history
        for metric, value in metrics.items():
            snapshot = MetricSnapshot(
                metric=metric,
                value=value,
                timestamp=datetime.now(UTC),
                details=details.get(metric.value, {}),
            )
            self._metric_history[metric].append(snapshot)

        self._current_metrics = metrics

        return QualityReport(
            metrics=metrics,
            alerts=alerts,
            overall_score=overall_score,
            recommendations=recommendations,
            details=details,
        )

    def _generate_recommendations(
        self, metrics: dict[QualityMetric, float], details: dict[str, Any]
    ) -> list[str]:
        """Generate actionable recommendations based on metrics."""
        recommendations = []

        if metrics[QualityMetric.EXECUTION_RATE] < self.thresholds.min_execution_rate:
            recommendations.append(
                "Improve query execution: Check network connectivity, "
                "increase retry attempts, or optimize query processing"
            )

        if metrics[QualityMetric.SOURCE_DIVERSITY] < self.thresholds.min_source_diversity:
            recommendations.append(
                "Increase source diversity: Expand search queries, "
                "use multiple search engines, or broaden search parameters"
            )

        if metrics[QualityMetric.SYNTHESIS_COHERENCE] < self.thresholds.min_synthesis_coherence:
            issues = details.get("coherence", {}).get("issues", [])
            if "No key findings" in issues:
                recommendations.append("Extract clear key findings from search results")
            if "Synthesis too brief" in issues:
                recommendations.append(
                    "Provide more comprehensive synthesis with detailed analysis"
                )

        if metrics[QualityMetric.CONTRADICTION_RATE] > self.thresholds.max_contradiction_rate:
            recommendations.append(
                "Resolve contradictions: Cross-reference sources, "
                "verify facts, and clarify temporal contexts"
            )

        if (
            metrics[QualityMetric.CONFIDENCE_CALIBRATION]
            < self.thresholds.min_confidence_calibration
        ):
            recommendations.append(
                "Improve confidence calibration: "
                "Adjust confidence scoring based on actual accuracy metrics"
            )

        return recommendations

    def get_metric_history(
        self, metric: QualityMetric, limit: int | None = None
    ) -> list[MetricSnapshot]:
        """Get historical snapshots for a specific metric."""
        history = list(self._metric_history[metric])
        if limit:
            return history[-limit:]
        return history

    def get_metric_trend(self, metric: QualityMetric, window_size: int = 10) -> dict[str, float]:
        """Calculate trend statistics for a metric."""
        history = list(self._metric_history[metric])

        if len(history) < 2:
            return {
                "current": history[-1].value if history else 0,
                "mean": history[-1].value if history else 0,
                "std": 0,
                "trend": 0,
            }

        recent = history[-window_size:] if len(history) >= window_size else history
        values = [s.value for s in recent]

        # Calculate trend (positive = improving, negative = declining)
        if len(values) >= 2:
            trend = (values[-1] - values[0]) / len(values)
        else:
            trend = 0

        return {
            "current": values[-1],
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "trend": trend,
            "min": min(values),
            "max": max(values),
        }

    def get_active_alerts(self, severity: AlertSeverity | None = None) -> list[QualityAlert]:
        """Get currently active alerts, optionally filtered by severity."""
        if severity:
            return [a for a in self._active_alerts if a.severity == severity]
        return self._active_alerts.copy()

    def clear_alerts(self) -> None:
        """Clear all active alerts."""
        self._active_alerts.clear()

    def get_summary_stats(self) -> dict[str, Any]:
        """Get summary statistics across all metrics."""
        return {
            "synthesis_count": self._synthesis_count,
            "current_metrics": self._current_metrics,
            "active_alerts": len(self._active_alerts),
            "critical_alerts": sum(
                1 for a in self._active_alerts if a.severity == AlertSeverity.CRITICAL
            ),
            "metric_trends": {metric: self.get_metric_trend(metric) for metric in QualityMetric},
        }
