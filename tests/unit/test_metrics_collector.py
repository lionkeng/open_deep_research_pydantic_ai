"""Unit tests for metrics collector enhancements."""

import pytest

from models.research_executor import OptimizationConfig, PatternAnalysis, PatternType
from services.metrics_collector import MetricsCollector


@pytest.mark.asyncio
async def test_record_synthesis_metrics_populates_snapshot():
    collector = MetricsCollector(OptimizationConfig())

    await collector.record_synthesis_metrics({"overall_quality": 0.8})

    assert collector.current_snapshot is not None
    assert collector.current_snapshot.quality_metrics["overall_quality"] == 0.8


def test_record_pattern_strength_accepts_models_and_mappings():
    collector = MetricsCollector(OptimizationConfig())
    collector.start_collection()

    model_pattern = PatternAnalysis(
        pattern_type=PatternType.CONVERGENCE,
        pattern_name="Consensus",
        description="Multiple sources align",
        strength=0.7,
        finding_ids=["1", "2"],
        confidence_factors={"confidence": 0.65},
    )

    dict_pattern = {
        "pattern_type": "divergence",
        "pattern_name": "Variance",
        "description": "Different directions",
        "confidence": 0.5,
    }

    collector.record_pattern_strength([model_pattern, dict_pattern])

    metrics = collector.current_snapshot.quality_metrics["pattern_strength"]
    assert metrics["count"] == 2
    assert metrics["max"] >= 0.5


@pytest.mark.asyncio
async def test_record_synthesis_metrics_noop_when_disabled():
    collector = MetricsCollector(OptimizationConfig(enable_metrics_collection=False))

    await collector.record_synthesis_metrics({"overall_quality": 0.9})

    assert collector.current_snapshot is None
    assert collector.snapshots == []
