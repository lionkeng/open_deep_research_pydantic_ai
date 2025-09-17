"""Unit tests for research executor tool helpers."""

from dataclasses import replace

import pytest
from unittest.mock import AsyncMock, MagicMock

from agents.research_executor_tools import (
    ResearchExecutorDependencies,
    analyze_patterns,
    assess_synthesis_quality,
    detect_contradictions,
    extract_hierarchical_findings,
    generate_executive_summary,
    identify_theme_clusters,
)
from models.research_executor import (
    ConfidenceLevel,
    Contradiction,
    ContradictionType,
    HierarchicalFinding,
    ImportanceLevel,
    PatternAnalysis,
    PatternType,
    ThemeCluster,
)


@pytest.fixture
def base_dependencies() -> ResearchExecutorDependencies:
    """Provide dependencies with mocks for core services."""

    synthesis_engine = MagicMock()
    synthesis_engine.extract_themes = AsyncMock(
        return_value=[{"text": "Synthetic finding", "confidence": 0.8, "importance": 0.7}]
    )
    synthesis_engine.cluster_findings = MagicMock(
        return_value=[
            {
                "name": "Cluster",
                "description": "Clustered findings",
                "findings": [],
                "coherence": 0.6,
                "importance": 0.5,
            }
        ]
    )

    contradiction_detector = MagicMock()
    contradiction_detector.detect_contradictions.return_value = [
        Contradiction(
            id="con-1",
            type=ContradictionType.SEMANTIC,
            evidence_indices=[0, 1],
            description="Conflict",
            confidence_score=0.75,
            resolution_suggestion="Investigate",
        )
    ]

    pattern_recognizer = MagicMock()
    pattern_recognizer.detect_patterns.return_value = [
        {
            "type": PatternType.CONVERGENCE,
            "name": "Pattern",
            "description": "Description",
            "strength": 0.6,
            "finding_indices": [0, 1],
            "implications": ["Implication"],
        }
    ]

    confidence_analyzer = MagicMock()

    metrics_collector = AsyncMock()

    return ResearchExecutorDependencies(
        synthesis_engine=synthesis_engine,
        contradiction_detector=contradiction_detector,
        pattern_recognizer=pattern_recognizer,
        confidence_analyzer=confidence_analyzer,
        metrics_collector=metrics_collector,
        original_query="test",
        search_results=[{"title": "Src", "content": "Content"}],
    )


@pytest.mark.asyncio
async def test_extract_hierarchical_findings_uses_synthesis_engine(base_dependencies):
    findings = await extract_hierarchical_findings(
        base_dependencies,
        "Example content",
        {"title": "Example", "url": "https://example.com", "type": "article"},
    )

    assert findings
    assert findings[0].finding.startswith("Synthetic finding")


@pytest.mark.asyncio
async def test_extract_hierarchical_findings_returns_cached(base_dependencies):
    cache_manager = MagicMock()
    cached_findings = [
        HierarchicalFinding(
            finding="Cached finding",
            supporting_evidence=[],
            confidence=ConfidenceLevel.MEDIUM,
            confidence_score=ConfidenceLevel.MEDIUM.to_score(),
            importance=ImportanceLevel.MEDIUM,
            importance_score=ImportanceLevel.MEDIUM.to_score(),
        )
    ]
    cache_manager.get.return_value = cached_findings

    deps = replace(base_dependencies, cache_manager=cache_manager)

    result = await extract_hierarchical_findings(
        deps,
        "Example content",
        {"title": "Example", "url": "https://example.com", "type": "article"},
    )

    assert result == cached_findings
    cache_manager.get.assert_called_once()
    assert deps.synthesis_engine.extract_themes.await_args_list == []
    cache_manager.set.assert_not_called()


@pytest.mark.asyncio
async def test_identify_theme_clusters_returns_converted_clusters(base_dependencies):
    results = await identify_theme_clusters(base_dependencies, [MagicMock(spec=HierarchicalFinding)])
    assert results
    assert isinstance(results[0], ThemeCluster)


@pytest.mark.asyncio
async def test_detect_contradictions_forwards_to_detector(base_dependencies):
    findings = [MagicMock(spec=HierarchicalFinding), MagicMock(spec=HierarchicalFinding)]
    contradictions = await detect_contradictions(base_dependencies, findings)
    assert contradictions
    base_dependencies.contradiction_detector.detect_contradictions.assert_called_once()


@pytest.mark.asyncio
async def test_analyze_patterns_combines_recognizer_results(base_dependencies):
    findings = [
        HierarchicalFinding(
            finding="A",
            supporting_evidence=[],
            confidence=ConfidenceLevel.HIGH,
            confidence_score=ConfidenceLevel.HIGH.to_score(),
            importance=ImportanceLevel.HIGH,
            importance_score=ImportanceLevel.HIGH.to_score(),
        ),
        HierarchicalFinding(
            finding="B",
            supporting_evidence=[],
            confidence=ConfidenceLevel.MEDIUM,
            confidence_score=ConfidenceLevel.MEDIUM.to_score(),
            importance=ImportanceLevel.MEDIUM,
            importance_score=ImportanceLevel.MEDIUM.to_score(),
        ),
    ]
    clusters = [
        ThemeCluster(
            theme_name="Cluster",
            description="",
            findings=findings,
            coherence_score=0.7,
            importance_score=0.6,
        )
    ]

    patterns = await analyze_patterns(base_dependencies, findings, clusters)
    assert patterns
    assert isinstance(patterns[0], PatternAnalysis)


@pytest.mark.asyncio
async def test_generate_executive_summary_handles_findings(base_dependencies):
    findings = [
        HierarchicalFinding(
            finding="Insight",
            supporting_evidence=["Evidence"],
            confidence=ConfidenceLevel.HIGH,
            confidence_score=ConfidenceLevel.HIGH.to_score(),
            importance=ImportanceLevel.CRITICAL,
            importance_score=ImportanceLevel.CRITICAL.to_score(),
        )
    ]
    contradictions = []
    patterns: list[PatternAnalysis] = []

    summary = await generate_executive_summary(base_dependencies, findings, contradictions, patterns)
    assert summary.key_findings
    assert "Overall confidence" in summary.confidence_assessment


@pytest.mark.asyncio
async def test_assess_synthesis_quality_returns_metrics(base_dependencies):
    findings = [
        HierarchicalFinding(
            finding="Insight",
            supporting_evidence=["Evidence"],
            confidence=ConfidenceLevel.MEDIUM,
            confidence_score=ConfidenceLevel.MEDIUM.to_score(),
            importance=ImportanceLevel.MEDIUM,
            importance_score=ImportanceLevel.MEDIUM.to_score(),
        )
    ]
    clusters = [
        ThemeCluster(
            theme_name="Cluster",
            description="",
            findings=findings,
            coherence_score=0.7,
            importance_score=0.6,
        )
    ]
    contradictions: list[Contradiction] = []

    metrics = await assess_synthesis_quality(base_dependencies, findings, clusters, contradictions)
    assert set(metrics.keys()) == {"completeness", "coherence", "average_confidence", "reliability", "overall_quality"}
    base_dependencies.metrics_collector.record_synthesis_metrics.assert_awaited()
