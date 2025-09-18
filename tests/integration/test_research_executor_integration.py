"""Integration tests for Research Executor components."""

from unittest.mock import MagicMock

import pytest

from agents.research_executor_tools import (
    ResearchExecutorDependencies,
    analyze_patterns,
)
from models.research_executor import (
    ConfidenceLevel,
    ImportanceLevel,
    ResearchSource,
    HierarchicalFinding,
    ThemeCluster,
    Contradiction,
    ExecutiveSummary,
    ResearchResults,
    OptimizationConfig,
    PatternAnalysis,
    PatternType,
)
from services.synthesis_engine import SynthesisEngine
from services.contradiction_detector import ContradictionDetector
from services.cache_manager import CacheManager
from services.metrics_collector import MetricsCollector


class TestResearchExecutorIntegration:
    """Test integration between Research Executor components."""

    @pytest.fixture
    def sample_findings(self) -> list[HierarchicalFinding]:
        """Create sample findings for testing."""
        source1 = ResearchSource(
            title="Research Paper 1",
            url="https://example.com/paper1",
            credibility_score=0.9,
            relevance_score=0.8,
        )

        source2 = ResearchSource(
            title="Industry Report",
            url="https://example.com/report",
            credibility_score=0.7,
            relevance_score=0.9,
        )

        findings = [
            HierarchicalFinding(
                finding="Machine learning models show 25% improvement in accuracy",
                supporting_evidence=["Benchmark results", "Cross-validation tests"],
                confidence=ConfidenceLevel.HIGH,
                importance=ImportanceLevel.CRITICAL,
                source=source1,
                category="performance",
            ),
            HierarchicalFinding(
                finding="Implementation costs increase by 40% with new approach",
                supporting_evidence=["Cost analysis", "Budget reports"],
                confidence=ConfidenceLevel.MEDIUM,
                importance=ImportanceLevel.HIGH,
                source=source2,
                category="economics",
            ),
            HierarchicalFinding(
                finding="Machine learning models show 15% improvement in speed",
                supporting_evidence=["Performance tests"],
                confidence=ConfidenceLevel.HIGH,
                importance=ImportanceLevel.HIGH,
                source=source1,
                category="performance",
            ),
            HierarchicalFinding(
                finding="Implementation costs decrease by 20% over time",
                supporting_evidence=["Long-term analysis"],
                confidence=ConfidenceLevel.LOW,
                importance=ImportanceLevel.MEDIUM,
                source=source2,
                category="economics",
            ),
        ]

        return findings

    def test_synthesis_engine_clustering(self, sample_findings):
        """Test SynthesisEngine clustering functionality."""
        engine = SynthesisEngine(min_cluster_size=2, max_clusters=3)

        # Cluster findings
        clusters = engine.cluster_findings(sample_findings)

        # Verify clustering results
        assert len(clusters) >= 1
        assert len(clusters) <= 3

        # Check cluster properties
        for cluster in clusters:
            assert isinstance(cluster, ThemeCluster)
            assert cluster.theme_name
            assert cluster.description
            assert len(cluster.findings) > 0
            assert 0.0 <= cluster.coherence_score <= 1.0
            assert 0.0 <= cluster.importance_score <= 1.0

        # Check that all findings are clustered
        clustered_findings = []
        for cluster in clusters:
            clustered_findings.extend(cluster.findings)
        assert len(clustered_findings) == len(sample_findings)

    def test_contradiction_detection(self, sample_findings):
        """Test ContradictionDetector functionality."""
        detector = ContradictionDetector()

        # Detect contradictions
        contradictions = detector.detect_contradictions(sample_findings)

        # Should find at least one contradiction (costs increase vs decrease)
        assert len(contradictions) >= 1

        # Check contradiction properties
        for cont in contradictions:
            assert isinstance(cont, Contradiction)
            assert cont.finding_1_id
            assert cont.finding_2_id
            assert cont.contradiction_type in ["direct", "partial"]
            assert cont.explanation
            assert cont.resolution_hint

    def test_synthesis_metrics_calculation(self, sample_findings):
        """Test synthesis metrics calculation."""
        engine = SynthesisEngine()

        # Cluster findings
        clusters = engine.cluster_findings(sample_findings)

        # Calculate metrics
        metrics = engine.calculate_synthesis_metrics(clusters)

        # Verify metrics
        assert "coverage" in metrics
        assert "coherence" in metrics
        assert "diversity" in metrics
        assert "confidence" in metrics

        # All metrics should be between 0 and 1
        for key, value in metrics.items():
            assert 0.0 <= value <= 1.0

    def test_research_results_integration(self, sample_findings):
        """Test complete ResearchResults integration."""
        # Create synthesis engine and detector
        engine = SynthesisEngine()
        detector = ContradictionDetector()

        # Process findings
        clusters = engine.cluster_findings(sample_findings)
        contradictions = detector.detect_contradictions(sample_findings)

        # Create executive summary
        summary = ExecutiveSummary(
            key_findings=[
                "ML models show significant accuracy improvements",
                "Cost implications are mixed and require further analysis",
            ],
            confidence_assessment="High confidence in performance gains, uncertainty in costs",
            critical_gaps=["Long-term cost projections need validation"],
            recommended_actions=["Conduct detailed cost-benefit analysis"],
        )

        # Create research results
        results = ResearchResults(
            query="ML implementation analysis",
            findings=sample_findings,
            theme_clusters=clusters,
            contradictions=contradictions,
            executive_summary=summary,
            sources=[f.source for f in sample_findings if f.source],
            overall_quality_score=0.75,
        )

        # Verify results
        assert results.query == "ML implementation analysis"
        assert len(results.findings) == 4
        assert len(results.theme_clusters) >= 1
        assert len(results.contradictions) >= 1
        assert results.executive_summary is not None
        assert results.overall_quality_score == 0.75

        # Test helper methods
        critical = results.get_critical_findings()
        assert len(critical) == 1
        assert critical[0].importance == ImportanceLevel.CRITICAL

        high_conf = results.get_high_confidence_findings()
        assert len(high_conf) == 2

        assert results.has_contradictions() is True
        # Note: needs_further_research returns True because there's a direct contradiction
        # even though quality > 0.7
        assert results.needs_further_research() is True

        # Test report generation
        report = results.to_report()
        assert "ML implementation analysis" in report
        assert "Executive Summary" in report or "Key Findings" in report

    def test_contradiction_pattern_analysis(self, sample_findings):
        """Test contradiction pattern analysis."""
        detector = ContradictionDetector()

        # Detect contradictions
        contradictions = detector.detect_contradictions(sample_findings)

        # Analyze patterns
        patterns = detector.analyze_contradiction_patterns(contradictions)

        # Verify pattern analysis
        assert "total_contradictions" in patterns
        assert "direct_contradictions" in patterns
        assert "partial_contradictions" in patterns
        assert "resolution_complexity" in patterns
        assert "requires_expert_review" in patterns

        assert patterns["total_contradictions"] >= 1
        assert patterns["resolution_complexity"] in ["low", "medium", "high"]

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        engine = SynthesisEngine()
        detector = ContradictionDetector()

        # Test with empty findings
        empty_clusters = engine.cluster_findings([])
        assert len(empty_clusters) == 1
        assert empty_clusters[0].theme_name == "General Findings"

        empty_contradictions = detector.detect_contradictions([])
        assert len(empty_contradictions) == 0

        # Test with single finding
        single_finding = HierarchicalFinding(
            finding="Single test finding",
            confidence=ConfidenceLevel.HIGH,
            importance=ImportanceLevel.MEDIUM,
        )

        single_cluster = engine.cluster_findings([single_finding])
        assert len(single_cluster) == 1
        assert len(single_cluster[0].findings) == 1

        single_contradiction = detector.detect_contradictions([single_finding])
        assert len(single_contradiction) == 0

    def test_confidence_importance_sync(self):
        """Test synchronization between confidence/importance levels and scores."""
        # Test with explicit scores - defaults to MEDIUM when only score provided
        finding = HierarchicalFinding(
            finding="Test finding",
            confidence_score=0.95,
            importance_score=0.95,
        )

        # When only score is provided, level defaults to MEDIUM
        assert finding.confidence == ConfidenceLevel.MEDIUM
        assert finding.importance == ImportanceLevel.MEDIUM
        assert finding.confidence_score == 0.95
        assert finding.importance_score == 0.95

        # Test with explicit levels
        finding2 = HierarchicalFinding(
            finding="Test finding 2",
            confidence=ConfidenceLevel.LOW,
            importance=ImportanceLevel.MEDIUM,
        )

        # When level is provided, score is calculated from level
        assert finding2.confidence_score == ConfidenceLevel.LOW.to_score()
        assert finding2.importance_score == ImportanceLevel.MEDIUM.to_score()

    @pytest.mark.asyncio
    async def test_pattern_analysis_cache_roundtrip(self, sample_findings):
        """Ensure cached pattern analyses deserialize without errors."""

        cache_manager = CacheManager(OptimizationConfig())
        metrics = MetricsCollector(OptimizationConfig())

        pattern = PatternAnalysis(
            pattern_type=PatternType.CONVERGENCE,
            pattern_name="Consensus",
            description="Multiple findings align",
            strength=0.7,
            finding_ids=["0", "1"],
            confidence_factors={"confidence": 0.65},
        )

        pattern_recognizer = MagicMock()
        pattern_recognizer.detect_patterns.return_value = [pattern]

        deps = ResearchExecutorDependencies(
            cache_manager=cache_manager,
            pattern_recognizer=pattern_recognizer,
            metrics_collector=metrics,
        )

        clusters = [
            ThemeCluster(
                theme_name="Cluster",
                description="",
                findings=sample_findings,
                coherence_score=0.7,
                importance_score=0.6,
            )
        ]

        first_run = await analyze_patterns(deps, sample_findings, clusters)
        second_run = await analyze_patterns(deps, sample_findings, clusters)

        assert first_run
        assert second_run
        assert isinstance(second_run[0], PatternAnalysis)
        assert pattern_recognizer.detect_patterns.call_count == 1


    def test_source_usage_recording(self):
        source = ResearchSource(title="Alpha", url="https://alpha.test", source_id="S1")
        finding = HierarchicalFinding(finding="Fact", source=source)
        results = ResearchResults(query="usage", findings=[finding], sources=[source])

        results.record_usage("S1", finding_id=finding.finding_id, cluster_id="cluster-1", contradiction_id="contradiction-1", pattern_id="pattern-1", report_section="final_report")
        usage = results.source_usage.get("S1")
        assert usage is not None
        assert usage.finding_ids == [finding.finding_id]
        assert usage.cluster_ids == ["cluster-1"]
        assert usage.contradiction_ids == ["contradiction-1"]
        assert usage.pattern_ids == ["pattern-1"]
        assert usage.report_sections == ["final_report"]
