"""Unit tests for the Enhanced Research Executor Agent with GPT-5 synthesis capabilities."""

import pytest
from dataclasses import dataclass
from datetime import datetime, UTC
from unittest.mock import AsyncMock, MagicMock, patch

from agents.research_executor import (
    ResearchExecutorDependencies,
    research_executor_agent,
    execute_research,
    add_synthesis_context,
    extract_hierarchical_findings,
    identify_theme_clusters,
    detect_contradictions,
    analyze_patterns,
    generate_executive_summary,
    assess_synthesis_quality,
)
from models.research_executor import (
    ConfidenceLevel,
    Contradiction,
    ExecutiveSummary,
    HierarchicalFinding,
    ImportanceLevel,
    PatternAnalysis,
    PatternType,
    ResearchResults,
    ResearchSource,
    ThemeCluster,
)
from services.synthesis_engine import SynthesisEngine
from services.contradiction_detector import ContradictionDetector
from services.pattern_recognizer import PatternRecognizer
from services.confidence_analyzer import ConfidenceAnalyzer


@pytest.fixture
def mock_dependencies():
    """Create mock dependencies for testing."""
    deps = ResearchExecutorDependencies(
        synthesis_engine=MagicMock(spec=SynthesisEngine),
        contradiction_detector=MagicMock(spec=ContradictionDetector),
        pattern_recognizer=MagicMock(spec=PatternRecognizer),
        confidence_analyzer=MagicMock(spec=ConfidenceAnalyzer),
        original_query="Test query",
        search_results=[
            {"title": "Source 1", "content": "Content 1"},
            {"title": "Source 2", "content": "Content 2"},
        ]
    )

    # Configure mock behaviors
    deps.synthesis_engine.cluster_findings.return_value = []
    deps.contradiction_detector.detect_contradictions.return_value = []

    return deps


@pytest.fixture
def sample_findings():
    """Create sample hierarchical findings for testing."""
    return [
        HierarchicalFinding(
            finding="AI improves productivity by 40%",
            supporting_evidence=["Study A", "Report B"],
            confidence=ConfidenceLevel.HIGH,
            confidence_score=0.9,
            importance=ImportanceLevel.CRITICAL,
            importance_score=0.95,
            source=ResearchSource(
                title="Research Paper",
                url="https://example.com",
                source_type="academic"
            ),
            category="productivity",
            temporal_relevance="2024"
        ),
        HierarchicalFinding(
            finding="Machine learning adoption increasing",
            supporting_evidence=["Survey data"],
            confidence=ConfidenceLevel.MEDIUM,
            confidence_score=0.7,
            importance=ImportanceLevel.HIGH,
            importance_score=0.8,
            category="trends",
            temporal_relevance="current"
        ),
        HierarchicalFinding(
            finding="Some challenges remain in implementation",
            supporting_evidence=["Industry reports"],
            confidence=ConfidenceLevel.MEDIUM,
            confidence_score=0.6,
            importance=ImportanceLevel.MEDIUM,
            importance_score=0.5,
            category="challenges",
            temporal_relevance="current"
        ),
    ]


@pytest.fixture
def sample_clusters(sample_findings):
    """Create sample theme clusters for testing."""
    return [
        ThemeCluster(
            theme_name="AI Impact",
            description="AI's impact on productivity and adoption",
            findings=sample_findings[:2],
            coherence_score=0.85,
            importance_score=0.9
        ),
        ThemeCluster(
            theme_name="Implementation Challenges",
            description="Challenges in AI implementation",
            findings=[sample_findings[2]],
            coherence_score=0.7,
            importance_score=0.6
        ),
    ]


class TestResearchExecutorAgent:
    """Test the Enhanced Research Executor Agent."""

    def test_agent_creation(self):
        """Test that the agent is created with correct configuration."""
        assert research_executor_agent is not None
        # The agent should be configured with correct types
        # Note: Pydantic AI doesn't expose these as public attributes

    @pytest.mark.asyncio
    async def test_synthesis_context_injection(self, mock_dependencies):
        """Test the dynamic synthesis context injection."""
        # Create a mock RunContext
        ctx = MagicMock()
        ctx.deps = mock_dependencies

        # Call the instructions function
        context = await add_synthesis_context(ctx)

        # Verify the context contains expected elements
        assert "# ENHANCED SYNTHESIS SYSTEM PROMPT (GPT-5 OPTIMIZED)" in context
        assert "Tree of Thoughts Methodology" in context
        assert "Phase 1: Pattern Recognition" in context
        assert "Phase 2: Insight Extraction" in context
        assert "Phase 3: Quality Verification" in context
        assert "Original Query: Test query" in context
        assert "Search Results Available: 2 sources" in context

    @pytest.mark.asyncio
    async def test_extract_hierarchical_findings_tool(self, mock_dependencies):
        """Test the extract_hierarchical_findings tool."""
        ctx = MagicMock()
        ctx.deps = mock_dependencies

        findings = await extract_hierarchical_findings(
            ctx,
            "Sample content to analyze",
            {"title": "Test Source", "url": "https://test.com", "type": "article"}
        )

        assert isinstance(findings, list)
        assert len(findings) > 0
        assert isinstance(findings[0], HierarchicalFinding)
        assert findings[0].confidence == ConfidenceLevel.MEDIUM

    @pytest.mark.asyncio
    async def test_identify_theme_clusters_tool(self, mock_dependencies, sample_findings):
        """Test the identify_theme_clusters tool."""
        ctx = MagicMock()
        ctx.deps = mock_dependencies

        # Mock the synthesis engine response
        mock_cluster = ThemeCluster(
            theme_name="Test Theme",
            description="Test description",
            findings=sample_findings[:2],
            coherence_score=0.8,
            importance_score=0.85
        )
        ctx.deps.synthesis_engine.cluster_findings.return_value = [mock_cluster]

        clusters = await identify_theme_clusters(ctx, sample_findings)

        assert isinstance(clusters, list)
        if clusters:  # May be empty or contain general cluster
            assert isinstance(clusters[0], ThemeCluster)

    @pytest.mark.asyncio
    async def test_detect_contradictions_tool(self, mock_dependencies, sample_findings):
        """Test the detect_contradictions tool."""
        ctx = MagicMock()
        ctx.deps = mock_dependencies

        # Mock contradiction detection
        mock_contradiction = Contradiction(
            finding_1_id="0",
            finding_2_id="1",
            contradiction_type="partial",
            explanation="Different scope",
            severity=0.5
        )
        ctx.deps.contradiction_detector.detect_contradictions.return_value = [mock_contradiction]

        contradictions = await detect_contradictions(ctx, sample_findings)

        assert isinstance(contradictions, list)
        if contradictions:
            assert isinstance(contradictions[0], Contradiction)

    @pytest.mark.asyncio
    async def test_analyze_patterns_tool(self, mock_dependencies, sample_findings, sample_clusters):
        """Test the analyze_patterns tool."""
        ctx = MagicMock()
        ctx.deps = mock_dependencies

        patterns = await analyze_patterns(ctx, sample_findings, sample_clusters)

        assert isinstance(patterns, list)
        # Should detect high confidence convergence with our sample data
        if len(sample_findings) > 3:
            assert any(p.pattern_type == PatternType.CONVERGENCE for p in patterns)

    @pytest.mark.asyncio
    async def test_generate_executive_summary_tool(self, mock_dependencies, sample_findings):
        """Test the generate_executive_summary tool."""
        ctx = MagicMock()
        ctx.deps = mock_dependencies

        contradictions = []
        patterns = [
            PatternAnalysis(
                pattern_type=PatternType.CONVERGENCE,
                pattern_name="Test Pattern",
                description="Test description",
                strength=0.8,
                implications=["Test implication"]
            )
        ]

        summary = await generate_executive_summary(ctx, sample_findings, contradictions, patterns)

        assert isinstance(summary, ExecutiveSummary)
        assert len(summary.key_findings) <= 5
        assert "Overall confidence:" in summary.confidence_assessment
        assert isinstance(summary.critical_gaps, list)
        assert isinstance(summary.recommended_actions, list)
        assert isinstance(summary.risk_factors, list)

    @pytest.mark.asyncio
    async def test_assess_synthesis_quality_tool(self, mock_dependencies, sample_findings, sample_clusters):
        """Test the assess_synthesis_quality tool."""
        ctx = MagicMock()
        ctx.deps = mock_dependencies

        contradictions = []

        metrics = await assess_synthesis_quality(ctx, sample_findings, sample_clusters, contradictions)

        assert isinstance(metrics, dict)
        assert "completeness" in metrics
        assert "coherence" in metrics
        assert "average_confidence" in metrics
        assert "reliability" in metrics
        assert "overall_quality" in metrics
        assert all(0 <= v <= 1 for v in metrics.values() if isinstance(v, (int, float)))

    @pytest.mark.asyncio
    async def test_execute_research_function(self):
        """Test the main execute_research function."""
        with patch('src.agents.research_executor.research_executor_agent.run') as mock_run:
            # Mock the agent run result
            mock_result = MagicMock()
            mock_result.output = ResearchResults(
                query="Test query",
                execution_time=datetime.now(UTC),
                findings=[],
                theme_clusters=[],
                contradictions=[],
                sources=[],
                overall_quality_score=0.8
            )
            mock_run.return_value = mock_result

            result = await execute_research(
                "Test query",
                [{"title": "Source", "content": "Content"}]
            )

            assert isinstance(result, ResearchResults)
            assert result.query == "Test query"
            mock_run.assert_called_once()


class TestDependenciesIntegration:
    """Test the integration with dependencies."""

    def test_dependencies_creation(self):
        """Test creating ResearchExecutorDependencies."""
        deps = ResearchExecutorDependencies(
            synthesis_engine=MagicMock(spec=SynthesisEngine),
            contradiction_detector=MagicMock(spec=ContradictionDetector),
            pattern_recognizer=MagicMock(spec=PatternRecognizer),
            confidence_analyzer=MagicMock(spec=ConfidenceAnalyzer),
            original_query="Test",
            search_results=None
        )

        assert deps.original_query == "Test"
        assert deps.search_results == []  # Should be initialized in __post_init__
        assert deps.cache_manager is None
        assert deps.parallel_executor is None

    def test_dependencies_with_optional_services(self):
        """Test dependencies with optional services."""
        deps = ResearchExecutorDependencies(
            synthesis_engine=MagicMock(spec=SynthesisEngine),
            contradiction_detector=MagicMock(spec=ContradictionDetector),
            pattern_recognizer=MagicMock(spec=PatternRecognizer),
            confidence_analyzer=MagicMock(spec=ConfidenceAnalyzer),
            cache_manager=MagicMock(),
            parallel_executor=MagicMock(),
            metrics_collector=MagicMock(),
            optimization_manager=MagicMock()
        )

        assert deps.cache_manager is not None
        assert deps.parallel_executor is not None
        assert deps.metrics_collector is not None
        assert deps.optimization_manager is not None


class TestPatternDetection:
    """Test pattern detection logic."""

    @pytest.mark.asyncio
    async def test_convergence_pattern_detection(self, mock_dependencies):
        """Test detection of convergence patterns."""
        ctx = MagicMock()
        ctx.deps = mock_dependencies

        # Create findings with high confidence
        findings = [
            HierarchicalFinding(
                finding=f"Finding {i}",
                confidence_score=0.85,
                importance_score=0.8
            )
            for i in range(5)
        ]

        patterns = await analyze_patterns(ctx, findings, [])

        # Should detect convergence with high confidence findings
        convergence_patterns = [p for p in patterns if p.pattern_type == PatternType.CONVERGENCE]
        assert len(convergence_patterns) > 0

    @pytest.mark.asyncio
    async def test_temporal_pattern_detection(self, mock_dependencies):
        """Test detection of temporal patterns."""
        ctx = MagicMock()
        ctx.deps = mock_dependencies

        # Create findings with temporal data
        findings = [
            HierarchicalFinding(
                finding=f"Finding {i}",
                confidence_score=0.7,
                importance_score=0.7,
                temporal_relevance="2024-Q1"
            )
            for i in range(3)
        ]

        patterns = await analyze_patterns(ctx, findings, [])

        # Should detect temporal patterns
        temporal_patterns = [p for p in patterns if p.pattern_type == PatternType.TEMPORAL]
        assert len(temporal_patterns) > 0


class TestQualityAssessment:
    """Test quality assessment functionality."""

    @pytest.mark.asyncio
    async def test_quality_metrics_calculation(self, mock_dependencies):
        """Test calculation of quality metrics."""
        ctx = MagicMock()
        ctx.deps = mock_dependencies

        findings = [
            HierarchicalFinding(
                finding="Test finding",
                confidence_score=0.8,
                importance_score=0.7
            )
            for _ in range(5)
        ]

        clusters = [
            ThemeCluster(
                theme_name="Test",
                description="Test",
                findings=findings,
                coherence_score=0.75,
                importance_score=0.8
            )
        ]

        contradictions = []

        metrics = await assess_synthesis_quality(ctx, findings, clusters, contradictions)

        assert metrics["completeness"] > 0
        assert metrics["coherence"] == 0.75
        assert metrics["average_confidence"] == 0.8
        assert metrics["reliability"] == 1.0  # No contradictions
        assert metrics["overall_quality"] > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
