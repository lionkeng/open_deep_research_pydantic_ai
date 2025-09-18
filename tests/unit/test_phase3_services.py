"""Unit tests for Phase 3 services: SearchOrchestrator, QualityMonitor, and SynthesisTools."""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.research_executor import (
    PatternType,
)
from models.search_query_models import ExecutionStrategy
from services.search_orchestrator import SearchQuery
from services.quality_monitor import (
    AlertSeverity,
    QualityMetric,
    QualityMonitor,
    QualityThresholds,
    SynthesisResult,  # Add missing import
)
from services.search_orchestrator import (
    CacheConfig,
    ExecutionStatus,
    QueryExecutionPlan,
    RetryConfig,
    SearchOrchestrator,
    SearchResult,
)
from services.synthesis_tools import (
    ContradictionConfig,
    ContradictionType,  # Add missing import
    HierarchyFactors,
    InformationHierarchy,  # Add missing import
    SynthesisTools,
    ThemeType,
)


class TestSearchOrchestrator:
    """Test the SearchOrchestrator service."""

    @pytest.fixture
    def mock_search_fn(self):
        """Create a mock search function."""
        async def search(query: SearchQuery) -> SearchResult:
            return SearchResult(
                query=query.query,
                results=[],
                metadata={"mock": True},
                timestamp=datetime.now(timezone.utc),
            )
        return search

    @pytest.fixture
    def orchestrator(self, mock_search_fn):
        """Create a SearchOrchestrator instance."""
        return SearchOrchestrator(
            search_fn=mock_search_fn,
            retry_config=RetryConfig(max_attempts=2, initial_delay_ms=10),
            cache_config=CacheConfig(enabled=True, ttl_seconds=60),
            max_workers=2,
        )

    @pytest.mark.asyncio
    async def test_sequential_execution(self, orchestrator):
        """Test sequential query execution."""
        queries = [
            SearchQuery(id="q1", query="test1", priority=1, rationale="Test query 1"),
            SearchQuery(id="q2", query="test2", priority=3, rationale="Test query 2"),
        ]

        results = await orchestrator.execute_sequential(queries)

        assert len(results) == 2
        assert all(result is not None for _, result in results)

    @pytest.mark.asyncio
    async def test_parallel_execution(self, orchestrator):
        """Test parallel query execution."""
        queries = [
            SearchQuery(id=f"q{i}", query=f"test{i}", priority=3, rationale=f"Test query {i}")
            for i in range(5)
        ]

        start_time = time.time()
        results = await orchestrator.execute_parallel(queries)
        execution_time = time.time() - start_time

        assert len(results) == 5
        assert all(result is not None for _, result in results)
        # Parallel should be faster than sequential (5 * delay)
        assert execution_time < 1.0

    @pytest.mark.asyncio
    async def test_hierarchical_execution(self, orchestrator):
        """Test hierarchical query execution based on priority."""
        queries = [
            SearchQuery(id="q1", query="low", priority=5, rationale="Low priority query"),
            SearchQuery(id="q2", query="high1", priority=1, rationale="High priority query 1"),
            SearchQuery(id="q3", query="medium", priority=3, rationale="Medium priority query"),
            SearchQuery(id="q4", query="high2", priority=1, rationale="High priority query 2"),
        ]

        results = await orchestrator.execute_hierarchical(queries)

        # Check that high priority queries were executed first
        query_order = [query.query for query, _ in results]
        assert query_order.index("high1") < query_order.index("medium")
        assert query_order.index("high2") < query_order.index("medium")
        assert query_order.index("medium") < query_order.index("low")

    @pytest.mark.asyncio
    async def test_retry_logic(self):
        """Test retry logic with exponential backoff."""
        call_count = 0

        async def failing_search(query: SearchQuery) -> SearchResult:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Network error")
            return SearchResult(
                query=query.query,
                results=[],
                metadata={"attempts": call_count},
                timestamp=datetime.now(timezone.utc),
            )

        orchestrator = SearchOrchestrator(
            search_fn=failing_search,
            retry_config=RetryConfig(max_attempts=3, initial_delay_ms=10),
        )

        query = SearchQuery(id="q1", query="test", rationale="Test query for retry")
        _, result = await orchestrator._execute_query(query)

        assert result is not None
        assert result.metadata["attempts"] == 2

    @pytest.mark.asyncio
    async def test_caching(self, orchestrator):
        """Test caching mechanism."""
        query = SearchQuery(id="q1", query="cached_test", rationale="Test query for caching")

        # First execution
        _, result1 = await orchestrator._execute_query(query)
        assert result1 is not None

        # Second execution should hit cache
        _, result2 = await orchestrator._execute_query(query)
        assert result2 is not None

        # Check that cache was used
        traces = orchestrator._execution_traces
        assert any(t.status == ExecutionStatus.CACHED for t in traces)

    @pytest.mark.asyncio
    async def test_execution_report(self, orchestrator):
        """Test execution report generation."""
        plan = QueryExecutionPlan(
            queries=[
                SearchQuery(id="q1", query="test1", rationale="Test query 1"),
                SearchQuery(id="q2", query="test2", rationale="Test query 2"),
            ],
            strategy=ExecutionStrategy.PARALLEL,
        )

        results, report = await orchestrator.execute_plan(plan)

        assert report.total_queries == 2
        assert report.executed_queries == 2
        assert report.execution_rate == 1.0
        assert report.strategy_used == ExecutionStrategy.PARALLEL
        assert len(report.traces) == 2

    @pytest.mark.asyncio
    async def test_cache_eviction(self):
        """Test LRU cache eviction."""
        orchestrator = SearchOrchestrator(
            search_fn=AsyncMock(return_value=SearchResult(
                query="test",
                results=[],
                metadata={},
                timestamp=datetime.now(timezone.utc),
            )),
            cache_config=CacheConfig(enabled=True, max_size=2),
        )

        # Fill cache beyond capacity
        for i in range(3):
            query = SearchQuery(id=f"q{i}", query=f"test{i}", rationale=f"Test query {i}")
            await orchestrator._execute_query(query)

        # First query should be evicted
        assert len(orchestrator._cache) == 2
        assert "test0" not in [q.query for q, _ in orchestrator._cache.values()]

    @pytest.mark.asyncio
    async def test_execution_stats(self, orchestrator):
        """Test execution statistics tracking."""
        queries = [
            SearchQuery(id=f"q{i}", query=f"test{i}", rationale=f"Test query {i}")
            for i in range(3)
        ]

        results = await orchestrator.execute_sequential(queries)
        stats = orchestrator.get_execution_stats()

        assert stats["total_executions"] == 3
        assert stats["success_rate"] == 1.0
        assert "average_execution_time_ms" in stats


class TestQualityMonitor:
    """Test the QualityMonitor service."""

    @pytest.fixture
    def monitor(self):
        """Create a QualityMonitor instance."""
        return QualityMonitor(
            thresholds=QualityThresholds(
                min_execution_rate=0.9,
                min_source_diversity=0.3,
            ),
            history_size=10,
            alert_cooldown_seconds=5,
        )

    def test_shannon_entropy_calculation(self, monitor):
        """Test Shannon entropy calculation for source diversity."""
        # Uniform distribution (maximum entropy)
        sources = ["A", "B", "C", "D"]
        entropy = monitor.calculate_shannon_entropy(sources)
        assert entropy == 1.0  # Maximum entropy for 4 unique sources

        # Single source (minimum entropy)
        sources = ["A", "A", "A", "A"]
        entropy = monitor.calculate_shannon_entropy(sources)
        assert entropy == 0.0

        # Mixed distribution
        sources = ["A", "A", "B", "B", "B", "C"]
        entropy = monitor.calculate_shannon_entropy(sources)
        assert 0 < entropy < 1.0

    def test_execution_rate_assessment(self, monitor):
        """Test execution rate metric assessment."""
        rate, details = monitor.assess_execution_rate(95, 100)
        assert rate == 0.95
        assert details["executed"] == 95
        assert details["failed"] == 5

    def test_pattern_accuracy_assessment(self, monitor):
        """Test pattern accuracy assessment."""
        synthesis = SynthesisResult(
            key_findings=["Finding 1"],
            synthesis="Test synthesis",
            confidence_score=0.8,
        )
        patterns = [PatternType.TEMPORAL, PatternType.CAUSAL]

        accuracy, details = monitor.assess_pattern_accuracy(synthesis, patterns)
        assert 0 <= accuracy <= 1.0
        assert len(details["patterns"]) == 2

    def test_synthesis_coherence_assessment(self, monitor):
        """Test synthesis coherence assessment."""
        # Good synthesis
        good_synthesis = SynthesisResult(
            key_findings=["This is a well-formed finding with adequate detail"],
            synthesis="This is a comprehensive synthesis with multiple paragraphs.\n\nIt has proper structure.",
            confidence_score=0.85,
        )
        coherence, details = monitor.assess_synthesis_coherence(good_synthesis)
        assert coherence > 0.5
        assert len(details["issues"]) < 2

        # Poor synthesis
        poor_synthesis = SynthesisResult(
            key_findings=[],
            synthesis="Too brief",
            confidence_score=1.5,  # Invalid
        )
        coherence, details = monitor.assess_synthesis_coherence(poor_synthesis)
        assert coherence < 0.5
        assert "No key findings" in details["issues"]

    def test_alert_generation(self, monitor):
        """Test alert generation with thresholds."""
        synthesis = SynthesisResult(
            key_findings=["Finding"],
            synthesis="Synthesis text",
            confidence_score=0.7,
        )
        search_results = []
        execution_stats = {
            "executed_queries": 80,
            "total_queries": 100,
        }

        report = monitor.assess_synthesis_quality(
            synthesis, search_results, execution_stats
        )

        # Should generate alert for low execution rate (80% < 90% threshold)
        assert any(
            alert.metric == QualityMetric.EXECUTION_RATE
            for alert in report.alerts
        )

    def test_metric_history_tracking(self, monitor):
        """Test historical metric tracking."""
        # Generate multiple assessments
        for i in range(3):
            synthesis = SynthesisResult(
                key_findings=["Finding"],
                synthesis="Synthesis",
                confidence_score=0.5 + i * 0.1,
            )
            monitor.assess_synthesis_quality(synthesis, [], {})

        # Check history
        history = monitor.get_metric_history(QualityMetric.SYNTHESIS_COHERENCE)
        assert len(history) == 3

    def test_metric_trend_calculation(self, monitor):
        """Test metric trend calculation."""
        # Generate trending data
        for i in range(5):
            synthesis = SynthesisResult(
                key_findings=["Finding"],
                synthesis="Synthesis",
                confidence_score=0.5 + i * 0.1,
            )
            monitor.assess_synthesis_quality(synthesis, [], {"executed_queries": 90 + i, "total_queries": 100})

        trend = monitor.get_metric_trend(QualityMetric.EXECUTION_RATE, window_size=5)
        assert trend["trend"] > 0  # Positive trend (improving)
        assert "mean" in trend
        assert "std" in trend

    def test_alert_cooldown(self, monitor):
        """Test alert cooldown mechanism."""
        execution_stats = {
            "executed_queries": 80,
            "total_queries": 100,
        }

        # First assessment should generate alert
        synthesis = SynthesisResult(
            key_findings=["Finding"],
            synthesis="Synthesis",
            confidence_score=0.7,
        )
        report1 = monitor.assess_synthesis_quality(synthesis, [], execution_stats)
        alert_count1 = len(report1.alerts)

        # Immediate second assessment should not generate duplicate alert
        report2 = monitor.assess_synthesis_quality(synthesis, [], execution_stats)
        alert_count2 = len(report2.alerts)

        assert alert_count2 < alert_count1  # Fewer alerts due to cooldown


class TestSynthesisTools:
    """Test the SynthesisTools service."""

    @pytest.fixture
    def tools(self):
        """Create a SynthesisTools instance."""
        return SynthesisTools(
            hierarchy_factors=HierarchyFactors(),
            contradiction_config=ContradictionConfig(),
        )

    def test_information_hierarchy_scoring(self, tools):
        """Test information hierarchy scoring."""
        # Highly relevant information
        score = tools.score_information_hierarchy(
            information="The query directly answers the main question with specific data points",
            query="main question",
            source_credibility=0.9,
            timestamp=datetime.now(timezone.utc),
        )
        assert score.level == InformationHierarchy.PRIMARY
        assert score.score > 0.7

        # Less relevant information
        score = tools.score_information_hierarchy(
            information="Some tangentially related content",
            query="completely different topic",
            source_credibility=0.3,
            timestamp=datetime.now(timezone.utc) - timedelta(days=365),
        )
        assert score.level in [InformationHierarchy.CONTEXTUAL, InformationHierarchy.TANGENTIAL]

    def test_relevance_calculation(self, tools):
        """Test relevance score calculation."""
        relevance = tools._calculate_relevance(
            "machine learning algorithms for data analysis",
            "machine learning algorithms"
        )
        assert relevance > 0.5  # High relevance

        relevance = tools._calculate_relevance(
            "cooking recipes",
            "machine learning"
        )
        assert relevance < 0.3  # Low relevance

    def test_specificity_calculation(self, tools):
        """Test specificity score calculation."""
        # Specific information with numbers and dates
        specific_text = "In 2024, the company achieved 45.7% growth with $2.3M revenue"
        specificity = tools._calculate_specificity(specific_text)
        assert specificity > 0.5

        # Vague information
        vague_text = "The company did well this year"
        specificity = tools._calculate_specificity(vague_text)
        assert specificity < 0.3

    def test_contradiction_detection(self, tools):
        """Test contradiction detection algorithms."""
        search_results = [
            SearchResult(
                query="test1",
                results=[
                    MagicMock(content="The stock price increased by 20% in Q3 2024"),
                ],
                metadata={},
                timestamp=datetime.now(timezone.utc),
            ),
            SearchResult(
                query="test2",
                results=[
                    MagicMock(content="The stock price decreased by 20% in Q3 2024"),
                ],
                metadata={},
                timestamp=datetime.now(timezone.utc),
            ),
        ]

        contradictions = tools.detect_contradictions(search_results)
        assert len(contradictions) > 0
        assert any(c.type == ContradictionType.FACTUAL for c in contradictions)

    def test_convergence_analysis(self, tools):
        """Test convergence analysis across sources."""
        search_results = [
            SearchResult(
                query="source1",
                results=[
                    MagicMock(content="AI adoption is increasing rapidly across industries"),
                ],
                metadata={},
                timestamp=datetime.now(timezone.utc),
            ),
            SearchResult(
                query="source2",
                results=[
                    MagicMock(content="Artificial intelligence adoption is growing quickly in various sectors"),
                ],
                metadata={},
                timestamp=datetime.now(timezone.utc),
            ),
            SearchResult(
                query="source3",
                results=[
                    MagicMock(content="AI implementation is accelerating across different industries"),
                ],
                metadata={},
                timestamp=datetime.now(timezone.utc),
            ),
        ]

        convergence_points = tools.analyze_convergence(search_results, min_sources=2)
        assert len(convergence_points) > 0
        assert convergence_points[0].support_count >= 2

    def test_theme_extraction(self, tools):
        """Test theme extraction with relationship mapping."""
        search_results = [
            SearchResult(
                query="test",
                results=[
                    MagicMock(content="Machine Learning is a key concept in AI. Deep Learning models are advancing rapidly. "
                                     "Google and Microsoft are leading companies in this space."),
                ],
                metadata={},
                timestamp=datetime.now(timezone.utc),
            ),
        ]

        themes = tools.extract_themes(search_results, min_frequency=1)
        assert len(themes) > 0
        # Should extract entities like Google, Microsoft
        assert any(theme.type == ThemeType.ENTITY for theme in themes)

    def test_pattern_matching(self, tools):
        """Test pattern matching capabilities."""
        search_results = [
            SearchResult(
                query="test",
                results=[
                    MagicMock(content="Sales increased from 2020 to 2024. "
                                     "This growth was caused by improved marketing strategies. "
                                     "As revenue increases, customer satisfaction also increases."),
                ],
                metadata={},
                timestamp=datetime.now(timezone.utc),
            ),
        ]

        patterns = tools.match_patterns(search_results)
        assert len(patterns) > 0

        # Should detect temporal pattern (from 2020 to 2024)
        assert any(p.type == PatternType.TEMPORAL for p in patterns)

        # Should detect causal pattern (caused by)
        assert any(p.type == PatternType.CAUSAL for p in patterns)

        # Should detect correlative pattern (as X increases, Y increases)
        assert any(p.type == PatternType.CORRELATION for p in patterns)

    def test_synthesis_score_calculation(self, tools):
        """Test overall synthesis score calculation."""
        from services.synthesis_tools import HierarchyScore, ConvergencePoint, Theme, Pattern

        hierarchy_scores = [
            HierarchyScore(
                level=InformationHierarchy.PRIMARY,
                score=0.8,
                factors={},
                reasoning="Test"
            )
        ]
        contradictions = []
        convergence_points = [
            ConvergencePoint(
                claim="Test claim",
                sources=["A", "B"],
                support_count=2,
                confidence=0.8,
                evidence=[]
            )
        ]
        themes = [
            Theme(
                type=ThemeType.CONCEPT,
                name="Test",
                description="Test theme",
                frequency=3,
                sources=["A"],
                related_themes=[],
                confidence=0.7
            )
        ]
        patterns = [
            Pattern(
                type=PatternType.TEMPORAL,
                description="Test pattern",
                evidence=[],
                strength=0.6
            )
        ]

        score = tools.calculate_synthesis_score(
            hierarchy_scores, contradictions, convergence_points, themes, patterns
        )
        assert 0 <= score <= 1.0

    def test_synthesis_recommendations(self, tools):
        """Test generation of synthesis recommendations."""
        from services.synthesis_tools import HierarchyScore

        # Poor hierarchy distribution
        hierarchy_scores = [
            HierarchyScore(
                level=InformationHierarchy.TANGENTIAL,
                score=0.3,
                factors={},
                reasoning="Test"
            )
        ] * 5

        recommendations = tools.get_synthesis_recommendations(
            hierarchy_scores, [], [], [], []
        )

        assert len(recommendations) > 0
        assert any("primary information" in r.lower() for r in recommendations)


class TestServiceIntegration:
    """Test integration between Phase 3 services."""

    @pytest.mark.asyncio
    async def test_orchestrator_with_quality_monitoring(self):
        """Test SearchOrchestrator integration with QualityMonitor."""
        orchestrator = SearchOrchestrator()
        monitor = QualityMonitor()

        # Execute queries
        plan = QueryExecutionPlan(
            queries=[
                SearchQuery(id="q1", query="test1", rationale="Test query 1"),
                SearchQuery(id="q2", query="test2", rationale="Test query 2"),
            ],
            strategy=ExecutionStrategy.PARALLEL,
        )

        results, report = await orchestrator.execute_plan(plan)

        # Assess quality
        synthesis = SynthesisResult(
            key_findings=["Finding from search"],
            synthesis="Synthesis of results",
            confidence_score=0.8,
        )

        execution_stats = {
            "executed_queries": report.executed_queries,
            "total_queries": report.total_queries,
        }

        quality_report = monitor.assess_synthesis_quality(
            synthesis, [r for _, r in results if r], execution_stats
        )

        assert quality_report.overall_score > 0
        assert QualityMetric.EXECUTION_RATE in quality_report.metrics

    @pytest.mark.asyncio
    async def test_synthesis_tools_with_search_results(self):
        """Test SynthesisTools processing SearchOrchestrator results."""
        orchestrator = SearchOrchestrator()
        tools = SynthesisTools()

        # Execute search
        queries = [
            SearchQuery(query="artificial intelligence trends"),
            SearchQuery(query="machine learning applications"),
        ]
        results = await orchestrator.execute_sequential(queries)

        # Process with synthesis tools
        search_results = [r for _, r in results if r]

        # Detect patterns
        patterns = tools.match_patterns(search_results)

        # Extract themes
        themes = tools.extract_themes(search_results)

        # Check convergence
        convergence = tools.analyze_convergence(search_results)

        assert isinstance(patterns, list)
        assert isinstance(themes, list)
        assert isinstance(convergence, list)

    @pytest.mark.asyncio
    async def test_full_pipeline_integration(self):
        """Test full integration of all Phase 3 services."""
        # Setup services
        orchestrator = SearchOrchestrator()
        monitor = QualityMonitor()
        tools = SynthesisTools()

        # Execute search plan
        plan = QueryExecutionPlan(
            queries=[
                SearchQuery(query="test query 1", priority=1),  # High priority
                SearchQuery(query="test query 2", priority=3),  # Medium priority
            ],
            strategy=ExecutionStrategy.HIERARCHICAL,
        )

        results, exec_report = await orchestrator.execute_plan(plan)

        # Process results with synthesis tools
        search_results = [r for _, r in results if r]

        hierarchy_scores = [
            tools.score_information_hierarchy(
                f"Information from {r.query}",
                r.query,
                source_credibility=0.7
            )
            for r in search_results
        ]

        contradictions = tools.detect_contradictions(search_results)
        convergence = tools.analyze_convergence(search_results)
        themes = tools.extract_themes(search_results)
        patterns = tools.match_patterns(search_results)

        # Calculate synthesis score
        synthesis_score = tools.calculate_synthesis_score(
            hierarchy_scores, contradictions, convergence, themes, patterns
        )

        # Create synthesis result
        synthesis = SynthesisResult(
            key_findings=[f"Finding from {t.name}" for t in themes[:3]],
            synthesis="Comprehensive synthesis of search results",
            confidence_score=synthesis_score,
            metadata={
                "patterns": [p.type for p in patterns],
                "contradictions": [c.type for c in contradictions],
                "hierarchy": [h.level for h in hierarchy_scores],
            }
        )

        # Assess quality
        quality_report = monitor.assess_synthesis_quality(
            synthesis,
            search_results,
            {
                "executed_queries": exec_report.executed_queries,
                "total_queries": exec_report.total_queries,
            }
        )

        # Verify integration
        assert exec_report.execution_rate > 0
        assert quality_report.overall_score > 0
        assert len(quality_report.recommendations) >= 0

        # Check for quality-driven improvements
        if quality_report.overall_score < 0.7:
            # Get recommendations
            recommendations = tools.get_synthesis_recommendations(
                hierarchy_scores, contradictions, convergence, themes, patterns
            )
            assert len(recommendations) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
