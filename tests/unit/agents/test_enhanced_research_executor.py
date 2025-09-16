"""Tests for the enhanced research executor implementation."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from agents.research_executor import (
    ResearchExecutorAgent,
    execute_research,
    ResearchExecutorDependencies,
    _generate_cache_key,
    _extract_findings_fallback
)
from models.research_executor import (
    ResearchResults,
    HierarchicalFinding,
    ThemeCluster,
    Contradiction,
    PatternAnalysis,
    ExecutiveSummary,
    ResearchSource,
    ConfidenceLevel,
    ImportanceLevel,
    PatternType,
    ResearchExecutorConfig
)


class TestEnhancedResearchExecutor:
    """Test suite for the enhanced research executor implementation."""

    @pytest.fixture
    def sample_search_results(self) -> List[Dict[str, Any]]:
        """Create sample search results for testing."""
        return [
            {
                "title": "Quantum Computing Advances",
                "content": "Recent advances in quantum computing have shown significant progress in quantum supremacy.",
                "url": "https://example.com/quantum1",
                "score": 0.95
            },
            {
                "title": "AI and Machine Learning Trends",
                "content": "Artificial intelligence continues to evolve with new machine learning algorithms.",
                "url": "https://example.com/ai1",
                "score": 0.88
            }
        ]

    @pytest.fixture
    def sample_hierarchical_findings(self) -> List[HierarchicalFinding]:
        """Create sample hierarchical findings for testing."""
        return [
            HierarchicalFinding(
                finding="Quantum computing has achieved significant breakthroughs",
                supporting_evidence=["Evidence from research paper 1", "Evidence from paper 2"],
                confidence=ConfidenceLevel.HIGH,
                confidence_score=0.9,
                importance=ImportanceLevel.HIGH,
                importance_score=0.85,
                source=ResearchSource(
                    title="Quantum Computing Research",
                    url="https://example.com/quantum",
                    source_type="academic"
                ),
                category="technology",
                temporal_relevance="recent"
            ),
            HierarchicalFinding(
                finding="AI algorithms are becoming more efficient",
                supporting_evidence=["Evidence from AI research"],
                confidence=ConfidenceLevel.MEDIUM,
                confidence_score=0.75,
                importance=ImportanceLevel.MEDIUM,
                importance_score=0.7,
                source=ResearchSource(
                    title="AI Research Journal",
                    url="https://example.com/ai",
                    source_type="academic"
                ),
                category="technology",
                temporal_relevance="current"
            )
        ]

    @pytest.fixture
    def mock_dependencies(self) -> ResearchExecutorDependencies:
        """Create mock dependencies for testing."""
        return ResearchExecutorDependencies(
            synthesis_engine=MagicMock(),
            contradiction_detector=MagicMock(),
            pattern_recognizer=MagicMock(),
            confidence_analyzer=MagicMock(),
            cache_manager=AsyncMock(),
            parallel_executor=MagicMock(),
            metrics_collector=MagicMock(),
            optimization_manager=MagicMock(),
            original_query="test query",
            search_results=[]
        )

    def test_research_executor_agent_initialization(self):
        """Test that ResearchExecutorAgent initializes correctly."""
        config = ResearchExecutorConfig()
        agent = ResearchExecutorAgent(config=config)

        assert agent.config == config

    def test_cache_key_generation(self):
        """Test cache key generation function."""
        key1 = _generate_cache_key("test", "data", {"key": "value"})
        key2 = _generate_cache_key("test", "data", {"key": "value"})
        key3 = _generate_cache_key("different", "data", {"key": "value"})

        assert key1 == key2  # Same inputs should generate same key
        assert key1 != key3  # Different inputs should generate different keys
        assert len(key1) == 16  # Key should be 16 characters

    def test_extract_findings_fallback(self, sample_search_results):
        """Test the fallback finding extraction method."""
        content = "Test content for extraction"
        metadata = {"title": "Test Title", "url": "https://test.com", "type": "test"}

        findings = _extract_findings_fallback(content, metadata)

        assert len(findings) == 1
        assert isinstance(findings[0], HierarchicalFinding)
        assert "Test content for extraction" in findings[0].finding
        assert findings[0].source.title == "Test Title"
        assert findings[0].source.url == "https://test.com"

    def test_extract_findings_fallback_no_metadata(self):
        """Test fallback extraction without metadata."""
        content = "Test content"

        findings = _extract_findings_fallback(content, None)

        assert len(findings) == 1
        assert findings[0].source.title == "Unknown"
        assert findings[0].source.url is None

    @pytest.mark.asyncio
    async def test_execute_research_basic(self, sample_search_results):
        """Test basic research execution functionality."""
        query = "test research query"

        # Mock the agent and its dependencies
        with patch('agents.research_executor.research_executor_agent') as mock_agent:
            mock_result = MagicMock()
            mock_result.output = ResearchResults(
                query="test research query",
                findings=[],
                theme_clusters=[],
                contradictions=[],
                patterns=[],
                executive_summary=ExecutiveSummary(
                    key_findings=["Test finding"],
                    confidence_assessment="High confidence",
                    critical_gaps=[],
                    recommended_actions=["Test action"],
                    risk_factors=[]
                ),
                overall_quality_score=0.8,
                synthesis_metadata={}
            )
            mock_agent.run = AsyncMock(return_value=mock_result)

            # Execute research
            result = await execute_research(query, sample_search_results)

            # Verify the agent was called
            mock_agent.run.assert_called_once()
            call_args = mock_agent.run.call_args
            assert "Synthesize research for: test research query" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_research_executor_agent_execute_research(self, sample_search_results):
        """Test ResearchExecutorAgent execute_research method."""
        config = ResearchExecutorConfig()
        agent = ResearchExecutorAgent(config=config)

        query = "test query"

        with patch('agents.research_executor.execute_research') as mock_execute:
            mock_execute.return_value = ResearchResults(
                query="test query",
                findings=[],
                theme_clusters=[],
                contradictions=[],
                patterns=[],
                executive_summary=ExecutiveSummary(
                    key_findings=["Test finding"],
                    confidence_assessment="High confidence",
                    critical_gaps=[],
                    recommended_actions=["Test action"],
                    risk_factors=[]
                ),
                overall_quality_score=0.8,
                synthesis_metadata={}
            )

            result = await agent.execute_research(query, sample_search_results)

            mock_execute.assert_called_once_with(query, sample_search_results)
            assert isinstance(result, ResearchResults)

    def test_research_executor_dependencies_creation(self):
        """Test creation of ResearchExecutorDependencies."""
        deps = ResearchExecutorDependencies(
            synthesis_engine=MagicMock(),
            contradiction_detector=MagicMock(),
            pattern_recognizer=MagicMock(),
            confidence_analyzer=MagicMock(),
            original_query="test query",
            search_results=[{"test": "data"}]
        )

        assert deps.original_query == "test query"
        assert len(deps.search_results) == 1
        assert deps.search_results[0]["test"] == "data"

    def test_research_executor_dependencies_post_init(self):
        """Test post_init behavior of ResearchExecutorDependencies."""
        deps = ResearchExecutorDependencies(
            synthesis_engine=MagicMock(),
            contradiction_detector=MagicMock(),
            pattern_recognizer=MagicMock(),
            confidence_analyzer=MagicMock(),
            original_query="test query"
            # search_results not provided, should default to empty list
        )

        assert deps.search_results == []

    def test_research_executor_config_defaults(self):
        """Test ResearchExecutorConfig default values."""
        config = ResearchExecutorConfig()

        assert config.enable_parallel_processing is True
        assert config.max_concurrent_tasks == 4
        assert config.synthesis_config is not None
        assert config.cache_config is not None

    @pytest.mark.parametrize("confidence_level,expected_score", [
        (ConfidenceLevel.HIGH, 0.9),
        (ConfidenceLevel.MEDIUM, 0.7),
        (ConfidenceLevel.LOW, 0.4),
        (ConfidenceLevel.UNCERTAIN, 0.2)
    ])
    def test_confidence_level_to_score(self, confidence_level, expected_score):
        """Test confidence level to score conversion."""
        assert confidence_level.to_score() == expected_score

    @pytest.mark.parametrize("score,expected_level", [
        (0.9, ConfidenceLevel.HIGH),
        (0.75, ConfidenceLevel.MEDIUM),
        (0.5, ConfidenceLevel.LOW),
        (0.2, ConfidenceLevel.UNCERTAIN)
    ])
    def test_confidence_level_from_score(self, score, expected_level):
        """Test confidence level from score conversion."""
        assert ConfidenceLevel.from_score(score) == expected_level

    @pytest.mark.parametrize("importance_level,expected_score", [
        (ImportanceLevel.CRITICAL, 1.0),
        (ImportanceLevel.HIGH, 0.8),
        (ImportanceLevel.MEDIUM, 0.5),
        (ImportanceLevel.LOW, 0.2)
    ])
    def test_importance_level_to_score(self, importance_level, expected_score):
        """Test importance level to score conversion."""
        assert importance_level.to_score() == expected_score