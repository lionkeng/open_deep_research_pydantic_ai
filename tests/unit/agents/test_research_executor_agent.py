"""
Comprehensive tests for the ResearchExecutorAgent.
"""

import asyncio
import pytest
from datetime import datetime
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.research_executor import ResearchExecutorAgent
from src.agents.base import ResearchDependencies, AgentConfiguration
from src.models.research_executor import ResearchResults, ResearchFinding, ResearchSource
from src.models.api_models import APIKeys, ResearchMetadata
from src.models.core import ResearchState, ResearchStage


class TestResearchExecutorAgent:
    """Test suite for ResearchExecutorAgent."""

    @pytest.fixture
    async def agent_dependencies(self):
        """Create mock dependencies for testing."""
        deps = ResearchDependencies(
            http_client=AsyncMock(),
            api_keys=APIKeys(),
            research_state=ResearchState(
                request_id="test-exec-123",
                user_id="test-user",
                session_id="test-session",
                user_query="Latest advancements in quantum computing",
                current_stage=ResearchStage.RESEARCH_EXECUTION
            ),
            metadata=ResearchMetadata(),
            usage=None
        )
        return deps

    @pytest.fixture
    def research_executor_agent(self, agent_dependencies):
        """Create a ResearchExecutorAgent instance."""
        config = AgentConfiguration(
            max_retries=3,
            timeout_seconds=30.0,
            temperature=0.7
        )
        agent = ResearchExecutorAgent(config=config)
        agent._deps = agent_dependencies
        return agent

    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = ResearchExecutorAgent()
        assert agent.agent_name == "research_executor"
        assert agent.agent is not None
        assert agent.result_validator is not None

    @pytest.mark.asyncio
    async def test_successful_research_execution(self, research_executor_agent, agent_dependencies):
        """Test successful research execution with multiple findings."""
        mock_result = MagicMock()
        mock_result.data = ResearchResults(
            query="Latest advancements in quantum computing",
            findings=[
                ResearchFinding(
                    finding="IBM achieved 433-qubit quantum processor",
                    confidence_level=0.95,
                    source=ResearchSource(
                        title="IBM Quantum Progress Report",
                        url="https://example.com/ibm-quantum",
                        publish_date="2024-01-15",
                        author="IBM Research",
                        credibility_score=0.98
                    ),
                    relevance_score=0.92,
                    key_insights=["Record qubit count", "Improved error rates"],
                    supporting_evidence=["Technical specifications", "Benchmark results"],
                    contradictions=[]
                ),
                ResearchFinding(
                    finding="Google demonstrates quantum supremacy in optimization",
                    confidence_level=0.88,
                    source=ResearchSource(
                        title="Google Quantum AI Publication",
                        url="https://example.com/google-quantum",
                        publish_date="2024-02-01",
                        author="Google Research",
                        credibility_score=0.97
                    ),
                    relevance_score=0.90,
                    key_insights=["New optimization algorithms", "Practical applications"],
                    supporting_evidence=["Algorithm details", "Performance metrics"],
                    contradictions=[]
                )
            ],
            total_sources_consulted=15,
            search_strategies_used=["academic_search", "news_search", "patent_search"],
            confidence_score=0.91,
            execution_metadata={
                "search_time": 5.2,
                "filtering_applied": True,
                "deduplication_performed": True
            }
        )

        with patch.object(research_executor_agent.agent, 'run', return_value=mock_result):
            result = await research_executor_agent.execute(agent_dependencies)

            assert isinstance(result, ResearchResults)
            assert len(result.findings) == 2
            assert result.total_sources_consulted > 0
            assert result.confidence_score > 0.8
            assert all(f.confidence_level > 0 for f in result.findings)

    @pytest.mark.asyncio
    async def test_research_with_contradictions(self, research_executor_agent, agent_dependencies):
        """Test handling of contradictory findings."""
        mock_result = MagicMock()
        mock_result.data = ResearchResults(
            query="Effect of coffee on health",
            findings=[
                ResearchFinding(
                    finding="Coffee consumption linked to reduced heart disease risk",
                    confidence_level=0.75,
                    source=ResearchSource(
                        title="Cardiovascular Health Study",
                        url="https://example.com/study1",
                        publish_date="2024-01-10",
                        author="Medical Journal A",
                        credibility_score=0.85
                    ),
                    relevance_score=0.88,
                    key_insights=["3-4 cups daily optimal"],
                    supporting_evidence=["10-year study", "50,000 participants"],
                    contradictions=["Some studies show increased anxiety"]
                ),
                ResearchFinding(
                    finding="High coffee intake may increase anxiety and sleep issues",
                    confidence_level=0.70,
                    source=ResearchSource(
                        title="Sleep and Anxiety Research",
                        url="https://example.com/study2",
                        publish_date="2024-01-20",
                        author="Psychology Journal B",
                        credibility_score=0.82
                    ),
                    relevance_score=0.85,
                    key_insights=["Dose-dependent effects", "Individual variation"],
                    supporting_evidence=["Meta-analysis", "Clinical trials"],
                    contradictions=["Cardiovascular benefits disputed"]
                )
            ],
            total_sources_consulted=25,
            search_strategies_used=["medical_database", "meta_analysis"],
            confidence_score=0.72,
            execution_metadata={
                "conflicting_evidence": True,
                "requires_interpretation": True
            }
        )

        with patch.object(research_executor_agent.agent, 'run', return_value=mock_result):
            result = await research_executor_agent.execute(agent_dependencies)

            assert any(len(f.contradictions) > 0 for f in result.findings)
            assert result.execution_metadata.get("conflicting_evidence") is True
            assert result.confidence_score < 0.8  # Lower confidence due to contradictions

    @pytest.mark.asyncio
    async def test_source_credibility_assessment(self, research_executor_agent, agent_dependencies):
        """Test that source credibility is properly assessed."""
        mock_result = MagicMock()
        mock_result.data = ResearchResults(
            query="Test query",
            findings=[
                ResearchFinding(
                    finding="High credibility finding",
                    confidence_level=0.90,
                    source=ResearchSource(
                        title="Peer-reviewed Journal",
                        url="https://journal.example.com",
                        publish_date="2024-01-01",
                        author="Renowned Institute",
                        credibility_score=0.95
                    ),
                    relevance_score=0.85,
                    key_insights=["Insight 1"],
                    supporting_evidence=["Evidence 1"],
                    contradictions=[]
                ),
                ResearchFinding(
                    finding="Lower credibility finding",
                    confidence_level=0.60,
                    source=ResearchSource(
                        title="Blog Post",
                        url="https://blog.example.com",
                        publish_date="2024-01-01",
                        author="Unknown Author",
                        credibility_score=0.45
                    ),
                    relevance_score=0.70,
                    key_insights=["Insight 2"],
                    supporting_evidence=["Anecdotal evidence"],
                    contradictions=[]
                )
            ],
            total_sources_consulted=10,
            search_strategies_used=["mixed_search"],
            confidence_score=0.75,
            execution_metadata={}
        )

        with patch.object(research_executor_agent.agent, 'run', return_value=mock_result):
            result = await research_executor_agent.execute(agent_dependencies)

            credibility_scores = [f.source.credibility_score for f in result.findings]
            assert max(credibility_scores) > 0.9
            assert min(credibility_scores) < 0.5
            assert all(0.0 <= score <= 1.0 for score in credibility_scores)

    @pytest.mark.asyncio
    async def test_search_strategy_selection(self, research_executor_agent, agent_dependencies):
        """Test that appropriate search strategies are selected."""
        test_cases = [
            ("Latest AI research papers", ["academic_search", "arxiv_search"]),
            ("Current stock market trends", ["financial_news", "market_data"]),
            ("Medical treatment guidelines", ["medical_database", "clinical_trials"]),
            ("Technology news today", ["news_search", "tech_blogs"])
        ]

        for query, expected_strategies in test_cases:
            agent_dependencies.research_state.user_query = query

            mock_result = MagicMock()
            mock_result.data = ResearchResults(
                query=query,
                findings=[
                    ResearchFinding(
                        finding="Test finding",
                        confidence_level=0.8,
                        source=ResearchSource(
                            title="Test",
                            url="https://test.com",
                            publish_date="2024-01-01",
                            author="Test",
                            credibility_score=0.8
                        ),
                        relevance_score=0.8,
                        key_insights=["Test"],
                        supporting_evidence=["Test"],
                        contradictions=[]
                    )
                ],
                total_sources_consulted=5,
                search_strategies_used=expected_strategies,
                confidence_score=0.8,
                execution_metadata={}
            )

            with patch.object(research_executor_agent.agent, 'run', return_value=mock_result):
                result = await research_executor_agent.execute(agent_dependencies)

                for strategy in expected_strategies:
                    assert strategy in result.search_strategies_used

    @pytest.mark.asyncio
    async def test_edge_case_no_findings(self, research_executor_agent, agent_dependencies):
        """Test handling when no relevant findings are found."""
        agent_dependencies.research_state.user_query = "Extremely obscure nonsensical topic xyz123"

        mock_result = MagicMock()
        mock_result.data = ResearchResults(
            query="Extremely obscure nonsensical topic xyz123",
            findings=[],
            total_sources_consulted=20,
            search_strategies_used=["general_search", "deep_search"],
            confidence_score=0.1,
            execution_metadata={
                "no_results_found": True,
                "search_exhausted": True
            }
        )

        with patch.object(research_executor_agent.agent, 'run', return_value=mock_result):
            result = await research_executor_agent.execute(agent_dependencies)

            assert len(result.findings) == 0
            assert result.confidence_score < 0.2
            assert result.execution_metadata.get("no_results_found") is True

    @pytest.mark.asyncio
    async def test_relevance_scoring(self, research_executor_agent, agent_dependencies):
        """Test that findings are properly scored for relevance."""
        mock_result = MagicMock()
        mock_result.data = ResearchResults(
            query="Machine learning algorithms",
            findings=[
                ResearchFinding(
                    finding="Deep learning breakthrough in computer vision",
                    confidence_level=0.85,
                    source=ResearchSource(
                        title="CV Research",
                        url="https://cv.example.com",
                        publish_date="2024-01-01",
                        author="Research Lab",
                        credibility_score=0.9
                    ),
                    relevance_score=0.95,  # Highly relevant
                    key_insights=["Direct ML application"],
                    supporting_evidence=["Algorithm details"],
                    contradictions=[]
                ),
                ResearchFinding(
                    finding="General AI trends in industry",
                    confidence_level=0.75,
                    source=ResearchSource(
                        title="Industry Report",
                        url="https://industry.example.com",
                        publish_date="2024-01-01",
                        author="Analyst Firm",
                        credibility_score=0.8
                    ),
                    relevance_score=0.60,  # Somewhat relevant
                    key_insights=["Broad AI trends"],
                    supporting_evidence=["Market data"],
                    contradictions=[]
                ),
                ResearchFinding(
                    finding="Hardware advances for computing",
                    confidence_level=0.70,
                    source=ResearchSource(
                        title="Hardware News",
                        url="https://hw.example.com",
                        publish_date="2024-01-01",
                        author="Tech Site",
                        credibility_score=0.75
                    ),
                    relevance_score=0.40,  # Tangentially relevant
                    key_insights=["GPU improvements"],
                    supporting_evidence=["Benchmarks"],
                    contradictions=[]
                )
            ],
            total_sources_consulted=30,
            search_strategies_used=["targeted_search"],
            confidence_score=0.78,
            execution_metadata={}
        )

        with patch.object(research_executor_agent.agent, 'run', return_value=mock_result):
            result = await research_executor_agent.execute(agent_dependencies)

            relevance_scores = [f.relevance_score for f in result.findings]
            assert max(relevance_scores) > 0.9
            assert min(relevance_scores) < 0.5
            # Should be sorted by relevance (highest first)
            assert relevance_scores == sorted(relevance_scores, reverse=True)

    @pytest.mark.asyncio
    async def test_metadata_tracking(self, research_executor_agent, agent_dependencies):
        """Test that execution metadata is properly tracked."""
        mock_result = MagicMock()
        mock_result.data = ResearchResults(
            query="Test query",
            findings=[
                ResearchFinding(
                    finding="Test",
                    confidence_level=0.8,
                    source=ResearchSource(
                        title="Test",
                        url="https://test.com",
                        publish_date="2024-01-01",
                        author="Test",
                        credibility_score=0.8
                    ),
                    relevance_score=0.8,
                    key_insights=["Test"],
                    supporting_evidence=["Test"],
                    contradictions=[]
                )
            ],
            total_sources_consulted=50,
            search_strategies_used=["comprehensive"],
            confidence_score=0.85,
            execution_metadata={
                "search_time": 12.5,
                "filtering_applied": True,
                "deduplication_performed": True,
                "sources_filtered_out": 35,
                "duplicate_findings_removed": 5,
                "search_iterations": 3,
                "api_calls_made": 8
            }
        )

        with patch.object(research_executor_agent.agent, 'run', return_value=mock_result):
            result = await research_executor_agent.execute(agent_dependencies)

            metadata = result.execution_metadata
            assert "search_time" in metadata
            assert metadata["search_time"] > 0
            assert metadata.get("filtering_applied") is True
            assert metadata.get("sources_filtered_out", 0) > 0

    @pytest.mark.asyncio
    async def test_error_handling(self, research_executor_agent, agent_dependencies):
        """Test error handling during research execution."""
        with patch.object(research_executor_agent.agent, 'run', side_effect=Exception("Search API failed")):
            with pytest.raises(Exception, match="Search API failed"):
                await research_executor_agent.execute(agent_dependencies)

    @pytest.mark.asyncio
    async def test_timeout_handling(self, research_executor_agent, agent_dependencies):
        """Test timeout handling for long-running searches."""
        async def delayed_response():
            await asyncio.sleep(10)
            return MagicMock()

        with patch.object(research_executor_agent.agent, 'run', side_effect=delayed_response):
            research_executor_agent.config.timeout_seconds = 0.1
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    research_executor_agent.execute(agent_dependencies),
                    timeout=0.2
                )
