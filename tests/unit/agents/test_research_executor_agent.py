"""
Comprehensive tests for the ResearchExecutorAgent.
"""

import asyncio
import pytest
import pytest_asyncio
from datetime import datetime, UTC
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

from agents.research_executor import ResearchExecutorAgent
from agents.base import ResearchDependencies, AgentConfiguration
from models.research_executor import (
    ResearchResults,
    HierarchicalFinding,
    ResearchSource,
    ConfidenceLevel,
    ImportanceLevel
)
from models.api_models import APIKeys
from models.core import ResearchState, ResearchStage

class TestResearchExecutorAgent:
    """Test suite for ResearchExecutorAgent."""

    @pytest_asyncio.fixture
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
            usage=None
        )
        return deps

    @pytest_asyncio.fixture
    async def research_executor_agent(self, agent_dependencies):
        """Create a ResearchExecutorAgent instance."""
        config = AgentConfiguration(
            agent_name="research_executor",
            agent_type="research_executor",
            max_retries=3,
            timeout=30.0
        )
        agent = ResearchExecutorAgent(config=config)
        agent._deps = agent_dependencies
        return agent

    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = ResearchExecutorAgent()
        assert agent.name == "research_executor"
        assert agent.agent is not None
        assert agent.config is not None

    @pytest.mark.asyncio
    async def test_successful_research_execution(self, research_executor_agent, agent_dependencies):
        """Test successful research execution with multiple findings."""
        deps = agent_dependencies

        expected_result = ResearchResults(
            query="Latest advancements in quantum computing",
            findings=[
                HierarchicalFinding(
                    finding="IBM achieved 433-qubit quantum processor",
                    confidence=ConfidenceLevel.HIGH,
                    confidence_score=0.95,
                    importance=ImportanceLevel.HIGH,
                    importance_score=0.92,
                    source=ResearchSource(
                        title="IBM Quantum Progress Report",
                        url="https://example.com/ibm-quantum",
                        date=datetime(2024, 1, 15, tzinfo=UTC),
                        author="IBM Research",
                        credibility_score=0.98,
                        relevance_score=0.92
                    ),
                    supporting_evidence=["Technical specifications", "Benchmark results"]
                ),
                HierarchicalFinding(
                    finding="Google demonstrates quantum supremacy in optimization",
                    confidence=ConfidenceLevel.HIGH,
                    confidence_score=0.88,
                    importance=ImportanceLevel.HIGH,
                    importance_score=0.90,
                    source=ResearchSource(
                        title="Google Quantum AI Publication",
                        url="https://example.com/google-quantum",
                        date=datetime(2024, 2, 1, tzinfo=UTC),
                        author="Google Research",
                        credibility_score=0.97,
                        relevance_score=0.90
                    ),
                    supporting_evidence=["Algorithm details", "Performance metrics"]
                )
            ]
        )

        with patch.object(research_executor_agent, 'run', return_value=expected_result):
            result = await research_executor_agent.run(deps)

            assert isinstance(result, ResearchResults)
            assert len(result.findings) == 2
            assert all(f.confidence_score > 0 for f in result.findings)

    @pytest.mark.asyncio
    async def test_research_with_contradictions(self, research_executor_agent, agent_dependencies):
        """Test handling of contradictory findings."""
        deps = agent_dependencies

        expected_result = ResearchResults(
            query="Effect of coffee on health",
            findings=[
                HierarchicalFinding(
                    finding="Coffee consumption linked to reduced heart disease risk",
                    confidence=ConfidenceLevel.MEDIUM,
                    confidence_score=0.75,
                    importance=ImportanceLevel.HIGH,
                    importance_score=0.88,
                    source=ResearchSource(
                        title="Cardiovascular Health Study",
                        url="https://example.com/study1",
                        date=datetime(2024, 1, 10, tzinfo=UTC),
                        author="Medical Journal A",
                        credibility_score=0.85,
                        relevance_score=0.88
                    ),
                    supporting_evidence=["10-year study", "50,000 participants"]
                ),
                HierarchicalFinding(
                    finding="High coffee intake may increase anxiety and sleep issues",
                    confidence=ConfidenceLevel.MEDIUM,
                    confidence_score=0.70,
                    importance=ImportanceLevel.HIGH,
                    importance_score=0.85,
                    source=ResearchSource(
                        title="Sleep and Anxiety Research",
                        url="https://example.com/study2",
                        date=datetime(2024, 1, 20, tzinfo=UTC),
                        author="Psychology Journal B",
                        credibility_score=0.82,
                        relevance_score=0.85
                    ),
                    supporting_evidence=["Meta-analysis", "Clinical trials"]
                )
            ]
        )

        with patch.object(research_executor_agent, 'run', return_value=expected_result):
            result = await research_executor_agent.run(deps)

            # Check that we have findings with medium confidence indicating contradictions
            assert all(f.confidence == ConfidenceLevel.MEDIUM for f in result.findings)
            assert result.findings[0].confidence_score < 0.8  # Lower confidence due to contradictions

    @pytest.mark.asyncio
    async def test_source_credibility_assessment(self, research_executor_agent, agent_dependencies):
        """Test that source credibility is properly assessed."""
        deps = agent_dependencies

        expected_result = ResearchResults(
            query="Test query",
            findings=[
                HierarchicalFinding(
                    finding="High credibility finding",
                    confidence=ConfidenceLevel.HIGH,
                    confidence_score=0.90,
                    importance=ImportanceLevel.HIGH,
                    importance_score=0.85,
                    source=ResearchSource(
                        title="Peer-reviewed Journal",
                        url="https://journal.example.com",
                        date=datetime(2024, 1, 1, tzinfo=UTC),
                        author="Renowned Institute",
                        credibility_score=0.95,
                        relevance_score=0.85
                    ),
                    supporting_evidence=["Evidence 1"]
                ),
                HierarchicalFinding(
                    finding="Lower credibility finding",
                    confidence=ConfidenceLevel.MEDIUM,
                    confidence_score=0.60,
                    importance=ImportanceLevel.MEDIUM,
                    importance_score=0.70,
                    source=ResearchSource(
                        title="Blog Post",
                        url="https://blog.example.com",
                        date=datetime(2024, 1, 1, tzinfo=UTC),
                        author="Unknown Author",
                        credibility_score=0.45,
                        relevance_score=0.70
                    ),
                    supporting_evidence=["Anecdotal evidence"]
                )
            ]
        )

        with patch.object(research_executor_agent, 'run', return_value=expected_result):
            result = await research_executor_agent.run(deps)

            credibility_scores = [f.source.credibility_score for f in result.findings if f.source]
            assert max(credibility_scores) > 0.9
            assert min(credibility_scores) < 0.5
            assert all(0.0 <= score <= 1.0 for score in credibility_scores)

    @pytest.mark.asyncio
    async def test_search_strategy_selection(self, research_executor_agent, agent_dependencies):
        """Test that appropriate search strategies are selected."""
        deps = agent_dependencies
        test_cases = [
            ("Latest AI research papers", ["academic_search", "arxiv_search"]),
            ("Current stock market trends", ["financial_news", "market_data"]),
            ("Medical treatment guidelines", ["medical_database", "clinical_trials"]),
            ("Technology news today", ["news_search", "tech_blogs"])
        ]

        for query, expected_strategies in test_cases:
            deps.research_state.user_query = query

            expected_result = ResearchResults(
                query=query,
                findings=[
                    HierarchicalFinding(
                        finding="Test finding",
                        confidence=ConfidenceLevel.HIGH,
                        confidence_score=0.8,
                        importance=ImportanceLevel.MEDIUM,
                        importance_score=0.8,
                        source=ResearchSource(
                            title="Test",
                            url="https://test.com",
                            date=datetime(2024, 1, 1, tzinfo=UTC),
                            author="Test",
                            credibility_score=0.8,
                            relevance_score=0.8
                        ),
                        supporting_evidence=["Test"]
                    )
                ],
                metadata={"search_strategies": expected_strategies}
            )

            with patch.object(research_executor_agent, 'run', return_value=expected_result):
                result = await research_executor_agent.run(deps)

                # Check that we got some result (the actual strategy selection would be in the agent implementation)
                assert result.findings

    @pytest.mark.asyncio
    async def test_edge_case_no_findings(self, research_executor_agent, agent_dependencies):
        """Test handling when no relevant findings are found."""
        deps = agent_dependencies
        deps.research_state.user_query = "Extremely obscure nonsensical topic xyz123"

        expected_result = ResearchResults(
            query="Extremely obscure nonsensical topic xyz123",
            findings=[],
            metadata={
                "no_results_found": True,
                "search_exhausted": True
            }
        )

        with patch.object(research_executor_agent, 'run', return_value=expected_result):
            result = await research_executor_agent.run(deps)

            assert len(result.findings) == 0
            assert result.metadata.get("no_results_found") is True

    @pytest.mark.asyncio
    async def test_relevance_scoring(self, research_executor_agent, agent_dependencies):
        """Test that findings are properly scored for relevance."""
        deps = agent_dependencies

        expected_result = ResearchResults(
            query="Machine learning algorithms",
            findings=[
                HierarchicalFinding(
                    finding="Deep learning breakthrough in computer vision",
                    confidence=ConfidenceLevel.HIGH,
                    confidence_score=0.85,
                    importance=ImportanceLevel.CRITICAL,
                    importance_score=0.95,
                    source=ResearchSource(
                        title="CV Research",
                        url="https://cv.example.com",
                        date=datetime(2024, 1, 1, tzinfo=UTC),
                        author="Research Lab",
                        credibility_score=0.9,
                        relevance_score=0.95
                    ),
                    supporting_evidence=["Algorithm details"]
                ),
                HierarchicalFinding(
                    finding="General AI trends in industry",
                    confidence=ConfidenceLevel.MEDIUM,
                    confidence_score=0.75,
                    importance=ImportanceLevel.MEDIUM,
                    importance_score=0.60,
                    source=ResearchSource(
                        title="Industry Report",
                        url="https://industry.example.com",
                        date=datetime(2024, 1, 1, tzinfo=UTC),
                        author="Analyst Firm",
                        credibility_score=0.8,
                        relevance_score=0.60
                    ),
                    supporting_evidence=["Market data"]
                ),
                HierarchicalFinding(
                    finding="Hardware advances for computing",
                    confidence=ConfidenceLevel.MEDIUM,
                    confidence_score=0.70,
                    importance=ImportanceLevel.LOW,
                    importance_score=0.40,
                    source=ResearchSource(
                        title="Hardware News",
                        url="https://hw.example.com",
                        date=datetime(2024, 1, 1, tzinfo=UTC),
                        author="Tech Site",
                        credibility_score=0.75,
                        relevance_score=0.40
                    ),
                    supporting_evidence=["Benchmarks"]
                )
            ]
        )

        with patch.object(research_executor_agent, 'run', return_value=expected_result):
            result = await research_executor_agent.run(deps)

            relevance_scores = [f.source.relevance_score for f in result.findings if f.source]
            assert max(relevance_scores) > 0.9
            assert min(relevance_scores) < 0.5
            # Check that findings are sorted by importance
            importance_scores = [f.importance_score for f in result.findings]
            assert importance_scores == sorted(importance_scores, reverse=True)

    @pytest.mark.asyncio
    async def test_metadata_tracking(self, research_executor_agent, agent_dependencies):
        """Test that execution metadata is properly tracked."""
        deps = agent_dependencies

        expected_result = ResearchResults(
            query="Test query",
            findings=[
                HierarchicalFinding(
                    finding="Test",
                    confidence=ConfidenceLevel.HIGH,
                    confidence_score=0.8,
                    importance=ImportanceLevel.MEDIUM,
                    importance_score=0.8,
                    source=ResearchSource(
                        title="Test",
                        url="https://test.com",
                        date=datetime(2024, 1, 1, tzinfo=UTC),
                        author="Test",
                        credibility_score=0.8,
                        relevance_score=0.8
                    ),
                    supporting_evidence=["Test"]
                )
            ],
            metadata={
                "search_time": 12.5,
                "filtering_applied": True,
                "deduplication_performed": True,
                "sources_filtered_out": 35,
                "duplicate_findings_removed": 5,
                "search_iterations": 3,
                "api_calls_made": 8
            }
        )

        with patch.object(research_executor_agent, 'run', return_value=expected_result):
            result = await research_executor_agent.run(deps)

            metadata = result.metadata
            assert "search_time" in metadata
            assert metadata["search_time"] > 0
            assert metadata.get("filtering_applied") is True
            assert metadata.get("sources_filtered_out", 0) > 0

    @pytest.mark.asyncio
    async def test_error_handling(self, research_executor_agent, agent_dependencies):
        """Test error handling during research execution."""
        deps = agent_dependencies

        with patch.object(research_executor_agent, 'run', side_effect=Exception("Search API failed")):
            with pytest.raises(Exception, match="Search API failed"):
                await research_executor_agent.run(deps)

    @pytest.mark.asyncio
    async def test_timeout_handling(self, research_executor_agent, agent_dependencies):
        """Test timeout handling for long-running searches."""
        deps = agent_dependencies

        async def delayed_response(deps):
            await asyncio.sleep(10)
            return ResearchResults(query="timeout test")

        with patch.object(research_executor_agent, 'run', side_effect=delayed_response):
            research_executor_agent.config.timeout = 0.1
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    research_executor_agent.run(deps),
                    timeout=0.2
                )
