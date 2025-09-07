"""
Comprehensive tests for the QueryTransformationAgent.
"""

import asyncio
import pytest
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.query_transformation import QueryTransformationAgent
from src.agents.base import ResearchDependencies, AgentConfiguration
from src.models.query_transformation import TransformedQuery
from src.models.api_models import APIKeys

class TestQueryTransformationAgent:
    """Test suite for QueryTransformationAgent."""

    @pytest.fixture
    async def agent_dependencies(self):
        """Create mock dependencies for testing."""
        deps = ResearchDependencies(
            http_client=AsyncMock(),
            api_keys=APIKeys(),
            research_state=ResearchState(
                request_id="test-456",
                user_id="test-user",
                session_id="test-session",
                user_query="How does machine learning work?",
                current_stage=ResearchStage.CLARIFICATION
            ),
            metadata=ResearchMetadata(),
            usage=None
        )
        return deps

    @pytest.fixture
    def transformation_agent(self, agent_dependencies):
        """Create a QueryTransformationAgent instance."""
        config = AgentConfiguration(
            agent_name="query_transformation",
            agent_type="transformation",
            max_retries=3,
            timeout_seconds=30.0,
            temperature=0.7
        )
        agent = QueryTransformationAgent(config=config)
        agent._deps = agent_dependencies
        return agent

    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = QueryTransformationAgent()
        assert agent.name == "query_transformation"
        assert agent.agent is not None
        assert agent.config is not None

    @pytest.mark.asyncio
    async def test_simple_query_transformation(self, transformation_agent, agent_dependencies):
        """Test transformation of simple queries."""
        mock_result = MagicMock()
        mock_result.data = TransformedQuery(
            original_query="How does machine learning work?",
            transformed_query="Explain the fundamental principles, algorithms, and applications of machine learning",
            supporting_questions=[
                "What are the main types of machine learning?",
                "How do neural networks learn from data?",
                "What are common machine learning algorithms?"
            ],
            transformation_rationale="Expanded query to cover key aspects of machine learning",
            specificity_score=0.8,
            missing_dimensions=[],
            clarification_responses={},
            transformation_metadata={"expansion_type": "comprehensive"}
        )

        with patch.object(transformation_agent.agent, 'run', return_value=mock_result):
            result = await transformation_agent.execute(agent_dependencies)

            assert isinstance(result, TransformedQuery)
            assert len(result.transformed_query) > len(result.original_query)
            assert len(result.supporting_questions) > 0
            assert result.specificity_score > 0.7

    @pytest.mark.asyncio
    async def test_complex_query_transformation(self, transformation_agent, agent_dependencies):
        """Test transformation of complex multi-part queries."""
        agent_dependencies.research_state.user_query = "Compare AI vs ML vs DL in terms of applications, performance, and future potential"

        mock_result = MagicMock()
        mock_result.data = TransformedQuery(
            original_query="Compare AI vs ML vs DL in terms of applications, performance, and future potential",
            transformed_query="Comprehensive comparison of Artificial Intelligence, Machine Learning, and Deep Learning across applications, performance metrics, and future potential",
            supporting_questions=[
                "What are the key differences between AI, ML, and DL?",
                "Which applications are best suited for each approach?",
                "How do they compare in terms of computational requirements?",
                "What are the future trends for each technology?"
            ],
            transformation_rationale="Structured comparison query with clear dimensions",
            specificity_score=0.9,
            missing_dimensions=["timeline", "industry focus"],
            clarification_responses={},
            transformation_metadata={"query_type": "comparison", "entities": 3}
        )

        with patch.object(transformation_agent.agent, 'run', return_value=mock_result):
            result = await transformation_agent.execute(agent_dependencies)

            assert result.specificity_score > 0.85
            assert len(result.supporting_questions) >= 3
            assert "comparison" in result.transformation_metadata.get("query_type", "")

    @pytest.mark.asyncio
    async def test_vague_query_transformation(self, transformation_agent, agent_dependencies):
        """Test transformation of vague queries."""
        agent_dependencies.research_state.user_query = "Tell me about technology"

        mock_result = MagicMock()
        mock_result.data = TransformedQuery(
            original_query="Tell me about technology",
            transformed_query="Overview of current technology trends including AI, cloud computing, IoT, and emerging technologies",
            supporting_questions=[
                "What are the most impactful technologies today?",
                "How is technology changing various industries?",
                "What emerging technologies show the most promise?"
            ],
            transformation_rationale="Query too broad, focused on current trends and impacts",
            specificity_score=0.4,
            missing_dimensions=["specific area", "timeframe", "application domain"],
            clarification_responses={},
            transformation_metadata={"needs_refinement": True}
        )

        with patch.object(transformation_agent.agent, 'run', return_value=mock_result):
            result = await transformation_agent.execute(agent_dependencies)

            assert result.specificity_score < 0.5
            assert len(result.missing_dimensions) > 0
            assert result.transformation_metadata.get("needs_refinement") is True

    @pytest.mark.asyncio
    async def test_query_with_temporal_context(self, transformation_agent, agent_dependencies):
        """Test transformation of queries with time context."""
        agent_dependencies.research_state.user_query = "Latest developments in quantum computing 2024"

        mock_result = MagicMock()
        mock_result.data = TransformedQuery(
            original_query="Latest developments in quantum computing 2024",
            transformed_query="Recent breakthroughs and developments in quantum computing technology during 2024",
            supporting_questions=[
                "What quantum computing milestones were achieved in 2024?",
                "Which companies made significant quantum advances?",
                "What are the practical applications emerging?"
            ],
            transformation_rationale="Focused on 2024 timeframe with emphasis on practical developments",
            specificity_score=0.85,
            missing_dimensions=[],
            clarification_responses={},
            transformation_metadata={"temporal_focus": "2024", "domain": "quantum_computing"}
        )

        with patch.object(transformation_agent.agent, 'run', return_value=mock_result):
            result = await transformation_agent.execute(agent_dependencies)

            assert "2024" in result.transformation_metadata.get("temporal_focus", "")
            assert result.specificity_score > 0.8

    @pytest.mark.asyncio
    async def test_supporting_questions_generation(self, transformation_agent, agent_dependencies):
        """Test generation of supporting questions."""
        mock_result = MagicMock()
        mock_result.data = TransformedQuery(
            original_query="How to implement neural networks?",
            transformed_query="Step-by-step guide to implementing neural networks from scratch",
            supporting_questions=[
                "What are the mathematical foundations needed?",
                "Which programming languages and frameworks to use?",
                "How to handle data preprocessing?",
                "How to optimize network architecture?",
                "How to evaluate model performance?"
            ],
            transformation_rationale="Practical implementation focus",
            specificity_score=0.75,
            missing_dimensions=["specific use case"],
            clarification_responses={},
            transformation_metadata={"focus": "implementation"}
        )

        with patch.object(transformation_agent.agent, 'run', return_value=mock_result):
            result = await transformation_agent.execute(agent_dependencies)

            assert len(result.supporting_questions) >= 3
            assert all(isinstance(q, str) and len(q) > 0 for q in result.supporting_questions)

    @pytest.mark.asyncio
    async def test_edge_case_empty_query(self, transformation_agent, agent_dependencies):
        """Test handling of empty query."""
        agent_dependencies.research_state.user_query = ""

        mock_result = MagicMock()
        mock_result.data = TransformedQuery(
            original_query="",
            transformed_query="",
            supporting_questions=[],
            transformation_rationale="No query provided",
            specificity_score=0.0,
            missing_dimensions=["query"],
            clarification_responses={},
            transformation_metadata={"error": "empty_query"}
        )

        with patch.object(transformation_agent.agent, 'run', return_value=mock_result):
            result = await transformation_agent.execute(agent_dependencies)

            assert result.specificity_score == 0.0
            assert result.transformation_metadata.get("error") == "empty_query"

    @pytest.mark.asyncio
    async def test_error_handling(self, transformation_agent, agent_dependencies):
        """Test error handling during transformation."""
        with patch.object(transformation_agent.agent, 'run', side_effect=Exception("Transformation failed")):
            with pytest.raises(Exception, match="Transformation failed"):
                await transformation_agent.execute(agent_dependencies)

    @pytest.mark.asyncio
    async def test_specificity_score_calculation(self, transformation_agent, agent_dependencies):
        """Test that specificity scores are properly calculated."""
        test_cases = [
            ("What is X?", 0.6),  # Simple what query
            ("How does X work in Y context during Z timeframe?", 0.9),  # Very specific
            ("Things", 0.1),  # Extremely vague
            ("Compare A, B, and C across dimensions D, E, F", 0.85)  # Structured comparison
        ]

        for query, expected_score_range in test_cases:
            agent_dependencies.research_state.user_query = query

            mock_result = MagicMock()
            mock_result.data = TransformedQuery(
                original_query=query,
                transformed_query=f"Transformed: {query}",
                supporting_questions=["Q1", "Q2"],
                transformation_rationale="Test",
                specificity_score=expected_score_range,
                missing_dimensions=[],
                clarification_responses={},
                transformation_metadata={}
            )

            with patch.object(transformation_agent.agent, 'run', return_value=mock_result):
                result = await transformation_agent.execute(agent_dependencies)
                assert 0.0 <= result.specificity_score <= 1.0

    @pytest.mark.asyncio
    async def test_clarification_responses_integration(self, transformation_agent, agent_dependencies):
        """Test integration of clarification responses."""
        agent_dependencies.research_state.user_query = "Machine learning applications"

        mock_result = MagicMock()
        mock_result.data = TransformedQuery(
            original_query="Machine learning applications",
            transformed_query="Machine learning applications in healthcare for diagnostic imaging",
            supporting_questions=["What ML models are used?", "What are accuracy rates?"],
            transformation_rationale="User clarified interest in healthcare diagnostics",
            specificity_score=0.9,
            missing_dimensions=[],
            clarification_responses={
                "domain": "healthcare",
                "specific_area": "diagnostic imaging"
            },
            transformation_metadata={"clarifications_applied": True}
        )

        with patch.object(transformation_agent.agent, 'run', return_value=mock_result):
            result = await transformation_agent.execute(agent_dependencies)

            assert len(result.clarification_responses) > 0
            assert result.clarification_responses.get("domain") == "healthcare"
            assert result.specificity_score > 0.85
