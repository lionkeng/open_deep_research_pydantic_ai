"""
Comprehensive tests for the ClarificationAgent.
"""

import asyncio
import json
import pytest
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.clarification import ClarificationAgent, ClarifyWithUser
from src.agents.base import ResearchDependencies, AgentConfiguration
from src.models.api_models import APIKeys, ResearchMetadata
from src.models.core import ResearchState, ResearchStage


class TestClarificationAgent:
    """Test suite for ClarificationAgent."""

    @pytest.fixture
    async def agent_dependencies(self):
        """Create mock dependencies for testing."""
        deps = ResearchDependencies(
            http_client=AsyncMock(),
            api_keys=APIKeys(),
            research_state=ResearchState(
                request_id="test-123",
                user_id="test-user",
                session_id="test-session",
                user_query="What is AI?",
                current_stage=ResearchStage.CLARIFICATION
            ),
            metadata=ResearchMetadata(),
            usage=None
        )
        return deps

    @pytest.fixture
    def clarification_agent(self, agent_dependencies):
        """Create a ClarificationAgent instance."""
        config = AgentConfiguration(
            max_retries=3,
            timeout_seconds=30.0,
            temperature=0.7
        )
        agent = ClarificationAgent(config=config)
        agent._deps = agent_dependencies
        return agent

    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = ClarificationAgent()
        assert agent.name == "clarification_agent"
        assert agent.agent is not None
        assert agent.config is not None

    @pytest.mark.asyncio
    async def test_needs_clarification_detection(self, clarification_agent, agent_dependencies):
        """Test detection of queries needing clarification."""
        # Mock agent to return clarification needed
        mock_result = MagicMock()
        mock_result.data = ClarifyWithUser(
            clarification_needed=True,
            clarification_question="Which aspect of AI interests you most?",
            original_query="What is AI?",
            transformed_query="What are the fundamental principles and applications of artificial intelligence?",
            confidence_score=0.6,
            reasoning="Query is too broad and could refer to multiple aspects"
        )

        with patch.object(clarification_agent.agent, 'run', return_value=mock_result):
            result = await clarification_agent.execute(agent_dependencies)

            assert isinstance(result, ClarifyWithUser)
            assert result.clarification_needed is True
            assert "AI" in result.clarification_question
            assert result.confidence_score < 0.8

    @pytest.mark.asyncio
    async def test_no_clarification_needed(self, clarification_agent, agent_dependencies):
        """Test queries that don't need clarification."""
        agent_dependencies.research_state.user_query = "What is the current price of Bitcoin in USD?"

        mock_result = MagicMock()
        mock_result.data = ClarifyWithUser(
            clarification_needed=False,
            clarification_question="",
            original_query="What is the current price of Bitcoin in USD?",
            transformed_query="What is the current price of Bitcoin in USD?",
            confidence_score=0.95,
            reasoning="Query is specific and clear"
        )

        with patch.object(clarification_agent.agent, 'run', return_value=mock_result):
            result = await clarification_agent.execute(agent_dependencies)

            assert isinstance(result, ClarifyWithUser)
            assert result.clarification_needed is False
            assert result.confidence_score > 0.9

    @pytest.mark.asyncio
    async def test_edge_case_empty_query(self, clarification_agent, agent_dependencies):
        """Test handling of empty query."""
        agent_dependencies.research_state.user_query = ""

        mock_result = MagicMock()
        mock_result.data = ClarifyWithUser(
            clarification_needed=True,
            clarification_question="Could you please provide a research question?",
            original_query="",
            transformed_query="",
            confidence_score=0.0,
            reasoning="No query provided"
        )

        with patch.object(clarification_agent.agent, 'run', return_value=mock_result):
            result = await clarification_agent.execute(agent_dependencies)

            assert result.clarification_needed is True
            assert result.confidence_score == 0.0

    @pytest.mark.asyncio
    async def test_ambiguous_query_handling(self, clarification_agent, agent_dependencies):
        """Test handling of ambiguous queries."""
        agent_dependencies.research_state.user_query = "Python performance"

        mock_result = MagicMock()
        mock_result.data = ClarifyWithUser(
            clarification_needed=True,
            clarification_question="Are you asking about Python programming language performance or something else?",
            original_query="Python performance",
            transformed_query="Python programming language performance optimization techniques",
            confidence_score=0.5,
            reasoning="Term 'Python' could refer to programming language or other contexts"
        )

        with patch.object(clarification_agent.agent, 'run', return_value=mock_result):
            result = await clarification_agent.execute(agent_dependencies)

            assert result.clarification_needed is True
            assert "Python" in result.clarification_question
            assert result.confidence_score < 0.7

    @pytest.mark.asyncio
    async def test_error_handling(self, clarification_agent, agent_dependencies):
        """Test error handling during clarification."""
        with patch.object(clarification_agent.agent, 'run', side_effect=Exception("API Error")):
            with pytest.raises(Exception, match="API Error"):
                await clarification_agent.execute(agent_dependencies)

    @pytest.mark.asyncio
    async def test_timeout_handling(self, clarification_agent, agent_dependencies):
        """Test timeout handling."""
        async def delayed_response():
            await asyncio.sleep(5)
            return MagicMock()

        with patch.object(clarification_agent.agent, 'run', side_effect=delayed_response):
            clarification_agent.config.timeout_seconds = 0.1
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    clarification_agent.execute(agent_dependencies),
                    timeout=0.2
                )

    @pytest.mark.asyncio
    async def test_result_validation(self, clarification_agent, agent_dependencies):
        """Test that results are properly validated."""
        mock_result = MagicMock()
        # Create invalid result missing required fields
        mock_result.data = {"invalid": "data"}

        with patch.object(clarification_agent.agent, 'run', return_value=mock_result):
            with pytest.raises(Exception):  # Should raise validation error
                await clarification_agent.execute(agent_dependencies)

    @pytest.mark.asyncio
    async def test_confidence_score_boundaries(self, clarification_agent, agent_dependencies):
        """Test confidence score remains within valid boundaries."""
        test_cases = [
            ("very clear query", 0.95, False),
            ("somewhat clear", 0.75, False),
            ("unclear query", 0.4, True),
            ("very ambiguous", 0.1, True)
        ]

        for query, confidence, needs_clarification in test_cases:
            agent_dependencies.research_state.user_query = query

            mock_result = MagicMock()
            mock_result.data = ClarifyWithUser(
                clarification_needed=needs_clarification,
                clarification_question="Test question" if needs_clarification else "",
                original_query=query,
                transformed_query=f"Transformed: {query}",
                confidence_score=confidence,
                reasoning="Test reasoning"
            )

            with patch.object(clarification_agent.agent, 'run', return_value=mock_result):
                result = await clarification_agent.execute(agent_dependencies)

                assert 0.0 <= result.confidence_score <= 1.0
                assert result.clarification_needed == needs_clarification

    @pytest.mark.asyncio
    async def test_query_transformation(self, clarification_agent, agent_dependencies):
        """Test that queries are properly transformed."""
        agent_dependencies.research_state.user_query = "AI in healthcare"

        mock_result = MagicMock()
        mock_result.data = ClarifyWithUser(
            clarification_needed=False,
            clarification_question="",
            original_query="AI in healthcare",
            transformed_query="Applications and impact of artificial intelligence in healthcare industry including diagnostics, treatment planning, and patient care",
            confidence_score=0.85,
            reasoning="Query expanded with relevant context"
        )

        with patch.object(clarification_agent.agent, 'run', return_value=mock_result):
            result = await clarification_agent.execute(agent_dependencies)

            assert len(result.transformed_query) > len(result.original_query)
            assert "healthcare" in result.transformed_query.lower()
            assert result.original_query == "AI in healthcare"
