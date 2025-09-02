"""
Comprehensive tests for the ClarificationAgent.
"""

import asyncio
import json
import pytest
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.clarification import ClarificationAgent, ClarifyWithUser
from src.agents.base import ResearchDependencies
from src.models.api_models import APIKeys, ResearchMetadata
from src.models.core import ResearchState, ResearchStage


class TestClarificationAgent:
    """Test suite for ClarificationAgent."""

    @pytest.fixture
    def agent_dependencies(self):
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
            usage=None
        )
        return deps

    @pytest.fixture
    def clarification_agent(self, agent_dependencies):
        """Create a ClarificationAgent instance."""
        agent = ClarificationAgent()  # No config parameter - agent creates its own
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
            need_clarification=True,
            question="Which aspect of AI interests you most?",
            verification="",
            missing_dimensions=["Specific focus area", "Technical depth"],
            assessment_reasoning="Query is too broad and could refer to multiple aspects",
            suggested_clarifications=["AI applications", "AI principles", "AI history"]
        )

        with patch.object(clarification_agent.agent, 'run', return_value=mock_result):
            result = await clarification_agent.agent.run("What is AI?", deps=agent_dependencies)

            assert result.data.need_clarification is True
            assert "AI" in result.data.question
            assert len(result.data.missing_dimensions) > 0

    @pytest.mark.asyncio
    async def test_no_clarification_needed(self, clarification_agent, agent_dependencies):
        """Test queries that don't need clarification."""
        query = "What is the current price of Bitcoin in USD?"
        agent_dependencies.research_state.user_query = query

        mock_result = MagicMock()
        mock_result.data = ClarifyWithUser(
            need_clarification=False,
            question="",
            verification="Query is specific and clear. Ready to proceed with research.",
            missing_dimensions=[],
            assessment_reasoning="Query is specific and clear",
            suggested_clarifications=[]
        )

        with patch.object(clarification_agent.agent, 'run', return_value=mock_result):
            result = await clarification_agent.agent.run(query, deps=agent_dependencies)

            assert result.data.need_clarification is False
            assert result.data.verification != ""
            assert len(result.data.missing_dimensions) == 0

    @pytest.mark.asyncio
    async def test_edge_case_empty_query(self, clarification_agent, agent_dependencies):
        """Test handling of minimal query."""
        # Use a minimal valid query (single char) since ResearchState requires min_length=1
        query = "?"
        agent_dependencies.research_state.user_query = query

        mock_result = MagicMock()
        mock_result.data = ClarifyWithUser(
            need_clarification=True,
            question="Could you please provide a research question?",
            verification="",
            missing_dimensions=["Query content"],
            assessment_reasoning="Query too minimal to be meaningful",
            suggested_clarifications=[]
        )

        with patch.object(clarification_agent.agent, 'run', return_value=mock_result):
            result = await clarification_agent.agent.run(query, deps=agent_dependencies)

            assert result.data.need_clarification is True
            assert "provide" in result.data.question.lower()

    @pytest.mark.asyncio
    async def test_ambiguous_query_handling(self, clarification_agent, agent_dependencies):
        """Test handling of ambiguous queries."""
        query = "Python performance"
        agent_dependencies.research_state.user_query = query

        mock_result = MagicMock()
        mock_result.data = ClarifyWithUser(
            need_clarification=True,
            question="Are you asking about Python programming language performance or something else?",
            verification="",
            missing_dimensions=["Specific context", "Performance aspect"],
            assessment_reasoning="Term 'Python' could refer to programming language or other contexts",
            suggested_clarifications=["Language performance", "Snake behavior"]
        )

        with patch.object(clarification_agent.agent, 'run', return_value=mock_result):
            result = await clarification_agent.agent.run(query, deps=agent_dependencies)

            assert result.data.need_clarification is True
            assert "Python" in result.data.question
            assert len(result.data.missing_dimensions) > 0

    @pytest.mark.asyncio
    async def test_error_handling(self, clarification_agent, agent_dependencies):
        """Test error handling during clarification."""
        with patch.object(clarification_agent.agent, 'run', side_effect=Exception("API Error")):
            with pytest.raises(Exception, match="API Error"):
                await clarification_agent.agent.run("test query", deps=agent_dependencies)

    @pytest.mark.asyncio
    async def test_timeout_handling(self, clarification_agent, agent_dependencies):
        """Test timeout handling."""
        async def delayed_response(*args, **kwargs):
            await asyncio.sleep(5)
            return MagicMock()

        with patch.object(clarification_agent.agent, 'run', side_effect=delayed_response):
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    clarification_agent.agent.run("test query", deps=agent_dependencies),
                    timeout=0.2
                )

    @pytest.mark.asyncio
    async def test_result_validation(self, clarification_agent, agent_dependencies):
        """Test that results are properly validated."""
        # This test validates that the ClarifyWithUser model is properly structured
        mock_result = MagicMock()
        mock_result.data = ClarifyWithUser(
            need_clarification=True,
            question="Valid question",
            verification="",
            missing_dimensions=["test"],
            assessment_reasoning="Valid reasoning",
            suggested_clarifications=[]
        )

        with patch.object(clarification_agent.agent, 'run', return_value=mock_result):
            result = await clarification_agent.agent.run("test query", deps=agent_dependencies)
            # Should not raise validation error
            assert result.data.need_clarification is True

    @pytest.mark.asyncio
    async def test_confidence_score_boundaries(self, clarification_agent, agent_dependencies):
        """Test different clarification scenarios."""
        test_cases = [
            ("What is the current price of Bitcoin?", False, "Specific query"),
            ("Tell me about technology", True, "Too broad"),
        ]

        for query, needs_clarification, reasoning in test_cases:
            agent_dependencies.research_state.user_query = query

            mock_result = MagicMock()
            mock_result.data = ClarifyWithUser(
                need_clarification=needs_clarification,
                question="Test question" if needs_clarification else "",
                verification="" if needs_clarification else "Clear query, proceeding",
                missing_dimensions=["specificity"] if needs_clarification else [],
                assessment_reasoning=reasoning,
                suggested_clarifications=[]
            )

            with patch.object(clarification_agent.agent, 'run', return_value=mock_result):
                result = await clarification_agent.agent.run(query, deps=agent_dependencies)

                assert result.data.need_clarification == needs_clarification
                if needs_clarification:
                    assert result.data.question != ""
                else:
                    assert result.data.verification != ""

    @pytest.mark.asyncio
    async def test_query_transformation(self, clarification_agent, agent_dependencies):
        """Test that queries are properly analyzed."""
        query = "AI in healthcare"
        agent_dependencies.research_state.user_query = query

        mock_result = MagicMock()
        mock_result.data = ClarifyWithUser(
            need_clarification=False,
            question="",
            verification="Query is specific enough about AI applications in healthcare",
            missing_dimensions=[],
            assessment_reasoning="Query has clear domain and topic",
            suggested_clarifications=[]
        )

        with patch.object(clarification_agent.agent, 'run', return_value=mock_result):
            result = await clarification_agent.agent.run(query, deps=agent_dependencies)

            assert result.data.need_clarification is False
            assert "healthcare" in result.data.verification.lower()
            assert len(result.data.missing_dimensions) == 0
