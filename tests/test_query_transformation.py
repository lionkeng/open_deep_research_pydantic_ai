"""Tests for the QueryTransformationAgent and related functionality."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime

from open_deep_research_with_pydantic_ai.agents.query_transformation import QueryTransformationAgent
from open_deep_research_with_pydantic_ai.agents.clarification import ClarificationAgent
from open_deep_research_with_pydantic_ai.agents.base import ResearchDependencies
from open_deep_research_with_pydantic_ai.models.research import ResearchState, TransformedQuery
from open_deep_research_with_pydantic_ai.models.api_models import APIKeys, ResearchMetadata
from pydantic_ai.usage import RunUsage
import httpx


class TestQueryTransformationAgent:
    """Test cases for QueryTransformationAgent."""

    @pytest.fixture
    def agent(self):
        """Create a QueryTransformationAgent instance for testing."""
        return QueryTransformationAgent()

    @pytest.fixture
    def sample_clarification_responses(self):
        """Sample clarification responses for testing."""
        return {
            "What time period are you interested in?": "The last 10 years, 2014-2024",
            "Are you focusing on a specific region?": "North America and Europe",
            "What aspect interests you most?": "Environmental impact and economic effects"
        }

    @pytest.fixture
    def sample_deps(self):
        """Create sample research dependencies."""
        return ResearchDependencies(
            http_client=Mock(),
            api_keys=APIKeys(),
            research_state=ResearchState(
                request_id="test-123",
                user_id="test-user",
                session_id="test-session",
                user_query="Test query"
            ),
            metadata=ResearchMetadata(),
            usage=RunUsage()
        )

    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.name == "query_transformation_agent"
        assert agent._output_type == TransformedQuery
        assert agent.agent is not None

    def test_system_prompt_structure(self, agent):
        """Test that the system prompt contains required elements."""
        prompt = agent._get_default_system_prompt()

        required_elements = [
            "research query transformation",
            "original research query",
            "clarification responses",
            "specific, focused research questions",
            "specificity scoring",
            "0.0-1.0"
        ]

        for element in required_elements:
            assert element in prompt.lower()

    def test_build_transformation_prompt(self, agent, sample_clarification_responses):
        """Test building the transformation prompt."""
        original_query = "climate change effects"
        conversation_context = ["User mentioned Arctic focus", "Previous discussion about ice caps"]

        prompt = agent._build_transformation_prompt(
            original_query,
            sample_clarification_responses,
            conversation_context
        )

        assert "Original Query: climate change effects" in prompt
        assert "Clarification Responses:" in prompt
        assert "What time period are you interested in?" in prompt
        assert "The last 10 years, 2014-2024" in prompt
        assert "Conversation Context:" in prompt
        assert "User mentioned Arctic focus" in prompt

    def test_create_fallback_transformation(self, agent, sample_clarification_responses):
        """Test creating a fallback transformation."""
        original_query = "artificial intelligence impact"

        result = agent._create_fallback_transformation(
            original_query,
            sample_clarification_responses
        )

        assert isinstance(result, TransformedQuery)
        assert result.original_query == original_query
        assert len(result.transformed_query) > len(original_query)
        assert result.specificity_score == 0.4
        assert result.transformation_metadata["method"] == "fallback"
        assert "during The last 10 years" in result.transformed_query or "2014-2024" in result.transformed_query

    def test_enhance_transformation_metadata(self, agent, sample_clarification_responses):
        """Test enhancing transformation metadata."""
        original_query = "AI impact"
        transformed_query = TransformedQuery(
            original_query=original_query,
            transformed_query="What are the specific economic impacts of AI on North American jobs from 2014-2024?",
            transformation_rationale="Test rationale",
            specificity_score=0.8,
            clarification_responses=sample_clarification_responses
        )

        result = agent._enhance_transformation_metadata(transformed_query, original_query)

        assert "original_word_count" in result.transformation_metadata
        assert "transformed_word_count" in result.transformation_metadata
        assert "word_overlap_ratio" in result.transformation_metadata
        assert "transformation_timestamp" in result.transformation_metadata
        assert result.transformation_metadata["agent"] == "query_transformation_agent"

    def test_basic_transformation_validation(self, agent):
        """Test basic transformation validation."""
        transformation = TransformedQuery(
            original_query="climate change",
            transformed_query="What are the economic impacts of climate change on agriculture in North America from 2010-2020?",
            transformation_rationale="Added specificity for region, timeframe, and impact type",
            specificity_score=0.8,
            clarification_responses={}
        )

        validation = agent._basic_transformation_validation(transformation)

        assert "scores" in validation
        assert "overall_score" in validation
        assert validation["validation_method"] == "basic_heuristic"
        assert 1 <= validation["overall_score"] <= 10

        # Check individual scores
        scores = validation["scores"]
        assert "specificity" in scores
        assert "researchability" in scores
        assert "intent_preservation" in scores
        assert "clarity" in scores

    @pytest.mark.asyncio
    async def test_transform_query_with_deps(self, agent, sample_clarification_responses, sample_deps):
        """Test query transformation with dependencies."""
        original_query = "renewable energy adoption"

        # Mock the agent's run method
        mock_result = TransformedQuery(
            original_query=original_query,
            transformed_query="How has renewable energy adoption progressed in North America and Europe from 2014-2024?",
            supporting_questions=["What policy changes drove adoption?"],
            transformation_rationale="Added geographical and temporal specificity",
            specificity_score=0.8,
            clarification_responses=sample_clarification_responses
        )

        with patch.object(agent, 'run', return_value=mock_result):
            result = await agent.transform_query(
                original_query=original_query,
                clarification_responses=sample_clarification_responses,
                deps=sample_deps
            )

        assert isinstance(result, TransformedQuery)
        assert result.original_query == original_query
        assert result.clarification_responses == sample_clarification_responses
        assert "transformation_timestamp" in result.transformation_metadata

    @pytest.mark.asyncio
    async def test_transform_query_without_deps(self, agent, sample_clarification_responses):
        """Test query transformation without dependencies (fallback mode)."""
        original_query = "machine learning applications"

        result = await agent.transform_query(
            original_query=original_query,
            clarification_responses=sample_clarification_responses
        )

        assert isinstance(result, TransformedQuery)
        assert result.original_query == original_query
        assert result.transformation_metadata.get("method") == "fallback"

    @pytest.mark.asyncio
    async def test_transform_query_error_handling(self, agent, sample_clarification_responses, sample_deps):
        """Test error handling in transform_query."""
        original_query = "test query"

        # Mock the run method to raise an exception
        with patch.object(agent, 'run', side_effect=Exception("Test error")):
            result = await agent.transform_query(
                original_query=original_query,
                clarification_responses=sample_clarification_responses,
                deps=sample_deps
            )

        assert isinstance(result, TransformedQuery)
        assert result.transformation_metadata.get("error") == "AI transformation failed"

    @pytest.mark.asyncio
    async def test_validate_transformation_quality(self, agent, sample_deps):
        """Test transformation quality validation."""
        transformation = TransformedQuery(
            original_query="climate change",
            transformed_query="What are climate change effects on Arctic wildlife from 2010-2020?",
            supporting_questions=["How have polar bear populations changed?"],
            transformation_rationale="Added temporal and geographical specificity",
            specificity_score=0.9,
            clarification_responses={}
        )

        validation = await agent.validate_transformation_quality(transformation, sample_deps)

        assert isinstance(validation, dict)
        assert "overall_score" in validation

    def test_fallback_with_empty_responses(self, agent):
        """Test fallback transformation with empty responses."""
        original_query = "blockchain technology"
        empty_responses = {}

        result = agent._create_fallback_transformation(original_query, empty_responses)

        assert result.original_query == original_query
        assert result.transformed_query == original_query  # Should remain unchanged
        assert result.transformation_metadata["method"] == "fallback"

    def test_fallback_with_irrelevant_responses(self, agent):
        """Test fallback transformation with irrelevant responses."""
        original_query = "quantum computing"
        irrelevant_responses = {
            "Do you like this topic?": "yes",
            "Any preferences?": "no",
            "Other questions?": "n/a"
        }

        result = agent._create_fallback_transformation(original_query, irrelevant_responses)

        # Should only include meaningful responses
        assert result.original_query == original_query
        # Query should remain mostly unchanged since responses are irrelevant
        assert len(result.transformed_query) <= len(original_query) + 10


class TestClarificationIntegration:
    """Test integration between ClarificationAgent and QueryTransformationAgent."""

    @pytest.fixture
    def clarification_agent(self):
        """Create a ClarificationAgent instance."""
        return ClarificationAgent()

    @pytest.fixture
    def transformation_agent(self):
        """Create a QueryTransformationAgent instance."""
        return QueryTransformationAgent()

    @pytest.fixture
    def sample_deps(self):
        """Create sample research dependencies."""
        return ResearchDependencies(
            http_client=Mock(),
            api_keys=APIKeys(),
            research_state=ResearchState(
                request_id="test-integration-123",
                user_id="test-user",
                session_id="test-session",
                user_query="Test integration query"
            ),
            metadata=ResearchMetadata(),
            usage=RunUsage()
        )

    def test_create_basic_transformation(self, clarification_agent):
        """Test basic transformation creation in clarification agent."""
        original_query = "renewable energy"
        clarification_responses = {
            "What time period?": "2020-2024",
            "Which region?": "Europe",
            "What aspect?": "economic impact"
        }

        result = clarification_agent._create_basic_transformation(
            original_query, clarification_responses
        )

        assert isinstance(result, TransformedQuery)
        assert result.original_query == original_query
        assert "during 2020-2024" in result.transformed_query
        assert "in Europe" in result.transformed_query
        assert result.specificity_score == 0.5
        assert result.transformation_metadata["method"] == "basic_fallback"

    @pytest.mark.asyncio
    async def test_process_clarification_responses_integration(self, clarification_agent, sample_deps):
        """Test processing clarification responses with transformation."""
        original_query = "artificial intelligence ethics"
        clarification_responses = {
            "What specific application?": "healthcare AI systems",
            "What timeframe?": "next 5 years",
            "What ethical concerns?": "privacy and bias"
        }

        # Mock the transformation agent in coordinator
        mock_transformation_agent = Mock()
        mock_transformed_query = TransformedQuery(
            original_query=original_query,
            transformed_query="What privacy and bias challenges will healthcare AI systems face in the next 5 years?",
            supporting_questions=["How can these ethical issues be addressed?"],
            transformation_rationale="Focused on specific application, timeframe, and ethical aspects",
            specificity_score=0.9,
            clarification_responses=clarification_responses
        )
        mock_transformation_agent.transform_query = AsyncMock(return_value=mock_transformed_query)

        from open_deep_research_with_pydantic_ai.agents.base import coordinator
        with patch.object(coordinator, 'agents', {"query_transformation_agent": mock_transformation_agent}):
            result = await clarification_agent.process_clarification_responses_with_transformation(
                original_query, clarification_responses, sample_deps
            )

        assert isinstance(result, TransformedQuery)
        assert result.transformed_query == mock_transformed_query.transformed_query
        assert sample_deps.research_state.clarified_query == result.transformed_query
        assert "transformed_query" in sample_deps.research_state.metadata

    @pytest.mark.asyncio
    async def test_process_clarification_responses_no_agent(self, clarification_agent, sample_deps):
        """Test processing when transformation agent is not available."""
        original_query = "machine learning applications"
        clarification_responses = {
            "What domain?": "healthcare",
            "What timeframe?": "current applications"
        }

        # Mock coordinator with no transformation agent
        from open_deep_research_with_pydantic_ai.agents.base import coordinator
        with patch.object(coordinator, 'agents', {}):
            result = await clarification_agent.process_clarification_responses_with_transformation(
                original_query, clarification_responses, sample_deps
            )

        assert isinstance(result, TransformedQuery)
        assert result.transformation_metadata["method"] == "basic_fallback"
        assert result.transformation_metadata["reason"] == "transformation_agent_unavailable"

    @pytest.mark.asyncio
    async def test_process_clarification_responses_error_handling(self, clarification_agent, sample_deps):
        """Test error handling in clarification response processing."""
        original_query = "test query"
        clarification_responses = {"test": "response"}

        # Mock transformation agent that raises an error
        mock_transformation_agent = Mock()
        mock_transformation_agent.transform_query = AsyncMock(side_effect=Exception("Test error"))

        from open_deep_research_with_pydantic_ai.agents.base import coordinator
        with patch.object(coordinator, 'agents', {"query_transformation_agent": mock_transformation_agent}):
            result = await clarification_agent.process_clarification_responses_with_transformation(
                original_query, clarification_responses, sample_deps
            )

        assert isinstance(result, TransformedQuery)
        assert result.transformation_metadata["method"] == "basic_fallback"


if __name__ == "__main__":
    pytest.main([__file__])
