"""Tests for the updated ClarificationAgent and BriefGeneratorAgent."""

from unittest.mock import patch
import pytest
import pytest_asyncio
from pydantic import SecretStr

from src.open_deep_research_with_pydantic_ai.agents.base import ResearchDependencies
from src.open_deep_research_with_pydantic_ai.agents.brief_generator import (
    BriefGeneratorAgent,
    ResearchBrief as BriefGeneratorResearchBrief,
)
from src.open_deep_research_with_pydantic_ai.agents.clarification import (
    ClarificationAgent,
    ClarifyWithUser,
)
from src.open_deep_research_with_pydantic_ai.models.api_models import APIKeys, ResearchMetadata
from src.open_deep_research_with_pydantic_ai.models.research import (
    ResearchStage,
    ResearchState,
)


@pytest_asyncio.fixture
async def mock_dependencies():
    """Create mock research dependencies for testing."""
    from unittest.mock import AsyncMock

    # Create a research state
    research_state = ResearchState(
        request_id="test-request-123",
        user_id="test-user",
        session_id="test-session",
        user_query="Test query about quantum computing",
        current_stage=ResearchStage.CLARIFICATION,
    )

    # Create API keys
    api_keys = APIKeys(openai=SecretStr("test-openai-key"))

    # Create metadata
    metadata = ResearchMetadata()

    # Create HTTP client mock
    http_client = AsyncMock()

    return ResearchDependencies(
        http_client=http_client,
        api_keys=api_keys,
        research_state=research_state,
        metadata=metadata,
    )


class TestClarifyWithUser:
    """Test the ClarifyWithUser model."""

    def test_clarify_with_user_creation(self):
        """Test creating a valid ClarifyWithUser."""
        result = ClarifyWithUser(
            need_clarification=True,
            question="What specific aspects of quantum computing are you interested in?",
            verification="I will start research on quantum computing basics",
        )

        assert result.need_clarification is True
        assert result.question == "What specific aspects of quantum computing are you interested in?"
        assert result.verification == "I will start research on quantum computing basics"

    def test_clarify_with_user_defaults(self):
        """Test ClarifyWithUser with default values."""
        result = ClarifyWithUser(
            need_clarification=False,
        )

        assert result.need_clarification is False
        assert result.question == ""
        assert result.verification == ""


class TestClarificationAgent:
    """Test the ClarificationAgent class."""

    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = ClarificationAgent()

        assert agent.name == "clarification_agent"
        # Check output type is configured correctly
        assert agent.agent._output_type == ClarifyWithUser  # pyright: ignore[reportPrivateUsage]

    def test_system_prompt_content(self):
        """Test that system prompt contains required content."""
        agent = ClarificationAgent()

        # Test that the agent has the correct default system prompt method
        prompt = agent._get_default_system_prompt()

        # Check for key elements adapted from Langgraph approach
        assert "research assistant" in prompt
        assert "clarifying question" in prompt
        assert "need_clarification" in prompt
        assert "verification" in prompt

    @pytest.mark.asyncio
    async def test_assess_query_needs_clarification(self, mock_dependencies: ResearchDependencies):
        """Test assessing a query that needs clarification."""
        agent = ClarificationAgent()

        # Vague query that should need clarification
        with patch.object(agent, 'run', return_value=ClarifyWithUser(
            need_clarification=True,
            question="What specific aspects of AI are you interested in?",
            verification=""
        )) as mock_run:
            result = await agent.assess_query("Tell me about AI", mock_dependencies)

            assert result.need_clarification is True
            assert "specific aspects" in result.question
            assert result.verification == ""
            mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_ask_another_question_limit_reached(self, mock_dependencies: ResearchDependencies):
        """Test question limit logic."""
        agent = ClarificationAgent()

        # Set up metadata showing we've reached the question limit
        mock_dependencies.research_state.metadata = {
            "clarification_count": 2
        }

        result = await agent.should_ask_another_question(mock_dependencies, max_questions=2)
        assert result is False


class TestBriefGeneratorAgent:
    """Test the BriefGeneratorAgent class."""

    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = BriefGeneratorAgent()

        assert agent.name == "brief_generator_agent"
        # Check output type is configured correctly
        assert agent.agent._output_type == BriefGeneratorResearchBrief  # pyright: ignore[reportPrivateUsage]

    def test_system_prompt_content(self):
        """Test that system prompt contains required content."""
        agent = BriefGeneratorAgent()

        # Test that the agent has the correct default system prompt method
        prompt = agent._get_default_system_prompt()

        # Check for key elements in the system prompt
        assert "messages that have been exchanged" in prompt
        assert "confidence score" in prompt
        assert "Missing Aspects" in prompt

    @pytest.mark.asyncio
    async def test_generate_from_conversation_basic(self, mock_dependencies: ResearchDependencies):
        """Test generating research brief from basic conversation."""
        agent = BriefGeneratorAgent()

        # Set up conversation history
        mock_dependencies.research_state.metadata = {
            "conversation_messages": ["I want to learn about quantum computing"]
        }

        # Mock the agent run to return a brief
        with patch.object(agent, 'run', return_value=BriefGeneratorResearchBrief(
            brief="I want to research quantum computing basics, including fundamental concepts, applications, and current developments.",
            confidence_score=0.8,
            missing_aspects=[]
        )) as mock_run:
            result = await agent.generate_from_conversation(mock_dependencies)

            assert result.brief.startswith("I want to research quantum computing")
            assert result.confidence_score == 0.8
            assert len(result.missing_aspects) == 0
            mock_run.assert_called_once()

            # Check that metadata is updated
            assert mock_dependencies.research_state.metadata["research_brief_text"] == result.brief
            assert mock_dependencies.research_state.metadata["research_brief_confidence"] == 0.8

    @pytest.mark.asyncio
    async def test_generate_from_conversation_no_history(self, mock_dependencies: ResearchDependencies):
        """Test generating research brief with no conversation history."""
        agent = BriefGeneratorAgent()

        # No conversation history, just the original query
        mock_dependencies.research_state.user_query = "Tell me about blockchain technology"
        mock_dependencies.research_state.metadata = {}

        with patch.object(agent, 'run', return_value=BriefGeneratorResearchBrief(
            brief="I want to research blockchain technology comprehensively.",
            confidence_score=0.6,
            missing_aspects=["specific use cases", "timeframe"]
        )) as mock_run:
            result = await agent.generate_from_conversation(mock_dependencies)

            assert result.confidence_score == 0.6
            assert len(result.missing_aspects) == 2
            # Check that metadata is updated
            assert mock_dependencies.research_state.metadata["research_brief_text"] == result.brief
            assert mock_dependencies.research_state.metadata["research_brief_confidence"] == 0.6
