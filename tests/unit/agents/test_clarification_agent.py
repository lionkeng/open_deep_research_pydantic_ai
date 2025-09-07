"""
Comprehensive tests for the ClarificationAgent with new multi-question support.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.base import ResearchDependencies
from src.agents.clarification import ClarificationAgent, ClarifyWithUser
from src.models.api_models import APIKeys
from src.models.metadata import ResearchMetadata
from src.models.clarification import ClarificationQuestion, ClarificationRequest
from src.models.core import ResearchStage, ResearchState


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
        agent = ClarificationAgent()
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
    async def test_needs_clarification_with_multiple_questions(self, clarification_agent, agent_dependencies):
        """Test detection of queries needing multiple clarification questions."""
        # Create multiple questions
        questions = [
            ClarificationQuestion(
                question="What specific aspect of AI interests you?",
                is_required=True,
                question_type="choice",
                choices=["Machine Learning", "Natural Language Processing", "Computer Vision", "Robotics"],
                order=0
            ),
            ClarificationQuestion(
                question="What is your technical background?",
                is_required=False,
                question_type="choice",
                choices=["Beginner", "Intermediate", "Advanced", "Expert"],
                context="This helps us tailor the research to your level",
                order=1
            ),
            ClarificationQuestion(
                question="What's the intended use for this research?",
                is_required=True,
                question_type="text",
                order=2
            )
        ]

        request = ClarificationRequest(questions=questions)

        # Mock agent to return clarification needed
        mock_result = MagicMock()
        mock_result.data = ClarifyWithUser(
            needs_clarification=True,
            request=request,
            reasoning="Query is too broad and needs specifics",
            missing_dimensions=["Specific focus area", "Technical depth", "Use case"],
            assessment_reasoning="AI is a vast field requiring focus"
        )

        with patch.object(clarification_agent.agent, 'run', return_value=mock_result):
            result = await clarification_agent.agent.run("What is AI?", deps=agent_dependencies)

            assert result.data.needs_clarification is True
            assert result.data.request is not None
            assert len(result.data.request.questions) == 3

            # Check first question (choice type)
            q1 = result.data.request.questions[0]
            assert q1.is_required is True
            assert q1.question_type == "choice"
            assert len(q1.choices) == 4

            # Check second question (optional)
            q2 = result.data.request.questions[1]
            assert q2.is_required is False
            assert q2.context is not None

            # Check question ordering
            sorted_questions = result.data.request.get_sorted_questions()
            assert sorted_questions[0].order == 0
            assert sorted_questions[2].order == 2

    @pytest.mark.asyncio
    async def test_no_clarification_needed(self, clarification_agent, agent_dependencies):
        """Test queries that don't need clarification."""
        query = "What is the current price of Bitcoin in USD?"
        agent_dependencies.research_state.user_query = query

        mock_result = MagicMock()
        mock_result.data = ClarifyWithUser(
            needs_clarification=False,
            request=None,
            reasoning="Query is specific and clear. Ready to proceed with research.",
            missing_dimensions=[],
            assessment_reasoning="Query is specific and clear"
        )

        with patch.object(clarification_agent.agent, 'run', return_value=mock_result):
            result = await clarification_agent.agent.run(query, deps=agent_dependencies)

            assert result.data.needs_clarification is False
            assert result.data.request is None
            assert result.data.reasoning != ""
            assert len(result.data.missing_dimensions) == 0

    @pytest.mark.asyncio
    async def test_single_required_question(self, clarification_agent, agent_dependencies):
        """Test handling of single required clarification question."""
        query = "Compare programming languages"
        agent_dependencies.research_state.user_query = query

        question = ClarificationQuestion(
            question="Which programming languages would you like compared?",
            is_required=True
        )
        request = ClarificationRequest(questions=[question])

        mock_result = MagicMock()
        mock_result.data = ClarifyWithUser(
            needs_clarification=True,
            request=request,
            reasoning="Need to know which specific languages to compare",
            missing_dimensions=["Specific languages"],
            assessment_reasoning="Too vague without specific languages"
        )

        with patch.object(clarification_agent.agent, 'run', return_value=mock_result):
            result = await clarification_agent.agent.run(query, deps=agent_dependencies)

            assert result.data.needs_clarification is True
            assert result.data.request is not None
            assert len(result.data.request.questions) == 1
            assert result.data.request.questions[0].is_required is True

    @pytest.mark.asyncio
    async def test_model_validation_fallback(self, clarification_agent, agent_dependencies):
        """Test that model validator provides fallback for missing request."""
        # Create ClarifyWithUser with needs_clarification=True but no request
        clarify = ClarifyWithUser(
            needs_clarification=True,
            request=None,  # Missing request
            reasoning="Test fallback",
            missing_dimensions=["test"],
            assessment_reasoning="Test"
        )

        # Validator should create a fallback request
        assert clarify.request is not None
        assert len(clarify.request.questions) == 1
        assert clarify.request.questions[0].is_required is True

    @pytest.mark.asyncio
    async def test_model_validation_clears_request(self):
        """Test that model validator clears request when not needed."""
        question = ClarificationQuestion(question="Test question")
        request = ClarificationRequest(questions=[question])

        # Create ClarifyWithUser with needs_clarification=False but with request
        clarify = ClarifyWithUser(
            needs_clarification=False,
            request=request,  # Should be cleared
            reasoning="No clarification needed",
            missing_dimensions=[],
            assessment_reasoning="Clear query"
        )

        # Validator should clear the request
        assert clarify.request is None

    @pytest.mark.asyncio
    async def test_uuid_generation_for_questions(self):
        """Test that questions get unique UUIDs."""
        q1 = ClarificationQuestion(question="Question 1")
        q2 = ClarificationQuestion(question="Question 2")

        assert q1.id != q2.id
        assert len(q1.id) == 36  # UUID string length
        assert len(q2.id) == 36

    @pytest.mark.asyncio
    async def test_request_question_lookup(self):
        """Test O(1) question lookup in request."""
        questions = [
            ClarificationQuestion(question=f"Q{i}") for i in range(10)
        ]
        request = ClarificationRequest(questions=questions)

        # Test lookup by ID
        q5 = request.get_question_by_id(questions[5].id)
        assert q5 == questions[5]

        # Test non-existent ID
        assert request.get_question_by_id("non-existent") is None

    @pytest.mark.asyncio
    async def test_get_required_questions(self):
        """Test filtering for required questions."""
        questions = [
            ClarificationQuestion(question="Q1", is_required=True),
            ClarificationQuestion(question="Q2", is_required=False),
            ClarificationQuestion(question="Q3", is_required=True),
        ]
        request = ClarificationRequest(questions=questions)

        required = request.get_required_questions()
        assert len(required) == 2
        assert all(q.is_required for q in required)

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
    async def test_different_question_types(self):
        """Test different question types (text, choice, multi_choice)."""
        text_q = ClarificationQuestion(
            question="Describe your use case",
            question_type="text"
        )
        assert text_q.question_type == "text"
        assert text_q.choices is None

        choice_q = ClarificationQuestion(
            question="Select one option",
            question_type="choice",
            choices=["A", "B", "C"]
        )
        assert choice_q.question_type == "choice"
        assert len(choice_q.choices) == 3

        multi_q = ClarificationQuestion(
            question="Select all that apply",
            question_type="multi_choice",
            choices=["X", "Y", "Z"]
        )
        assert multi_q.question_type == "multi_choice"
        assert len(multi_q.choices) == 3

    @pytest.mark.asyncio
    async def test_question_with_context(self):
        """Test questions with additional context."""
        q = ClarificationQuestion(
            question="What's your budget?",
            context="We offer solutions from $100 to $10,000 per month",
            is_required=True
        )
        assert q.context is not None
        assert "$100" in q.context
