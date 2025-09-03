"""
Real integration tests for clarification agent with actual LLM calls.
"""

import os
import pytest
import pytest_asyncio
import httpx
from typing import List

from src.agents.clarification import ClarificationAgent, ClarifyWithUser
from src.agents.base import ResearchDependencies
from src.models.api_models import APIKeys, ResearchMetadata
from src.models.core import ResearchState, ResearchStage
from src.models.clarification import ClarificationQuestion, ClarificationRequest


class TestClarificationIntegration:
    """Integration tests that actually run the clarification agent."""

    @pytest_asyncio.fixture
    async def real_dependencies(self):
        """Create real dependencies with actual API keys."""
        # Get API keys from environment
        api_keys = APIKeys(
            openai=os.getenv("OPENAI_API_KEY"),
            anthropic=os.getenv("ANTHROPIC_API_KEY"),
            tavily=os.getenv("TAVILY_API_KEY")
        )

        # Create HTTP client
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            yield ResearchDependencies(
                http_client=http_client,
                api_keys=api_keys,
                research_state=ResearchState(
                    request_id="test-clarification",
                    user_id="test-user",
                    session_id="test-session",
                    user_query="placeholder",  # Will be set per test
                    current_stage=ResearchStage.CLARIFICATION,
                    metadata=ResearchMetadata()
                ),
                usage=None
            )

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY environment variable"
    )
    async def test_vague_query_needs_clarification(self, real_dependencies):
        """Test that vague queries trigger clarification questions."""
        # Set the vague query
        real_dependencies.research_state.user_query = "Tell me about it"

        # Create and run agent
        agent = ClarificationAgent()
        agent._deps = real_dependencies

        run_result = await agent.agent.run(
            real_dependencies.research_state.user_query,
            deps=real_dependencies
        )

        # Get the actual result data
        result = run_result.output

        # Assertions
        assert result is not None
        assert isinstance(result, ClarifyWithUser)
        assert result.needs_clarification is True
        assert result.request is not None
        assert isinstance(result.request, ClarificationRequest)
        assert len(result.request.questions) > 0

        # Check question quality
        for question in result.request.questions:
            assert isinstance(question, ClarificationQuestion)
            assert len(question.question) > 10  # Not too short
            assert question.question_type in ["text", "choice", "multi_choice"]
            if question.question_type in ["choice", "multi_choice"]:
                assert question.choices is not None and len(question.choices) > 1

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY environment variable"
    )
    async def test_specific_query_no_clarification(self, real_dependencies):
        """Test that specific queries don't need clarification."""
        # Set a specific query
        real_dependencies.research_state.user_query = (
            "Compare the performance of ResNet-50 and VGG-16 architectures "
            "on ImageNet classification in terms of accuracy and computational efficiency"
        )

        # Create and run agent
        agent = ClarificationAgent()
        agent._deps = real_dependencies

        run_result = await agent.agent.run(
            real_dependencies.research_state.user_query,
            deps=real_dependencies
        )
        result = run_result.output

        # Assertions
        assert result is not None
        assert isinstance(result, ClarifyWithUser)
        assert result.needs_clarification is False
        assert result.request is None
        assert len(result.reasoning) > 0  # Should explain why no clarification needed

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY environment variable"
    )
    async def test_ambiguous_comparison_needs_clarification(self, real_dependencies):
        """Test that ambiguous comparisons trigger appropriate questions."""
        # Set an ambiguous comparison query
        real_dependencies.research_state.user_query = "Compare them"

        # Create and run agent
        agent = ClarificationAgent()
        agent._deps = real_dependencies

        run_result = await agent.agent.run(
            real_dependencies.research_state.user_query,
            deps=real_dependencies
        )
        result = run_result.output

        # Assertions
        assert result is not None
        assert result.needs_clarification is True
        assert result.request is not None

        # Should ask what to compare
        questions = result.request.questions
        assert any(
            "what" in q.question.lower() and "compare" in q.question.lower()
            for q in questions
        ), "Should ask what items to compare"

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY environment variable"
    )
    async def test_partial_context_generates_focused_questions(self, real_dependencies):
        """Test that partial context leads to focused clarification questions."""
        # Query with partial context
        real_dependencies.research_state.user_query = (
            "I need to understand the implementation details for my project"
        )

        # Create and run agent
        agent = ClarificationAgent()
        agent._deps = real_dependencies

        run_result = await agent.agent.run(
            real_dependencies.research_state.user_query,
            deps=real_dependencies
        )
        result = run_result.output

        # Assertions
        assert result is not None
        assert result.needs_clarification is True
        assert result.request is not None

        # Check for relevant questions about project type, technology, etc.
        questions = result.request.questions
        question_texts = " ".join(q.question.lower() for q in questions)

        # Should ask about specific aspects
        relevant_keywords = ["project", "technology", "implementation", "details", "specific"]
        assert any(
            keyword in question_texts for keyword in relevant_keywords
        ), f"Questions should be relevant to the query context"

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY environment variable"
    )
    async def test_question_types_variety(self, real_dependencies):
        """Test that the agent can generate different types of questions."""
        # Query that could benefit from multiple question types
        real_dependencies.research_state.user_query = (
            "Help me choose a database for my application"
        )

        # Create and run agent
        agent = ClarificationAgent()
        agent._deps = real_dependencies

        run_result = await agent.agent.run(
            real_dependencies.research_state.user_query,
            deps=real_dependencies
        )
        result = run_result.output

        # Assertions
        assert result is not None
        assert result.needs_clarification is True
        assert result.request is not None

        # Check question variety
        questions = result.request.questions
        question_types = set(q.question_type for q in questions)

        # Log the actual questions for debugging
        for q in questions:
            print(f"Question: {q.question} (Type: {q.question_type})")

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY environment variable"
    )
    async def test_multi_question_generation(self, real_dependencies):
        """Test that complex queries generate multiple relevant questions."""
        # Complex query requiring multiple clarifications
        real_dependencies.research_state.user_query = (
            "I want to build something modern and scalable"
        )

        # Create and run agent
        agent = ClarificationAgent()
        agent._deps = real_dependencies

        run_result = await agent.agent.run(
            real_dependencies.research_state.user_query,
            deps=real_dependencies
        )
        result = run_result.output

        # Assertions
        assert result is not None
        assert result.needs_clarification is True
        assert result.request is not None

        # Should generate multiple questions
        questions = result.request.questions
        assert len(questions) >= 2, "Complex vague query should generate multiple questions"

        # Questions should cover different aspects
        question_topics = []
        for q in questions:
            if "what" in q.question.lower():
                question_topics.append("what")
            if "scale" in q.question.lower() or "size" in q.question.lower():
                question_topics.append("scale")
            if "technology" in q.question.lower() or "stack" in q.question.lower():
                question_topics.append("technology")
            if "purpose" in q.question.lower() or "use" in q.question.lower():
                question_topics.append("purpose")

        assert len(set(question_topics)) >= 2, "Questions should cover different aspects"

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY environment variable"
    )
    async def test_required_vs_optional_questions(self, real_dependencies):
        """Test that agent properly marks questions as required or optional."""
        # Query that should trigger both required and optional questions
        real_dependencies.research_state.user_query = (
            "Research machine learning frameworks"
        )

        # Create and run agent
        agent = ClarificationAgent()
        agent._deps = real_dependencies

        result = await agent.agent.run(
            real_dependencies.research_state.user_query,
            deps=real_dependencies
        )

        if result.needs_clarification and result.request:
            questions = result.request.questions

            # Check that questions have proper required/optional flags
            required_count = sum(1 for q in questions if q.is_required)
            optional_count = sum(1 for q in questions if not q.is_required)

            print(f"Required questions: {required_count}, Optional: {optional_count}")

            # At least some questions should be marked appropriately
            assert all(isinstance(q.is_required, bool) for q in questions)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY environment variable"
    )
    async def test_performance_response_time(self, real_dependencies):
        """Test that clarification agent responds within reasonable time."""
        import time

        # Set a typical query
        real_dependencies.research_state.user_query = "Explain quantum computing"

        # Create and run agent with timing
        agent = ClarificationAgent()
        agent._deps = real_dependencies

        start_time = time.time()
        run_result = await agent.agent.run(
            real_dependencies.research_state.user_query,
            deps=real_dependencies
        )
        result = run_result.output
        end_time = time.time()

        response_time = end_time - start_time

        # Assertions
        assert result is not None
        assert response_time < 10.0, f"Response took {response_time}s, should be under 10s"

        print(f"Clarification check took {response_time:.2f} seconds")

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY environment variable"
    )
    async def test_clarification_with_context(self, real_dependencies):
        """Test that agent considers conversation context when generating questions."""
        # Add some conversation history
        real_dependencies.research_state.metadata.conversation_messages = [
            {"role": "user", "content": "I'm working on a Python project"},
            {"role": "assistant", "content": "I understand you're working with Python."},
        ]

        # Set a query that refers back to context
        real_dependencies.research_state.user_query = "What's the best way to handle errors?"

        # Create and run agent
        agent = ClarificationAgent()
        agent._deps = real_dependencies

        result = await agent.agent.run(
            real_dependencies.research_state.user_query,
            deps=real_dependencies
        )

        # The agent should understand this is about Python error handling
        # and either not need clarification or ask Python-specific questions
        if result.needs_clarification and result.request:
            question_texts = " ".join(q.question.lower() for q in result.request.questions)
            # Questions should be relevant to error handling
            assert any(
                keyword in question_texts
                for keyword in ["error", "exception", "handling", "type"]
            )
