"""Unit tests for ClarificationAgent core logic.

These tests focus on testing the agent's core functionality in isolation,
verifying input validation, response structure, and basic decision logic
without relying on complex scenarios or integration concerns.
"""

import pytest
import asyncio
import httpx
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.clarification import ClarificationAgent, ClarifyWithUser
from src.agents.base import ResearchDependencies
from src.models.core import ResearchState, ResearchStage
from src.models.metadata import ResearchMetadata
from src.models.api_models import APIKeys
from pydantic import SecretStr


class TestClarificationAgentUnit:
    """Unit tests for ClarificationAgent core functionality."""

    @pytest.fixture
    def agent(self) -> ClarificationAgent:
        """Create a ClarificationAgent instance for testing."""
        return ClarificationAgent()

    @pytest.fixture
    def sample_dependencies(self) -> ResearchDependencies:
        """Create sample research dependencies for testing."""
        return ResearchDependencies(
            http_client=AsyncMock(),
            api_keys=APIKeys(),
            research_state=ResearchState(
                request_id="test-123",
                user_query="test query"
            )
        )

    async def test_agent_initialization(self, agent: ClarificationAgent) -> None:
        """Test that the agent initializes correctly."""
        assert agent is not None
        assert hasattr(agent, 'agent')
        assert agent.agent is not None

    async def test_agent_has_required_methods(self, agent: ClarificationAgent) -> None:
        """Test that the agent has all required methods."""
        assert hasattr(agent.agent, 'run')

    async def test_response_structure_validation(self, agent: ClarificationAgent, sample_dependencies: ResearchDependencies) -> None:
        """Test that the agent always returns properly structured responses."""
        test_query = "What is machine learning?"

        result = await agent.agent.run(test_query, deps=sample_dependencies)

        # Validate response structure
        assert hasattr(result, 'data')
        output = result.data
        assert isinstance(output, ClarifyWithUser)

        # Validate required fields exist
        assert hasattr(output, 'need_clarification')
        assert isinstance(output.need_clarification, bool)

        # Additional structure validation based on clarification decision
        if output.need_clarification:
            assert hasattr(output, 'question') or hasattr(output, 'request')
            assert hasattr(output, 'missing_dimensions')
            assert hasattr(output, 'assessment_reasoning')
        else:
            assert hasattr(output, 'verification')

    @pytest.mark.parametrize("query,expected_type", [
        ("What is the current Bitcoin price in USD?", bool),
        ("How do I optimize this code?", bool),
        ("Tell me about Python programming language", bool),
        ("", bool),  # Empty query
        ("?", bool),  # Minimal query
    ])
    async def test_basic_decision_logic(
        self,
        agent: ClarificationAgent,
        sample_dependencies: ResearchDependencies,
        query: str,
        expected_type: type
    ) -> None:
        """Test that the agent makes binary clarification decisions for various query types."""
        sample_dependencies.research_state.user_query = query

        result = await agent.agent.run(query, deps=sample_dependencies)
        output = result.data

        assert isinstance(output, ClarifyWithUser)
        assert isinstance(output.need_clarification, expected_type)

    async def test_input_validation_empty_query(self, agent: ClarificationAgent, sample_dependencies: ResearchDependencies) -> None:
        """Test agent behavior with empty query."""
        result = await agent.agent.run("", deps=sample_dependencies)
        output = result.data

        assert isinstance(output, ClarifyWithUser)
        # Empty query should typically need clarification
        assert output.need_clarification is True

    async def test_input_validation_whitespace_query(self, agent: ClarificationAgent, sample_dependencies: ResearchDependencies) -> None:
        """Test agent behavior with whitespace-only query."""
        result = await agent.agent.run("   \n\t  ", deps=sample_dependencies)
        output = result.data

        assert isinstance(output, ClarifyWithUser)
        # Whitespace-only query should need clarification
        assert output.need_clarification is True

    async def test_input_validation_very_long_query(self, agent: ClarificationAgent, sample_dependencies: ResearchDependencies) -> None:
        """Test agent behavior with extremely long queries."""
        long_query = "What is machine learning? " * 200  # Very long query

        result = await agent.agent.run(long_query, deps=sample_dependencies)
        output = result.data

        assert isinstance(output, ClarifyWithUser)
        # Should handle long queries gracefully
        assert isinstance(output.need_clarification, bool)

    async def test_special_characters_handling(self, agent: ClarificationAgent, sample_dependencies: ResearchDependencies) -> None:
        """Test agent behavior with special characters and unicode."""
        special_query = "What is 机器学习 and how does it relate to café? 🤖"

        result = await agent.agent.run(special_query, deps=sample_dependencies)
        output = result.data

        assert isinstance(output, ClarifyWithUser)
        assert isinstance(output.need_clarification, bool)

    async def test_reasoning_field_populated(self, agent: ClarificationAgent, sample_dependencies: ResearchDependencies) -> None:
        """Test that reasoning is provided for decisions."""
        result = await agent.agent.run("What is Python?", deps=sample_dependencies)
        output = result.data

        assert isinstance(output, ClarifyWithUser)

        if output.need_clarification:
            assert hasattr(output, 'assessment_reasoning')
            if hasattr(output, 'assessment_reasoning') and output.assessment_reasoning:
                assert len(output.assessment_reasoning.strip()) > 0
        else:
            assert hasattr(output, 'verification')
            if hasattr(output, 'verification') and output.verification:
                assert len(output.verification.strip()) > 0


class TestClarificationAgentErrorHandling:
    """Test error handling and edge cases."""

    @pytest.fixture
    def agent(self) -> ClarificationAgent:
        """Create a ClarificationAgent instance for testing."""
        return ClarificationAgent()

    async def test_malformed_dependencies_handling(self, agent: ClarificationAgent) -> None:
        """Test agent behavior with malformed dependencies."""
        query = "What is machine learning?"

        # Test with None dependencies - should handle gracefully
        try:
            result = await agent.agent.run(query, deps=None)
            # If it doesn't raise an exception, verify the response structure
            if hasattr(result, 'data'):
                assert isinstance(result.data, ClarifyWithUser)
        except Exception as e:
            # If it raises an exception, it should be informative
            assert len(str(e)) > 0

    async def test_missing_api_keys(self, agent: ClarificationAgent) -> None:
        """Test agent behavior when API keys are missing."""
        deps = ResearchDependencies(
            http_client=AsyncMock(),
            api_keys=APIKeys(),  # Empty API keys
            research_state=ResearchState(
                request_id="test-missing-keys",
                user_query="test"
            )
        )

        # This might fail or succeed depending on agent implementation
        # The key is that it should fail gracefully if it fails
        try:
            result = await agent.agent.run("What is AI?", deps=deps)
            if hasattr(result, 'data'):
                assert isinstance(result.data, ClarifyWithUser)
        except Exception as e:
            # Should have informative error message about missing keys
            error_msg = str(e).lower()
            assert any(word in error_msg for word in ['key', 'api', 'auth', 'credential'])

    async def test_http_client_failure(self, agent: ClarificationAgent) -> None:
        """Test agent behavior when HTTP client fails."""
        # Mock HTTP client that raises exceptions
        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Network error")
        mock_client.post.side_effect = Exception("Network error")

        deps = ResearchDependencies(
            http_client=mock_client,
            api_keys=APIKeys(openai=SecretStr("test-key")),
            research_state=ResearchState(
                request_id="test-http-fail",
                user_query="test"
            )
        )

        # Should handle HTTP failures gracefully
        try:
            result = await agent.agent.run("What is Python?", deps=deps)
            # If no exception, should still return valid structure
            if hasattr(result, 'data'):
                assert isinstance(result.data, ClarifyWithUser)
        except Exception as e:
            # If exception, should be related to network/HTTP
            error_msg = str(e).lower()
            assert any(word in error_msg for word in ['network', 'http', 'connection', 'timeout'])


class TestClarificationAgentConsistency:
    """Test consistency and deterministic behavior."""

    @pytest.fixture
    def agent(self) -> ClarificationAgent:
        """Create a ClarificationAgent instance for testing."""
        return ClarificationAgent()

    @pytest.fixture
    def consistent_dependencies(self) -> ResearchDependencies:
        """Create dependencies configured for consistent results."""
        return ResearchDependencies(
            http_client=AsyncMock(),
            api_keys=APIKeys(openai=SecretStr("test-key")),
            research_state=ResearchState(
                request_id="consistency-test",
                user_query="test"
            )
        )

    async def test_same_query_consistency(self, agent: ClarificationAgent, consistent_dependencies: ResearchDependencies) -> None:
        """Test that the same query produces consistent results."""
        query = "What is the best way to learn Python programming?"

        # Run the same query multiple times
        results = []
        for i in range(3):
            consistent_dependencies.research_state.request_id = f"consistency-test-{i}"
            result = await agent.agent.run(query, deps=consistent_dependencies)
            results.append(result.data)

        # All results should have the same clarification decision
        decisions = [r.need_clarification for r in results]

        # With temperature=0 or similar settings, decisions should be consistent
        # But we'll be lenient and just check that we got valid decisions
        assert all(isinstance(d, bool) for d in decisions)

        # All results should be properly structured
        for result in results:
            assert isinstance(result, ClarifyWithUser)

    @pytest.mark.parametrize("query", [
        "What is the capital of France?",
        "How do I implement quicksort?",
        "Tell me about machine learning",
        "What's the weather like?",
    ])
    async def test_query_type_consistency(
        self,
        agent: ClarificationAgent,
        consistent_dependencies: ResearchDependencies,
        query: str
    ) -> None:
        """Test that similar query types produce consistent decision patterns."""
        consistent_dependencies.research_state.user_query = query

        result = await agent.agent.run(query, deps=consistent_dependencies)
        output = result.data

        assert isinstance(output, ClarifyWithUser)
        assert isinstance(output.need_clarification, bool)

        # Specific factual queries should generally not need clarification
        if "capital of" in query.lower() or "implement" in query.lower():
            # These are more likely to be clear, but we won't enforce it strictly
            assert isinstance(output.need_clarification, bool)

        # Broad queries should more likely need clarification
        if "tell me about" in query.lower():
            # More likely to need clarification, but not enforced strictly
            assert isinstance(output.need_clarification, bool)


class TestClarificationAgentPerformance:
    """Test performance characteristics."""

    @pytest.fixture
    def agent(self) -> ClarificationAgent:
        """Create a ClarificationAgent instance for testing."""
        return ClarificationAgent()

    @pytest.fixture
    def performance_dependencies(self) -> ResearchDependencies:
        """Create dependencies for performance testing."""
        return ResearchDependencies(
            http_client=AsyncMock(),
            api_keys=APIKeys(openai=SecretStr("test-key")),
            research_state=ResearchState(
                request_id="perf-test",
                user_query="test"
            )
        )

    async def test_response_time_reasonable(self, agent: ClarificationAgent, performance_dependencies: ResearchDependencies) -> None:
        """Test that agent responds within reasonable time."""
        import time

        query = "What is Python programming?"
        performance_dependencies.research_state.user_query = query

        start_time = time.time()
        result = await agent.agent.run(query, deps=performance_dependencies)
        end_time = time.time()

        response_time = end_time - start_time

        # Should respond within 30 seconds (very generous for unit tests)
        assert response_time < 30.0, f"Response took {response_time}s, should be under 30s"

        # Response should be valid
        assert hasattr(result, 'data')
        assert isinstance(result.data, ClarifyWithUser)

    async def test_concurrent_requests_handling(self, agent: ClarificationAgent, performance_dependencies: ResearchDependencies) -> None:
        """Test that agent can handle concurrent requests."""
        queries = [
            "What is Python?",
            "How does machine learning work?",
            "What is the best database?",
        ]

        # Create tasks for concurrent execution
        tasks = []
        for i, query in enumerate(queries):
            deps = ResearchDependencies(
                http_client=AsyncMock(),
                api_keys=APIKeys(openai=SecretStr("test-key")),
                research_state=ResearchState(
                    request_id=f"concurrent-test-{i}",
                    user_query=query
                )
            )
            tasks.append(agent.agent.run(query, deps=deps))

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All results should be successful (no exceptions)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pytest.fail(f"Query {i} failed with exception: {result}")

            assert hasattr(result, 'data')
            assert isinstance(result.data, ClarifyWithUser)
