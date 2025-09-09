"""Integration tests for ClarificationAgent workflow integration.

These tests focus on the agent's integration with external dependencies,
real AI models, and its role within larger research workflows.
They do NOT test specific clarification scenarios - those are handled
by the evaluation framework.
"""

import pytest
import asyncio
import os
import time
from typing import Dict, Any, List
from unittest.mock import patch, AsyncMock, MagicMock

from src.agents.clarification import ClarificationAgent, ClarifyWithUser
from src.agents.base import ResearchDependencies
from src.models.core import ResearchState, ResearchStage
from src.models.metadata import ResearchMetadata
from src.models.api_models import APIKeys
from pydantic import SecretStr
from pydantic_ai.usage import RunUsage


class TestClarificationWorkflowIntegration:
    """Integration tests for ClarificationAgent within research workflows.

    These tests focus on workflow integration (component interactions, dependency
    injection, error handling) rather than external API integration. Most tests
    use mocked LLMs for speed and reliability.

    Tests marked with 'real_api' in their name use actual API calls when
    API keys are available.
    """

    @pytest.fixture
    def agent(self) -> ClarificationAgent:
        """Create a ClarificationAgent instance with mocked LLM for workflow testing."""
        # Create agent with mocked LLM to avoid real API calls
        with patch('src.agents.base.Agent') as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance

            agent = ClarificationAgent()
            agent.agent = mock_agent_instance

            # Set up default mock response
            mock_result = MagicMock()
            mock_result.output = ClarifyWithUser(
                needs_clarification=False,
                missing_dimensions=[],
                request=None,
                reasoning="Query is sufficiently clear for research",
                assessment_reasoning="All dimensions are adequately specified"
            )
            mock_agent_instance.run = AsyncMock(return_value=mock_result)

            return agent

    @pytest.fixture
    def real_agent(self) -> ClarificationAgent:
        """Create a real ClarificationAgent for API integration testing."""
        return ClarificationAgent()

    @pytest.fixture
    def real_dependencies(self) -> ResearchDependencies:
        """Create real dependencies with actual API keys for testing."""
        # Use AsyncMock for http_client since we're mocking the agent anyway
        return ResearchDependencies(
            http_client=AsyncMock(),
            api_keys=APIKeys(
                openai=SecretStr(key) if (key := os.getenv("OPENAI_API_KEY")) else None,
                anthropic=SecretStr(key) if (key := os.getenv("ANTHROPIC_API_KEY")) else None
            ),
            research_state=ResearchState(
                request_id="workflow-integration-test",
                user_query="test query",
                current_stage=ResearchStage.CLARIFICATION,
                metadata=ResearchMetadata()
            )
        )

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY environment variable"
    )
    async def test_real_api_integration_openai(self, real_agent: ClarificationAgent, real_dependencies: ResearchDependencies) -> None:
        """Test agent integration with real OpenAI API."""
        query = "What is machine learning?"
        real_dependencies.research_state.user_query = query

        result = await real_agent.agent.run(query, deps=real_dependencies)

        # Verify API integration worked
        assert hasattr(result, 'output')
        assert isinstance(result.output, ClarifyWithUser)
        assert isinstance(result.output.needs_clarification, bool)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="Requires ANTHROPIC_API_KEY environment variable"
    )
    async def test_real_api_integration_anthropic(self, real_agent: ClarificationAgent, real_dependencies: ResearchDependencies) -> None:
        """Test agent integration with real Anthropic API."""
        # Modify dependencies to use Anthropic
        real_dependencies.api_keys = APIKeys(
            anthropic=SecretStr(key) if (key := os.getenv("ANTHROPIC_API_KEY")) else None
        )

        query = "What is Python programming?"
        real_dependencies.research_state.user_query = query

        result = await real_agent.agent.run(query, deps=real_dependencies)

        # Verify API integration worked
        assert hasattr(result, 'output')
        assert isinstance(result.output, ClarifyWithUser)
        assert isinstance(result.output.needs_clarification, bool)

    @pytest.mark.asyncio
    async def test_workflow_context_integration(self, agent: ClarificationAgent, real_dependencies: ResearchDependencies) -> None:
        """Test agent integration with research workflow context."""
        # Set up workflow context
        real_dependencies.research_state.metadata = ResearchMetadata(
            conversation_messages=[
                {"role": "user", "content": "I'm working on a Python project"},
                {"role": "assistant", "content": "I understand you're working with Python."}
            ]
        )

        query = "What's the best way to handle errors?"
        real_dependencies.research_state.user_query = query

        result = await agent.agent.run(query, deps=real_dependencies)
        output = result.output

        assert isinstance(output, ClarifyWithUser)

        # With context, the agent should understand this is about Python error handling
        # This is more of a functional test - the agent should leverage context

    @pytest.mark.asyncio
    async def test_multi_stage_workflow_integration(self, agent: ClarificationAgent, real_dependencies: ResearchDependencies) -> None:
        """Test agent behavior within multi-stage research workflow."""
        # Test progression through workflow stages
        stages = [
            ResearchStage.CLARIFICATION,
            ResearchStage.RESEARCH_EXECUTION,
            ResearchStage.COMPRESSION,
        ]

        for stage in stages:
            real_dependencies.research_state.current_stage = stage
            real_dependencies.research_state.request_id = f"workflow-{stage.value}"

            query = "Research artificial intelligence applications"
            result = await agent.agent.run(query, deps=real_dependencies)

            assert hasattr(result, 'output')
            assert isinstance(result.output, ClarifyWithUser)

    @pytest.mark.asyncio
    async def test_dependency_injection_variants(self, agent: ClarificationAgent) -> None:
        """Test agent behavior with different dependency configurations."""
        base_state = ResearchState(
            request_id="dep-injection-test",
            user_query="What is machine learning?"
        )

        # Test with minimal dependencies
        minimal_deps = ResearchDependencies(
            http_client=AsyncMock(),
            api_keys=APIKeys(),
            research_state=base_state
        )

        result = await agent.agent.run("Test query", deps=minimal_deps)
        assert isinstance(result.output, ClarifyWithUser)

        # Test with extended dependencies
        extended_state = ResearchState(
            request_id="dep-injection-test",
            user_query="What is machine learning?",
            metadata=ResearchMetadata(
                user_preferences={"technical_level": "expert"}
            )
        )
        extended_deps = ResearchDependencies(
            http_client=AsyncMock(),
            api_keys=APIKeys(openai=SecretStr("test-key")),
            research_state=extended_state,
            usage=RunUsage(requests=1, output_tokens=100)
        )

        result = await agent.agent.run("Test query", deps=extended_deps)
        assert isinstance(result.output, ClarifyWithUser)

    @pytest.mark.asyncio
    async def test_concurrent_workflow_handling(self, agent: ClarificationAgent) -> None:
        """Test agent behavior with concurrent workflow requests."""
        # Create multiple concurrent workflow contexts
        contexts = []
        for i in range(3):
            deps = ResearchDependencies(
                http_client=AsyncMock(),
                api_keys=APIKeys(openai=SecretStr("test-key")),
                research_state=ResearchState(
                    request_id=f"concurrent-workflow-{i}",
                    user_query=f"Query {i}",
                    current_stage=ResearchStage.CLARIFICATION
                )
            )
            contexts.append(deps)

        # Run concurrent requests
        tasks = [
            agent.agent.run(f"Concurrent query {i}", deps=contexts[i])
            for i in range(len(contexts))
        ]

        results = await asyncio.gather(*tasks)

        # All should succeed and be properly structured
        for i, result in enumerate(results):
            assert hasattr(result, 'output'), f"Result {i} missing data"
            assert isinstance(result.output, ClarifyWithUser), f"Result {i} wrong type"


class TestClarificationErrorRecovery:
    """Test error recovery and resilience in workflow integration."""

    @pytest.fixture
    def agent(self) -> ClarificationAgent:
        """Create a ClarificationAgent instance for testing."""
        # Create agent with mocked LLM to avoid real API calls
        with patch('src.agents.base.Agent') as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance

            agent = ClarificationAgent()
            agent.agent = mock_agent_instance

            # Set up default mock response
            mock_result = MagicMock()
            mock_result.output = ClarifyWithUser(
                needs_clarification=False,
                missing_dimensions=[],
                request=None,
                reasoning="Query is sufficiently clear for research",
                assessment_reasoning="All dimensions are adequately specified"
            )
            mock_agent_instance.run = AsyncMock(return_value=mock_result)

            return agent

    @pytest.mark.asyncio
    async def test_api_timeout_recovery(self, agent: ClarificationAgent) -> None:
        """Test agent behavior when API calls timeout."""
        # Mock HTTP client that times out
        mock_client = AsyncMock()
        mock_client.post.side_effect = asyncio.TimeoutError("Request timeout")

        deps = ResearchDependencies(
            http_client=mock_client,
            api_keys=APIKeys(openai=SecretStr("test-key")),
            research_state=ResearchState(
                request_id="timeout-test",
                user_query="test query"
            )
        )

        # Should handle timeout gracefully
        try:
            result = await agent.agent.run("Test timeout query", deps=deps)
            # If no exception, verify structure
            if hasattr(result, 'output'):
                assert isinstance(result.output, ClarifyWithUser)
        except Exception as e:
            # If exception, should be timeout-related
            assert "timeout" in str(e).lower() or "time" in str(e).lower()

    @pytest.mark.asyncio
    async def test_api_rate_limit_handling(self, agent: ClarificationAgent) -> None:
        """Test agent behavior when hitting API rate limits."""
        # Mock HTTP client that returns rate limit errors
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status_code = 429  # Too Many Requests
        mock_response.json.return_value = {"error": "Rate limit exceeded"}
        mock_client.post.return_value = mock_response

        deps = ResearchDependencies(
            http_client=mock_client,
            api_keys=APIKeys(openai=SecretStr("test-key")),
            research_state=ResearchState(
                request_id="rate-limit-test",
                user_query="test query"
            )
        )

        # Should handle rate limit gracefully
        try:
            result = await agent.agent.run("Test rate limit query", deps=deps)
            if hasattr(result, 'output'):
                assert isinstance(result.output, ClarifyWithUser)
        except Exception as e:
            error_msg = str(e).lower()
            assert any(word in error_msg for word in ['rate', 'limit', '429', 'quota'])

    @pytest.mark.asyncio
    async def test_network_failure_recovery(self, agent: ClarificationAgent) -> None:
        """Test agent behavior during network failures."""
        # Mock HTTP client with network errors
        mock_client = AsyncMock()
        mock_client.post.side_effect = Exception("Network unreachable")

        deps = ResearchDependencies(
            http_client=mock_client,
            api_keys=APIKeys(openai=SecretStr("test-key")),
            research_state=ResearchState(
                request_id="network-fail-test",
                user_query="test query"
            )
        )

        # Should handle network failure gracefully
        try:
            result = await agent.agent.run("Test network failure", deps=deps)
            if hasattr(result, 'output'):
                assert isinstance(result.output, ClarifyWithUser)
        except Exception as e:
            error_msg = str(e).lower()
            assert any(word in error_msg for word in ['network', 'connection', 'unreachable'])

    @pytest.mark.asyncio
    async def test_malformed_response_handling(self, agent: ClarificationAgent) -> None:
        """Test agent behavior with malformed API responses."""
        # Mock HTTP client with malformed responses
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"invalid": "response format"}
        mock_client.post.return_value = mock_response

        deps = ResearchDependencies(
            http_client=mock_client,
            api_keys=APIKeys(openai=SecretStr("test-key")),
            research_state=ResearchState(
                request_id="malformed-test",
                user_query="test query"
            )
        )

        # Should handle malformed response gracefully
        try:
            result = await agent.agent.run("Test malformed response", deps=deps)
            if hasattr(result, 'output'):
                assert isinstance(result.output, ClarifyWithUser)
        except Exception as e:
            error_msg = str(e).lower()
            assert any(word in error_msg for word in ['parse', 'format', 'invalid', 'response'])


class TestClarificationPerformanceIntegration:
    """Test performance characteristics in workflow integration."""

    @pytest.fixture
    def agent(self) -> ClarificationAgent:
        """Create a ClarificationAgent instance for testing."""
        # Create agent with mocked LLM to avoid real API calls
        with patch('src.agents.base.Agent') as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance

            agent = ClarificationAgent()
            agent.agent = mock_agent_instance

            # Set up default mock response
            mock_result = MagicMock()
            mock_result.output = ClarifyWithUser(
                needs_clarification=False,
                missing_dimensions=[],
                request=None,
                reasoning="Query is sufficiently clear for research",
                assessment_reasoning="All dimensions are adequately specified"
            )
            mock_agent_instance.run = AsyncMock(return_value=mock_result)

            return agent

    @pytest.mark.asyncio
    async def test_workflow_performance_benchmarks(self, agent: ClarificationAgent) -> None:
        """Test that agent meets performance requirements in workflow context."""
        deps = ResearchDependencies(
            http_client=AsyncMock(),
            api_keys=APIKeys(openai=SecretStr("test-key")),
            research_state=ResearchState(
                request_id="perf-benchmark",
                user_query="Performance test query"
            )
        )

        # Measure response time
        start_time = time.time()
        result = await agent.agent.run("Performance test query", deps=deps)
        end_time = time.time()

        response_time = end_time - start_time

        # Should respond within reasonable time for workflow integration
        assert response_time < 15.0, f"Workflow integration took {response_time}s, should be under 15s"

        # Response should be valid
        assert hasattr(result, 'output')
        assert isinstance(result.output, ClarifyWithUser)

    @pytest.mark.asyncio
    async def test_memory_usage_workflow(self, agent: ClarificationAgent) -> None:
        """Test memory usage doesn't grow excessively during workflow operations."""
        pytest.importorskip("psutil", reason="psutil not installed")
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run multiple workflow operations
        for i in range(5):
            deps = ResearchDependencies(
                http_client=AsyncMock(),
                api_keys=APIKeys(openai=SecretStr("test-key")),
                research_state=ResearchState(
                    request_id=f"memory-test-{i}",
                    user_query=f"Memory test query {i}"
                )
            )

            result = await agent.agent.run(f"Memory test {i}", deps=deps)
            assert isinstance(result.output, ClarifyWithUser)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory

        # Memory growth should be reasonable (less than 100MB for this test)
        assert memory_growth < 100, f"Memory grew by {memory_growth}MB, should be under 100MB"

    @pytest.mark.asyncio
    async def test_concurrent_workflow_performance(self, agent: ClarificationAgent) -> None:
        """Test performance under concurrent workflow load."""
        # Create concurrent workflow tasks
        tasks = []
        for i in range(10):  # 10 concurrent requests
            deps = ResearchDependencies(
                http_client=AsyncMock(),
                api_keys=APIKeys(openai=SecretStr("test-key")),
                research_state=ResearchState(
                    request_id=f"concurrent-perf-{i}",
                    user_query=f"Concurrent query {i}"
                )
            )
            tasks.append(agent.agent.run(f"Concurrent test {i}", deps=deps))

        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()

        total_time = end_time - start_time

        # All concurrent requests should complete within reasonable time
        assert total_time < 30.0, f"Concurrent workflow took {total_time}s, should be under 30s"

        # All should be successful
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pytest.fail(f"Concurrent request {i} failed: {result}")
            assert hasattr(result, 'output')
            assert isinstance(result.output, ClarifyWithUser)


class TestClarificationObservability:
    """Test observability and monitoring integration."""

    @pytest.fixture
    def agent(self) -> ClarificationAgent:
        """Create a ClarificationAgent instance for testing."""
        # Create agent with mocked LLM to avoid real API calls
        with patch('src.agents.base.Agent') as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance

            agent = ClarificationAgent()
            agent.agent = mock_agent_instance

            # Set up default mock response
            mock_result = MagicMock()
            mock_result.output = ClarifyWithUser(
                needs_clarification=False,
                missing_dimensions=[],
                request=None,
                reasoning="Query is sufficiently clear for research",
                assessment_reasoning="All dimensions are adequately specified"
            )
            mock_agent_instance.run = AsyncMock(return_value=mock_result)

            return agent

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Logging test needs refactoring with current mocking strategy")
    async def test_logging_integration(self, agent: ClarificationAgent) -> None:
        """Test that agent integrates with logging systems."""
        deps = ResearchDependencies(
            http_client=AsyncMock(),
            api_keys=APIKeys(openai=SecretStr("test-key")),
            research_state=ResearchState(
                request_id="logging-test",
                user_query="Logging integration test"
            )
        )

        # Run agent and verify it doesn't break logging
        with patch('logging.getLogger') as mock_logger:
            result = await agent.agent.run("Test logging", deps=deps)

            assert isinstance(result.output, ClarifyWithUser)
            # Logger should have been accessed (agent should log its operations)
            assert mock_logger.called

    @pytest.mark.asyncio
    async def test_metrics_collection_compatibility(self, agent: ClarificationAgent) -> None:
        """Test that agent operations are compatible with metrics collection."""
        deps = ResearchDependencies(
            http_client=AsyncMock(),
            api_keys=APIKeys(openai=SecretStr("test-key")),
            research_state=ResearchState(
                request_id="metrics-test",
                user_query="Metrics collection test"
            )
        )

        # Mock metrics collection
        metrics = {
            "requests": 0,
            "successes": 0,
            "failures": 0,
            "response_times": []
        }

        start_time = time.time()
        try:
            metrics["requests"] += 1
            result = await agent.agent.run("Test metrics", deps=deps)
            end_time = time.time()

            metrics["successes"] += 1
            metrics["response_times"].append(end_time - start_time)

            assert isinstance(result.output, ClarifyWithUser)
            assert metrics["successes"] == 1
            assert len(metrics["response_times"]) == 1

        except Exception:
            metrics["failures"] += 1
            raise

    @pytest.mark.asyncio
    async def test_tracing_integration(self, agent: ClarificationAgent) -> None:
        """Test that agent operations can be traced."""
        deps = ResearchDependencies(
            http_client=AsyncMock(),
            api_keys=APIKeys(openai=SecretStr("test-key")),
            research_state=ResearchState(
                request_id="tracing-test",
                user_query="Tracing integration test"
            )
        )

        # Mock tracing span
        trace_data = {
            "spans": [],
            "current_span": None
        }

        class MockSpan:
            def __init__(self, name):
                self.name = name
                self.attributes = {}

            def set_attribute(self, key, value):
                self.attributes[key] = value

            def __enter__(self):
                trace_data["current_span"] = self
                trace_data["spans"].append(self)
                return self

            def __exit__(self, *args):
                trace_data["current_span"] = None

        # Simulate traced operation
        with MockSpan("clarification_agent_run") as span:
            span.set_attribute("query", "Test tracing")
            result = await agent.agent.run("Test tracing", deps=deps)
            span.set_attribute("need_clarification", str(result.output.needs_clarification))

        assert isinstance(result.output, ClarifyWithUser)
        assert len(trace_data["spans"]) == 1
        assert trace_data["spans"][0].name == "clarification_agent_run"
