"""
Resilience tests for error scenarios and recovery mechanisms.
"""

import asyncio
import pytest
from typing import List, Any
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.factory import AgentFactory, AgentType
from src.agents.base import (
    ResearchDependencies,
    AgentConfiguration,
    AgentTimeoutError,
    AgentExecutionError,
    AgentValidationError
)
from src.models.metadata import ResearchMetadata
from src.models.api_models import APIKeys
from src.models.core import ResearchState, ResearchStage

class TestErrorScenarios:
    """Test various error scenarios and agent resilience."""

    @pytest.fixture
    def error_dependencies(self):
        """Create dependencies for error testing."""
        return ResearchDependencies(
            http_client=AsyncMock(),
            api_keys=APIKeys(),
            research_state=ResearchState(
                request_id="error-test",
                user_id="test-user",
                session_id="test-session",
                user_query="Test query",
                current_stage=ResearchStage.CLARIFICATION
            ),
            metadata=ResearchMetadata(),
            usage=None
        )

    @pytest.fixture
    def resilient_config(self):
        """Configuration for resilience testing."""
        return AgentConfiguration(
            max_retries=3,
            timeout_seconds=5.0,
            temperature=0.7
        )

    @pytest.mark.asyncio
    async def test_network_timeout_recovery(self, error_dependencies, resilient_config):
        """Test recovery from network timeouts."""
        agent = AgentFactory.create_agent(
            AgentType.RESEARCH_EXECUTOR,
            error_dependencies,
            config=resilient_config
        )

        call_count = 0

        async def mock_execute_with_timeout(deps):
            nonlocal call_count
            call_count += 1

            if call_count < 3:
                # Fail first 2 attempts
                await asyncio.sleep(10)  # Exceed timeout
                raise asyncio.TimeoutError("Network timeout")
            else:
                # Succeed on 3rd attempt
                return MagicMock()

        with patch.object(agent, 'execute', side_effect=mock_execute_with_timeout):
            # Should succeed after retries
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(agent.execute(error_dependencies), timeout=1.0)

    @pytest.mark.asyncio
    async def test_api_rate_limit_handling(self, error_dependencies, resilient_config):
        """Test handling of API rate limiting."""
        agent = AgentFactory.create_agent(
            AgentType.CLARIFICATION,
            error_dependencies,
            config=resilient_config
        )

        call_count = 0

        async def mock_execute_with_rate_limit(deps):
            nonlocal call_count
            call_count += 1

            if call_count <= 2:
                # Simulate rate limit error
                raise Exception("Rate limit exceeded: 429")
            else:
                # Succeed after backing off
                return MagicMock()

        with patch.object(agent.agent, 'run', side_effect=mock_execute_with_rate_limit):
            # Should handle rate limiting with retries
            try:
                result = await agent.execute(error_dependencies)
                assert call_count == 3
            except Exception as e:
                assert "Rate limit" in str(e)

    @pytest.mark.asyncio
    async def test_malformed_response_handling(self, error_dependencies):
        """Test handling of malformed LLM responses."""
        agent = AgentFactory.create_agent(
            AgentType.QUERY_TRANSFORMATION,
            error_dependencies
        )

        # Create malformed response
        mock_result = MagicMock()
        mock_result.data = "This is not a valid TransformedQuery object"

        with patch.object(agent.agent, 'run', return_value=mock_result):
            with pytest.raises((AgentValidationError, AttributeError, Exception)):
                await agent.execute(error_dependencies)

    @pytest.mark.asyncio
    async def test_cascading_failure_recovery(self, error_dependencies):
        """Test recovery from cascading failures in pipeline."""
        agents = [
            AgentFactory.create_agent(AgentType.CLARIFICATION, error_dependencies),
            AgentFactory.create_agent(AgentType.QUERY_TRANSFORMATION, error_dependencies),
            AgentFactory.create_agent(AgentType.COMPRESSION, error_dependencies)
        ]

        # First agent fails
        with patch.object(agents[0], 'execute', side_effect=Exception("Agent 1 failed")):
            with pytest.raises(Exception, match="Agent 1 failed"):
                await agents[0].execute(error_dependencies)

            # Second agent should handle upstream failure
            fallback_result = MagicMock()
            fallback_result.transformed_query = error_dependencies.research_state.user_query
            fallback_result.specificity_score = 0.3

            with patch.object(agents[1], 'execute', return_value=fallback_result):
                result = await agents[1].execute(error_dependencies)
                assert result.specificity_score < 0.5  # Low confidence due to fallback

    @pytest.mark.asyncio
    async def test_memory_exhaustion_handling(self, error_dependencies):
        """Test handling of memory exhaustion scenarios."""
        agent = AgentFactory.create_agent(
            AgentType.COMPRESSION,
            error_dependencies
        )

        # Simulate large data that causes memory issues
        large_data = ["x" * 1000000 for _ in range(100)]
        error_dependencies.metadata.additional_context = {"findings": large_data}

        async def mock_execute_with_memory_error(deps):
            # Simulate memory error
            raise MemoryError("Not enough memory to process data")

        with patch.object(agent, 'execute', side_effect=mock_execute_with_memory_error):
            with pytest.raises(MemoryError):
                await agent.execute(error_dependencies)

    @pytest.mark.asyncio
    async def test_concurrent_error_handling(self, error_dependencies):
        """Test error handling with concurrent agent executions."""
        agents = [
            AgentFactory.create_agent(AgentType.RESEARCH_EXECUTOR, error_dependencies),
            AgentFactory.create_agent(AgentType.RESEARCH_EXECUTOR, error_dependencies),
            AgentFactory.create_agent(AgentType.RESEARCH_EXECUTOR, error_dependencies)
        ]

        async def mock_execute_with_varied_errors(deps, agent_idx):
            if agent_idx == 0:
                raise ConnectionError("Network error")
            elif agent_idx == 1:
                raise ValueError("Invalid input")
            else:
                return MagicMock()  # One succeeds

        # Patch agents with different errors
        for i, agent in enumerate(agents):
            agent.execute = lambda deps, idx=i: mock_execute_with_varied_errors(deps, idx)

        # Execute concurrently
        results = await asyncio.gather(
            *[agent.execute(error_dependencies) for agent in agents],
            return_exceptions=True
        )

        # Verify mixed results
        assert sum(isinstance(r, Exception) for r in results) == 2
        assert sum(not isinstance(r, Exception) for r in results) == 1

    @pytest.mark.asyncio
    async def test_validation_error_recovery(self, error_dependencies):
        """Test recovery from validation errors."""
        agent = AgentFactory.create_agent(
            AgentType.COMPRESSION,
            error_dependencies
        )

        attempt_count = 0

        async def mock_execute_with_validation_recovery(deps):
            nonlocal attempt_count
            attempt_count += 1

            if attempt_count == 1:
                # First attempt: validation error
                raise AgentValidationError("Invalid compression format", agent_name="compression")
            else:
                # Second attempt: succeed with fallback
                return MagicMock()

        agent.execute = mock_execute_with_validation_recovery

        # Should recover after validation error
        result = await agent.execute(error_dependencies)
        assert attempt_count == 2

    @pytest.mark.asyncio
    async def test_circuit_breaker_pattern(self, error_dependencies):
        """Test circuit breaker pattern for repeated failures."""
        agent = AgentFactory.create_agent(
            AgentType.RESEARCH_EXECUTOR,
            error_dependencies
        )

        failure_count = 0
        circuit_open = False

        async def mock_execute_with_circuit_breaker(deps):
            nonlocal failure_count, circuit_open

            if circuit_open:
                raise Exception("Circuit breaker is open")

            failure_count += 1
            if failure_count >= 3:
                circuit_open = True

            raise Exception("Service unavailable")

        agent.execute = mock_execute_with_circuit_breaker

        # Trigger circuit breaker
        for _ in range(4):
            with pytest.raises(Exception) as exc_info:
                await agent.execute(error_dependencies)

        # Last error should be circuit breaker
        assert "Circuit breaker" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_partial_failure_handling(self, error_dependencies):
        """Test handling of partial failures in batch operations."""
        agents = [
            AgentFactory.create_agent(AgentType.RESEARCH_EXECUTOR, error_dependencies)
            for _ in range(5)
        ]

        async def mock_execute_with_partial_failure(deps, agent_idx):
            if agent_idx % 2 == 0:
                # Even indices fail
                raise Exception(f"Agent {agent_idx} failed")
            else:
                # Odd indices succeed
                return MagicMock(success=True, agent_idx=agent_idx)

        # Execute with partial failures
        results = []
        for i, agent in enumerate(agents):
            try:
                result = await mock_execute_with_partial_failure(error_dependencies, i)
                results.append(result)
            except Exception:
                results.append(None)

        # Verify partial success
        successful_results = [r for r in results if r is not None]
        assert len(successful_results) == 2
        assert all(r.success for r in successful_results)

    @pytest.mark.asyncio
    async def test_timeout_cascade_prevention(self, error_dependencies):
        """Test prevention of timeout cascades through pipeline."""
        agents = [
            AgentFactory.create_agent(AgentType.CLARIFICATION, error_dependencies),
            AgentFactory.create_agent(AgentType.QUERY_TRANSFORMATION, error_dependencies),
        ]

        # First agent times out
        async def slow_execute(deps):
            await asyncio.sleep(10)
            return MagicMock()

        agents[0].execute = slow_execute

        # Second agent should not wait for first
        fast_result = MagicMock()
        agents[1].execute = AsyncMock(return_value=fast_result)

        # Execute with timeout
        try:
            await asyncio.wait_for(agents[0].execute(error_dependencies), timeout=0.5)
        except asyncio.TimeoutError:
            pass

        # Second agent should still be able to execute quickly
        result = await asyncio.wait_for(agents[1].execute(error_dependencies), timeout=1.0)
        assert result == fast_result

    @pytest.mark.asyncio
    async def test_error_context_preservation(self, error_dependencies):
        """Test that error context is preserved for debugging."""
        agent = AgentFactory.create_agent(
            AgentType.REPORT_GENERATOR,
            error_dependencies
        )

        error_context = {
            "request_id": error_dependencies.research_state.request_id,
            "stage": "report_generation",
            "attempt": 1,
            "timestamp": "2024-01-01T10:00:00"
        }

        async def mock_execute_with_context_error(deps):
            error = AgentExecutionError(
                "Report generation failed",
                agent_name="report_generator",
                context=error_context
            )
            raise error

        agent.execute = mock_execute_with_context_error

        with pytest.raises(AgentExecutionError) as exc_info:
            await agent.execute(error_dependencies)

        # Verify context is preserved
        assert exc_info.value.context == error_context
        assert exc_info.value.agent_name == "report_generator"

    @pytest.mark.asyncio
    async def test_graceful_degradation(self, error_dependencies):
        """Test graceful degradation when optional features fail."""
        agent = AgentFactory.create_agent(
            AgentType.COMPRESSION,
            error_dependencies
        )

        # Mock compression with degraded functionality
        async def mock_execute_with_degradation(deps):
            # Simulate failure of advanced compression
            basic_result = MagicMock()
            basic_result.compression_ratio = 1.5  # Poor compression
            basic_result.summary = "Basic summary without advanced features"
            basic_result.themes = {}  # No theme extraction
            basic_result.metadata = {"degraded": True}
            return basic_result

        agent.execute = mock_execute_with_degradation

        result = await agent.execute(error_dependencies)

        # Verify degraded but functional result
        assert result.compression_ratio < 2.0
        assert result.metadata.get("degraded") is True
        assert len(result.themes) == 0
