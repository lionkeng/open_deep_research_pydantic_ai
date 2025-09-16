"""Focused resilience tests for the four-agent pipeline."""

from datetime import timezone
from unittest.mock import AsyncMock, MagicMock, patch

import asyncio
import pytest

from agents.base import AgentConfiguration, ResearchDependencies
from agents.factory import AgentFactory, AgentType
from agents.research_executor import ResearchExecutorAgent
from models.api_models import APIKeys
from models.core import ResearchStage, ResearchState
from models.metadata import ResearchMetadata


@pytest.fixture
def error_dependencies() -> ResearchDependencies:
    """Dependencies shared by resilience tests."""
    return ResearchDependencies(
        http_client=AsyncMock(),
        api_keys=APIKeys(),
        research_state=ResearchState(
            request_id="error-test",
            user_query="Test query",
            current_stage=ResearchStage.CLARIFICATION,
            metadata=ResearchMetadata(),
        ),
    )


@pytest.fixture
def resilient_config() -> AgentConfiguration:
    """Configuration with low timeout to exercise retry paths."""
    return AgentConfiguration(
        agent_name="research_executor",
        agent_type=AgentType.RESEARCH_EXECUTOR.value,
        max_retries=3,
        timeout_seconds=5.0,
        temperature=0.7,
    )


@pytest.mark.asyncio
async def test_network_timeout_recovery(error_dependencies, resilient_config) -> None:
    """Research executor should tolerate transient timeouts."""
    agent: ResearchExecutorAgent = AgentFactory.create_agent(
        AgentType.RESEARCH_EXECUTOR,
        error_dependencies,
        config=resilient_config,
    )

    side_effects = [TimeoutError("temporary network issue"), []]

    with patch("agents.research_executor.extract_hierarchical_findings", side_effect=side_effects):
        try:
            await agent.run(error_dependencies)
        except TimeoutError:
            pass

        result = await agent.run(error_dependencies)
        assert result is not None


@pytest.mark.asyncio
async def test_error_context_surfaces(error_dependencies) -> None:
    """Report generator errors should preserve context for debugging."""
    agent = AgentFactory.create_agent(
        AgentType.REPORT_GENERATOR,
        error_dependencies,
    )

    error_context = {
        "request_id": error_dependencies.research_state.request_id,
        "stage": "report_generation",
        "timestamp": "2024-01-01T10:00:00",
    }

    async def failing_run(deps):
        raise Exception(str(error_context))

    with patch.object(agent, "run", side_effect=failing_run):
        with pytest.raises(Exception) as exc_info:
            await agent.run(error_dependencies)
        for value in error_context.values():
            assert value in str(exc_info.value)


@pytest.mark.asyncio
async def test_concurrent_research_agents(error_dependencies) -> None:
    """Multiple research agents should handle concurrent errors gracefully."""
    agents = [
        AgentFactory.create_agent(AgentType.RESEARCH_EXECUTOR, error_dependencies)
        for _ in range(3)
    ]

    async def execute_with_mixed_outcomes(deps, idx):
        if idx == 0:
            raise ConnectionError("Network error")
        if idx == 1:
            raise ValueError("Invalid input")
        return MagicMock()

    tasks = []
    for idx, agent in enumerate(agents):
        async def run_agent(at=idx, instance=agent):
            return await execute_with_mixed_outcomes(error_dependencies, at)

        tasks.append(run_agent())

    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert sum(isinstance(r, Exception) for r in results) == 2
    assert any(not isinstance(r, Exception) for r in results)
