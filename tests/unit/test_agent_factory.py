"""Updated tests for AgentFactory without compression agent."""

from typing import List
from unittest.mock import MagicMock

import pytest

from agents.base import AgentConfiguration, ResearchDependencies
from agents.factory import (
    AgentFactory,
    AgentType,
    create_clarification_agent,
    create_query_transformation_agent,
    create_research_executor_agent,
    create_report_generator_agent,
)
from models.api_models import APIKeys
from models.core import ResearchStage, ResearchState
from models.metadata import ResearchMetadata


@pytest.fixture
def mock_dependencies() -> ResearchDependencies:
    """Provide shared dependencies for agent creation tests."""
    return ResearchDependencies(
        http_client=MagicMock(),
        api_keys=APIKeys(),
        research_state=ResearchState(
            request_id="factory-test",
            user_query="How does AI impact healthcare?",
            current_stage=ResearchStage.CLARIFICATION,
            metadata=ResearchMetadata(),
        ),
    )


def test_create_agents(mock_dependencies: ResearchDependencies) -> None:
    """Ensure each agent type can be instantiated via the factory."""
    for agent_type in [
        AgentType.CLARIFICATION,
        AgentType.QUERY_TRANSFORMATION,
        AgentType.RESEARCH_EXECUTOR,
        AgentType.REPORT_GENERATOR,
    ]:
        agent = AgentFactory.create_agent(agent_type, mock_dependencies)
        assert agent is not None
        assert hasattr(agent, "run")


def test_get_available_agents() -> None:
    """Factory should report available agent types in the new pipeline."""
    available: List[AgentType] = AgentFactory.get_available_agents()
    assert set(available) == {
        AgentType.CLARIFICATION,
        AgentType.QUERY_TRANSFORMATION,
        AgentType.RESEARCH_EXECUTOR,
        AgentType.REPORT_GENERATOR,
    }


def test_default_config_round_trip() -> None:
    """Setting and retrieving default configuration should round-trip."""
    config = AgentConfiguration(
        agent_name="research_executor",
        agent_type=AgentType.RESEARCH_EXECUTOR.value,
        max_retries=5,
        timeout_seconds=120.0,
    )
    AgentFactory.set_default_config(AgentType.RESEARCH_EXECUTOR, config)
    retrieved = AgentFactory.get_default_config(AgentType.RESEARCH_EXECUTOR)
    assert retrieved == config


def test_create_agent_batch(mock_dependencies: ResearchDependencies) -> None:
    """Batch agent creation should succeed for supported agents."""
    specs = [
        (AgentType.CLARIFICATION, mock_dependencies, None),
        (
            AgentType.QUERY_TRANSFORMATION,
            mock_dependencies,
            AgentConfiguration(
                agent_name="query_transformation",
                agent_type=AgentType.QUERY_TRANSFORMATION.value,
            ),
        ),
        (AgentType.RESEARCH_EXECUTOR, mock_dependencies, None),
    ]

    agents = AgentFactory.create_agent_batch(specs)
    assert len(agents) == 3
    assert all(hasattr(agent, "run") for agent in agents)


def test_convenience_creators(mock_dependencies: ResearchDependencies) -> None:
    """Convenience factory helpers should return functional agents."""
    assert hasattr(create_clarification_agent(mock_dependencies), "run")
    assert hasattr(create_query_transformation_agent(mock_dependencies), "run")
    assert hasattr(create_research_executor_agent(mock_dependencies), "run")
    assert hasattr(create_report_generator_agent(mock_dependencies), "run")
