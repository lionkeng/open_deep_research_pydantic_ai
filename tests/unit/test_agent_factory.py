"""
Comprehensive tests for the AgentFactory.
"""

import pytest
from typing import Dict, Any
from unittest.mock import MagicMock, patch

from agents.factory import AgentFactory, AgentType
from agents.base import (
    BaseResearchAgent,
    ResearchDependencies,
    AgentConfiguration
)
from models.api_models import APIKeys
from models.metadata import ResearchMetadata
from models.core import ResearchState, ResearchStage


class TestAgentFactory:
    """Test suite for AgentFactory."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for testing."""
        return ResearchDependencies(
            http_client=MagicMock(),
            api_keys=APIKeys(),
            research_state=ResearchState(
                request_id="test-factory",
                user_id="test-user",
                session_id="test-session",
                user_query="Test query",
                current_stage=ResearchStage.CLARIFICATION,
                metadata=ResearchMetadata()
            ),
            usage=None
        )

    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        return AgentConfiguration(
            max_retries=5,
            timeout_seconds=60.0,
            temperature=0.8
        )

    def test_factory_singleton_pattern(self):
        """Test that factory uses appropriate patterns."""
        # AgentFactory should be accessible via class methods
        assert hasattr(AgentFactory, 'create_agent')
        assert hasattr(AgentFactory, 'register_default_config')
        assert hasattr(AgentFactory, 'create_agent_batch')

    def test_create_clarification_agent(self, mock_dependencies):
        """Test creation of clarification agent."""
        agent = AgentFactory.create_agent(
            AgentType.CLARIFICATION,
            mock_dependencies
        )

        assert agent is not None
        assert hasattr(agent, 'execute')
        assert hasattr(agent, 'agent_name')

    def test_create_query_transformation_agent(self, mock_dependencies):
        """Test creation of query transformation agent."""
        agent = AgentFactory.create_agent(
            AgentType.QUERY_TRANSFORMATION,
            mock_dependencies
        )

        assert agent is not None
        assert hasattr(agent, 'execute')


    def test_create_research_executor_agent(self, mock_dependencies):
        """Test creation of research executor agent."""
        agent = AgentFactory.create_agent(
            AgentType.RESEARCH_EXECUTOR,
            mock_dependencies
        )

        assert agent is not None
        assert hasattr(agent, 'execute')

    def test_create_compression_agent(self, mock_dependencies):
        """Test creation of compression agent."""
        agent = AgentFactory.create_agent(
            AgentType.COMPRESSION,
            mock_dependencies
        )

        assert agent is not None
        assert hasattr(agent, 'execute')

    def test_create_report_generator_agent(self, mock_dependencies):
        """Test creation of report generator agent."""
        agent = AgentFactory.create_agent(
            AgentType.REPORT_GENERATOR,
            mock_dependencies
        )

        assert agent is not None
        assert hasattr(agent, 'execute')

    def test_create_agent_with_custom_config(self, mock_dependencies, test_config):
        """Test agent creation with custom configuration."""
        agent = AgentFactory.create_agent(
            AgentType.CLARIFICATION,
            mock_dependencies,
            config=test_config
        )

        assert agent is not None
        assert agent.config.max_retries == 5
        assert agent.config.timeout_seconds == 60.0

    def test_invalid_agent_type(self, mock_dependencies):
        """Test that invalid agent type raises error."""
        with pytest.raises(ValueError):
            AgentFactory.create_agent(
                "INVALID_TYPE",  # type: ignore
                mock_dependencies
            )

    def test_register_default_config(self, test_config):
        """Test registering default configuration for agent type."""
        AgentFactory.register_default_config(
            AgentType.CLARIFICATION,
            test_config
        )

        # Verify the config is stored
        assert AgentType.CLARIFICATION in AgentFactory._default_configs
        assert AgentFactory._default_configs[AgentType.CLARIFICATION] == test_config

    def test_create_agent_uses_default_config(self, mock_dependencies, test_config):
        """Test that agents use registered default configs."""
        # Register a default config
        AgentFactory.register_default_config(
            AgentType.COMPRESSION,
            test_config
        )

        # Create agent without specifying config
        agent = AgentFactory.create_agent(
            AgentType.COMPRESSION,
            mock_dependencies
        )

        assert agent.config.max_retries == test_config.max_retries
        assert agent.config.timeout_seconds == test_config.timeout_seconds

    def test_batch_agent_creation(self, mock_dependencies, test_config):
        """Test creating multiple agents in batch."""
        agent_specs = [
            (AgentType.CLARIFICATION, mock_dependencies, None),
            (AgentType.QUERY_TRANSFORMATION, mock_dependencies, test_config),
            (AgentType.COMPRESSION, mock_dependencies, None),
        ]

        agents = AgentFactory.create_agent_batch(agent_specs)

        assert len(agents) == 3
        assert all(agent is not None for agent in agents)
        assert all(hasattr(agent, 'execute') for agent in agents)

    def test_agent_type_enum_coverage(self):
        """Test that all agent types in enum are supported."""
        expected_types = [
            AgentType.CLARIFICATION,
            AgentType.QUERY_TRANSFORMATION,
            AgentType.RESEARCH_EXECUTOR,
            AgentType.COMPRESSION,
            AgentType.REPORT_GENERATOR
        ]

        for agent_type in expected_types:
            assert agent_type in AgentType

    def test_factory_agent_registry(self):
        """Test that factory maintains proper agent registry."""
        registry = AgentFactory._agent_registry

        # Check that all agent types are registered
        assert AgentType.CLARIFICATION in registry
        assert AgentType.QUERY_TRANSFORMATION in registry
        assert AgentType.RESEARCH_EXECUTOR in registry
        assert AgentType.COMPRESSION in registry
        assert AgentType.REPORT_GENERATOR in registry

    def test_agent_creation_with_dependencies(self, mock_dependencies):
        """Test that agents receive dependencies correctly."""
        agent = AgentFactory.create_agent(
            AgentType.CLARIFICATION,
            mock_dependencies
        )

        # Agent should have access to dependencies
        assert agent._deps is not None
        assert agent._deps.api_keys is not None
        assert agent._deps.research_state is not None

    def test_configuration_override(self, mock_dependencies):
        """Test that custom config overrides defaults."""
        # Set a default
        default_config = AgentConfiguration(
            max_retries=3,
            timeout_seconds=30.0
        )
        AgentFactory.register_default_config(
            AgentType.COMPRESSION,
            default_config
        )

        # Create with override
        override_config = AgentConfiguration(
            max_retries=10,
            timeout_seconds=120.0
        )
        agent = AgentFactory.create_agent(
            AgentType.COMPRESSION,
            mock_dependencies,
            config=override_config
        )

        # Override should take precedence
        assert agent.config.max_retries == 10
        assert agent.config.timeout_seconds == 120.0

    def test_factory_error_handling(self, mock_dependencies):
        """Test factory error handling."""
        # Test with None dependencies
        with pytest.raises(Exception):
            AgentFactory.create_agent(
                AgentType.CLARIFICATION,
                None  # type: ignore
            )

    def test_agent_type_from_string(self):
        """Test conversion from string to AgentType."""
        test_cases = [
            ("CLARIFICATION", AgentType.CLARIFICATION),
            ("QUERY_TRANSFORMATION", AgentType.QUERY_TRANSFORMATION),
            ("RESEARCH_EXECUTOR", AgentType.RESEARCH_EXECUTOR),
            ("COMPRESSION", AgentType.COMPRESSION),
            ("REPORT_GENERATOR", AgentType.REPORT_GENERATOR),
        ]

        for string_value, expected_type in test_cases:
            assert AgentType(string_value) == expected_type

    @pytest.mark.parametrize("agent_type", list(AgentType))
    def test_all_agent_types_creatable(self, agent_type, mock_dependencies):
        """Test that all agent types can be created."""
        agent = AgentFactory.create_agent(
            agent_type,
            mock_dependencies
        )

        assert agent is not None
        assert isinstance(agent, BaseResearchAgent)

    def test_agent_factory_convenience_functions(self, mock_dependencies):
        """Test convenience functions for agent creation."""
        # Test individual convenience functions
        clarification = create_clarification_agent(mock_dependencies)
        assert clarification is not None

        query_transform = create_query_transformation_agent(mock_dependencies)
        assert query_transform is not None

        research_exec = create_research_executor_agent(mock_dependencies)
        assert research_exec is not None

        compression = create_compression_agent(mock_dependencies)
        assert compression is not None

        report_gen = create_report_generator_agent(mock_dependencies)
        assert report_gen is not None

    def test_factory_thread_safety(self, mock_dependencies):
        """Test that factory is thread-safe for concurrent creation."""
        import threading
        import time

        agents = []
        errors = []

        def create_agent():
            try:
                agent = AgentFactory.create_agent(
                    AgentType.CLARIFICATION,
                    mock_dependencies
                )
                agents.append(agent)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for _ in range(10):
            t = threading.Thread(target=create_agent)
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Check results
        assert len(errors) == 0
        assert len(agents) == 10
        assert all(agent is not None for agent in agents)


# Import convenience functions for testing
from agents.factory import (
    create_clarification_agent,
    create_query_transformation_agent,
    create_research_executor_agent,
    create_compression_agent,
    create_report_generator_agent
)
