"""Agent factory for creating individual agent instances."""

from enum import Enum
from typing import Any, Dict, Optional, Type

import logfire

from agents.base import AgentConfiguration, BaseResearchAgent, ResearchDependencies
from agents.brief_generator import BriefGeneratorAgent
from agents.clarification import ClarificationAgent
from agents.compression import CompressionAgent
from agents.query_transformation import QueryTransformationAgent
from agents.report_generator import ReportGeneratorAgent
from agents.research_executor import ResearchExecutorAgent


class AgentType(str, Enum):
    """Enumeration of available agent types."""
    
    CLARIFICATION = "clarification"
    QUERY_TRANSFORMATION = "query_transformation"
    BRIEF_GENERATOR = "brief_generator"
    RESEARCH_EXECUTOR = "research_executor"
    COMPRESSION = "compression"
    REPORT_GENERATOR = "report_generator"


class AgentFactory:
    """Factory for creating research agents.
    
    Simple factory pattern that creates new agent instances on demand.
    No complex pooling or caching - just straightforward agent creation.
    """
    
    # Registry of agent classes
    _agent_registry: Dict[AgentType, Type[BaseResearchAgent]] = {
        AgentType.CLARIFICATION: ClarificationAgent,
        AgentType.QUERY_TRANSFORMATION: QueryTransformationAgent,
        AgentType.BRIEF_GENERATOR: BriefGeneratorAgent,
        AgentType.RESEARCH_EXECUTOR: ResearchExecutorAgent,
        AgentType.COMPRESSION: CompressionAgent,
        AgentType.REPORT_GENERATOR: ReportGeneratorAgent,
    }
    
    # Default configurations per agent type
    _default_configs: Dict[AgentType, AgentConfiguration] = {}
    
    
    @classmethod
    def create_agent(
        cls,
        agent_type: AgentType,
        dependencies: ResearchDependencies,
        config: Optional[AgentConfiguration] = None
    ) -> BaseResearchAgent[Any, Any]:
        """Create an agent instance with proper configuration.
        
        Simple factory method that creates a new agent instance.
        
        Args:
            agent_type: Type of agent to create
            dependencies: Research dependencies for the agent
            config: Optional configuration override
            
        Returns:
            Configured agent instance
            
        Raises:
            ValueError: If agent type is not registered
        """
        return cls._create_agent_internal(agent_type, dependencies, config)
    
    @classmethod
    def _create_agent_internal(
        cls,
        agent_type: AgentType,
        dependencies: ResearchDependencies,
        config: Optional[AgentConfiguration] = None
    ) -> BaseResearchAgent[Any, Any]:
        """Internal method to create an agent instance."""
        if agent_type not in cls._agent_registry:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        agent_class = cls._agent_registry[agent_type]
        
        # Use provided config or default
        effective_config = config or cls.get_default_config(agent_type)
        if not effective_config:
            effective_config = AgentConfiguration(
                agent_name=f"{agent_type.value}_agent",
                agent_type=agent_type.value
            )
        
        logfire.debug(f"Creating new {agent_type.value} agent instance")
        
        # Create agent instance - simple and direct
        agent = agent_class(config=effective_config)
        
        # Initialize with dependencies
        agent._deps = dependencies
        
        return agent
    
    @classmethod
    def register_agent(
        cls,
        agent_type: AgentType,
        agent_class: Type[BaseResearchAgent],
        default_config: Optional[AgentConfiguration] = None
    ) -> None:
        """Register a new agent type with the factory.
        
        Args:
            agent_type: Type identifier for the agent
            agent_class: Agent class to register
            default_config: Optional default configuration
        """
        cls._agent_registry[agent_type] = agent_class
        if default_config:
            cls._default_configs[agent_type] = default_config
        logfire.info(f"Registered agent type: {agent_type.value}")
    
    @classmethod
    def get_default_config(cls, agent_type: AgentType) -> Optional[AgentConfiguration]:
        """Get default configuration for an agent type.
        
        Args:
            agent_type: Type of agent
            
        Returns:
            Default configuration or None
        """
        return cls._default_configs.get(agent_type)
    
    @classmethod
    def set_default_config(
        cls,
        agent_type: AgentType,
        config: AgentConfiguration
    ) -> None:
        """Set default configuration for an agent type.
        
        Args:
            agent_type: Type of agent
            config: Configuration to set as default
        """
        cls._default_configs[agent_type] = config
    
    @classmethod
    def get_available_agents(cls) -> list[AgentType]:
        """Get list of available agent types.
        
        Returns:
            List of registered agent types
        """
        return list(cls._agent_registry.keys())
    
    @classmethod
    def create_agent_batch(
        cls,
        agent_configs: list[tuple[AgentType, ResearchDependencies, Optional[AgentConfiguration]]]
    ) -> list[BaseResearchAgent[Any, Any]]:
        """Create multiple agents in batch.
        
        Args:
            agent_configs: List of (agent_type, dependencies, config) tuples
            
        Returns:
            List of created agents
        """
        agents = []
        for agent_type, deps, config in agent_configs:
            try:
                agent = cls.create_agent(agent_type, deps, config)
                agents.append(agent)
            except Exception as e:
                logfire.error(f"Failed to create {agent_type.value} agent: {e}")
                raise
        
        return agents


# Convenience functions for creating specific agents
def create_clarification_agent(
    dependencies: ResearchDependencies,
    config: Optional[AgentConfiguration] = None
) -> ClarificationAgent:
    """Create a clarification agent."""
    return AgentFactory.create_agent(AgentType.CLARIFICATION, dependencies, config)


def create_query_transformation_agent(
    dependencies: ResearchDependencies,
    config: Optional[AgentConfiguration] = None
) -> QueryTransformationAgent:
    """Create a query transformation agent."""
    return AgentFactory.create_agent(AgentType.QUERY_TRANSFORMATION, dependencies, config)


def create_brief_generator_agent(
    dependencies: ResearchDependencies,
    config: Optional[AgentConfiguration] = None
) -> BriefGeneratorAgent:
    """Create a brief generator agent."""
    return AgentFactory.create_agent(AgentType.BRIEF_GENERATOR, dependencies, config)


def create_research_executor_agent(
    dependencies: ResearchDependencies,
    config: Optional[AgentConfiguration] = None
) -> ResearchExecutorAgent:
    """Create a research executor agent."""
    return AgentFactory.create_agent(AgentType.RESEARCH_EXECUTOR, dependencies, config)


def create_compression_agent(
    dependencies: ResearchDependencies,
    config: Optional[AgentConfiguration] = None
) -> CompressionAgent:
    """Create a compression agent."""
    return AgentFactory.create_agent(AgentType.COMPRESSION, dependencies, config)


def create_report_generator_agent(
    dependencies: ResearchDependencies,
    config: Optional[AgentConfiguration] = None
) -> ReportGeneratorAgent:
    """Create a report generator agent."""
    return AgentFactory.create_agent(AgentType.REPORT_GENERATOR, dependencies, config)