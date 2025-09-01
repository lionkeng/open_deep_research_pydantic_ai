"""Agents package for Open Deep Research with Pydantic AI.

This package contains all research agents with enhanced base functionality including:
- Type-safe generic base agent with template method pattern
- Advanced error handling with retry mechanisms
- Performance monitoring and metrics collection
- Tool management and conversation context handling
- Agent factory pattern for creation and pooling
- Full integration with Pydantic AI framework and Logfire logging
"""

# Enhanced base agent system
from .base import (
    AgentConfiguration,
    AgentConfigurationError,
    AgentError,
    AgentExecutionError,
    AgentStatus,
    AgentTimeoutError,
    AgentValidationError,
    BaseResearchAgent,
    ConversationMixin,
    PerformanceMetrics,
    PerformanceMonitoringMixin,
    ToolMixin,
)

# Phase 2 agent imports
from .brief_generator import brief_generator_agent
from .compression import compression_agent

# Factory system for agent creation and management
from .factory import (
    AgentCreationError,
    AgentFactory,
    AgentPool,
    AgentPoolConfig,
    AgentPoolError,
    AgentRegistrationError,
    AgentRegistry,
    AgentType,
    FactoryError,
    create_agent,
    get_agent_factory,
    get_agent_type_from_string,
    register_agent_type,
)
from .query_transformation import query_transformation_agent
from .report_generator import report_generator_agent
from .research_executor import research_executor_agent

__all__ = [
    # Enhanced base agent system
    "BaseResearchAgent",
    "AgentConfiguration",
    "AgentStatus",
    "PerformanceMetrics",
    "AgentError",
    "AgentExecutionError",
    "AgentValidationError",
    "AgentConfigurationError",
    "AgentTimeoutError",
    "ToolMixin",
    "ConversationMixin",
    "PerformanceMonitoringMixin",
    # Factory system
    "AgentType",
    "AgentPoolConfig",
    "AgentFactory",
    "AgentRegistry",
    "AgentPool",
    "FactoryError",
    "AgentRegistrationError",
    "AgentCreationError",
    "AgentPoolError",
    # Convenience functions
    "get_agent_factory",
    "create_agent",
    "register_agent_type",
    "get_agent_type_from_string",
    # Phase 2 agents
    "query_transformation_agent",
    "brief_generator_agent",
    "research_executor_agent",
    "compression_agent",
    "report_generator_agent",
]
