"""Agents package for Open Deep Research with Pydantic AI.

This package contains all research agents with enhanced base functionality including:
- Type-safe generic base agent with template method pattern
- Advanced error handling with retry mechanisms
- Performance monitoring and metrics collection
- Tool management and conversation context handling
- Simple agent factory pattern for creation
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
from .compression import compression_agent

# Factory system for agent creation and management
from .factory import (
    AgentFactory,
    AgentType,
    create_clarification_agent,
    create_compression_agent,
    create_query_transformation_agent,
    create_report_generator_agent,
    create_research_executor_agent,
)
from .query_transformation import query_transformation_agent
from .report_generator import report_generator_agent
from .research_executor import (
    ResearchExecutorAgent,
    get_research_executor_agent,
    research_executor_agent,
)

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
    "AgentFactory",
    # Factory convenience functions
    "create_clarification_agent",
    "create_query_transformation_agent",
    "create_research_executor_agent",
    "create_compression_agent",
    "create_report_generator_agent",
    # Phase 2 agents
    "query_transformation_agent",
    "research_executor_agent",
    "get_research_executor_agent",
    "ResearchExecutorAgent",
    "compression_agent",
    "report_generator_agent",
]
