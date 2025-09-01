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
    BaseResearchAgent,
    AgentConfiguration,
    AgentStatus,
    PerformanceMetrics,
    AgentError,
    AgentExecutionError,
    AgentValidationError,
    AgentConfigurationError,
    AgentTimeoutError,
    ToolMixin,
    ConversationMixin,
    PerformanceMonitoringMixin,
)

# Factory system for agent creation and management
from .factory import (
    AgentType,
    AgentPoolConfig,
    AgentFactory,
    AgentRegistry,
    AgentPool,
    FactoryError,
    AgentRegistrationError,
    AgentCreationError,
    AgentPoolError,
    get_agent_factory,
    create_agent,
    register_agent_type,
    get_agent_type_from_string,
)

# Legacy imports for backward compatibility (to be deprecated)
try:
    from open_deep_research_with_pydantic_ai.agents.base import (
        ResearchDependencies,
        coordinator,
    )
    from open_deep_research_with_pydantic_ai.agents.brief_generator import brief_generator_agent
    from open_deep_research_with_pydantic_ai.agents.clarification import clarification_agent
    from open_deep_research_with_pydantic_ai.agents.compression import compression_agent
    from open_deep_research_with_pydantic_ai.agents.report_generator import report_generator_agent
    from open_deep_research_with_pydantic_ai.agents.research_executor import research_executor_agent
except ImportError:
    # These may not exist yet during migration
    pass

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
]

# Legacy exports (backward compatibility - to be deprecated)
legacy_exports = [
    "ResearchDependencies",
    "coordinator",
    "brief_generator_agent",
    "clarification_agent",
    "compression_agent",
    "report_generator_agent",
    "research_executor_agent",
]

# Add legacy exports if they exist
for export in legacy_exports:
    if export in globals():
        __all__.append(export)