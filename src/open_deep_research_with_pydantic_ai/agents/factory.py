"""Agent factory pattern for creating and managing research agents.

This module provides a comprehensive factory system for research agents including:
- Type-safe agent registration and creation
- Agent pooling for performance optimization
- Batch agent creation capabilities
- Configuration management and validation
- Integration with the enhanced base agent system
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any

import logfire
from pydantic import BaseModel, ConfigDict, Field

from .base import (
    AgentConfiguration,
    AgentError, 
    BaseResearchAgent,
    ResearchDependencies,
)


class AgentType(str, Enum):
    """Enumeration of available research agent types."""

    CLARIFICATION = "clarification"
    BRIEF_GENERATOR = "brief_generator"
    QUERY_TRANSFORMATION = "query_transformation"
    RESEARCH_EXECUTOR = "research_executor"
    COMPRESSION = "compression"
    REPORT_GENERATOR = "report_generator"
    VALIDATION = "validation"
    SYNTHESIS = "synthesis"


class AgentPoolConfig(BaseModel):
    """Configuration for agent pooling behavior."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    max_size: int = Field(default=10, ge=1, le=100, description="Maximum pool size")
    min_size: int = Field(default=1, ge=0, description="Minimum pool size")
    initial_size: int = Field(default=2, ge=0, description="Initial pool size")
    max_idle_time: float = Field(default=300.0, gt=0, description="Max idle time in seconds")
    enable_monitoring: bool = Field(default=True, description="Enable pool monitoring")
    cleanup_interval: float = Field(default=60.0, gt=0, description="Cleanup interval in seconds")

    def model_post_init(self, __context: Any) -> None:
        """Validate pool configuration constraints."""
        if self.min_size > self.max_size:
            raise ValueError("min_size cannot be greater than max_size")
        if self.initial_size > self.max_size:
            raise ValueError("initial_size cannot be greater than max_size")


class FactoryError(AgentError):
    """Base exception for factory-related errors."""
    pass


class AgentRegistrationError(FactoryError):
    """Exception raised during agent registration."""
    pass


class AgentCreationError(FactoryError):
    """Exception raised during agent creation."""
    pass


class AgentPoolError(FactoryError):
    """Exception raised during pool operations."""
    pass


@dataclass
class PooledAgent:
    """Wrapper for pooled agent instances."""

    agent: BaseResearchAgent
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    usage_count: int = 0

    def mark_used(self) -> None:
        """Mark agent as recently used."""
        self.last_used = time.time()
        self.usage_count += 1

    def is_expired(self, max_idle_time: float) -> bool:
        """Check if agent has exceeded idle time."""
        return time.time() - self.last_used > max_idle_time


class AgentRegistry:
    """Registry for managing agent types and their configurations."""

    def __init__(self):
        self._agent_classes: dict[AgentType, type[BaseResearchAgent]] = {}
        self._default_configs: dict[AgentType, AgentConfiguration] = {}
        self._factory_functions: dict[AgentType, Callable] = {}
        self._pool_configs: dict[AgentType, AgentPoolConfig] = {}

    def register_agent_type(
        self,
        agent_type: AgentType,
        agent_class: type[BaseResearchAgent],
        default_config: AgentConfiguration | None = None,
        pool_config: AgentPoolConfig | None = None,
        factory_function: Callable | None = None
    ) -> None:
        """Register an agent type with the registry.
        
        Args:
            agent_type: Type of agent to register
            agent_class: Agent class (must inherit from BaseResearchAgent)
            default_config: Default configuration for this agent type
            pool_config: Pool configuration for this agent type
            factory_function: Custom factory function (optional)
            
        Raises:
            AgentRegistrationError: If registration fails
        """
        try:
            # Validate agent class
            if not issubclass(agent_class, BaseResearchAgent):
                raise AgentRegistrationError(
                    "Agent class must inherit from BaseResearchAgent",
                    "registry",
                    {"agent_type": agent_type.value, "agent_class": agent_class.__name__}
                )

            # Store registration data
            self._agent_classes[agent_type] = agent_class

            if default_config:
                self._default_configs[agent_type] = default_config

            if pool_config:
                self._pool_configs[agent_type] = pool_config

            if factory_function:
                self._factory_functions[agent_type] = factory_function

            logfire.info(
                "Agent type registered successfully",
                agent_type=agent_type.value,
                agent_class=agent_class.__name__,
                has_default_config=default_config is not None,
                has_pool_config=pool_config is not None,
                has_factory_function=factory_function is not None
            )

        except Exception as e:
            raise AgentRegistrationError(
                f"Failed to register agent type '{agent_type.value}': {str(e)}",
                "registry",
                {"agent_type": agent_type.value, "error": str(e)}
            ) from e

    def get_agent_class(self, agent_type: AgentType) -> type[BaseResearchAgent] | None:
        """Get registered agent class for a type."""
        return self._agent_classes.get(agent_type)

    def get_default_config(self, agent_type: AgentType) -> AgentConfiguration | None:
        """Get default configuration for an agent type."""
        return self._default_configs.get(agent_type)

    def get_pool_config(self, agent_type: AgentType) -> AgentPoolConfig | None:
        """Get pool configuration for an agent type."""
        return self._pool_configs.get(agent_type)

    def get_factory_function(self, agent_type: AgentType) -> Callable | None:
        """Get custom factory function for an agent type."""
        return self._factory_functions.get(agent_type)

    def is_registered(self, agent_type: AgentType) -> bool:
        """Check if an agent type is registered."""
        return agent_type in self._agent_classes

    def list_registered_types(self) -> list[AgentType]:
        """List all registered agent types."""
        return list(self._agent_classes.keys())

    def get_registration_stats(self) -> dict[str, Any]:
        """Get registration statistics."""
        return {
            "total_registered": len(self._agent_classes),
            "with_default_config": len(self._default_configs),
            "with_pool_config": len(self._pool_configs),
            "with_factory_function": len(self._factory_functions),
            "registered_types": [t.value for t in self._agent_classes.keys()]
        }


class AgentPool:
    """Pool manager for a specific agent type."""

    def __init__(self, agent_type: AgentType, config: AgentPoolConfig):
        self.agent_type = agent_type
        self.config = config
        self._pool: list[PooledAgent] = []
        self._lock = asyncio.Lock()
        self._stats = {
            "created": 0,
            "reused": 0,
            "expired": 0,
            "errors": 0
        }
        self._cleanup_task: asyncio.Task | None = None

        # Start cleanup task if monitoring enabled
        if config.enable_monitoring:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def get_agent(
        self,
        agent_class: type[BaseResearchAgent],
        config: AgentConfiguration,
        dependencies: ResearchDependencies
    ) -> BaseResearchAgent:
        """Get an agent from the pool or create a new one.
        
        Args:
            agent_class: Agent class to instantiate
            config: Agent configuration
            dependencies: Agent dependencies
            
        Returns:
            Agent instance (pooled or new)
            
        Raises:
            AgentPoolError: If agent creation fails
        """
        async with self._lock:
            try:
                # Try to get from pool first
                for pooled_agent in self._pool[:]:
                    if not pooled_agent.is_expired(self.config.max_idle_time):
                        self._pool.remove(pooled_agent)
                        pooled_agent.mark_used()
                        self._stats["reused"] += 1

                        logfire.debug(
                            "Agent retrieved from pool",
                            agent_type=self.agent_type.value,
                            pool_size=len(self._pool),
                            usage_count=pooled_agent.usage_count
                        )

                        return pooled_agent.agent

                # Create new agent if pool empty or all expired
                agent = agent_class(config=config, dependencies=dependencies)
                self._stats["created"] += 1

                logfire.debug(
                    "New agent created for pool",
                    agent_type=self.agent_type.value,
                    pool_size=len(self._pool)
                )

                return agent

            except Exception as e:
                self._stats["errors"] += 1
                raise AgentPoolError(
                    f"Failed to get agent from pool: {str(e)}",
                    "pool_manager",
                    {"agent_type": self.agent_type.value, "error": str(e)}
                ) from e

    async def return_agent(self, agent: BaseResearchAgent) -> None:
        """Return an agent to the pool.
        
        Args:
            agent: Agent to return to pool
        """
        async with self._lock:
            try:
                if len(self._pool) < self.config.max_size:
                    pooled_agent = PooledAgent(agent=agent)
                    self._pool.append(pooled_agent)

                    logfire.debug(
                        "Agent returned to pool",
                        agent_type=self.agent_type.value,
                        pool_size=len(self._pool)
                    )
                else:
                    logfire.debug(
                        "Agent discarded (pool full)",
                        agent_type=self.agent_type.value,
                        pool_size=len(self._pool),
                        max_size=self.config.max_size
                    )
            except Exception as e:
                logfire.error(
                    "Failed to return agent to pool",
                    agent_type=self.agent_type.value,
                    error=str(e)
                )

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of expired agents."""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._cleanup_expired_agents()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logfire.error("Pool cleanup error", error=str(e))

    async def _cleanup_expired_agents(self) -> None:
        """Remove expired agents from the pool."""
        async with self._lock:
            expired_count = 0
            self._pool = [
                pooled_agent for pooled_agent in self._pool
                if not pooled_agent.is_expired(self.config.max_idle_time)
            ]

            if expired_count > 0:
                self._stats["expired"] += expired_count
                logfire.debug(
                    "Cleaned up expired agents",
                    agent_type=self.agent_type.value,
                    expired_count=expired_count,
                    remaining_count=len(self._pool)
                )

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        return {
            "agent_type": self.agent_type.value,
            "current_size": len(self._pool),
            "max_size": self.config.max_size,
            "min_size": self.config.min_size,
            "stats": self._stats.copy(),
            "config": self.config.model_dump()
        }

    async def shutdown(self) -> None:
        """Shutdown the pool and cleanup resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        async with self._lock:
            self._pool.clear()

        logfire.info("Agent pool shut down", agent_type=self.agent_type.value)


class AgentFactory:
    """Main factory for creating and managing research agents."""

    def __init__(self):
        self.registry = AgentRegistry()
        self._pools: dict[AgentType, AgentPool] = {}
        self._global_stats = {
            "agents_created": 0,
            "agents_from_pool": 0,
            "factory_errors": 0
        }

    def register_agent_type(
        self,
        agent_type: AgentType,
        agent_class: type[BaseResearchAgent],
        default_config: AgentConfiguration | None = None,
        pool_config: AgentPoolConfig | None = None,
        factory_function: Callable | None = None
    ) -> None:
        """Register an agent type with the factory.
        
        Args:
            agent_type: Type of agent to register
            agent_class: Agent class implementation
            default_config: Default configuration
            pool_config: Pool configuration
            factory_function: Custom factory function
        """
        # Register with registry
        self.registry.register_agent_type(
            agent_type, agent_class, default_config, pool_config, factory_function
        )

        # Create pool if configured
        if pool_config:
            self._pools[agent_type] = AgentPool(agent_type, pool_config)

    async def create_agent(
        self,
        agent_type: AgentType,
        dependencies: ResearchDependencies,
        config: AgentConfiguration | None = None,
        use_pool: bool = True
    ) -> BaseResearchAgent:
        """Create an agent instance.
        
        Args:
            agent_type: Type of agent to create
            dependencies: Agent dependencies
            config: Optional configuration override
            use_pool: Whether to use agent pooling
            
        Returns:
            Agent instance
            
        Raises:
            AgentCreationError: If agent creation fails
        """
        try:
            # Validate agent type is registered
            if not self.registry.is_registered(agent_type):
                available = [t.value for t in self.registry.list_registered_types()]
                raise AgentCreationError(
                    f"Agent type '{agent_type.value}' not registered. Available: {available}",
                    "factory",
                    {"agent_type": agent_type.value, "available_types": available}
                )

            # Get agent class and configuration
            agent_class = self.registry.get_agent_class(agent_type)
            if not agent_class:
                raise AgentCreationError(
                    f"No agent class found for type '{agent_type.value}'",
                    "factory",
                    {"agent_type": agent_type.value}
                )

            # Use provided config or default
            effective_config = config or self.registry.get_default_config(agent_type)
            if not effective_config:
                # Create basic config
                effective_config = AgentConfiguration(
                    agent_name=f"{agent_type.value}_agent",
                    agent_type=agent_type.value
                )

            # Try pool first if enabled
            if use_pool and agent_type in self._pools:
                agent = await self._pools[agent_type].get_agent(
                    agent_class, effective_config, dependencies
                )
                self._global_stats["agents_from_pool"] += 1
                return agent

            # Use custom factory function if available
            factory_fn = self.registry.get_factory_function(agent_type)
            if factory_fn:
                agent = factory_fn(config=effective_config, dependencies=dependencies)
            else:
                # Standard creation
                agent = agent_class(config=effective_config, dependencies=dependencies)

            self._global_stats["agents_created"] += 1

            logfire.info(
                "Agent created successfully",
                agent_type=agent_type.value,
                agent_name=effective_config.agent_name,
                from_pool=False
            )

            return agent

        except Exception as e:
            self._global_stats["factory_errors"] += 1

            if isinstance(e, AgentCreationError | AgentRegistrationError):
                raise

            raise AgentCreationError(
                f"Failed to create agent of type '{agent_type.value}': {str(e)}",
                "factory",
                {"agent_type": agent_type.value, "error": str(e)}
            ) from e

    async def create_agents_batch(
        self,
        specs: list[dict[str, Any]],
        dependencies: ResearchDependencies,
        parallel: bool = True
    ) -> list[BaseResearchAgent]:
        """Create multiple agents in batch.
        
        Args:
            specs: List of agent specifications with 'type' and optional 'config'
            dependencies: Shared dependencies for all agents
            parallel: Whether to create agents in parallel
            
        Returns:
            List of created agents
        """
        if parallel:
            return await self._create_agents_parallel(specs, dependencies)
        else:
            return await self._create_agents_sequential(specs, dependencies)

    async def _create_agents_sequential(
        self,
        specs: list[dict[str, Any]],
        dependencies: ResearchDependencies
    ) -> list[BaseResearchAgent]:
        """Create agents sequentially."""
        agents = []
        for spec in specs:
            agent = await self.create_agent(
                AgentType(spec["type"]),
                dependencies,
                spec.get("config"),
                spec.get("use_pool", True)
            )
            agents.append(agent)
        return agents

    async def _create_agents_parallel(
        self,
        specs: list[dict[str, Any]],
        dependencies: ResearchDependencies
    ) -> list[BaseResearchAgent]:
        """Create agents in parallel."""
        tasks = []
        for spec in specs:
            task = asyncio.create_task(
                self.create_agent(
                    AgentType(spec["type"]),
                    dependencies,
                    spec.get("config"),
                    spec.get("use_pool", True)
                )
            )
            tasks.append(task)

        return await asyncio.gather(*tasks)

    async def return_agent_to_pool(
        self,
        agent: BaseResearchAgent,
        agent_type: AgentType
    ) -> None:
        """Return an agent to its pool.
        
        Args:
            agent: Agent to return
            agent_type: Type of agent
        """
        if agent_type in self._pools:
            await self._pools[agent_type].return_agent(agent)

    def get_factory_stats(self) -> dict[str, Any]:
        """Get comprehensive factory statistics."""
        pool_stats = {}
        for agent_type, pool in self._pools.items():
            pool_stats[agent_type.value] = pool.get_stats()

        return {
            "global_stats": self._global_stats.copy(),
            "registry_stats": self.registry.get_registration_stats(),
            "pool_stats": pool_stats,
            "registered_types": [t.value for t in self.registry.list_registered_types()]
        }

    async def shutdown(self) -> None:
        """Shutdown the factory and all pools."""
        shutdown_tasks = [pool.shutdown() for pool in self._pools.values()]
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)

        self._pools.clear()
        logfire.info("Agent factory shut down")


# Global factory instance
_global_factory: AgentFactory | None = None


def get_agent_factory() -> AgentFactory:
    """Get the global agent factory instance.
    
    Returns:
        Global factory instance (created if needed)
    """
    global _global_factory
    if _global_factory is None:
        _global_factory = AgentFactory()
        logfire.info("Global agent factory created")
    return _global_factory


async def create_agent(
    agent_type: AgentType,
    dependencies: ResearchDependencies,
    config: AgentConfiguration | None = None,
    use_pool: bool = True
) -> BaseResearchAgent:
    """Convenience function to create an agent using the global factory.
    
    Args:
        agent_type: Type of agent to create
        dependencies: Agent dependencies  
        config: Optional configuration override
        use_pool: Whether to use agent pooling
        
    Returns:
        Created agent instance
    """
    factory = get_agent_factory()
    return await factory.create_agent(agent_type, dependencies, config, use_pool)


def register_agent_type(
    agent_type: AgentType,
    agent_class: type[BaseResearchAgent],
    default_config: AgentConfiguration | None = None,
    pool_config: AgentPoolConfig | None = None,
    factory_function: Callable | None = None
) -> None:
    """Convenience function to register an agent type with the global factory.
    
    Args:
        agent_type: Type of agent to register
        agent_class: Agent class implementation
        default_config: Default configuration
        pool_config: Pool configuration
        factory_function: Custom factory function
    """
    factory = get_agent_factory()
    factory.register_agent_type(
        agent_type, agent_class, default_config, pool_config, factory_function
    )


@lru_cache(maxsize=32)
def get_agent_type_from_string(type_str: str) -> AgentType:
    """Get AgentType enum from string with caching.
    
    Args:
        type_str: Agent type as string
        
    Returns:
        AgentType enum value
        
    Raises:
        ValueError: If type string is invalid
    """
    try:
        return AgentType(type_str)
    except ValueError:
        available = [t.value for t in AgentType]
        raise ValueError(f"Invalid agent type '{type_str}'. Available: {available}") from None
