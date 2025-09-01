"""Enhanced base agent classes with dependency injection, performance monitoring, and factory support."""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, TypeVar, Generic, Callable, Awaitable

import httpx
import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import RunUsage

from open_deep_research_with_pydantic_ai.core.config import config
from open_deep_research_with_pydantic_ai.core.events import (
    AgentDelegationEvent,
    StreamingUpdateEvent,
    research_event_bus,
)
from open_deep_research_with_pydantic_ai.core.logging import configure_logging
from open_deep_research_with_pydantic_ai.models.api_models import APIKeys, ResearchMetadata
from open_deep_research_with_pydantic_ai.models.research import ResearchState


# Enhanced exception system for agent errors
class AgentError(Exception):
    """Base exception for agent-related errors."""
    
    def __init__(self, message: str, agent_name: str | None = None, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.agent_name = agent_name
        self.context = context or {}
        self.timestamp = time.time()


class AgentExecutionError(AgentError):
    """Raised when agent execution fails."""
    pass


class AgentValidationError(AgentError):
    """Raised when agent input/output validation fails."""
    pass


class AgentConfigurationError(AgentError):
    """Raised when agent configuration is invalid."""
    pass


class AgentTimeoutError(AgentError):
    """Raised when agent execution times out."""
    pass


class AgentStatus(Enum):
    """Agent execution status enumeration."""
    IDLE = auto()
    INITIALIZING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    TIMEOUT = auto()


class PerformanceMetrics(BaseModel):
    """Performance metrics for agent execution."""
    
    execution_time: float = Field(default=0.0, description="Total execution time in seconds")
    token_usage: dict[str, int] = Field(default_factory=dict, description="Token usage statistics")
    api_calls: int = Field(default=0, description="Number of API calls made")
    cache_hits: int = Field(default=0, description="Number of cache hits")
    error_count: int = Field(default=0, description="Number of errors encountered")
    retry_count: int = Field(default=0, description="Number of retries attempted")
    memory_usage: float = Field(default=0.0, description="Peak memory usage in MB")
    success_rate: float = Field(default=1.0, description="Success rate (0.0 to 1.0)")
    
    def update_execution_time(self, start_time: float) -> None:
        """Update execution time from start timestamp."""
        self.execution_time = time.time() - start_time
    
    def record_success(self) -> None:
        """Record a successful execution."""
        # Success rate calculation would need total attempts tracking
        pass
    
    def record_failure(self) -> None:
        """Record a failed execution."""
        self.error_count += 1
    
    def record_retry(self) -> None:
        """Record a retry attempt."""
        self.retry_count += 1


class AgentConfiguration(BaseModel):
    """Configuration for agent creation and behavior."""
    
    agent_name: str = Field(description="Name identifier for the agent")
    agent_type: str = Field(description="Type identifier for the agent")
    model: str | None = Field(default=None, description="LLM model to use")
    system_prompt: str | None = Field(default=None, description="System prompt override")
    timeout: float = Field(default=300.0, description="Execution timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    enable_monitoring: bool = Field(default=True, description="Enable performance monitoring")
    enable_caching: bool = Field(default=False, description="Enable response caching")
    custom_settings: dict[str, Any] = Field(default_factory=dict, description="Agent-specific settings")
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"  # Allow additional fields for agent-specific config


@dataclass
class ResearchDependencies:
    """Shared dependencies for research agents."""

    http_client: httpx.AsyncClient
    api_keys: APIKeys  # Changed from dict to typed model
    research_state: ResearchState
    metadata: ResearchMetadata | None = None  # Added typed metadata
    usage: RunUsage | None = None
    stream_callback: Any | None = None

    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = ResearchMetadata()


DepsT = TypeVar("DepsT", bound=ResearchDependencies)
OutputT = TypeVar("OutputT", bound=BaseModel)


# Mixin classes for enhanced functionality
class ToolMixin:
    """Mixin for tool management functionality."""
    
    def __init__(self):
        self._tools: dict[str, Callable] = {}
    
    def register_tool(self, name: str, tool: Callable) -> None:
        """Register a tool for the agent."""
        self._tools[name] = tool
        logfire.debug(f"Registered tool: {name}")
    
    def get_tool(self, name: str) -> Callable | None:
        """Get a registered tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())


class ConversationMixin:
    """Mixin for conversation context management."""
    
    def __init__(self):
        self._conversation_history: list[ModelMessage] = []
        self._context_window_size: int = 10
    
    def add_message(self, message: ModelMessage) -> None:
        """Add a message to conversation history."""
        self._conversation_history.append(message)
        # Trim to context window size
        if len(self._conversation_history) > self._context_window_size:
            self._conversation_history = self._conversation_history[-self._context_window_size:]
    
    def get_conversation_context(self) -> list[ModelMessage]:
        """Get recent conversation context."""
        return self._conversation_history.copy()
    
    def clear_conversation(self) -> None:
        """Clear conversation history."""
        self._conversation_history.clear()
    
    def set_context_window_size(self, size: int) -> None:
        """Set the conversation context window size."""
        self._context_window_size = max(1, size)


class PerformanceMonitoringMixin:
    """Mixin for performance monitoring functionality."""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self._execution_start_time: float | None = None
        self._hooks: dict[str, list[Callable]] = {
            "before_execution": [],
            "after_execution": [],
            "on_error": [],
            "on_retry": [],
        }
    
    def start_execution_timer(self) -> None:
        """Start timing execution."""
        self._execution_start_time = time.time()
    
    def end_execution_timer(self) -> None:
        """End timing and update metrics."""
        if self._execution_start_time:
            self.metrics.update_execution_time(self._execution_start_time)
            self._execution_start_time = None
    
    def add_hook(self, event: str, hook: Callable) -> None:
        """Add a lifecycle hook."""
        if event in self._hooks:
            self._hooks[event].append(hook)
    
    async def execute_hooks(self, event: str, context: dict[str, Any] | None = None) -> None:
        """Execute hooks for a given event."""
        for hook in self._hooks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(context or {})
                else:
                    hook(context or {})
            except Exception as e:
                logfire.error(f"Hook execution failed: {e}")


class BaseResearchAgent[DepsT: ResearchDependencies, OutputT: BaseModel](
    ABC, ToolMixin, ConversationMixin, PerformanceMonitoringMixin
):
    """Enhanced base class for all research agents with generic typing and monitoring."""

    def __init__(
        self,
        config: AgentConfiguration | None = None,
        dependencies: DepsT | None = None,
    ):
        """Initialize the enhanced research agent.

        Args:
            config: Agent configuration (uses defaults if None)
            dependencies: Shared dependencies (can be provided later)
        """
        # Initialize mixins
        ToolMixin.__init__(self)
        ConversationMixin.__init__(self)
        PerformanceMonitoringMixin.__init__(self)
        
        # Set configuration with defaults
        self.config = config or AgentConfiguration(
            agent_name=self.__class__.__name__,
            agent_type=self.__class__.__name__.lower().replace("agent", ""),
        )
        self.name = self.config.agent_name
        self.dependencies = dependencies
        self.status = AgentStatus.IDLE
        
        # Get model configuration
        from open_deep_research_with_pydantic_ai.core.config import config as global_config
        model_config = global_config.get_model_config()
        self.model = self.config.model or model_config["model"]
        self._output_type = self._get_output_type()

        # Handle output type properly
        if self._output_type is None:
            # Default to dict output
            actual_output_type = dict
        else:
            actual_output_type = self._output_type

        # Create the Pydantic AI agent with proper configuration
        self.agent: Agent[DepsT, OutputT] = Agent(
            model=self.model,
            retries=self.config.max_retries,
            deps_type=type(dependencies) if dependencies else ResearchDependencies,
            output_type=actual_output_type,
            system_prompt=self.config.system_prompt or self._get_default_system_prompt(),
        )

        # Register tools for the agent
        self._register_tools()

        # Ensure logfire is configured before logging
        configure_logging()
        logfire.info(
            f"Initialized {self.name} agent",
            agent_type=self.config.agent_type,
            model=self.model,
        )
        
        self.status = AgentStatus.IDLE

    @abstractmethod
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for this agent."""
        pass
    
    @abstractmethod
    def _get_output_type(self) -> type[OutputT] | None:
        """Get the output type for this agent. Override in subclasses."""
        pass

    def _register_tools(self) -> None:
        """Register agent-specific tools. Override in subclasses."""
        # This is intentionally empty - subclasses can override if needed
        return
    
    def update_dependencies(self, dependencies: DepsT) -> None:
        """Update agent dependencies."""
        self.dependencies = dependencies
    
    def get_agent_info(self) -> dict[str, Any]:
        """Get comprehensive agent information."""
        return {
            "name": self.name,
            "type": self.config.agent_type,
            "model": self.model,
            "status": self.status.name,
            "metrics": self.metrics.model_dump(),
            "tools": self.list_tools(),
            "config": self.config.model_dump(),
        }

    async def run(
        self,
        prompt: str,
        deps: DepsT | None = None,
        message_history: list[ModelMessage] | None = None,
        stream: bool = False,
    ) -> OutputT:
        """Run the agent with the given prompt.

        Args:
            prompt: User prompt or task description
            deps: Agent dependencies (uses stored dependencies if None)
            message_history: Previous conversation history
            stream: Whether to stream the response

        Returns:
            Agent output of the specified type
        """
        # Use provided dependencies or stored ones
        actual_deps = deps or self.dependencies
        if not actual_deps:
            raise AgentConfigurationError(
                "No dependencies provided and no stored dependencies available",
                agent_name=self.name
            )
        
        # Update status and start monitoring
        self.status = AgentStatus.RUNNING
        self.start_execution_timer()
        
        # Execute before_execution hooks
        await self.execute_hooks("before_execution", {"prompt": prompt})
        
        try:
            # Emit streaming update if callback provided
            if actual_deps.stream_callback and stream:
                await research_event_bus.emit(
                    StreamingUpdateEvent(
                        _request_id=actual_deps.research_state.request_id,
                        content=f"{self.name} processing...",
                        stage=actual_deps.research_state.current_stage,
                    )
                )

            # Run the agent with proper error handling
            try:
                result = await self.agent.run(
                    prompt,
                    deps=actual_deps,
                    message_history=message_history or self.get_conversation_context(),
                    usage=actual_deps.usage,  # Pass usage for tracking
                )
                
                # Update metrics and status on success
                self.metrics.record_success()
                self.status = AgentStatus.COMPLETED
                
            except Exception as e:
                # Record metrics for errors and retries
                self.metrics.record_failure()
                await self.execute_hooks("on_error", {"error": e})
                
                # Check if it's a recoverable error
                if "rate limit" in str(e).lower():
                    self.metrics.record_retry()
                    await self.execute_hooks("on_retry", {"reason": "rate_limit"})
                    raise ModelRetry(f"Rate limit hit, retrying: {e}") from e
                elif "timeout" in str(e).lower():
                    self.metrics.record_retry()
                    await self.execute_hooks("on_retry", {"reason": "timeout"})
                    raise ModelRetry(f"Request timeout, retrying: {e}") from e
                else:
                    self.status = AgentStatus.FAILED
                    raise AgentExecutionError(
                        f"Agent execution failed: {e}",
                        agent_name=self.name,
                        context={"prompt": prompt, "error": str(e)}
                    ) from e

            # Log completion and update conversation history
            logfire.info(
                f"{self.name} completed",
                request_id=actual_deps.research_state.request_id,
                usage=result.usage() if result.usage() else None,
                execution_time=self.metrics.execution_time,
            )
            
            # Store conversation context if message history was provided
            if message_history:
                for msg in message_history:
                    self.add_message(msg)

            return result.output

        except ModelRetry:
            # Let Pydantic-AI handle retries
            self.status = AgentStatus.RUNNING  # Keep running status during retries
            raise
        except AgentExecutionError:
            # Re-raise agent execution errors as-is
            raise
        except Exception as e:
            # Handle unexpected errors
            self.status = AgentStatus.FAILED
            self.metrics.record_failure()
            
            logfire.error(
                f"{self.name} failed",
                request_id=actual_deps.research_state.request_id if actual_deps else "unknown",
                error=str(e),
                exc_info=True,
            )
            
            raise AgentExecutionError(
                f"Unexpected error in {self.name}: {e}",
                agent_name=self.name,
                context={"prompt": prompt, "error": str(e)}
            ) from e
        
        finally:
            # Always complete monitoring and execute after hooks
            self.end_execution_timer()
            await self.execute_hooks("after_execution", {
                "status": self.status.name,
                "execution_time": self.metrics.execution_time,
            })

    async def delegate_to_agent(
        self,
        ctx: RunContext[DepsT],
        target_agent: "BaseResearchAgent[DepsT, Any]",
        prompt: str,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Delegate a task to another agent.

        Args:
            ctx: Current run context
            target_agent: Agent to delegate to
            prompt: Task prompt for the target agent
            context: Additional context for delegation

        Returns:
            Result from the delegated agent
        """
        # Emit delegation event
        await research_event_bus.emit(
            AgentDelegationEvent(
                _request_id=ctx.deps.research_state.request_id,
                from_agent=self.name,
                to_agent=target_agent.name,
                task_description=prompt,
                context=context,
            )
        )

        # Run the target agent with shared dependencies and usage tracking
        result = await target_agent.run(
            prompt=prompt,
            deps=ctx.deps,
            message_history=None,  # Start fresh for delegated task
            stream=ctx.deps.stream_callback is not None,
        )

        logfire.info(
            f"Delegation completed: {self.name} -> {target_agent.name}",
            request_id=ctx.deps.research_state.request_id,
        )

        return result


class MultiAgentCoordinator:
    """Coordinator for managing multiple research agents."""

    def __init__(self):
        """Initialize the multi-agent coordinator."""
        self.agents: dict[str, BaseResearchAgent[Any, Any]] = {}
        self._initialized = False
        self._agents_lock = asyncio.Lock()  # Lock for thread-safe agent registration

    def register_agent(self, agent: BaseResearchAgent[Any, Any]) -> None:
        """Register an agent with the coordinator (sync version for module init).

        Args:
            agent: Agent to register
        """
        # This is safe without lock during module import (single-threaded)
        self.agents[agent.name] = agent
        logfire.info(f"Registered agent: {agent.name}")

    async def register_agent_async(self, agent: BaseResearchAgent[Any, Any]) -> None:
        """Register an agent with the coordinator (async version for runtime).

        Args:
            agent: Agent to register
        """
        async with self._agents_lock:
            self.agents[agent.name] = agent
            logfire.info(f"Registered agent: {agent.name}")

    async def get_agent(self, name: str) -> BaseResearchAgent[Any, Any] | None:
        """Get a registered agent by name.

        Args:
            name: Agent name

        Returns:
            The agent if found, None otherwise
        """
        async with self._agents_lock:
            return self.agents.get(name)

    async def execute_workflow(
        self,
        user_query: str,
        http_client: httpx.AsyncClient,
        api_keys: APIKeys,
        stream_callback: Any | None = None,
    ) -> ResearchState:
        """Execute the complete research workflow.

        Args:
            user_query: User's research query
            http_client: HTTP client for API calls
            api_keys: API keys for various services
            stream_callback: Callback for streaming updates

        Returns:
            Final research state with results
        """
        # Create initial research state
        import uuid

        research_state = ResearchState(
            request_id=str(uuid.uuid4()),
            user_id="api-user",
            session_id=None,
            user_query=user_query,
        )

        # Execute workflow stages
        # This will be implemented when we have all agents ready
        logfire.info(
            "Starting research workflow",
            request_id=research_state.request_id,
            query=user_query,
        )

        return research_state

    async def get_stats(self) -> dict[str, Any]:
        """Get coordinator statistics.

        Returns:
            Dictionary with coordinator stats
        """
        async with self._agents_lock:
            return {
                "registered_agents": list(self.agents.keys()),
                "total_agents": len(self.agents),
            }


# Global coordinator instance
coordinator = MultiAgentCoordinator()