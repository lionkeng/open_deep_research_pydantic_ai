"""Enhanced base agent classes with dependency injection and performance monitoring."""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from models.research_plan_models import TransformedQuery
    from models.search_query_models import SearchQueryBatch

import httpx
import logfire
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import RunUsage

from core.config import config as global_config
from core.events import (
    StreamingUpdateEvent,
    research_event_bus,
)
from core.logging import configure_logging
from models.api_models import APIKeys
from models.core import ResearchState
from services.source_repository import AbstractSourceRepository


# Enhanced exception system for agent errors
class AgentError(Exception):
    """Base exception for agent-related errors."""

    def __init__(
        self, message: str, agent_name: str | None = None, context: dict[str, Any] | None = None
    ):
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
    custom_settings: dict[str, Any] = Field(
        default_factory=dict, description="Agent-specific settings"
    )

    model_config = ConfigDict(extra="allow")  # Allow additional fields for agent-specific config


@dataclass
class ResearchDependencies:
    """Shared dependencies for research agents.

    Metadata is accessed through research_state.metadata, which is now
    properly typed as ResearchMetadata instead of dict[str, Any].
    """

    http_client: httpx.AsyncClient
    api_keys: APIKeys  # Changed from dict to typed model
    research_state: ResearchState
    # Removed redundant metadata field - access via research_state.metadata
    usage: RunUsage | None = None
    stream_callback: Any | None = None
    search_results: list[dict[str, Any]] = field(default_factory=list)
    source_repository: AbstractSourceRepository | None = None

    def get_transformed_query(self) -> "TransformedQuery | None":
        """Return the transformed query stored on metadata, if any."""

        return getattr(self.research_state.metadata.query, "transformed_query", None)

    def get_search_query_batch(self) -> "SearchQueryBatch | None":
        """Return the stored search query batch without copying."""

        transformed_query = self.get_transformed_query()
        if transformed_query is None:
            return None
        return transformed_query.search_queries


DepsT = TypeVar("DepsT", bound=ResearchDependencies)
OutputT = TypeVar("OutputT", bound=BaseModel)


# Mixin classes for enhanced functionality
class ToolMixin:
    """Mixin for tool management functionality."""

    def __init__(self):
        self._tools: dict[str, Callable[..., Any]] = {}

    def register_tool(self, name: str, tool: Callable[..., Any]) -> None:
        """Register a tool for the agent."""
        self._tools[name] = tool
        logfire.debug(f"Registered tool: {name}")

    def get_tool(self, name: str) -> Callable[..., Any] | None:
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
            self._conversation_history = self._conversation_history[-self._context_window_size :]

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

    def start_execution_timer(self) -> None:
        """Start timing execution."""
        self._execution_start_time = time.time()

    def end_execution_timer(self) -> None:
        """End timing and update metrics."""
        if self._execution_start_time:
            self.metrics.update_execution_time(self._execution_start_time)
            self._execution_start_time = None


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

        model_config = global_config.get_model_config()
        self.model = self.config.model or model_config["model"]
        self._output_type = self._get_output_type()

        # Create the Pydantic AI agent with proper configuration
        self.agent: Agent[DepsT | ResearchDependencies, OutputT] = Agent(
            model=self.model,
            retries=self.config.max_retries,
            deps_type=type(dependencies) if dependencies else ResearchDependencies,
            output_type=self._output_type,
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

    def _format_conversation_context(
        self, conversation: list[Any], query: str | None = None, max_messages: int = 3
    ) -> str:
        """Format conversation history for the prompt.

        This consolidated method handles both dict and BaseModel message formats.

        Args:
            conversation: List of conversation messages (dict or BaseModel)
            query: Optional current query to append
            max_messages: Maximum number of recent messages to include

        Returns:
            Formatted conversation context string
        """
        if not conversation:
            if query:
                return f"Initial Query: {query}\n(No prior conversation)"
            return "No prior conversation context."

        formatted = []
        # Take the last N messages
        recent_messages = conversation[-max_messages:]

        for msg in recent_messages:
            # Extract role and content uniformly from any message format
            try:
                # Try dict-style access first, fall back to attribute access
                role = (msg.get("role") if isinstance(msg, dict) else msg.role) or "unknown"
                content = (msg.get("content") if isinstance(msg, dict) else msg.content) or ""
                formatted.append(f"{str(role).capitalize()}: {str(content)}")
            except (AttributeError, TypeError, KeyError):
                # Fallback for unknown formats
                formatted.append(str(msg))

        result = "Recent Conversation:\n" + "\n".join(formatted)

        # Append current query if provided (for clarification agent)
        if query:
            result += f"\nCurrent Query: {query}"

        return result

    @abstractmethod
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for this agent."""
        pass

    @abstractmethod
    def _get_output_type(self) -> type[OutputT]:
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
        deps: DepsT | None = None,
        message_history: list[ModelMessage] | None = None,
        stream: bool = False,
    ) -> OutputT:
        """Run the agent with the given prompt.

        Args:
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
                agent_name=self.name,
            )

        # Update status and start monitoring
        self.status = AgentStatus.RUNNING
        self.start_execution_timer()

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

                # Check if it's a recoverable error
                if "rate limit" in str(e).lower():
                    self.metrics.record_retry()
                    raise ModelRetry(f"Rate limit hit, retrying: {e}") from e
                if "timeout" in str(e).lower():
                    self.metrics.record_retry()
                    raise ModelRetry(f"Request timeout, retrying: {e}") from e
                self.status = AgentStatus.FAILED
                raise AgentExecutionError(
                    f"Agent execution failed: {e}",
                    agent_name=self.name,
                    context={
                        "user_prompt": (
                            self.get_conversation_context()[-1]
                            if self.get_conversation_context()
                            else None
                        ),
                        "error": str(e),
                    },
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
                context={
                    "user_prompt": (
                        self.get_conversation_context()[-1]
                        if self.get_conversation_context()
                        else None
                    ),
                    "error": str(e),
                },
            ) from e

        finally:
            # Always complete monitoring and execute after hooks
            self.end_execution_timer()
