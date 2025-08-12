"""Base agent classes with dependency injection support."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
import logfire
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import Usage

from open_deep_research_with_pydantic_ai.core.config import config
from open_deep_research_with_pydantic_ai.core.events import (
    AgentDelegationEvent,
    StreamingUpdateEvent,
    research_event_bus,
)
from open_deep_research_with_pydantic_ai.models.api_models import APIKeys, ResearchMetadata
from open_deep_research_with_pydantic_ai.models.research import ResearchState


@dataclass
class ResearchDependencies:
    """Shared dependencies for research agents."""

    http_client: httpx.AsyncClient
    api_keys: APIKeys  # Changed from dict to typed model
    research_state: ResearchState
    metadata: ResearchMetadata | None = None  # Added typed metadata
    usage: Usage | None = None
    stream_callback: Any | None = None

    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = ResearchMetadata()


DepsT = TypeVar("DepsT", bound=ResearchDependencies)
OutputT = TypeVar("OutputT", bound=BaseModel)


class BaseResearchAgent[DepsT: ResearchDependencies, OutputT: BaseModel](ABC):
    """Base class for all research agents."""

    def __init__(
        self,
        name: str,
        model: str | None = None,
        system_prompt: str | None = None,
        output_type: type[OutputT] | None = None,
    ):
        """Initialize the base research agent.

        Args:
            name: Agent name for identification
            model: LLM model to use (None for config default)
            system_prompt: System prompt for the agent
            output_type: Expected output type
        """
        self.name = name

        # Get model configuration
        model_config = config.get_model_config(model)
        self.model = model_config["model"]
        self._output_type = output_type

        # Handle output type properly
        if output_type is None:
            # Default to dict output
            actual_output_type = dict
        else:
            actual_output_type = output_type

        # Create the Pydantic AI agent with proper configuration
        self.agent: Agent[ResearchDependencies, Any] = Agent(
            model=self.model,
            retries=model_config.get("retries", 3),
            deps_type=ResearchDependencies,
            output_type=actual_output_type,
            system_prompt=system_prompt or self._get_default_system_prompt(),
        )

        # Register tools for the agent
        self._register_tools()

        logfire.info(f"Initialized {self.name} agent", model=model)

    @abstractmethod
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for this agent."""
        pass

    def _register_tools(self) -> None:
        """Register agent-specific tools. Override in subclasses."""
        # This is intentionally empty - subclasses can override if needed
        return

    async def run(
        self,
        prompt: str,
        deps: DepsT,
        message_history: list[ModelMessage] | None = None,
        stream: bool = False,
    ) -> OutputT:
        """Run the agent with the given prompt.

        Args:
            prompt: User prompt or task description
            deps: Agent dependencies
            message_history: Previous conversation history
            stream: Whether to stream the response

        Returns:
            Agent output of the specified type
        """
        try:
            # Emit streaming update if callback provided
            if deps.stream_callback and stream:
                await research_event_bus.emit(
                    StreamingUpdateEvent(
                        _request_id=deps.research_state.request_id,
                        content=f"{self.name} processing...",
                        stage=deps.research_state.current_stage,
                    )
                )

            # Run the agent with proper error handling
            try:
                result = await self.agent.run(
                    prompt,
                    deps=deps,
                    message_history=message_history,
                    usage=deps.usage,  # Pass usage for tracking
                )
            except Exception as e:
                # Check if it's a recoverable error
                if "rate limit" in str(e).lower():
                    raise ModelRetry(f"Rate limit hit, retrying: {e}") from e
                elif "timeout" in str(e).lower():
                    raise ModelRetry(f"Request timeout, retrying: {e}") from e
                else:
                    raise

            logfire.info(
                f"{self.name} completed",
                request_id=deps.research_state.request_id,
                usage=result.usage() if result.usage() else None,
            )

            return result.output

        except ModelRetry:
            # Let Pydantic-AI handle retries
            raise
        except Exception as e:
            logfire.error(
                f"{self.name} failed",
                request_id=deps.research_state.request_id,
                error=str(e),
                exc_info=True,
            )
            raise

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
        # IMPORTANT: Pass ctx.usage to track usage across delegated agents
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
