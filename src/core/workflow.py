"""Main workflow orchestrator for the streamlined 4-agent research system."""

from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

import httpx
import logfire

from agents.base import ResearchDependencies
from agents.factory import AgentFactory, AgentType
from core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from core.config import config as global_config
from core.events import (
    emit_error,
    emit_research_started,
    emit_stage_completed,
    emit_stage_started,
    emit_streaming_update,
)
from interfaces.clarification_flow import handle_clarification_with_review
from models.api_models import APIKeys, ConversationMessage
from models.clarification import ClarificationRequest, ClarificationResponse
from models.core import (
    ResearchMetadata,
    ResearchStage,
    ResearchState,
)
from models.priority import Priority
from models.research_executor import ResearchResults
from models.search_query_models import (
    ExecutionStrategy as BatchExecutionStrategy,
)
from models.search_query_models import (
    SearchQueryBatch,
)
from services.search import WebSearchService
from services.search_orchestrator import (
    ExecutionStrategy as SearchExecutionStrategy,
)
from services.search_orchestrator import (
    QueryExecutionPlan,
    SearchOrchestrator,
)
from services.search_orchestrator import (
    SearchQuery as OrchestratorQuery,
)
from services.search_orchestrator import (
    SearchResult as OrchestratorResult,
)
from services.source_repository import InMemorySourceRepository

try:  # optional embeddings
    from services.embeddings import EmbeddingService, OpenAIEmbeddingBackend
except Exception:  # pragma: no cover - optional
    EmbeddingService = None  # type: ignore[assignment]
    OpenAIEmbeddingBackend = None  # type: ignore[assignment]


class ResearchWorkflow:
    """Orchestrator for the streamlined 4-agent research workflow.

    Pipeline: CLARIFICATION → QUERY_TRANSFORMATION → RESEARCH_EXECUTION →
              REPORT_GENERATION

    The Query Transformation Agent now produces both SearchQueryBatch (for search execution)
    and ResearchPlan (for report structure), eliminating the need for a separate Brief Generator.
    """

    def __init__(self):
        """Initialize the research workflow."""
        self.agent_factory = AgentFactory
        self._initialized = False
        self._search_service = WebSearchService()

        # Concurrent processing configuration
        self._max_concurrent_tasks = 5
        self._task_timeout = 300.0  # 5 minutes per task

        # Create circuit breaker with AgentType as key type
        default_config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=60.0,
            half_open_max_attempts=2,
            name="workflow_circuit_breaker",
        )

        self.circuit_breaker: CircuitBreaker[AgentType] = CircuitBreaker(
            config=default_config,
            fallback_factory=self._create_fallback,
        )

        # Per-agent circuit breaker configurations
        self.agent_configs = self._create_agent_configs()

        # Legacy attributes for backward compatibility
        self._consecutive_errors: dict[AgentType, int] = {}
        self._last_error_time: dict[AgentType, float] = {}
        self._circuit_open: dict[AgentType, bool] = {}

    def _ensure_initialized(self) -> None:
        """Ensure all agents are registered."""
        if not self._initialized:
            # Agents are auto-registered when imported
            self._initialized = True
            logfire.info(
                "Streamlined research workflow initialized",
                agents=[agent.value for agent in AgentType],
                max_concurrent_tasks=self._max_concurrent_tasks,
            )

    def _create_agent_configs(self) -> dict[AgentType, dict[str, Any]]:
        """Create agent-specific configurations."""
        return {
            # Critical agents - more lenient settings
            AgentType.RESEARCH_EXECUTOR: {
                "critical": True,
                "config": CircuitBreakerConfig(
                    failure_threshold=5,
                    success_threshold=2,
                    timeout_seconds=60.0,
                    half_open_max_attempts=3,
                    name="critical_research_executor",
                ),
            },
            # Important agents - balanced settings
            AgentType.REPORT_GENERATOR: {
                "critical": False,
                "config": CircuitBreakerConfig(
                    failure_threshold=3,
                    success_threshold=2,
                    timeout_seconds=45.0,
                    half_open_max_attempts=2,
                    name="important_report_generator",
                ),
            },
            # Optional agents - fail fast
            AgentType.QUERY_TRANSFORMATION: {
                "critical": True,  # Now critical since it produces SearchQueryBatch
                "config": CircuitBreakerConfig(
                    failure_threshold=3,
                    success_threshold=2,
                    timeout_seconds=45.0,
                    half_open_max_attempts=2,
                    name="critical_query_transformation",
                ),
            },
            # Supporting agents
            AgentType.CLARIFICATION: {
                "critical": False,
                "config": CircuitBreakerConfig(
                    failure_threshold=3,
                    success_threshold=2,
                    timeout_seconds=30.0,
                    half_open_max_attempts=2,
                    name="support_clarification",
                ),
            },
        }

    def _create_fallback(self, agent_type: AgentType) -> dict[str, Any]:
        """Create fallback response for failed agents."""
        fallbacks = {
            AgentType.CLARIFICATION: {
                "needs_clarification": False,
                "confidence": 0.0,
                "fallback": True,
            },
            AgentType.QUERY_TRANSFORMATION: {
                "transformed_query": "",
                "confidence": 0.0,
                "fallback": True,
            },
            AgentType.RESEARCH_EXECUTOR: {
                "results": [],
                "from_cache": True,
                "fallback": True,
            },
            AgentType.REPORT_GENERATOR: {
                "report": "Report generation failed",
                "fallback": True,
            },
        }
        return fallbacks.get(agent_type, {"error": "No fallback available"})

    def _configure_synthesis_deps(self, deps: ResearchDependencies) -> None:
        """Apply synthesis-related feature flags and services just-in-time.

        Ensures consistent behavior across HTTP and direct modes without
        re-reading environment variables per run. Values are sourced from
        the global config singleton.
        """
        # Apply feature flags
        deps.enable_embedding_similarity = global_config.enable_embedding_similarity
        deps.enable_llm_clean_merge = global_config.enable_llm_clean_merge
        deps.similarity_threshold = global_config.embedding_similarity_threshold

        # Attach embedding service if enabled and not already present
        if (
            deps.enable_embedding_similarity
            and getattr(deps, "embedding_service", None) is None
            and EmbeddingService is not None
            and OpenAIEmbeddingBackend is not None
        ):
            openai_key = global_config.openai_api_key
            if openai_key:
                try:
                    deps.embedding_service = EmbeddingService(
                        backend=OpenAIEmbeddingBackend(api_key=openai_key),
                    )
                except Exception:
                    deps.embedding_service = None

        # Observability
        logfire.info(
            "Synthesis feature flags",
            embedding_similarity=deps.enable_embedding_similarity,
            similarity_threshold=deps.similarity_threshold,
            llm_clean_merge=deps.enable_llm_clean_merge,
            embedding_backend=(
                "openai" if getattr(deps, "embedding_service", None) else "disabled"
            ),
        )

    async def _run_agent_with_circuit_breaker(
        self,
        agent_type: AgentType,
        deps: ResearchDependencies,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run an agent with circuit breaker protection."""
        agent_config = self.agent_configs.get(agent_type, {})

        try:
            # Create the agent instance from factory
            agent = self.agent_factory.create_agent(agent_type, deps, None)

            # Run with appropriate method based on agent type
            if agent_type == AgentType.CLARIFICATION:
                # Run clarification agent
                result = await agent.run(deps)
            elif agent_type == AgentType.QUERY_TRANSFORMATION:
                # Run query transformation to get TransformedQuery
                result = await agent.run(deps)
            elif agent_type == AgentType.RESEARCH_EXECUTOR:
                transformed_query = deps.get_transformed_query()
                if transformed_query is None:
                    raise ValueError("No transformed query available for research execution")

                logfire.info(
                    "Passing search queries to Research Executor",
                    queries=[q.query for q in transformed_query.search_queries.queries],
                    num_queries=len(transformed_query.search_queries.queries),
                )

                result = await agent.run(deps)
            elif agent_type == AgentType.REPORT_GENERATOR:
                # Report generator can access research plan from metadata if needed
                result = await agent.run(deps)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")

            return result

        except Exception as e:
            logfire.error(f"Agent {agent_type.value} failed: {e}")
            if agent_config.get("critical", False):
                raise
            fallback = self._create_fallback(agent_type)
            if agent_type == AgentType.RESEARCH_EXECUTOR:
                return ResearchResults(
                    query=deps.research_state.user_query,
                    findings=[],
                )
            return fallback

    async def _execute_two_phase_clarification(
        self, deps: ResearchDependencies, user_query: str
    ) -> bool:
        """Execute two-phase clarification system.

        Phase 1: Initial clarification check
        Phase 2: Query transformation (produces SearchQueryBatch and ResearchPlan)
        """
        research_state = deps.research_state

        try:
            skip_assessment = False
            # Fast-path: if we already have a completed clarification response
            # and are not awaiting further input, skip running the agent again.
            if (
                research_state.metadata
                and not research_state.metadata.clarification.awaiting_clarification
                and research_state.metadata.clarification.response is not None
            ):
                # If there's a request, validate completeness; otherwise treat as complete.
                try:
                    is_complete = (
                        research_state.metadata.is_clarification_complete()
                        if research_state.metadata.clarification.request is not None
                        else True
                    )
                except Exception:
                    is_complete = True

                if is_complete:
                    await emit_stage_started(research_state.request_id, ResearchStage.CLARIFICATION)
                    await emit_streaming_update(
                        research_state.request_id,
                        "Using previously provided clarification to proceed...",
                        ResearchStage.CLARIFICATION,
                    )
                    await emit_stage_completed(
                        research_state.request_id,
                        ResearchStage.CLARIFICATION,
                        True,
                        {"used_existing_response": True},
                    )
                    # Proceed directly to transformation
                    skip_assessment = True

            # Phase 1: Clarification Assessment
            if not skip_assessment:
                logfire.info("Phase 1: Clarification assessment")

                await emit_stage_started(research_state.request_id, ResearchStage.CLARIFICATION)
                await emit_streaming_update(
                    research_state.request_id,
                    "Analyzing your query for clarity and completeness...",
                    ResearchStage.CLARIFICATION,
                )

                try:
                    # Run clarification agent
                    clarification_result = await self._run_agent_with_circuit_breaker(
                        AgentType.CLARIFICATION, deps
                    )

                    # Store clarification metadata
                    if research_state.metadata:
                        research_state.metadata.clarification.assessment = {
                            "needs_clarification": clarification_result.needs_clarification,
                            "confidence": 0.8,  # Default confidence for now
                            "clarification_type": "general",  # Default type
                            "assessment_reasoning": clarification_result.reasoning,
                            "missing_dimensions": clarification_result.missing_dimensions,
                        }

                        if clarification_result.needs_clarification:
                            research_state.metadata.clarification.request = (
                                clarification_result.request
                            )
                            research_state.metadata.clarification.awaiting_clarification = True

                    await emit_stage_completed(
                        research_state.request_id,
                        ResearchStage.CLARIFICATION,
                        True,
                        clarification_result,
                    )

                    # If clarification needed, handle it
                    if clarification_result.needs_clarification:
                        logfire.info("Clarification needed, initiating follow-up flow")
                        request_model = clarification_result.request
                        if not request_model:
                            logfire.warning(
                                "Clarification requested but no question payload provided"
                            )
                            return False

                        if research_state.metadata:
                            research_state.metadata.clarification.request = request_model
                            research_state.metadata.clarification.awaiting_clarification = True

                        callback = getattr(deps, "clarification_callback", None)
                        clarification_response = None
                        if callback:
                            clarification_response = await callback(request_model, research_state)
                            if not clarification_response:
                                if getattr(callback, "_default_cli_handler", False):
                                    logfire.info(
                                        "No clarification provided via CLI (non-interactive); "
                                        "proceeding"
                                    )
                                    if research_state.metadata:
                                        clarification_meta = research_state.metadata.clarification
                                        clarification_meta.awaiting_clarification = False
                                        clarification_meta.request = None
                                        clarification_meta.response = None
                                else:
                                    logfire.info("Clarification pending via external handler")
                                    return False
                        else:
                            clarification_response = await handle_clarification_with_review(
                                request=request_model,
                                original_query=research_state.user_query,
                            )
                            if not clarification_response:
                                logfire.info("Clarification still pending")
                                return False

                        if research_state.metadata and clarification_response:
                            research_state.metadata.clarification.response = clarification_response
                            research_state.metadata.clarification.awaiting_clarification = False

                except Exception as e:
                    logfire.error(f"Clarification failed: {e}")
                    # Continue without clarification
                    await emit_error(
                        research_state.request_id,
                        ResearchStage.CLARIFICATION,
                        "ClarificationError",
                        str(e),
                        recoverable=True,
                    )

            # Phase 2: Query Transformation (produces SearchQueryBatch and ResearchPlan)
            logfire.info("Phase 2: Query transformation with research planning")

            await emit_stage_started(research_state.request_id, ResearchStage.QUERY_TRANSFORMATION)
            await emit_streaming_update(
                research_state.request_id,
                "Transforming your query and creating research plan...",
                ResearchStage.QUERY_TRANSFORMATION,
            )

            try:
                # Run query transformation agent to get TransformedQuery
                enhanced_query = await self._run_agent_with_circuit_breaker(
                    AgentType.QUERY_TRANSFORMATION, deps
                )

                # Store the TransformedQuery object directly for downstream access
                research_state.metadata.query.transformed_query = enhanced_query

                await emit_stage_completed(
                    research_state.request_id,
                    ResearchStage.QUERY_TRANSFORMATION,
                    True,
                    enhanced_query,
                )

                logfire.info(
                    "Query transformation completed",
                    original_query=enhanced_query.original_query,
                    search_queries=[q.query for q in enhanced_query.search_queries.queries],
                    num_search_queries=len(enhanced_query.search_queries.queries),
                    num_objectives=len(enhanced_query.research_plan.objectives),
                    execution_strategy=enhanced_query.search_queries.execution_strategy,
                    confidence=enhanced_query.confidence_score,
                    transformation_rationale=enhanced_query.transformation_rationale,
                )

            except Exception as e:
                logfire.error(f"Query transformation failed: {e}")
                await emit_error(
                    research_state.request_id,
                    ResearchStage.QUERY_TRANSFORMATION,
                    "TransformationError",
                    str(e),
                    recoverable=False,
                )
                raise

            return True

        except Exception as e:
            # Log phase completion failure
            logfire.error(f"Two-phase clarification system failed: {e}", exc_info=True)
            await emit_error(
                research_state.request_id,
                research_state.current_stage,
                "TwoPhaseError",
                str(e),
                recoverable=False,
            )
            raise

    async def run(
        self,
        user_query: str,
        api_keys: APIKeys | None = None,
        conversation_history: list[ConversationMessage] | None = None,
        request_id: str | None = None,
        stream_callback: Any | None = None,
        clarification_callback: (
            Callable[[ClarificationRequest, ResearchState], Awaitable[ClarificationResponse | None]]
            | None
        ) = None,
    ) -> ResearchState:
        """Execute the complete research workflow end‑to‑end.

        Pipeline: CLARIFICATION → QUERY_TRANSFORMATION → RESEARCH_EXECUTION →
        REPORT_GENERATION. This method creates an async HTTP client and delegates
        execution to `_run_with_http_client` so that the same logic works for
        both direct mode and HTTP mode.

        Notes:
        - An `httpx.AsyncClient` is created even in direct mode so downstream
          services (e.g., source validation) can reuse a shared client.
        - Synthesis feature flags and EmbeddingService are configured just in
          time at execution/report stages via `_configure_synthesis_deps`.

        Args:
            user_query: The user's research query.
            api_keys: API keys for providers; falls back to env‑derived defaults.
            conversation_history: Optional prior messages to seed context.
            request_id: Optional request identifier for tracing.
            stream_callback: Optional callback or truthy flag to emit streaming updates.
            clarification_callback: Optional handler to collect clarification answers.

        Returns:
            Final `ResearchState` containing results or error details.
        """
        self._ensure_initialized()

        # Initialize research state
        research_state = ResearchState(
            user_query=user_query,
            current_stage=ResearchStage.CLARIFICATION,
            request_id=request_id or ResearchState.generate_request_id(),
            user_id="default",  # Explicitly set to match the default
            session_id=None,  # Explicitly set to match the default
            metadata=ResearchMetadata(),
        )

        # Store conversation history if provided
        if conversation_history:
            research_state.metadata.conversation_messages = conversation_history

        # Emit start event
        await emit_research_started(research_state.request_id, user_query)

        # Create HTTP client and delegate to helper
        async with httpx.AsyncClient() as http_client:
            return await self._run_with_http_client(
                http_client=http_client,
                research_state=research_state,
                api_keys=api_keys,
                stream_callback=stream_callback,
                clarification_callback=clarification_callback,
            )

    async def _run_with_http_client(
        self,
        *,
        http_client: httpx.AsyncClient,
        research_state: ResearchState,
        api_keys: APIKeys | None,
        stream_callback: Any | None,
        clarification_callback: (
            Callable[[ClarificationRequest, ResearchState], Awaitable[ClarificationResponse | None]]
            | None
        ),
    ) -> ResearchState:
        """Run the workflow with an existing HTTP client (shared logic for direct/HTTP modes)."""

        # Create dependencies
        deps = ResearchDependencies(
            http_client=http_client,
            api_keys=api_keys or APIKeys(),
            research_state=research_state,
        )

        # Clarification callback resolution
        if clarification_callback is None:

            async def default_cli_callback(
                request: ClarificationRequest, state: ResearchState
            ) -> ClarificationResponse | None:
                return await handle_clarification_with_review(
                    request=request,
                    original_query=state.user_query,
                )

            default_cli_callback._default_cli_handler = True  # type: ignore[attr-defined]

            deps.clarification_callback = default_cli_callback
        else:
            deps.clarification_callback = clarification_callback
        deps.source_repository = InMemorySourceRepository()

        try:
            # Execute two-phase clarification (includes query transformation)
            ready_to_continue = await self._execute_two_phase_clarification(
                deps, research_state.user_query
            )
            if not ready_to_continue:
                return research_state

            research_state.advance_stage()  # Move to RESEARCH_EXECUTION

            # Execute remaining stages with concurrent processing where possible
            await self._execute_research_stages(deps, stream_callback)

            # Mark as complete
            research_state.current_stage = ResearchStage.COMPLETED
            research_state.completed_at = datetime.now()

            duration = (datetime.now() - research_state.started_at).total_seconds()

            logfire.info(
                "Research workflow completed successfully",
                request_id=research_state.request_id,
                duration=duration,
            )

            # Emit research completed event for CLI and other consumers
            from core.events import ResearchCompletedEvent, research_event_bus

            await research_event_bus.emit(
                ResearchCompletedEvent(
                    _request_id=research_state.request_id,
                    report=research_state.final_report,
                    success=True,
                    duration_seconds=duration,
                    error_message=None,
                )
            )

            return research_state

        except Exception as e:
            logfire.error(f"Research workflow failed: {e}", exc_info=True)
            research_state.set_error(str(e))
            await emit_error(
                research_state.request_id,
                research_state.current_stage,
                "WorkflowError",
                str(e),
                recoverable=False,
            )

            # Emit research completed event with failure status
            duration = (datetime.now() - research_state.started_at).total_seconds()
            from core.events import ResearchCompletedEvent, research_event_bus

            await research_event_bus.emit(
                ResearchCompletedEvent(
                    _request_id=research_state.request_id,
                    report=research_state.final_report,
                    success=False,
                    duration_seconds=duration,
                    error_message=str(e),
                )
            )

            return research_state

    async def _execute_research_stages(
        self, deps: ResearchDependencies, stream_callback: Any | None = None
    ) -> None:
        """Execute the main research stages after clarification and transformation."""

        research_state = deps.research_state

        # Configure synthesis features just-in-time for execution/report stages
        self._configure_synthesis_deps(deps)

        # Stage 1: Research Execution
        await emit_stage_started(research_state.request_id, ResearchStage.RESEARCH_EXECUTION)
        await emit_streaming_update(
            research_state.request_id,
            "Executing research queries and gathering information...",
            ResearchStage.RESEARCH_EXECUTION,
        )

        try:
            # Research executor now receives SearchQueryBatch directly
            if not deps.search_results:
                batch = deps.get_search_query_batch()
                if batch is None:
                    raise ValueError("Missing SearchQueryBatch prior to search execution")
                deps.search_results = await self._execute_search_queries(batch, deps)

            results = await self._run_agent_with_circuit_breaker(AgentType.RESEARCH_EXECUTOR, deps)
            research_state.research_results = results

            await emit_stage_completed(
                research_state.request_id,
                ResearchStage.RESEARCH_EXECUTION,
                True,
                {"findings_count": (len(getattr(results, "findings", [])) if results else 0)},
            )

            research_state.advance_stage()  # Move to REPORT_GENERATION

        except Exception as e:
            logfire.error(f"Research execution failed: {e}")
            await emit_error(
                research_state.request_id,
                ResearchStage.RESEARCH_EXECUTION,
                "ResearchExecutionError",
                str(e),
                recoverable=False,
            )
            raise

        # Stage 2: Report Generation
        await emit_stage_started(research_state.request_id, ResearchStage.REPORT_GENERATION)
        await emit_streaming_update(
            research_state.request_id,
            "Generating comprehensive research report...",
            ResearchStage.REPORT_GENERATION,
        )

        try:
            report = await self._run_agent_with_circuit_breaker(AgentType.REPORT_GENERATOR, deps)
            research_state.final_report = report

            await emit_stage_completed(
                research_state.request_id,
                ResearchStage.REPORT_GENERATION,
                True,
                {"report_sections": len(report.sections) if hasattr(report, "sections") else 0},
            )

        except Exception as e:
            logfire.error(f"Report generation failed: {e}")
            await emit_error(
                research_state.request_id,
                ResearchStage.REPORT_GENERATION,
                "ReportGenerationError",
                str(e),
                recoverable=False,
            )
            raise

    async def _execute_search_queries(
        self, batch: SearchQueryBatch, deps: ResearchDependencies
    ) -> list[dict[str, Any]]:
        """Execute search queries for the current research state."""

        if not batch.queries:
            logfire.info("No search queries provided for execution")
            return []

        plan = self._build_query_execution_plan(batch)

        orchestrator = SearchOrchestrator(search_fn=self._orchestrated_search)
        query_results, _ = await orchestrator.execute_plan(plan)

        aggregated: list[dict[str, Any]] = []
        for query, result in query_results:
            if not result:
                continue
            for item in result.results:
                if hasattr(item, "model_dump"):
                    data = item.model_dump()
                elif isinstance(item, dict):
                    data = item
                else:
                    data = {"content": str(item)}

                content = data.get("content") or data.get("snippet") or ""
                aggregated.append(
                    {
                        "query": query.query,
                        "title": data.get("title", ""),
                        "url": data.get("url", ""),
                        "snippet": data.get("snippet", ""),
                        "content": content,
                        "score": data.get("score", 0.0),
                        "metadata": data.get("metadata", {}),
                    }
                )

        # Record execution metadata if available
        execution_meta = getattr(deps.research_state.metadata, "execution", None)
        if execution_meta is not None:
            execution_meta.results = aggregated
            execution_meta.status = "completed"

        return aggregated

    def _build_query_execution_plan(self, batch: SearchQueryBatch) -> QueryExecutionPlan:
        """Convert `SearchQueryBatch` to orchestrator execution plan."""

        strategy_map = {
            BatchExecutionStrategy.SEQUENTIAL: SearchExecutionStrategy.SEQUENTIAL,
            BatchExecutionStrategy.PARALLEL: SearchExecutionStrategy.PARALLEL,
            BatchExecutionStrategy.HIERARCHICAL: SearchExecutionStrategy.HIERARCHICAL,
            BatchExecutionStrategy.ADAPTIVE: SearchExecutionStrategy.SEQUENTIAL,
        }

        orchestrator_queries: list[OrchestratorQuery] = []
        for query in batch.queries:
            priority = self._map_query_priority(query.priority)
            context: dict[str, Any] = {
                "max_results": query.max_results,
                "search_sources": [source.value for source in query.search_sources],
                "expected_result_type": query.expected_result_type,
            }
            if query.temporal_context:
                context["temporal_context"] = query.temporal_context.model_dump()

            orchestrator_queries.append(
                OrchestratorQuery(
                    id=query.id,
                    query=query.query,
                    priority=priority,
                    context=context,
                )
            )

        return QueryExecutionPlan(
            queries=orchestrator_queries,
            strategy=strategy_map.get(batch.execution_strategy, SearchExecutionStrategy.SEQUENTIAL),
        )

    def _map_query_priority(self, priority: int | None) -> int:
        """Map numeric priority, returning a default if None."""
        return priority if priority is not None else Priority.DEFAULT_PRIORITY

    async def _orchestrated_search(self, query: OrchestratorQuery) -> OrchestratorResult:
        """Bridge function that executes a query through WebSearchService."""

        context = query.context or {}
        max_results = context.get("max_results", 5)
        provider = context.get("provider")

        response = await self._search_service.search(
            query.query,
            num_results=max_results,
            provider=provider,
        )

        return OrchestratorResult(
            query=query.query,
            results=[res.model_dump() for res in response.results],
            metadata={
                "source": response.source,
                "total_results": response.total_results,
            },
        )

    async def resume_research(
        self,
        research_state: ResearchState,
        api_keys: APIKeys | None = None,
        stream_callback: Any | None = None,
        clarification_callback: (
            Callable[[ClarificationRequest, ResearchState], Awaitable[ClarificationResponse | None]]
            | None
        ) = None,
    ) -> ResearchState:
        """Resume a research workflow from a given state.

        Args:
            research_state: Previous research state to resume from
            api_keys: API keys for various services
            stream_callback: Optional callback for streaming updates

        Returns:
            Updated research state
        """
        self._ensure_initialized()

        # Create HTTP client
        async with httpx.AsyncClient() as http_client:
            # Create dependencies with existing state
            deps = ResearchDependencies(
                http_client=http_client,
                api_keys=api_keys or APIKeys(),
                research_state=research_state,
            )
            if clarification_callback is None:

                async def default_cli_callback(
                    request: ClarificationRequest, state: ResearchState
                ) -> ClarificationResponse | None:
                    return await handle_clarification_with_review(
                        request=request,
                        original_query=state.user_query,
                    )

                deps.clarification_callback = default_cli_callback
            else:
                deps.clarification_callback = clarification_callback

            try:
                # Resume from current stage
                current_stage = research_state.current_stage

                if current_stage == ResearchStage.CLARIFICATION:
                    # Check if we're waiting for clarification response
                    if (
                        research_state.metadata
                        and research_state.metadata.clarification.awaiting_clarification
                    ):
                        # Still waiting for user response
                        logfire.info("Still awaiting clarification response")
                        return research_state

                    # Re-execute two-phase clarification system
                    ready_to_continue = await self._execute_two_phase_clarification(
                        deps, research_state.user_query
                    )
                    if not ready_to_continue:
                        return research_state
                    research_state.advance_stage()
                    return await self.resume_research(
                        research_state,
                        api_keys,
                        stream_callback,
                        clarification_callback,
                    )

                if current_stage == ResearchStage.QUERY_TRANSFORMATION:
                    # Query transformation should complete in two-phase
                    research_state.advance_stage()
                    return await self.resume_research(
                        research_state,
                        api_keys,
                        stream_callback,
                        clarification_callback,
                    )

                if current_stage in [
                    ResearchStage.RESEARCH_EXECUTION,
                    ResearchStage.REPORT_GENERATION,
                ]:
                    # Execute remaining stages
                    self._configure_synthesis_deps(deps)
                    await self._execute_research_stages(deps, stream_callback)

                    # Mark as complete
                    research_state.current_stage = ResearchStage.COMPLETED
                    research_state.completed_at = datetime.now()

                    # Emit research completed event for resumed research
                    duration = (datetime.now() - research_state.started_at).total_seconds()
                    from core.events import ResearchCompletedEvent, research_event_bus

                    await research_event_bus.emit(
                        ResearchCompletedEvent(
                            _request_id=research_state.request_id,
                            report=research_state.final_report,
                            success=True,
                            duration_seconds=duration,
                            error_message=None,
                        )
                    )

                    return research_state

                if current_stage == ResearchStage.COMPLETED:
                    logfire.info("Research already completed")
                    return research_state

                logfire.warning(f"Unknown stage: {current_stage}")
                return research_state

            except Exception as e:
                logfire.error(f"Resume failed: {e}", exc_info=True)
                research_state.set_error(str(e))

                # Emit research completed event with failure status for resumed research
                duration = (datetime.now() - research_state.started_at).total_seconds()
                from core.events import ResearchCompletedEvent, research_event_bus

                await research_event_bus.emit(
                    ResearchCompletedEvent(
                        _request_id=research_state.request_id,
                        report=research_state.final_report,
                        success=False,
                        duration_seconds=duration,
                        error_message=str(e),
                    )
                )

                return research_state


# Create a module-level singleton instance for API usage
workflow = ResearchWorkflow()
