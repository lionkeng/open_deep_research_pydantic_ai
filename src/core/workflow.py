"""Main workflow orchestrator for the streamlined 4-agent research system."""

from datetime import datetime
from typing import Any

import httpx
import logfire

from agents.base import ResearchDependencies
from agents.factory import AgentFactory, AgentType
from core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from core.events import (
    emit_error,
    emit_research_started,
    emit_stage_completed,
    emit_stage_started,
    emit_streaming_update,
)
from interfaces.clarification_flow import handle_clarification_with_review
from models.api_models import APIKeys, ConversationMessage
from models.core import (
    ResearchMetadata,
    ResearchStage,
    ResearchState,
)
from models.research_executor import ResearchFinding
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
    QueryPriority as SearchQueryPriority,
)
from services.search_orchestrator import (
    SearchQuery as OrchestratorQuery,
)
from services.search_orchestrator import (
    SearchResult as OrchestratorResult,
)


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
                # Retrieve transformed query from metadata and pass search queries via dependencies
                if (
                    deps.research_state.metadata
                    and deps.research_state.metadata.query.transformed_query
                ):
                    transformed_query_data = deps.research_state.metadata.query.transformed_query
                    # Reconstruct SearchQueryBatch from the stored data
                    from models.search_query_models import SearchQueryBatch

                    search_queries_data = transformed_query_data.get("search_queries", {})
                    search_queries = SearchQueryBatch.model_validate(search_queries_data)
                    # Pass search queries to the Research Executor via dependencies
                    # for debugging purposes, we log the queries here
                    logfire.info(
                        "Passing search queries to Research Executor",
                        queries=[q.query for q in search_queries.queries],
                        num_queries=len(search_queries.queries),
                    )
                    deps.search_queries = search_queries
                    result = await agent.run(deps)
                else:
                    raise ValueError("No transformed query available for research execution")
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
            return self._create_fallback(agent_type)

    async def _execute_two_phase_clarification(
        self, deps: ResearchDependencies, user_query: str
    ) -> None:
        """Execute two-phase clarification system.

        Phase 1: Initial clarification check
        Phase 2: Query transformation (produces SearchQueryBatch and ResearchPlan)
        """
        research_state = deps.research_state

        try:
            # Phase 1: Clarification Assessment
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
                        research_state.metadata.clarification.request = clarification_result.request
                        research_state.metadata.clarification.awaiting_clarification = True

                await emit_stage_completed(
                    research_state.request_id,
                    ResearchStage.CLARIFICATION,
                    True,
                    clarification_result,
                )

                # If clarification needed, handle it
                if clarification_result.needs_clarification:
                    logfire.info("Clarification needed, entering interactive flow")
                    # Handle clarification flow
                    if clarification_result.request:
                        clarification_response = await handle_clarification_with_review(
                            request=clarification_result.request,
                            original_query=research_state.user_query,
                        )
                        if not clarification_response:
                            logfire.info("Clarification still pending")
                            return
                        # Store the clarification response in metadata
                        if research_state.metadata and clarification_response:
                            research_state.metadata.clarification.response = clarification_response
                    else:
                        logfire.warn("Clarification needed but no request provided")
                        return

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

                # Store the entire TransformedQuery in metadata for inter-agent communication
                research_state.metadata.query.transformed_query = {
                    "original_query": enhanced_query.original_query,
                    "search_queries": enhanced_query.search_queries.model_dump(),
                    "research_plan": enhanced_query.research_plan.model_dump(),
                    "transformation_rationale": enhanced_query.transformation_rationale,
                    "confidence_score": enhanced_query.confidence_score,
                }

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
    ) -> ResearchState:
        """Execute the complete research workflow.

        Pipeline: CLARIFICATION → QUERY_TRANSFORMATION → RESEARCH_EXECUTION →
              REPORT_GENERATION

        Args:
            user_query: The user's research query
            api_keys: API keys for various services
            conversation_history: Previous conversation messages
            request_id: Unique request identifier
            stream_callback: Optional callback for streaming updates

        Returns:
            Final research state with results
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

        # Create HTTP client
        async with httpx.AsyncClient() as http_client:
            # Create dependencies
            deps = ResearchDependencies(
                http_client=http_client,
                api_keys=api_keys or APIKeys(),
                research_state=research_state,
            )

            try:
                # Execute two-phase clarification (includes query transformation)
                await self._execute_two_phase_clarification(deps, user_query)

                research_state.advance_stage()  # Move to RESEARCH_EXECUTION

                # Execute remaining stages with concurrent processing where possible
                await self._execute_research_stages(research_state, deps, stream_callback)

                # Mark as complete
                research_state.current_stage = ResearchStage.COMPLETED
                research_state.completed_at = datetime.now()

                logfire.info(
                    "Research workflow completed successfully",
                    request_id=research_state.request_id,
                    duration=(datetime.now() - research_state.started_at).total_seconds(),
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
                return research_state

    async def _execute_research_stages(
        self,
        research_state: ResearchState,
        deps: ResearchDependencies,
        stream_callback: Any | None = None,
    ) -> None:
        """Execute the main research stages after clarification and transformation."""

        # Stage 1: Research Execution
        await emit_stage_started(research_state.request_id, ResearchStage.RESEARCH_EXECUTION)
        await emit_streaming_update(
            research_state.request_id,
            "Executing research queries and gathering information...",
            ResearchStage.RESEARCH_EXECUTION,
        )

        try:
            # Research executor now receives SearchQueryBatch directly
            if not getattr(deps, "search_results", None):
                deps.search_results = await self._execute_search_queries(deps)

            results = await self._run_agent_with_circuit_breaker(AgentType.RESEARCH_EXECUTOR, deps)

            if isinstance(results, list):
                research_state.findings = results
            else:
                research_state.research_results = results
                hierarchical = list(getattr(results, "findings", []))
                research_state.findings = [
                    ResearchFinding.from_hierarchical(finding) for finding in hierarchical
                ]

            await emit_stage_completed(
                research_state.request_id,
                ResearchStage.RESEARCH_EXECUTION,
                True,
                {"findings_count": len(getattr(results, "findings", []))},
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

    async def _execute_search_queries(self, deps: ResearchDependencies) -> list[dict[str, Any]]:
        """Execute search queries for the current research state."""

        batch = getattr(deps, "search_queries", None)
        if not batch or not getattr(batch, "queries", None):
            logfire.info("No search queries provided for execution")
            return []

        plan = self._build_query_execution_plan(batch)

        orchestrator = SearchOrchestrator(search_fn=self._orchestrated_search)
        query_results, report = await orchestrator.execute_plan(plan)

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

    def _map_query_priority(self, priority: int | None) -> SearchQueryPriority:
        """Map numeric priority to orchestrator priority enum."""

        if priority is None or priority <= 2:
            return SearchQueryPriority.HIGH
        if priority <= 4:
            return SearchQueryPriority.MEDIUM
        return SearchQueryPriority.LOW

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
                    await self._execute_two_phase_clarification(deps, research_state.user_query)
                    research_state.advance_stage()
                    return await self.resume_research(research_state, api_keys, stream_callback)

                if current_stage == ResearchStage.QUERY_TRANSFORMATION:
                    # Query transformation should complete in two-phase
                    research_state.advance_stage()
                    return await self.resume_research(research_state, api_keys, stream_callback)

                if current_stage in [
                    ResearchStage.RESEARCH_EXECUTION,
                    ResearchStage.REPORT_GENERATION,
                ]:
                    # Execute remaining stages
                    await self._execute_research_stages(research_state, deps, stream_callback)

                    # Mark as complete
                    research_state.current_stage = ResearchStage.COMPLETED
                    research_state.completed_at = datetime.now()

                    return research_state

                if current_stage == ResearchStage.COMPLETED:
                    logfire.info("Research already completed")
                    return research_state

                logfire.warning(f"Unknown stage: {current_stage}")
                return research_state

            except Exception as e:
                logfire.error(f"Resume failed: {e}", exc_info=True)
                research_state.set_error(str(e))
                return research_state
