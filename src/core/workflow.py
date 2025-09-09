"""Main workflow orchestrator for the streamlined 5-agent research system."""

from datetime import datetime
from typing import Any

import httpx
import logfire

from agents.base import ResearchDependencies
from agents.factory import AgentFactory, AgentType
from core.events import (
    emit_error,
    emit_research_started,
    emit_stage_completed,
    emit_stage_started,
    emit_streaming_update,
)
from interfaces.clarification_flow import handle_clarification_with_review
from src.core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from src.models.api_models import APIKeys, ConversationMessage
from src.models.core import (
    ResearchMetadata,
    ResearchStage,
    ResearchState,
)


class ResearchWorkflow:
    """Orchestrator for the streamlined 5-agent research workflow.

    Pipeline: CLARIFICATION → QUERY_TRANSFORMATION → RESEARCH_EXECUTION →
              COMPRESSION → REPORT_GENERATION

    The Query Transformation Agent now produces both SearchQueryBatch (for search execution)
    and ResearchPlan (for report structure), eliminating the need for a separate Brief Generator.
    """

    def __init__(self):
        """Initialize the research workflow."""
        self.agent_factory = AgentFactory
        self._initialized = False

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

        # Per-agent circuit breaker configurations (no Brief Generator)
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
                "Streamlined 5-agent workflow initialized",
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
            AgentType.COMPRESSION: {
                "critical": False,
                "config": CircuitBreakerConfig(
                    failure_threshold=3,
                    success_threshold=2,
                    timeout_seconds=30.0,
                    half_open_max_attempts=2,
                    name="support_compression",
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
            AgentType.COMPRESSION: {
                "compressed_text": "",
                "ratio": 1.0,
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
                # Extract SearchQueryBatch from transformed query
                enhanced_query = deps.research_state.metadata.query.enhanced_query
                if enhanced_query and hasattr(enhanced_query, "search_queries"):
                    result = await agent.run(deps)
                else:
                    raise ValueError("No search queries available for research execution")
            elif agent_type == AgentType.COMPRESSION:
                # Get research plan from transformed query
                enhanced_query = deps.research_state.metadata.query.enhanced_query
                result = await agent.run(deps)
            elif agent_type == AgentType.REPORT_GENERATOR:
                # Get research plan from transformed query
                enhanced_query = deps.research_state.metadata.query.enhanced_query
                result = await agent.run(deps)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")

            return result

        except Exception as e:
            logfire.error(f"Agent {agent_type.value} failed: {e}")
            if agent_config.get("critical", False):
                raise
            return self._create_fallback(agent_type)

    async def _execute_three_phase_clarification(
        self, deps: ResearchDependencies, user_query: str
    ) -> None:
        """Execute three-phase clarification system.

        Phase 1: Initial clarification check
        Phase 2: Query transformation (produces SearchQueryBatch and ResearchPlan)
        Phase 3: Direct to research execution (no Brief Generator)
        """
        research_state = deps.research_state
        phase_results = {}

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
                        "confidence": clarification_result.confidence_score,
                        "clarification_type": clarification_result.clarification_type,
                        "assessment_reasoning": clarification_result.reasoning,
                        "missing_dimensions": clarification_result.missing_dimensions,
                    }

                    if clarification_result.needs_clarification:
                        research_state.metadata.clarification.request = (
                            clarification_result.clarification_request
                        )
                        research_state.metadata.clarification.awaiting_clarification = True

                await emit_stage_completed(
                    research_state.request_id,
                    ResearchStage.CLARIFICATION,
                    True,
                    clarification_result,
                )

                phase_results["clarification"] = clarification_result

                # If clarification needed, handle it
                if clarification_result.needs_clarification:
                    logfire.info("Clarification needed, entering interactive flow")
                    # Handle clarification flow
                    clarification_handled = await handle_clarification_with_review(
                        research_state=research_state,
                        clarification_result=clarification_result,
                        deps=deps,
                    )
                    if not clarification_handled:
                        logfire.info("Clarification still pending")
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

                # Store the enhanced query with both SearchQueryBatch and ResearchPlan
                if research_state.metadata:
                    research_state.metadata.query.enhanced_query = enhanced_query
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
                    num_search_queries=len(enhanced_query.search_queries.queries),
                    num_objectives=len(enhanced_query.research_plan.objectives),
                    execution_strategy=enhanced_query.search_queries.execution_strategy,
                    confidence=enhanced_query.confidence_score,
                )

                phase_results["transformation"] = enhanced_query

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

            # Phase 3 is now direct research execution (no Brief Generator)
            logfire.info(
                "Three-phase clarification system completed, proceeding to research execution",
                has_clarification=phase_results.get("clarification") is not None,
                has_transformation=phase_results.get("transformation") is not None,
            )

        except Exception as e:
            # Log phase completion failure
            logfire.error(f"Three-phase clarification system failed: {e}", exc_info=True)
            await emit_error(
                research_state.request_id,
                research_state.current_stage,
                "ThreePhaseError",
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
              COMPRESSION → REPORT_GENERATION

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
            request_id=request_id or f"req_{datetime.now().isoformat()}",
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
                # Execute three-phase clarification (includes query transformation)
                await self._execute_three_phase_clarification(deps, user_query)

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
                research_state.current_stage = ResearchStage.FAILED
                research_state.error = str(e)
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
            findings = await self._run_agent_with_circuit_breaker(AgentType.RESEARCH_EXECUTOR, deps)
            research_state.findings = findings

            await emit_stage_completed(
                research_state.request_id,
                ResearchStage.RESEARCH_EXECUTION,
                True,
                {
                    "findings_count": len(findings.key_findings)
                    if hasattr(findings, "key_findings")
                    else 0
                },
            )

            research_state.advance_stage()  # Move to COMPRESSION

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

        # Stage 2: Compression
        await emit_stage_started(research_state.request_id, ResearchStage.COMPRESSION)
        await emit_streaming_update(
            research_state.request_id,
            "Compressing and organizing research findings...",
            ResearchStage.COMPRESSION,
        )

        try:
            compressed = await self._run_agent_with_circuit_breaker(AgentType.COMPRESSION, deps)
            research_state.compressed_findings = compressed

            if research_state.metadata:
                research_state.metadata.compression.full = compressed

            await emit_stage_completed(
                research_state.request_id,
                ResearchStage.COMPRESSION,
                True,
                {"compression_ratio": getattr(compressed, "compression_ratio", 0.5)},
            )

            research_state.advance_stage()  # Move to REPORT_GENERATION

        except Exception as e:
            logfire.error(f"Compression failed: {e}")
            # Compression is not critical, continue with uncompressed findings
            research_state.compressed_findings = research_state.findings
            research_state.advance_stage()

        # Stage 3: Report Generation
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

                    # Re-execute three-phase clarification system
                    await self._execute_three_phase_clarification(deps, research_state.user_query)
                    research_state.advance_stage()
                    return await self.resume_research(research_state, api_keys, stream_callback)

                elif current_stage == ResearchStage.QUERY_TRANSFORMATION:
                    # Query transformation should complete in three-phase
                    research_state.advance_stage()
                    return await self.resume_research(research_state, api_keys, stream_callback)

                elif current_stage in [
                    ResearchStage.RESEARCH_EXECUTION,
                    ResearchStage.COMPRESSION,
                    ResearchStage.REPORT_GENERATION,
                ]:
                    # Execute remaining stages
                    await self._execute_research_stages(research_state, deps, stream_callback)

                    # Mark as complete
                    research_state.current_stage = ResearchStage.COMPLETED
                    research_state.completed_at = datetime.now()

                    return research_state

                elif current_stage == ResearchStage.COMPLETED:
                    logfire.info("Research already completed")
                    return research_state

                else:
                    logfire.warning(f"Unknown stage: {current_stage}")
                    return research_state

            except Exception as e:
                logfire.error(f"Resume failed: {e}", exc_info=True)
                research_state.current_stage = ResearchStage.FAILED
                research_state.error = str(e)
                return research_state
