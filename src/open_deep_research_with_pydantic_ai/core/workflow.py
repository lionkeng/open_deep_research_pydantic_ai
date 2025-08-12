"""Main workflow orchestrator for the research pipeline."""

from datetime import datetime
from typing import Any

import httpx
import logfire
from pydantic_ai.usage import Usage

from open_deep_research_with_pydantic_ai.agents.base import (
    ResearchDependencies,
    coordinator,
)
from open_deep_research_with_pydantic_ai.agents.brief_generator import brief_generator_agent
from open_deep_research_with_pydantic_ai.agents.clarification import clarification_agent
from open_deep_research_with_pydantic_ai.agents.compression import compression_agent
from open_deep_research_with_pydantic_ai.agents.report_generator import report_generator_agent
from open_deep_research_with_pydantic_ai.agents.research_executor import (
    research_executor_agent,
)
from open_deep_research_with_pydantic_ai.core.context import get_current_context
from open_deep_research_with_pydantic_ai.core.events import (
    emit_error,
    emit_research_started,
)
from open_deep_research_with_pydantic_ai.models.api_models import APIKeys, ResearchMetadata
from open_deep_research_with_pydantic_ai.models.research import ResearchStage, ResearchState


class ResearchWorkflow:
    """Orchestrator for the complete research workflow."""

    def __init__(self):
        """Initialize the research workflow."""
        self.coordinator = coordinator
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Ensure all agents are registered."""
        if not self._initialized:
            # Agents are auto-registered when imported
            self._initialized = True
            logfire.info(
                "Research workflow initialized",
                agents=list(self.coordinator.agents.keys()),
            )

    async def execute_research(
        self,
        user_query: str,
        api_keys: APIKeys | None = None,
        stream_callback: Any | None = None,
        request_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> ResearchState:
        """Execute the complete research workflow.

        Args:
            user_query: User's research query
            api_keys: API keys for various services
            stream_callback: Optional callback for streaming updates
            request_id: Optional request ID (will generate if not provided)
            user_id: Optional user ID for isolation (defaults to context or "default")
            session_id: Optional session ID for user

        Returns:
            Final research state with results
        """
        self._ensure_initialized()

        # Get user context
        context = get_current_context()
        if user_id is None:
            user_id = context.user_id
        if session_id is None:
            session_id = context.session_id

        # Create research state with scoped request ID
        if request_id is None:
            request_id = ResearchState.generate_request_id(user_id, session_id)
        research_state = ResearchState(
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            user_query=user_query,
        )

        # Create HTTP client
        async with httpx.AsyncClient() as http_client:
            # Create dependencies
            deps = ResearchDependencies(
                http_client=http_client,
                api_keys=api_keys or APIKeys(),
                research_state=research_state,
                metadata=ResearchMetadata(),
                usage=Usage(),
                stream_callback=stream_callback,
            )

            try:
                # Start research (move from PENDING to CLARIFICATION)
                research_state.start_research()

                # Emit research started event
                await emit_research_started(request_id, user_query)

                # Stage 1: User Clarification
                logfire.info("Stage 1: Clarifying query", request_id=request_id)
                clarification_result = await clarification_agent.clarify_query(user_query, deps)

                if not clarification_result.is_clear:
                    # If clarification needed, return with questions
                    research_state.metadata["clarifying_questions"] = (
                        clarification_result.clarifying_questions
                    )
                    research_state.metadata["warnings"] = clarification_result.warnings
                    research_state.set_error("Clarification needed")
                    return research_state

                research_state.advance_stage()
                research_state.clarified_query = clarification_result.clarified_query

                # Stage 2: Research Brief Generation
                logfire.info("Stage 2: Generating research brief", request_id=request_id)
                research_brief = await brief_generator_agent.generate_brief(
                    clarification_result.clarified_query,
                    clarification_result.estimated_complexity,
                    deps,
                )
                research_state.research_brief = research_brief
                research_state.advance_stage()

                # Stage 3: Research Execution
                logfire.info("Stage 3: Executing research", request_id=request_id)
                findings = await research_executor_agent.execute_research(
                    research_brief,
                    deps,
                    max_parallel_tasks=3,
                )
                research_state.findings = findings
                research_state.advance_stage()

                # Stage 4: Compression
                logfire.info("Stage 4: Compressing findings", request_id=request_id)
                compressed_findings = await compression_agent.compress_findings(
                    findings,
                    research_brief.key_questions,
                    deps,
                )
                # Store the summary in compressed_findings field
                research_state.compressed_findings = compressed_findings.summary
                # Store the full object in metadata for later use
                research_state.metadata["compressed_findings_full"] = compressed_findings
                research_state.advance_stage()

                # Stage 5: Report Generation
                logfire.info("Stage 5: Generating final report", request_id=request_id)
                final_report = await report_generator_agent.generate_report(
                    research_brief,
                    findings,
                    compressed_findings,
                    deps,
                )

                # Store the final report in research state
                research_state.final_report = final_report
                research_state.advance_stage()
                research_state.completed_at = datetime.now()

                # Workflow complete
                logfire.info(
                    "Research workflow completed successfully",
                    request_id=request_id,
                    total_usage=deps.usage,
                )

                return research_state

            except Exception as e:
                logfire.error(
                    "Research workflow failed",
                    request_id=request_id,
                    error=str(e),
                    exc_info=True,
                )

                # Emit error event
                await emit_error(
                    request_id,
                    research_state.current_stage,
                    type(e).__name__,
                    str(e),
                )

                # Update state with error
                research_state.set_error(str(e))
                return research_state

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
                metadata=ResearchMetadata(),
                usage=Usage(),
                stream_callback=stream_callback,
            )

            try:
                # Resume from current stage
                current_stage = research_state.current_stage

                if current_stage == ResearchStage.CLARIFICATION:
                    # Re-run clarification
                    clarification_result = await clarification_agent.clarify_query(
                        research_state.user_query, deps
                    )
                    if clarification_result.is_clear:
                        research_state.clarified_query = clarification_result.clarified_query
                        research_state.advance_stage()
                        return await self.resume_research(research_state, api_keys, stream_callback)

                elif current_stage == ResearchStage.BRIEF_GENERATION:
                    if research_state.clarified_query:
                        research_brief = await brief_generator_agent.generate_brief(
                            research_state.clarified_query,
                            "medium",  # Default complexity
                            deps,
                        )
                        research_state.research_brief = research_brief
                        research_state.advance_stage()
                        return await self.resume_research(research_state, api_keys, stream_callback)

                elif current_stage == ResearchStage.RESEARCH_EXECUTION:
                    if research_state.research_brief:
                        findings = await research_executor_agent.execute_research(
                            research_state.research_brief,
                            deps,
                        )
                        research_state.findings = findings
                        research_state.advance_stage()
                        return await self.resume_research(research_state, api_keys, stream_callback)

                elif current_stage == ResearchStage.COMPRESSION:
                    if research_state.findings and research_state.research_brief:
                        compressed = await compression_agent.compress_findings(
                            research_state.findings,
                            research_state.research_brief.key_questions,
                            deps,
                        )
                        research_state.compressed_findings = compressed.summary
                        research_state.metadata["compressed_findings_full"] = compressed
                        research_state.advance_stage()
                        return await self.resume_research(research_state, api_keys, stream_callback)

                elif current_stage == ResearchStage.REPORT_GENERATION:
                    if (
                        research_state.research_brief
                        and research_state.findings
                        and research_state.compressed_findings
                    ):
                        # Reconstruct compressed findings from metadata if available
                        from open_deep_research_with_pydantic_ai.agents.compression import (
                            CompressedFindings,
                        )

                        if "compressed_findings_full" in research_state.metadata:
                            # Use the full object stored in metadata
                            compressed = CompressedFindings(
                                **research_state.metadata["compressed_findings_full"]
                            )
                        else:
                            # Fallback to basic reconstruction
                            compressed = CompressedFindings(
                                summary=research_state.compressed_findings or "",
                                key_insights=[],
                                themes={},
                            )

                        report = await report_generator_agent.generate_report(
                            research_state.research_brief,
                            research_state.findings,
                            compressed,
                            deps,
                        )

                        research_state.final_report = report
                        research_state.advance_stage()
                        research_state.completed_at = datetime.now()

                return research_state

            except Exception as e:
                logfire.error(
                    "Research resume failed",
                    request_id=research_state.request_id,
                    error=str(e),
                    exc_info=True,
                )
                research_state.set_error(str(e))
                return research_state


# Global workflow instance
workflow = ResearchWorkflow()
