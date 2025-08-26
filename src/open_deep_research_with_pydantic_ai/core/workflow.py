"""Main workflow orchestrator with integrated three-phase clarification system."""

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
from open_deep_research_with_pydantic_ai.agents.query_transformation import (
    query_transformation_agent,
)
from open_deep_research_with_pydantic_ai.agents.report_generator import report_generator_agent
from open_deep_research_with_pydantic_ai.agents.research_executor import (
    research_executor_agent,
)

# Removed old clarification system imports - now using direct agents
from open_deep_research_with_pydantic_ai.core.context import get_current_context
from open_deep_research_with_pydantic_ai.core.events import (
    emit_error,
    emit_research_started,
)
from open_deep_research_with_pydantic_ai.models.api_models import APIKeys, ResearchMetadata
from open_deep_research_with_pydantic_ai.models.research import (
    ResearchBrief,
    ResearchStage,
    ResearchState,
)


class ResearchWorkflow:
    """Orchestrator for the complete research workflow with three-phase clarification system."""

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
                "Research workflow initialized with three-phase clarification system",
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
        """Execute the complete research workflow with integrated three-phase clarification.

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

                # Execute the integrated three-phase clarification system
                await self._execute_three_phase_clarification(research_state, deps, user_query)

                research_state.advance_stage()

                # Stage 4: Research Execution
                logfire.info("Stage 4: Executing research", request_id=request_id)

                # Get the enhanced brief text from metadata
                brief_text = research_state.metadata.get(
                    "research_brief_text", "No research brief available"
                )

                # Create a minimal ResearchBrief from our enhanced text
                questions = [q.strip() + "?" for q in brief_text.split("?") if q.strip()][:5]
                if not questions:
                    questions = ["What are the key aspects of this topic?"]

                minimal_brief = ResearchBrief(
                    topic=user_query,
                    objectives=["Research and understand the topic comprehensively"],
                    key_questions=questions,
                    scope=brief_text[:500],  # Use first part of brief as scope
                    priority_areas=["General overview", "Key concepts", "Applications"],
                )

                findings = await research_executor_agent.execute_research(
                    minimal_brief,
                    deps,
                    max_parallel_tasks=3,
                )
                research_state.findings = findings
                research_state.advance_stage()

                # Stage 5: Compression
                logfire.info("Stage 5: Compressing findings", request_id=request_id)
                key_questions = [q.strip() for q in brief_text.split("?") if q.strip()][:5]
                compressed_findings = await compression_agent.compress_findings(
                    findings,
                    key_questions,
                    deps,
                )
                research_state.compressed_findings = compressed_findings.summary
                research_state.metadata["compressed_findings_full"] = compressed_findings
                research_state.advance_stage()

                # Stage 6: Report Generation
                logfire.info("Stage 6: Generating final report", request_id=request_id)

                # Recreate minimal ResearchBrief for report generation
                minimal_brief = ResearchBrief(
                    topic=user_query,
                    objectives=["Research and understand the topic comprehensively"],
                    key_questions=questions,
                    scope=brief_text[:500],
                    priority_areas=["General overview", "Key concepts", "Applications"],
                )

                final_report = await report_generator_agent.generate_report(
                    minimal_brief,
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
                    "Research workflow completed successfully with three-phase clarification",
                    request_id=request_id,
                    total_usage=deps.usage,
                    clarification_enhanced=True,
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

    async def _execute_three_phase_clarification(
        self, research_state: ResearchState, deps: ResearchDependencies, user_query: str
    ) -> None:
        """Execute the integrated three-phase clarification system.

        Phase 1: Enhanced Clarification Assessment
        Phase 2: Query Transformation
        Phase 3: Enhanced Brief Generation
        """

        # Phase 1: Enhanced Clarification Assessment
        logfire.info("Phase 1: Enhanced clarification assessment")

        # Create a simple prompt for clarification assessment
        clarification_prompt = f"Assess if this query needs clarification: {user_query}"
        clarification_result = await clarification_agent.run(clarification_prompt, deps)

        # Store clarification assessment
        if not research_state.metadata:
            research_state.metadata = {}
        research_state.metadata["clarification_assessment"] = {
            "needs_clarification": clarification_result.need_clarification,
            "question": clarification_result.question,
            "verification": clarification_result.verification,
        }

        # Handle clarification if needed (simplified for non-interactive)
        clarification_responses = {}
        if clarification_result.need_clarification and clarification_result.question:
            import sys

            if sys.stdin.isatty():  # Interactive mode (CLI)
                try:
                    from open_deep_research_with_pydantic_ai.interfaces.cli_clarification import (
                        ask_single_clarification_question,
                    )

                    # Get user response for the clarification question
                    user_response = ask_single_clarification_question(clarification_result.question)
                    if user_response:
                        clarification_responses[clarification_result.question] = user_response

                        # Add to conversation messages
                        if "conversation_messages" not in research_state.metadata:
                            research_state.metadata["conversation_messages"] = []
                        research_state.metadata["conversation_messages"].extend(
                            [
                                {"role": "assistant", "content": clarification_result.question},
                                {"role": "user", "content": user_response},
                            ]
                        )

                except (KeyboardInterrupt, EOFError):
                    logfire.info("User cancelled clarification")
            else:
                # Non-interactive mode - store for HTTP handling
                research_state.metadata.update(
                    {
                        "awaiting_clarification": True,
                        "clarification_question": clarification_result.question,
                    }
                )
                return  # Exit early for HTTP handling

        # Store clarification responses in metadata
        research_state.metadata["clarification_responses"] = clarification_responses

        # Phase 2: Query Transformation
        logfire.info("Phase 2: Query transformation")

        try:
            transformed_query = await query_transformation_agent.transform_query(
                original_query=user_query,
                clarification_responses=clarification_responses,
                conversation_context=research_state.metadata.get("conversation_messages", []),
                deps=deps,
            )

            # Store comprehensive transformation data
            research_state.metadata["transformed_query"] = {
                "original_query": transformed_query.original_query,
                "transformed_query": transformed_query.transformed_query,
                "supporting_questions": transformed_query.supporting_questions,
                "transformation_rationale": transformed_query.transformation_rationale,
                "specificity_score": transformed_query.specificity_score,
                "missing_dimensions": transformed_query.missing_dimensions,
                "clarification_responses": transformed_query.clarification_responses,
                "transformation_metadata": transformed_query.transformation_metadata,
            }

            logfire.info(
                "Query transformation completed",
                specificity_score=transformed_query.specificity_score,
                supporting_questions_count=len(transformed_query.supporting_questions),
            )

        except Exception as e:
            logfire.warning(f"Query transformation failed, proceeding without: {e}")
            # Continue without transformation data - brief generator has fallback

        # Phase 3: Enhanced Brief Generation
        logfire.info("Phase 3: Enhanced brief generation")

        # The brief generator will automatically use transformation data if available
        # and fall back to conversation-based generation if not
        brief_result = await brief_generator_agent.generate_from_conversation(deps)

        # Ensure brief is stored in metadata for downstream stages
        research_state.metadata["research_brief_text"] = brief_result.brief
        research_state.metadata["research_brief_confidence"] = brief_result.confidence_score

        logfire.info(
            "Three-phase clarification system completed",
            confidence=brief_result.confidence_score,
            brief_length=len(brief_result.brief),
            has_transformation="transformed_query" in research_state.metadata,
            clarification_count=len(clarification_responses),
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
                metadata=ResearchMetadata(),
                usage=Usage(),
                stream_callback=stream_callback,
            )

            try:
                # Resume from current stage
                current_stage = research_state.current_stage

                if current_stage == ResearchStage.CLARIFICATION:
                    # Check if we're waiting for clarification response
                    if research_state.metadata and research_state.metadata.get(
                        "awaiting_clarification"
                    ):
                        # Still waiting for user response in HTTP mode
                        logfire.info("Still awaiting clarification response")
                        return research_state

                    # Re-execute three-phase clarification system
                    await self._execute_three_phase_clarification(
                        research_state, deps, research_state.user_query
                    )
                    research_state.advance_stage()
                    return await self.resume_research(research_state, api_keys, stream_callback)

                elif current_stage == ResearchStage.BRIEF_GENERATION:
                    # Brief generation complete, advance to research execution
                    research_state.advance_stage()
                    return await self.resume_research(research_state, api_keys, stream_callback)

                elif current_stage == ResearchStage.RESEARCH_EXECUTION:
                    brief_text = (
                        research_state.metadata.get("research_brief_text")
                        if research_state.metadata
                        else None
                    )
                    if brief_text:
                        # Create minimal ResearchBrief for compatibility
                        from open_deep_research_with_pydantic_ai.models.research import (
                            ResearchBrief,
                        )

                        questions = [q.strip() + "?" for q in brief_text.split("?") if q.strip()][
                            :5
                        ]
                        if not questions:
                            questions = ["What are the key aspects of this topic?"]

                        minimal_brief = ResearchBrief(
                            topic=research_state.user_query,
                            objectives=["Research and understand the topic comprehensively"],
                            key_questions=questions,
                            scope=brief_text[:500],
                            priority_areas=["General overview", "Key concepts", "Applications"],
                        )

                        findings = await research_executor_agent.execute_research(
                            minimal_brief,
                            deps,
                            max_parallel_tasks=3,
                        )
                        research_state.findings = findings
                        research_state.advance_stage()
                        return await self.resume_research(research_state, api_keys, stream_callback)

                elif current_stage == ResearchStage.COMPRESSION:
                    brief_text = (
                        research_state.metadata.get("research_brief_text")
                        if research_state.metadata
                        else None
                    )
                    if research_state.findings and brief_text:
                        # Extract key questions from brief text (simplified approach)
                        key_questions = [q.strip() for q in brief_text.split("?") if q.strip()][:5]
                        compressed = await compression_agent.compress_findings(
                            research_state.findings,
                            key_questions,
                            deps,
                        )
                        research_state.compressed_findings = compressed.summary
                        research_state.metadata["compressed_findings_full"] = compressed
                        research_state.advance_stage()
                        return await self.resume_research(research_state, api_keys, stream_callback)

                elif current_stage == ResearchStage.REPORT_GENERATION:
                    brief_text = (
                        research_state.metadata.get("research_brief_text")
                        if research_state.metadata
                        else None
                    )
                    if (
                        brief_text
                        and research_state.findings
                        and research_state.compressed_findings
                    ):
                        # Reconstruct compressed findings from metadata if available
                        from open_deep_research_with_pydantic_ai.agents.compression import (
                            CompressedFindings,
                        )
                        from open_deep_research_with_pydantic_ai.models.research import (
                            ResearchBrief,
                        )

                        if "compressed_findings_full" in research_state.metadata:
                            # Use the full object stored in metadata
                            compressed = research_state.metadata["compressed_findings_full"]
                        else:
                            # Fallback to basic reconstruction
                            compressed = CompressedFindings(
                                summary=research_state.compressed_findings or "",
                                key_insights=[],
                                themes={},
                            )

                        # Recreate minimal ResearchBrief for report generation
                        questions = [q.strip() + "?" for q in brief_text.split("?") if q.strip()][
                            :5
                        ]
                        if not questions:
                            questions = ["What are the key aspects of this topic?"]

                        minimal_brief = ResearchBrief(
                            topic=research_state.user_query,
                            objectives=["Research and understand the topic comprehensively"],
                            key_questions=questions,
                            scope=brief_text[:500],
                            priority_areas=["General overview", "Key concepts", "Applications"],
                        )

                        report = await report_generator_agent.generate_report(
                            minimal_brief,
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

    async def execute_planning_only(
        self,
        user_query: str,
        api_keys: APIKeys | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> ResearchState:
        """Execute only the three-phase clarification and brief generation.

        This is perfect for API endpoints that just need the research brief.

        Args:
            user_query: User's research query
            api_keys: API keys for various services
            user_id: Optional user ID for isolation
            session_id: Optional session ID for user

        Returns:
            Research state with completed brief generation
        """
        self._ensure_initialized()

        # Get user context
        context = get_current_context()
        if user_id is None:
            user_id = context.user_id
        if session_id is None:
            session_id = context.session_id

        # Create research state
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
            )

            try:
                # Start research and execute three-phase clarification
                research_state.start_research()
                await emit_research_started(request_id, user_query)

                # Execute integrated three-phase system
                await self._execute_three_phase_clarification(research_state, deps, user_query)

                # Mark planning as complete
                research_state.current_stage = ResearchStage.BRIEF_GENERATION
                research_state.advance_stage()

                logfire.info(
                    "Planning phase completed with three-phase clarification",
                    request_id=request_id,
                    brief_confidence=research_state.metadata.get("research_brief_confidence", 0.0),
                )

                return research_state

            except Exception as e:
                logfire.error(
                    "Planning phase failed",
                    request_id=request_id,
                    error=str(e),
                    exc_info=True,
                )
                research_state.set_error(str(e))
                return research_state


# Global workflow instance
workflow = ResearchWorkflow()
