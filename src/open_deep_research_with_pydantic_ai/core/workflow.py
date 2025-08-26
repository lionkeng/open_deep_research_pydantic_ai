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

# Removed old clarification system imports - now using direct agents
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

                # Stage 1: Interactive Clarification and Brief Generation
                logfire.info("Stage 1: Clarification and brief generation", request_id=request_id)

                # Step 1: Try to generate research brief from conversation
                brief_result = await brief_generator_agent.generate_from_conversation(deps)

                # Step 2: Check if confidence is sufficient to proceed
                from open_deep_research_with_pydantic_ai.core.config import config

                if (
                    brief_result.confidence_score < config.research_brief_confidence_threshold
                    and config.research_interactive
                ):
                    # Check if we should ask another question
                    if await clarification_agent.should_ask_another_question(
                        deps, config.max_clarification_questions
                    ):
                        # Get clarification from user
                        clarification = await clarification_agent.assess_query(user_query, deps)

                        if clarification.need_clarification:
                            logfire.info("Clarification needed", question=clarification.question)

                            # Store clarification in metadata
                            if not research_state.metadata:
                                research_state.metadata = {}
                            research_state.metadata.update(
                                {
                                    "awaiting_clarification": True,
                                    "clarification_question": clarification.question,
                                    "conversation_messages": research_state.metadata.get(
                                        "conversation_messages", []
                                    )
                                    + [user_query],
                                }
                            )

                            # Check environment for interaction mode
                            import sys

                            if sys.stdin.isatty():  # Interactive terminal (CLI mode)
                                try:
                                    # Use the CLI interface for clarification
                                    from open_deep_research_with_pydantic_ai.interfaces.cli_clarification import (  # noqa: E501
                                        ask_single_clarification_question,
                                    )

                                    user_response = ask_single_clarification_question(
                                        clarification.question
                                    )

                                    if user_response:
                                        # Add response to conversation
                                        conversation = research_state.metadata.get(
                                            "conversation_messages", []
                                        )
                                        conversation.extend([clarification.question, user_response])
                                        research_state.metadata["conversation_messages"] = (
                                            conversation
                                        )

                                        # Generate new brief with updated conversation
                                        brief_result = (
                                            await brief_generator_agent.generate_from_conversation(
                                                deps
                                            )
                                        )
                                        logfire.info(
                                            "Updated brief after clarification",
                                            confidence=brief_result.confidence_score,
                                        )
                                    else:
                                        # User cancelled, proceed with original brief
                                        logfire.info("Clarification cancelled by user")

                                except (KeyboardInterrupt, EOFError):
                                    # User cancelled, proceed with original
                                    logfire.info("Clarification cancelled by user")
                            else:
                                # Non-interactive environment (HTTP mode or CI) - store question and
                                # pause workflow
                                logfire.info(
                                    "Non-interactive environment detected, storing clarification "
                                    "for HTTP response"
                                )
                                research_state.metadata.update(
                                    {
                                        "awaiting_clarification": True,
                                        "clarification_question": clarification.question,
                                    }
                                )
                                # Return early - workflow paused for HTTP client to handle
                                return research_state
                        else:
                            logfire.info(
                                "No clarification needed", verification=clarification.verification
                            )

                # The brief is already stored in metadata by the agent
                # Get the brief text from metadata for downstream stages
                brief_text = research_state.metadata.get("research_brief_text", brief_result.brief)

                logfire.info(
                    "Research brief generated",
                    confidence=brief_result.confidence_score,
                    brief_length=len(brief_result.brief),
                    missing_aspects=brief_result.missing_aspects,
                )

                research_state.advance_stage()

                # Stage 2: Research Execution
                logfire.info("Stage 2: Executing research", request_id=request_id)
                # Create a minimal ResearchBrief from our text
                brief_text = research_state.metadata.get(
                    "research_brief_text", "No research brief available"
                )

                # Create a minimal ResearchBrief object for compatibility
                from open_deep_research_with_pydantic_ai.models.research import ResearchBrief

                # Extract questions from the brief text
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

                # Stage 3: Compression
                logfire.info("Stage 3: Compressing findings", request_id=request_id)
                # Extract key questions from the brief text (simplified approach)
                brief_text = research_state.metadata.get(
                    "research_brief_text", "No research brief available"
                )
                key_questions = [q.strip() for q in brief_text.split("?") if q.strip()][:5]
                compressed_findings = await compression_agent.compress_findings(
                    findings,
                    key_questions,
                    deps,
                )
                # Store the summary in compressed_findings field
                research_state.compressed_findings = compressed_findings.summary
                # Store the full object in metadata for later use
                research_state.metadata["compressed_findings_full"] = compressed_findings
                research_state.advance_stage()

                # Stage 4: Report Generation
                logfire.info("Stage 4: Generating final report", request_id=request_id)
                # Recreate the minimal ResearchBrief for report generation
                brief_text = research_state.metadata.get(
                    "research_brief_text", "No research brief available"
                )
                questions = [q.strip() + "?" for q in brief_text.split("?") if q.strip()][:5]
                if not questions:
                    questions = ["What are the key aspects of this topic?"]

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
                    # Handle clarification resume - simplified approach

                    # Check if we're waiting for clarification response
                    if research_state.metadata and research_state.metadata.get(
                        "awaiting_clarification"
                    ):
                        # Still waiting for user response in HTTP mode
                        logfire.info("Still awaiting clarification response")
                        return research_state

                    # Generate brief from current conversation state
                    await brief_generator_agent.generate_from_conversation(deps)

                    # Brief is already stored in metadata by the agent
                    # Just advance to the next stage

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


# Global workflow instance
workflow = ResearchWorkflow()
