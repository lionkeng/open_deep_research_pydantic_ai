"""Integration tests for the complete research workflow with new clarification approach."""

from unittest.mock import patch
import pytest
import pytest_asyncio
from pydantic import SecretStr

from open_deep_research_with_pydantic_ai.core.workflow import workflow
from open_deep_research_with_pydantic_ai.models.api_models import APIKeys
from open_deep_research_with_pydantic_ai.models.research import (
    ResearchStage,
    ResearchState,
    ResearchFinding,
    ResearchReport,
)
from open_deep_research_with_pydantic_ai.agents.brief_generator import ResearchBrief as BriefGeneratorResearchBrief
from open_deep_research_with_pydantic_ai.agents.clarification import ClarifyWithUser
from open_deep_research_with_pydantic_ai.agents.compression import CompressedFindings


@pytest_asyncio.fixture
async def mock_api_keys() -> APIKeys:
    """Create mock API keys."""
    return APIKeys(openai=SecretStr("test-openai-key"))


class TestWorkflowStage1Integration:
    """Test the integrated Stage 1 workflow (clarification + brief generation)."""

    @pytest.mark.asyncio
    async def test_workflow_high_confidence_brief_no_clarification(self, mock_api_keys: APIKeys):
        """Test workflow when initial brief has high confidence - no clarification needed."""

        # Mock brief generator to return high confidence
        with patch('open_deep_research_with_pydantic_ai.agents.brief_generator.brief_generator_agent.generate_from_conversation') as mock_brief:
            mock_brief.return_value = BriefGeneratorResearchBrief(
                brief="I want to research quantum computing applications in cryptography, focusing on current implementations and security implications.",
                confidence_score=0.9,  # High confidence - should skip clarification
                missing_aspects=[]
            )

            # Mock the downstream agents to avoid full workflow execution
            with patch('open_deep_research_with_pydantic_ai.agents.research_executor.research_executor_agent.execute_research') as mock_research:
                mock_research.return_value = []

                with patch('open_deep_research_with_pydantic_ai.agents.compression.compression_agent.compress_findings') as mock_compress:
                    mock_compress.return_value = CompressedFindings(summary="Test summary", key_insights=[], themes={})

                    with patch('open_deep_research_with_pydantic_ai.agents.report_generator.report_generator_agent.generate_report') as mock_report:
                        mock_report.return_value = ResearchReport(
                            title="Test Report",
                            executive_summary="Test summary",
                            introduction="Test intro",
                            methodology="Test method",
                            sections=[],
                            conclusion="Test conclusion"
                        )

                        # Execute the workflow
                        result = await workflow.execute_research(
                            user_query="I want to research quantum computing",
                            api_keys=mock_api_keys
                        )

                        # Verify workflow completed successfully
                        assert result.current_stage in [ResearchStage.COMPLETED, ResearchStage.REPORT_GENERATION]
                        assert result.error_message is None
                        assert result.final_report is not None
                        assert mock_brief.call_count == 1  # Brief generated once

    @pytest.mark.asyncio
    async def test_workflow_low_confidence_brief_with_clarification_interactive(self, mock_api_keys: APIKeys):
        """Test workflow when brief needs clarification in interactive mode."""

        # Mock configuration to enable interactive mode
        with patch('open_deep_research_with_pydantic_ai.core.config.config') as mock_config:
            mock_config.research_interactive = True
            mock_config.max_clarification_questions = 2
            mock_config.research_brief_confidence_threshold = 0.7

            # Mock brief generator to return low confidence initially, then high after clarification
            brief_calls = [
                BriefGeneratorResearchBrief(
                    brief="I want to research AI",
                    confidence_score=0.5,  # Low confidence - needs clarification
                    missing_aspects=["specific domain", "timeframe"]
                ),
                BriefGeneratorResearchBrief(
                    brief="I want to research AI applications in healthcare diagnostics, focusing on recent developments in the last 2 years.",
                    confidence_score=0.9,  # High confidence after clarification
                    missing_aspects=[]
                )
            ]

            with patch('open_deep_research_with_pydantic_ai.agents.brief_generator.brief_generator_agent.generate_from_conversation', side_effect=brief_calls) as mock_brief:

                # Mock clarification agent run method
                with patch('open_deep_research_with_pydantic_ai.agents.clarification.clarification_agent.run') as mock_clarify:
                    mock_clarify.return_value = ClarifyWithUser(
                        need_clarification=True,
                        question="What specific area of AI are you interested in?",
                        verification=""
                    )

                    # Mock CLI interaction (simulating interactive terminal)
                    with patch('sys.stdin.isatty', return_value=True):
                        with patch('open_deep_research_with_pydantic_ai.interfaces.cli_clarification.ask_single_clarification_question') as mock_cli:
                            mock_cli.return_value = "I'm interested in AI for healthcare diagnostics"

                            # Mock downstream agents
                            with patch('open_deep_research_with_pydantic_ai.agents.research_executor.research_executor_agent.execute_research', return_value=[]):
                                with patch('open_deep_research_with_pydantic_ai.agents.compression.compression_agent.compress_findings') as mock_compress:
                                    mock_compress.return_value = CompressedFindings(summary="Test summary", key_insights=[], themes={})

                                    with patch('open_deep_research_with_pydantic_ai.agents.report_generator.report_generator_agent.generate_report') as mock_report:
                                        mock_report.return_value = ResearchReport(
                                            title="AI Healthcare Report",
                                            executive_summary="Test summary",
                                            introduction="Test intro",
                                            methodology="Test method",
                                            sections=[],
                                            conclusion="Test conclusion"
                                        )

                                        # Execute the workflow
                                        result = await workflow.execute_research(
                                            user_query="I want to research AI",
                                            api_keys=mock_api_keys
                                        )

                                        # Verify workflow completed successfully with clarification
                                        assert result.current_stage in [ResearchStage.COMPLETED, ResearchStage.REPORT_GENERATION]
                                        assert result.error_message is None
                                        assert result.final_report is not None
                                        assert mock_brief.call_count == 2  # Brief generated twice (before and after clarification)
                                        assert mock_cli.called  # CLI clarification was called

    @pytest.mark.asyncio
    async def test_workflow_low_confidence_brief_http_mode(self, mock_api_keys: APIKeys):
        """Test workflow pauses for HTTP clarification when not in terminal mode."""

        # Mock configuration to enable interactive mode
        with patch('open_deep_research_with_pydantic_ai.core.config.config') as mock_config:
            mock_config.research_interactive = True
            mock_config.max_clarification_questions = 2
            mock_config.research_brief_confidence_threshold = 0.7

            # Mock brief generator to return low confidence
            with patch('open_deep_research_with_pydantic_ai.agents.brief_generator.brief_generator_agent.generate_from_conversation') as mock_brief:
                mock_brief.return_value = BriefGeneratorResearchBrief(
                    brief="I want to research AI",
                    confidence_score=0.5,  # Low confidence - needs clarification
                    missing_aspects=["specific domain"]
                )

                # Mock clarification agent run method
                with patch('open_deep_research_with_pydantic_ai.agents.clarification.clarification_agent.run') as mock_clarify:
                    mock_clarify.return_value = ClarifyWithUser(
                        need_clarification=True,
                        question="What specific area of AI are you interested in?",
                        verification=""
                    )

                    # Mock non-interactive environment (HTTP mode)
                    with patch('sys.stdin.isatty', return_value=False):

                        # Execute the workflow
                        result = await workflow.execute_research(
                            user_query="I want to research AI",
                            api_keys=mock_api_keys
                        )

                        # Verify workflow paused for HTTP clarification
                        assert result.current_stage == ResearchStage.CLARIFICATION
                        assert result.metadata["awaiting_clarification"] is True
                        assert result.metadata["clarification_question"] == "What specific area of AI are you interested in?"
                        assert result.final_report is None  # Should not complete

    @pytest.mark.asyncio
    async def test_workflow_question_limit_reached(self, mock_api_keys: APIKeys):
        """Test workflow continues when question limit is reached."""

        # Mock configuration
        with patch('open_deep_research_with_pydantic_ai.core.config.config') as mock_config:
            mock_config.research_interactive = True
            mock_config.max_clarification_questions = 1  # Low limit
            mock_config.research_brief_confidence_threshold = 0.7

            # Mock brief generator to return low confidence
            with patch('open_deep_research_with_pydantic_ai.agents.brief_generator.brief_generator_agent.generate_from_conversation') as mock_brief:
                mock_brief.return_value = BriefGeneratorResearchBrief(
                    brief="I want to research AI",
                    confidence_score=0.5,  # Low confidence but we'll hit question limit
                    missing_aspects=["specific domain"]
                )

                # Mock clarification agent - workflow continues due to question limit
                # Mock downstream agents
                with patch('open_deep_research_with_pydantic_ai.agents.research_executor.research_executor_agent.execute_research', return_value=[]):
                    with patch('open_deep_research_with_pydantic_ai.agents.compression.compression_agent.compress_findings') as mock_compress:
                        mock_compress.return_value = CompressedFindings(summary="Test summary", key_insights=[], themes={})

                        with patch('open_deep_research_with_pydantic_ai.agents.report_generator.report_generator_agent.generate_report') as mock_report:
                            mock_report.return_value = ResearchReport(
                                title="AI Report",
                                executive_summary="Test summary",
                                introduction="Test intro",
                                methodology="Test method",
                                sections=[],
                                conclusion="Test conclusion"
                            )

                            # Execute the workflow
                            result = await workflow.execute_research(
                                user_query="I want to research AI",
                                api_keys=mock_api_keys
                            )

                            # Verify workflow completed despite low confidence (hit question limit)
                            assert result.current_stage in [ResearchStage.COMPLETED, ResearchStage.REPORT_GENERATION]
                            assert result.error_message is None
                            assert result.final_report is not None


class TestWorkflowResumeIntegration:
    """Test workflow resume functionality for HTTP clarification flow."""

    @pytest.mark.asyncio
    async def test_resume_research_after_clarification(self, mock_api_keys: APIKeys):
        """Test resuming research after user provides clarification via HTTP."""

        # Create a research state that's paused for clarification
        research_state = ResearchState(
            request_id="test-resume-123",
            user_id="test-user",
            session_id="test-session",
            user_query="I want to research AI",
            current_stage=ResearchStage.CLARIFICATION,
            metadata={
                "conversation_messages": [
                    "I want to research AI",
                    "What specific area of AI are you interested in?",
                    "Machine learning applications in healthcare"
                ],
                "awaiting_clarification": False,  # User responded, ready to resume
                "research_brief_text": "I want to research machine learning applications in healthcare, focusing on diagnostic tools and patient outcomes.",
                "research_brief_confidence": 0.9
            }
        )

        # Mock brief generator to return updated brief with conversation
        with patch('open_deep_research_with_pydantic_ai.agents.brief_generator.brief_generator_agent.generate_from_conversation') as mock_brief:
            mock_brief.return_value = BriefGeneratorResearchBrief(
                brief="I want to research machine learning applications in healthcare, focusing on diagnostic tools and patient outcomes.",
                confidence_score=0.9,  # High confidence after clarification
                missing_aspects=[]
            )

            # Mock downstream agents
            mock_findings = [
                ResearchFinding(
                    content="AI in healthcare has shown significant promise in diagnostic applications",
                    source="http://example.com/ai-healthcare",
                    relevance_score=0.9,
                    confidence=0.8
                )
            ]
            with patch('open_deep_research_with_pydantic_ai.agents.research_executor.research_executor_agent.execute_research', return_value=mock_findings):
                with patch('open_deep_research_with_pydantic_ai.agents.compression.compression_agent.compress_findings') as mock_compress:
                    mock_compress.return_value = CompressedFindings(summary="Healthcare ML summary", key_insights=[], themes={})

                    with patch('open_deep_research_with_pydantic_ai.agents.report_generator.report_generator_agent.generate_report') as mock_report:
                        mock_report.return_value = ResearchReport(
                            title="ML Healthcare Report",
                            executive_summary="Healthcare ML analysis",
                            introduction="Test intro",
                            methodology="Test method",
                            sections=[],
                            conclusion="Test conclusion"
                        )

                        # Resume the workflow
                        result = await workflow.resume_research(
                            research_state=research_state,
                            api_keys=mock_api_keys
                        )

                        # Verify workflow completed successfully
                        assert result.current_stage in [ResearchStage.COMPLETED, ResearchStage.REPORT_GENERATION]
                        assert result.error_message is None
                        assert result.final_report is not None
                        assert result.final_report.title == "ML Healthcare Report"
                        assert mock_brief.called  # Brief was regenerated with updated conversation


class TestWorkflowErrorHandling:
    """Test workflow error handling and recovery."""

    @pytest.mark.asyncio
    async def test_workflow_handles_brief_generation_error(self, mock_api_keys: APIKeys):
        """Test workflow handles errors in brief generation gracefully."""

        # Mock brief generator to raise an exception
        with patch('open_deep_research_with_pydantic_ai.agents.brief_generator.brief_generator_agent.generate_from_conversation') as mock_brief:
            mock_brief.side_effect = Exception("Brief generation failed")

            # Execute the workflow
            result = await workflow.execute_research(
                user_query="I want to research AI",
                api_keys=mock_api_keys
            )

            # Verify workflow handled error gracefully
            assert result.current_stage == ResearchStage.COMPLETED
            assert result.error_message == "Brief generation failed"
            assert result.final_report is None

    @pytest.mark.asyncio
    async def test_workflow_handles_clarification_error(self, mock_api_keys: APIKeys):
        """Test workflow handles errors in clarification gracefully."""

        # Mock configuration to enable interactive mode
        with patch('open_deep_research_with_pydantic_ai.core.config.config') as mock_config:
            mock_config.research_interactive = True
            mock_config.max_clarification_questions = 2
            mock_config.research_brief_confidence_threshold = 0.7

            # Mock brief generator to return low confidence
            with patch('open_deep_research_with_pydantic_ai.agents.brief_generator.brief_generator_agent.generate_from_conversation') as mock_brief:
                mock_brief.return_value = BriefGeneratorResearchBrief(
                    brief="I want to research AI",
                    confidence_score=0.5,  # Low confidence - needs clarification
                    missing_aspects=["specific domain"]
                )

                # Mock clarification agent to raise error
                with patch('open_deep_research_with_pydantic_ai.agents.clarification.clarification_agent.run') as mock_clarify:
                    mock_clarify.side_effect = Exception("Clarification failed")

                    # Execute the workflow
                    result = await workflow.execute_research(
                        user_query="I want to research AI",
                        api_keys=mock_api_keys
                    )

                    # Verify workflow handled error gracefully
                    assert result.current_stage == ResearchStage.COMPLETED
                    assert result.error_message == "Clarification failed"
                    assert result.final_report is None
