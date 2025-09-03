"""Integration tests for multi-question clarification system."""

from unittest.mock import MagicMock, patch

import pytest

from src.interfaces.cli_multi_clarification import (
    ask_choice_question,
    ask_multi_choice_question,
    ask_text_question,
    handle_multi_clarification_cli,
)
from src.models.api_models import ResearchMetadata
from src.models.clarification import (
    ClarificationAnswer,
    ClarificationQuestion,
    ClarificationRequest,
    ClarificationResponse,
)
from src.utils.serialization import (
    deserialize_clarification_request,
    deserialize_clarification_response,
    format_clarification_for_display,
    format_response_for_display,
    serialize_clarification_request,
    serialize_clarification_response,
)


class TestMultiClarificationIntegration:
    """Integration tests for the multi-question clarification system."""

    @pytest.fixture
    def sample_request(self):
        """Create a sample multi-question clarification request."""
        return ClarificationRequest(
            questions=[
                ClarificationQuestion(
                    question="What is your technical expertise level?",
                    is_required=True,
                    question_type="choice",
                    choices=["Beginner", "Intermediate", "Advanced", "Expert"],
                    order=0,
                ),
                ClarificationQuestion(
                    question="Which aspects interest you most?",
                    is_required=False,
                    question_type="multi_choice",
                    choices=["Performance", "Security", "Scalability", "Cost", "Ease of use"],
                    order=1,
                ),
                ClarificationQuestion(
                    question="What is your primary use case?",
                    is_required=True,
                    question_type="text",
                    context="This helps us tailor the research to your needs",
                    order=2,
                ),
            ]
        )

    @pytest.fixture
    def sample_response(self, sample_request):
        """Create a sample response to the request."""
        return ClarificationResponse(
            request_id=sample_request.id,
            answers=[
                ClarificationAnswer(
                    question_id=sample_request.questions[0].id,
                    answer="Intermediate",
                    skipped=False,
                ),
                ClarificationAnswer(
                    question_id=sample_request.questions[1].id,
                    answer="Performance, Scalability",
                    skipped=False,
                ),
                ClarificationAnswer(
                    question_id=sample_request.questions[2].id,
                    answer="Building a real-time data processing pipeline",
                    skipped=False,
                ),
            ],
        )

    def test_serialization_round_trip(self, sample_request, sample_response):
        """Test that serialization and deserialization work correctly."""
        # Serialize request
        request_json = serialize_clarification_request(sample_request)
        assert isinstance(request_json, str)

        # Deserialize request
        request_data = deserialize_clarification_request(request_json)
        reconstructed_request = ClarificationRequest(**request_data)
        assert len(reconstructed_request.questions) == len(sample_request.questions)

        # Serialize response
        response_json = serialize_clarification_response(sample_response)
        assert isinstance(response_json, str)

        # Deserialize response
        response_data = deserialize_clarification_response(response_json)
        reconstructed_response = ClarificationResponse(**response_data)
        assert len(reconstructed_response.answers) == len(sample_response.answers)

    def test_display_formatting(self, sample_request, sample_response):
        """Test human-readable formatting."""
        # Format request for display
        request_display = format_clarification_for_display(sample_request)
        assert "Clarification Questions:" in request_display
        assert "[Required]" in request_display
        assert "[Optional]" in request_display
        assert "What is your technical expertise level?" in request_display

        # Format response for display
        response_display = format_response_for_display(sample_response, sample_request)
        assert "Clarification Responses:" in response_display
        assert "Intermediate" in response_display
        assert "Performance, Scalability" in response_display

    def test_metadata_helper_methods(self, sample_request, sample_response):
        """Test ResearchMetadata helper methods."""
        metadata = ResearchMetadata()

        # Initially no questions pending
        assert metadata.is_clarification_complete()
        assert len(metadata.get_pending_questions()) == 0

        # Add clarification request
        metadata.clarification_request = sample_request

        # Now should have pending questions
        assert not metadata.is_clarification_complete()
        pending = metadata.get_pending_questions()
        assert len(pending) == 2  # Two required questions

        # Add response for one question
        metadata.add_clarification_response(
            sample_request.questions[0].id,
            "Intermediate"
        )

        # Should still have one pending
        pending = metadata.get_pending_questions()
        assert len(pending) == 1
        assert pending[0].id == sample_request.questions[2].id

        # Add response for the other required question
        metadata.add_clarification_response(
            sample_request.questions[2].id,
            "Building a data pipeline"
        )

        # Debug - check what we have
        if metadata.clarification_response:
            print(f"Answers count: {len(metadata.clarification_response.answers)}")
            print(f"Question IDs in request: {[q.id for q in sample_request.questions]}")
            for answer in metadata.clarification_response.answers:
                answer_text = answer.answer if not answer.skipped else '[skipped]'
                print(f"  - Answer for question {answer.question_id}: {answer_text}")
            errors = metadata.clarification_response.validate_against_request(sample_request)
            if errors:
                print(f"Validation errors: {errors}")

        # Should now be complete (optional question can be skipped)
        assert metadata.is_clarification_complete()

    @patch("sys.stdin.isatty")
    @patch("rich.prompt.Prompt.ask")
    @patch("rich.prompt.IntPrompt.ask")
    def test_cli_text_question(self, mock_int_prompt, mock_prompt, mock_isatty):
        """Test CLI handling of text questions."""
        mock_isatty.return_value = True
        mock_prompt.return_value = "My answer"

        question = ClarificationQuestion(
            question="What is your use case?",
            is_required=True,
            question_type="text",
        )

        console = MagicMock()
        answer = ask_text_question(question, console)

        assert answer == "My answer"
        mock_prompt.assert_called_once()

    @patch("sys.stdin.isatty")
    @patch("rich.prompt.IntPrompt.ask")
    def test_cli_choice_question(self, mock_int_prompt, mock_isatty):
        """Test CLI handling of choice questions."""
        mock_isatty.return_value = True
        mock_int_prompt.return_value = 2

        question = ClarificationQuestion(
            question="Select level",
            is_required=True,
            question_type="choice",
            choices=["Low", "Medium", "High"],
        )

        console = MagicMock()
        answer = ask_choice_question(question, console)

        assert answer == "Medium"
        mock_int_prompt.assert_called_once()

    @patch("sys.stdin.isatty")
    @patch("rich.prompt.Prompt.ask")
    def test_cli_multi_choice_question(self, mock_prompt, mock_isatty):
        """Test CLI handling of multi-choice questions."""
        mock_isatty.return_value = True
        mock_prompt.return_value = "1,3"

        question = ClarificationQuestion(
            question="Select features",
            is_required=True,
            question_type="multi_choice",
            choices=["Feature A", "Feature B", "Feature C"],
        )

        console = MagicMock()
        answer = ask_multi_choice_question(question, console)

        assert answer == "Feature A, Feature C"
        mock_prompt.assert_called_once()

    @pytest.mark.asyncio
    @patch("sys.stdin.isatty")
    @patch("src.interfaces.cli_multi_clarification.ask_text_question")
    @patch("src.interfaces.cli_multi_clarification.ask_choice_question")
    @patch("src.interfaces.cli_multi_clarification.ask_multi_choice_question")
    async def test_full_cli_flow(
        self,
        mock_multi_choice,
        mock_choice,
        mock_text,
        mock_isatty,
        sample_request,
    ):
        """Test complete CLI clarification flow."""
        mock_isatty.return_value = True
        mock_choice.return_value = "Intermediate"
        mock_multi_choice.return_value = "Performance, Security"
        mock_text.return_value = "Building APIs"

        console = MagicMock()
        response = await handle_multi_clarification_cli(
            sample_request,
            "Original query",
            console
        )

        assert response is not None
        assert len(response.answers) == 3
        assert response.answers[0].answer == "Intermediate"
        assert response.answers[1].answer == "Performance, Security"
        assert response.answers[2].answer == "Building APIs"

    def test_response_validation(self, sample_request):
        """Test response validation against request."""
        # Create incomplete response (missing required question)
        incomplete_response = ClarificationResponse(
            request_id=sample_request.id,
            answers=[
                ClarificationAnswer(
                    question_id=sample_request.questions[0].id,
                    answer="Intermediate",
                    skipped=False,
                ),
                # Skipping optional question is OK
                ClarificationAnswer(
                    question_id=sample_request.questions[1].id,
                    skipped=True,
                ),
                # Missing required question[2]
            ],
        )

        errors = incomplete_response.validate_against_request(sample_request)
        assert len(errors) > 0
        assert "Required question" in errors[0]

        # Create response with invalid choice
        invalid_choice_response = ClarificationResponse(
            request_id=sample_request.id,
            answers=[
                ClarificationAnswer(
                    question_id=sample_request.questions[0].id,
                    answer="Invalid Level",  # Not in choices
                    skipped=False,
                ),
                ClarificationAnswer(
                    question_id=sample_request.questions[1].id,
                    skipped=True,
                ),
                ClarificationAnswer(
                    question_id=sample_request.questions[2].id,
                    answer="Some use case",
                    skipped=False,
                ),
            ],
        )

        errors = invalid_choice_response.validate_against_request(sample_request)
        assert len(errors) > 0
        assert "Invalid choice" in errors[0]
