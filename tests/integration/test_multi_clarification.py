"""Integration tests for multi-question clarification system."""

from unittest.mock import MagicMock, patch

import pytest

from interfaces.cli_multi_clarification import (
    ask_choice_question,
    ask_multi_choice_question,
    ask_text_question,
    handle_multi_clarification_cli,
)
from models.clarification import (
    ClarificationAnswer,
    ClarificationChoice,
    ClarificationQuestion,
    ClarificationRequest,
    ClarificationResponse,
)
from models.metadata import ResearchMetadata
from utils.serialization import (
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
                    choices=[
                        ClarificationChoice(id="beginner", label="Beginner"),
                        ClarificationChoice(id="intermediate", label="Intermediate"),
                        ClarificationChoice(id="advanced", label="Advanced"),
                        ClarificationChoice(id="expert", label="Expert"),
                    ],
                    order=0,
                ),
                ClarificationQuestion(
                    question="Which aspects interest you most?",
                    is_required=False,
                    question_type="multi_choice",
                    choices=[
                        ClarificationChoice(id="performance", label="Performance"),
                        ClarificationChoice(id="security", label="Security"),
                        ClarificationChoice(id="scalability", label="Scalability"),
                        ClarificationChoice(id="cost", label="Cost"),
                        ClarificationChoice(id="ease", label="Ease of use"),
                    ],
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
                    selection={"id": "intermediate"},
                    skipped=False,
                ),
                ClarificationAnswer(
                    question_id=sample_request.questions[1].id,
                    selections=[{"id": "performance"}, {"id": "scalability"}],
                    skipped=False,
                ),
                ClarificationAnswer(
                    question_id=sample_request.questions[2].id,
                    text="Building a real-time data processing pipeline",
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
        assert "Performance" in response_display and "Scalability" in response_display

    def test_metadata_helper_methods(self, sample_request, sample_response):
        """Test ResearchMetadata helper methods."""
        metadata = ResearchMetadata()

        # Initially no questions pending
        assert metadata.is_clarification_complete()
        assert len(metadata.get_pending_questions()) == 0

        # Add clarification request
        metadata.clarification.request = sample_request

        # Now should have pending questions
        assert not metadata.is_clarification_complete()
        pending = metadata.get_pending_questions()
        assert len(pending) == 2  # Two required questions

        # Add response for one question
        # Only text helper is supported; choice/multi_choice require structured path
        metadata.add_clarification_response(
            sample_request.questions[2].id,
            "Building a data pipeline",
        )

        # Should still have one pending
        pending = metadata.get_pending_questions()
        # One required text answered; one required choice still pending
        assert len(pending) == 1
        assert pending[0].id == sample_request.questions[0].id

        # Add response for the other required question
        # Structured choice answer added directly into response
        sel = {"id": "intermediate"}
        if metadata.clarification.response is None:
            metadata.clarification.response = ClarificationResponse(
                request_id=sample_request.id,
                answers=[ClarificationAnswer(question_id=sample_request.questions[0].id, selection=sel)],
            )
        else:
            metadata.clarification.response.answers.append(
                ClarificationAnswer(question_id=sample_request.questions[0].id, selection=sel)
            )
            metadata.clarification.response.model_post_init(None)

        # Debug - check what we have
        if metadata.clarification.response:
            print(f"Answers count: {len(metadata.clarification.response.answers)}")
            print(f"Question IDs in request: {[q.id for q in sample_request.questions]}")
            for answer in metadata.clarification.response.answers:
                print(f"  - Answer for question {answer.question_id}: [structured]")
            errors = metadata.clarification.response.validate_against_request(sample_request)
            if errors:
                print(f"Validation errors: {errors}")

        # Should now be complete (optional question can be skipped)
        assert metadata.is_clarification_complete()

    @patch("interfaces.cli_multi_clarification.sys.stdin.isatty")
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

    @patch("interfaces.cli_multi_clarification.has_interactive", False)
    @patch("interfaces.cli_multi_clarification.sys.stdin.isatty")
    @patch("rich.prompt.IntPrompt.ask")
    def test_cli_choice_question(self, mock_int_prompt, mock_isatty):
        """Test CLI handling of choice questions."""
        mock_isatty.return_value = True
        mock_int_prompt.return_value = 2

        question = ClarificationQuestion(
            question="Select level",
            is_required=True,
            question_type="choice",
            choices=[
                ClarificationChoice(id="low", label="Low"),
                ClarificationChoice(id="med", label="Medium"),
                ClarificationChoice(id="high", label="High"),
            ],
        )

        console = MagicMock()
        answer = ask_choice_question(question, console)

        assert answer is not None and answer.id == "med"
        mock_int_prompt.assert_called_once()

    @patch("interfaces.cli_multi_clarification.has_interactive", False)
    @patch("interfaces.cli_multi_clarification.sys.stdin.isatty")
    @patch("rich.prompt.IntPrompt.ask")
    @patch("rich.prompt.Prompt.ask")
    def test_cli_choice_question_with_specify(self, mock_prompt, mock_int_prompt, mock_isatty):
        """Test CLI handling of '(please specify)' choice with follow-up input."""
        mock_isatty.return_value = True
        # Select the first option
        mock_int_prompt.return_value = 1
        # Then provide details
        mock_prompt.return_value = "Seattle, WA"

        question = ClarificationQuestion(
            question=(
                "What geographic market(s) should the AEO-focused digital marketing "
                "strategy target?"
            ),
            is_required=True,
            question_type="choice",
            choices=[
                ClarificationChoice(
                    id="city",
                    label="Single city / metro area",
                    requires_details=True,
                    details_prompt="Enter city",
                ),
                ClarificationChoice(
                    id="state",
                    label="Single state / province",
                    requires_details=True,
                    details_prompt="Enter state",
                ),
                ClarificationChoice(id="national", label="National (country) level"),
            ],
        )

        console = MagicMock()
        answer = ask_choice_question(question, console)

        assert answer is not None and answer.id == "city" and answer.details == "Seattle, WA"

    @patch("interfaces.cli_multi_clarification.has_interactive", False)
    @patch("interfaces.cli_multi_clarification.sys.stdin.isatty")
    @patch("rich.prompt.Prompt.ask")
    def test_cli_multi_choice_question(self, mock_prompt, mock_isatty):
        """Test CLI handling of multi-choice questions."""
        mock_isatty.return_value = True
        mock_prompt.return_value = "1,3"

        question = ClarificationQuestion(
            question="Select features",
            is_required=True,
            question_type="multi_choice",
            choices=[
                ClarificationChoice(id="A", label="Feature A"),
                ClarificationChoice(id="B", label="Feature B"),
                ClarificationChoice(id="C", label="Feature C"),
            ],
        )

        console = MagicMock()
        answer = ask_multi_choice_question(question, console)

        assert answer is not None and [sel.id for sel in answer] == ["A", "C"]
        mock_prompt.assert_called_once()

    @patch("interfaces.cli_multi_clarification.has_interactive", False)
    @patch("interfaces.cli_multi_clarification.sys.stdin.isatty")
    @patch("rich.prompt.Prompt.ask")
    def test_cli_multi_choice_with_mixed_specify(self, mock_prompt, mock_isatty):
        """Test multi-choice where one selection requires specification."""
        mock_isatty.return_value = True
        # User selects 1 and 2
        mock_prompt.side_effect = ["1,2", "Germany"]

        question = ClarificationQuestion(
            question="Target regions",
            is_required=True,
            question_type="multi_choice",
            choices=[
                ClarificationChoice(id="nat", label="National (country) level"),
                ClarificationChoice(
                    id="multi",
                    label="Multiple countries",
                    requires_details=True,
                    details_prompt="Which countries?",
                ),
                ClarificationChoice(id="other", label="Other", is_other=True, details_prompt="Specify other"),
            ],
        )

        console = MagicMock()
        answer = ask_multi_choice_question(question, console)
        assert answer is not None
        # Expect selections for nat and multi with details
        assert [sel.id for sel in answer] == ["nat", "multi"]
        assert answer[1].details == "Germany"
        # Two prompts: one for selections, one for details
        assert mock_prompt.call_count == 2

    @pytest.mark.asyncio
    @patch("interfaces.cli_multi_clarification.sys.stdin.isatty")
    @patch("interfaces.cli_multi_clarification.ask_text_question")
    @patch("interfaces.cli_multi_clarification.ask_choice_question")
    @patch("interfaces.cli_multi_clarification.ask_multi_choice_question")
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
        mock_choice.return_value = {"id": "intermediate"}
        mock_multi_choice.return_value = [{"id": "performance"}, {"id": "security"}]
        mock_text.return_value = "Building APIs"

        console = MagicMock()
        response = await handle_multi_clarification_cli(
            sample_request,
            "Original query",
            console
        )

        assert response is not None
        assert len(response.answers) == 3
        assert response.answers[0].selection is not None and response.answers[0].selection.id == "intermediate"
        assert response.answers[1].selections is not None and [sel.id for sel in response.answers[1].selections] == ["performance", "security"]
        assert response.answers[2].text == "Building APIs"

    def test_response_validation(self, sample_request):
        """Test response validation against request."""
        # Create incomplete response (missing required question)
        incomplete_response = ClarificationResponse(
            request_id=sample_request.id,
            answers=[
                ClarificationAnswer(
                    question_id=sample_request.questions[0].id,
                    selection={"id": "intermediate"},
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
                    selection={"id": "invalid"},  # Not in choices
                    skipped=False,
                ),
                ClarificationAnswer(
                    question_id=sample_request.questions[1].id,
                    skipped=True,
                ),
                ClarificationAnswer(
                    question_id=sample_request.questions[2].id,
                    text="Some use case",
                    skipped=False,
                ),
            ],
        )

        errors = invalid_choice_response.validate_against_request(sample_request)
        assert len(errors) > 0
        assert "Invalid choice" in errors[0]

    def test_validation_accepts_details_required_structured_answers(self):
        """Validation should accept structured selections when details are required."""
        req = ClarificationRequest(
            questions=[
                ClarificationQuestion(
                    question="Market targeting",
                    is_required=True,
                    question_type="choice",
                    choices=[
                        ClarificationChoice(
                            id="city", label="Single city / metro area", requires_details=True
                        ),
                        ClarificationChoice(id="national", label="National (country) level"),
                        ClarificationChoice(id="other", label="Other", is_other=True),
                    ],
                ),
                ClarificationQuestion(
                    question="Regions",
                    is_required=True,
                    question_type="multi_choice",
                    choices=[
                        ClarificationChoice(
                            id="multi", label="Multiple countries", requires_details=True
                        ),
                        ClarificationChoice(
                            id="state", label="Single state / province", requires_details=True
                        ),
                        ClarificationChoice(id="other", label="Other", is_other=True),
                    ],
                ),
            ]
        )

        resp = ClarificationResponse(
            request_id=req.id,
            answers=[
                ClarificationAnswer(
                    question_id=req.questions[0].id,
                    selection={"id": "city", "details": "Paris"},
                ),
                ClarificationAnswer(
                    question_id=req.questions[1].id,
                    selections=[
                        {"id": "multi", "details": "Germany"},
                        {"id": "state", "details": "Bavaria"},
                        {"id": "other", "details": "DACH"},
                    ],
                ),
            ],
        )

        errors = resp.validate_against_request(req)
        assert errors == []
