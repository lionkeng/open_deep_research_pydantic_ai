"""Tests for clarification models."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from models.clarification import (
    ChoiceSelection,
    ClarificationAnswer,
    ClarificationChoice,
    ClarificationQuestion,
    ClarificationRequest,
    ClarificationResponse,
)


class TestClarificationQuestion:
    """Tests for ClarificationQuestion model."""

    def test_create_basic_question(self):
        """Test creating a basic text question."""
        q = ClarificationQuestion(
            question="What is your budget?",
            is_required=True
        )
        assert q.id is not None
        assert len(q.id) == 36  # UUID string length
        assert q.question_type == "text"
        assert q.choices is None
        assert q.order == 0
        assert q.is_required is True

    def test_create_choice_question(self):
        """Test creating a single-choice question."""
        q = ClarificationQuestion(
            question="Select priority",
            question_type="choice",
            choices=[
                ClarificationChoice(id="speed", label="Speed"),
                ClarificationChoice(id="accuracy", label="Accuracy"),
                ClarificationChoice(id="cost", label="Cost"),
            ],
        )
        assert q.choices is not None
        assert len(q.choices) == 3
        assert q.question_type == "choice"
        assert any(c.label == "Speed" for c in q.choices)

    def test_create_multi_choice_question(self):
        """Test creating a multi-choice question."""
        q = ClarificationQuestion(
            question="Select features",
            question_type="multi_choice",
            choices=[
                ClarificationChoice(id="a", label="Feature A"),
                ClarificationChoice(id="b", label="Feature B"),
                ClarificationChoice(id="c", label="Feature C"),
            ],
            is_required=False,
        )
        assert q.question_type == "multi_choice"
        assert q.choices is not None
        assert len(q.choices) == 3
        assert q.is_required is False

    def test_question_immutability(self):
        """Test that questions are immutable."""
        q = ClarificationQuestion(question="Test")
        with pytest.raises(ValidationError):
            q.question = "Modified"  # type: ignore

    def test_invalid_choice_question_no_choices(self):
        """Test that choice questions require choices."""
        with pytest.raises(ValidationError, match="Choices must be provided"):
            _ = ClarificationQuestion(
                question="Select one",
                question_type="choice"
            )

    def test_invalid_text_question_with_choices(self):
        """Test that text questions cannot have choices."""
        with pytest.raises(ValidationError, match="Choices should not be provided"):
            _ = ClarificationQuestion(
                question="Enter text",
                question_type="text",
                choices=[ClarificationChoice(id="a", label="A")]
            )

    def test_question_with_context(self):
        """Test question with additional context."""
        q = ClarificationQuestion(
            question="What version?",
            context="We support Python 3.8+",
            order=5
        )
        assert q.context == "We support Python 3.8+"
        assert q.order == 5


class TestClarificationAnswer:
    """Tests for ClarificationAnswer model."""

    def test_valid_answer(self):
        """Test creating a valid answer."""
        a = ClarificationAnswer(question_id="test-id", text="My answer")
        assert not a.skipped
        assert a.text == "My answer"
        assert a.answered_at.tzinfo == UTC

    def test_skipped_answer(self):
        """Test creating a skipped answer."""
        a = ClarificationAnswer(
            question_id="test-id",
            skipped=True
        )
        assert a.skipped
        assert a.text is None and a.selection is None and a.selections is None

    def test_invalid_answer_both_answer_and_skipped(self):
        """Test that answer cannot have both value and skipped=True."""
        with pytest.raises(ValidationError, match="Cannot have both content and skipped=True"):
            _ = ClarificationAnswer(question_id="test-id", text="Answer", skipped=True)

    def test_invalid_answer_neither_answer_nor_skipped(self):
        """Test that answer must have either value or skipped=True."""
        with pytest.raises(ValidationError, match="Content must be provided"):
            _ = ClarificationAnswer(question_id="test-id")

    def test_invalid_empty_answer(self):
        """Test that empty string answers are invalid."""
        with pytest.raises(ValidationError, match="Content must be provided"):
            _ = ClarificationAnswer(question_id="test-id", text="   ")

    def test_custom_timestamp(self):
        """Test answer with custom timestamp."""
        custom_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        a = ClarificationAnswer(question_id="test-id", text="Answer", answered_at=custom_time)
        assert a.answered_at == custom_time


class TestClarificationRequest:
    """Tests for ClarificationRequest model."""

    def test_create_request(self):
        """Test creating a clarification request."""
        questions = [
            ClarificationQuestion(question="Q1", order=1),
            ClarificationQuestion(question="Q2", order=0)
        ]
        req = ClarificationRequest(questions=questions)

        assert len(req.questions) == 2
        assert req.id is not None
        assert req.created_at.tzinfo == UTC

        # Test sorting
        sorted_q = req.get_sorted_questions()
        assert sorted_q[0].question == "Q2"  # order=0
        assert sorted_q[1].question == "Q1"  # order=1

    def test_empty_request_invalid(self):
        """Test that request must have at least one question."""
        from pydantic import ValidationError as PydanticValidationError
        with pytest.raises(PydanticValidationError, match="List should have at least 1 item"):
            _ = ClarificationRequest(questions=[])

    def test_o1_lookup_performance(self):
        """Test O(1) lookup for questions by ID."""
        # Create request with maximum allowed questions (10)
        questions = [
            ClarificationQuestion(question=f"Q{i}", order=i)
            for i in range(10)
        ]
        req = ClarificationRequest(questions=questions)

        # Test O(1) lookup
        q5 = req.get_question_by_id(questions[5].id)
        assert q5 == questions[5]

        # Test non-existent ID
        assert req.get_question_by_id("non-existent") is None

    def test_get_required_questions(self):
        """Test filtering for required questions only."""
        questions = [
            ClarificationQuestion(question="Q1", is_required=True),
            ClarificationQuestion(question="Q2", is_required=False),
            ClarificationQuestion(question="Q3", is_required=True)
        ]
        req = ClarificationRequest(questions=questions)

        required = req.get_required_questions()
        assert len(required) == 2
        assert all(q.is_required for q in required)
        assert all(q.question in ["Q1", "Q3"] for q in required)

    def test_request_with_context(self):
        """Test request with additional context."""
        req = ClarificationRequest(
            questions=[ClarificationQuestion(question="Q1")],
            context="Research about Python programming"
        )
        assert req.context == "Research about Python programming"

    def test_choice_id_auto_generation_and_dedup(self):
        """Choices without IDs or with duplicate IDs get unique UUIDs assigned."""
        q = ClarificationQuestion(
            question="Pick one",
            question_type="choice",
            choices=[
                ClarificationChoice(id="dup", label="A"),
                ClarificationChoice(id="dup", label="B"),  # duplicate id
                ClarificationChoice(id="", label="C"),      # missing id
            ],
        )
        assert q.choices is not None
        ids = [c.id for c in q.choices]
        assert all(isinstance(i, str) and len(i) > 0 for i in ids)
        assert len(set(ids)) == len(ids)  # all unique


class TestClarificationResponse:
    """Tests for ClarificationResponse model."""

    def test_create_response(self):
        """Test creating a clarification response."""
        answers = [
            ClarificationAnswer(question_id="q1", text="A1"),
            ClarificationAnswer(question_id="q2", text="A2"),
        ]
        resp = ClarificationResponse(
            request_id="req-123",
            answers=answers
        )

        assert resp.request_id == "req-123"
        assert len(resp.answers) == 2
        assert resp.completed_at.tzinfo == UTC

    def test_empty_response_invalid(self):
        """Test that response must have at least one answer."""
        with pytest.raises(ValidationError, match="Response must have at least one answer"):
            _ = ClarificationResponse(request_id="req-123", answers=[])

    def test_o1_answer_lookup(self):
        """Test O(1) lookup for answers by question ID."""
        answers = [ClarificationAnswer(question_id=f"q{i}", text=f"A{i}") for i in range(50)]
        resp = ClarificationResponse(request_id="req", answers=answers)

        # Test O(1) lookup
        a25 = resp.get_answer_for_question("q25")
        assert a25 is not None
        assert a25.text == "A25"

        # Test non-existent question ID
        assert resp.get_answer_for_question("non-existent") is None

    def test_validate_against_request_success(self):
        """Test validating a complete response against request."""
        questions = [
            ClarificationQuestion(question="Q1", is_required=True),
            ClarificationQuestion(question="Q2", is_required=False)
        ]
        req = ClarificationRequest(questions=questions)

        answers = [
            ClarificationAnswer(question_id=questions[0].id, text="A1"),
            ClarificationAnswer(question_id=questions[1].id, skipped=True),
        ]
        resp = ClarificationResponse(request_id=req.id, answers=answers)

        errors = resp.validate_against_request(req)
        assert len(errors) == 0

    def test_validate_missing_required_question(self):
        """Test validation detects missing required questions."""
        questions = [
            ClarificationQuestion(question="Q1", is_required=True),
            ClarificationQuestion(question="Q2", is_required=True)
        ]
        req = ClarificationRequest(questions=questions)

        # Only answer first question
        answers = [ClarificationAnswer(question_id=questions[0].id, text="A1")]
        resp = ClarificationResponse(request_id=req.id, answers=answers)

        errors = resp.validate_against_request(req)
        assert len(errors) == 1
        assert "Required question" in errors[0]
        assert questions[1].id in errors[0]

    def test_validate_skipped_required_question(self):
        """Test validation detects skipped required questions."""
        q = ClarificationQuestion(question="Q1", is_required=True)
        req = ClarificationRequest(questions=[q])

        resp = ClarificationResponse(
            request_id=req.id,
            answers=[ClarificationAnswer(question_id=q.id, skipped=True)]
        )

        errors = resp.validate_against_request(req)
        assert len(errors) == 1
        assert "Required question" in errors[0]

    def test_validate_unknown_question_id(self):
        """Test validation detects answers for unknown questions."""
        req = ClarificationRequest(
            questions=[ClarificationQuestion(question="Q1", is_required=False)]
        )

        resp = ClarificationResponse(
            request_id=req.id,
            answers=[ClarificationAnswer(question_id="unknown-id", text="A")],
        )

        errors = resp.validate_against_request(req)
        assert len(errors) == 1
        assert "Unknown question ID" in errors[0]

    def test_validate_invalid_choice(self):
        """Test validation detects invalid choice answers."""
        q = ClarificationQuestion(
            question="Select one",
            question_type="choice",
            choices=[
                ClarificationChoice(id="a", label="Option A"),
                ClarificationChoice(id="b", label="Option B"),
                ClarificationChoice(id="c", label="Option C"),
            ],
        )
        req = ClarificationRequest(questions=[q])

        resp = ClarificationResponse(
            request_id=req.id,
            answers=[ClarificationAnswer(question_id=q.id, selection=ChoiceSelection(id="d"))],
        )

        errors = resp.validate_against_request(req)
        assert len(errors) == 1
        assert "Invalid choice id 'd'" in errors[0]

    def test_validate_invalid_multi_choice(self):
        """Test validation detects invalid multi-choice answers."""
        q = ClarificationQuestion(
            question="Select multiple",
            question_type="multi_choice",
            choices=[
                ClarificationChoice(id="A", label="A"),
                ClarificationChoice(id="B", label="B"),
                ClarificationChoice(id="C", label="C"),
            ],
        )
        req = ClarificationRequest(questions=[q])

        resp = ClarificationResponse(
            request_id=req.id,
            answers=[
                ClarificationAnswer(
                    question_id=q.id,
                    selections=[
                        ChoiceSelection(id="A"),
                        ChoiceSelection(id="D"),
                        ChoiceSelection(id="E")
                    ],
                )
            ],
        )

        errors = resp.validate_against_request(req)
        assert len(errors) == 1
        assert "Invalid choice ids ['D', 'E']" in errors[0]

    def test_validate_valid_choices(self):
        """Test validation accepts valid choice answers."""
        q1 = ClarificationQuestion(
            question="Single choice",
            question_type="choice",
            choices=[
                ClarificationChoice(id="yes", label="Yes"),
                ClarificationChoice(id="no", label="No"),
            ],
        )
        q2 = ClarificationQuestion(
            question="Multi choice",
            question_type="multi_choice",
            choices=[
                ClarificationChoice(id="r", label="Red"),
                ClarificationChoice(id="g", label="Green"),
                ClarificationChoice(id="b", label="Blue"),
            ],
        )
        req = ClarificationRequest(questions=[q1, q2])

        resp = ClarificationResponse(
            request_id=req.id,
            answers=[
                ClarificationAnswer(question_id=q1.id, selection=ChoiceSelection(id="yes")),
                ClarificationAnswer(
                    question_id=q2.id,
                    selections=[ChoiceSelection(id="r"), ChoiceSelection(id="b")],
                ),
            ],
        )

        errors = resp.validate_against_request(req)
        assert len(errors) == 0
