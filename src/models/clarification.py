"""Data models for the clarification system with UUID-based identification."""

from datetime import UTC, datetime
from typing import Any, Final, Literal, Self
from uuid import uuid4

from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

# Constants for validation
MAX_QUESTION_LENGTH: Final[int] = 500
MAX_CONTEXT_LENGTH: Final[int] = 1000
MAX_ANSWER_LENGTH: Final[int] = 2000
MAX_CHOICES: Final[int] = 20
MAX_CHOICE_LENGTH: Final[int] = 200


# Custom Exceptions
class ClarificationError(Exception):
    """Base exception for clarification-related errors."""
    pass


class ValidationError(ClarificationError):
    """Raised when clarification validation fails."""
    pass


class InvalidChoiceError(ValidationError):
    """Raised when an invalid choice is provided."""
    def __init__(self, invalid_choice: str, valid_choices: list[str]):
        self.invalid_choice = invalid_choice
        self.valid_choices = valid_choices
        super().__init__(
            f"Invalid choice '{invalid_choice}'. Valid choices: {', '.join(valid_choices)}"
        )


class MissingRequiredAnswerError(ValidationError):
    """Raised when a required question is not answered."""
    def __init__(self, question_id: str):
        self.question_id = question_id
        super().__init__(f"Required question '{question_id}' not answered")


class ClarificationQuestion(BaseModel):
    """Individual clarification question with metadata."""

    model_config = {"frozen": True}

    id: str = Field(default_factory=lambda: str(uuid4()))
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH)
    is_required: bool = True
    question_type: Literal["text", "choice", "multi_choice"] = "text"
    choices: list[str] | None = Field(default=None, max_length=MAX_CHOICES)
    context: str | None = Field(default=None, max_length=MAX_CONTEXT_LENGTH)
    order: int = 0

    @field_validator('question', 'context')
    @classmethod
    def validate_string_fields(cls, v: str | None) -> str | None:
        """Validate and sanitize string fields."""
        if v is None:
            return None
        # Strip whitespace and ensure non-empty
        v = v.strip()
        if not v:
            raise ValueError("Field cannot be empty or whitespace-only")
        return v
    
    @field_validator('choices')
    @classmethod
    def validate_choices(cls, v: list[str] | None) -> list[str] | None:
        """Validate choice options."""
        if v is None:
            return None
        # Validate each choice
        validated_choices = []
        for choice in v:
            choice = choice.strip()
            if not choice:
                raise ValueError("Choice cannot be empty")
            if len(choice) > MAX_CHOICE_LENGTH:
                raise ValueError(f"Choice too long (max {MAX_CHOICE_LENGTH} chars)")
            validated_choices.append(choice)
        return validated_choices
    
    @model_validator(mode='after')
    def validate_choices_consistency(self) -> Self:
        """Validate that choices are provided for choice questions."""
        if self.question_type in ("choice", "multi_choice") and not self.choices:
            raise ValueError(f"Choices must be provided for {self.question_type} questions")
        if self.question_type == "text" and self.choices:
            raise ValueError("Choices should not be provided for text questions")
        return self


class ClarificationAnswer(BaseModel):
    """Answer to a clarification question."""

    question_id: str
    answer: str | None = Field(default=None, max_length=MAX_ANSWER_LENGTH)
    skipped: bool = False
    answered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator('answer')
    @classmethod
    def validate_answer(cls, v: str | None, info) -> str | None:
        """Validate and sanitize answer."""
        if v is None:
            return None
        # Strip whitespace
        v = v.strip()
        if not v:
            return None  # Will be caught by consistency check
        return v
    
    @model_validator(mode='after')
    def validate_answer_consistency(self) -> Self:
        """Validate that answer and skipped states are consistent."""
        if not self.skipped and (self.answer is None or not self.answer.strip()):
            raise ValueError("Answer must be provided if not skipped")
        if self.skipped and self.answer is not None:
            raise ValueError("Cannot have both answer and skipped=True")
        return self


class ClarificationRequest(BaseModel):
    """Collection of clarification questions to ask the user."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    questions: list[ClarificationQuestion] = Field(..., min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    context: str | None = Field(default=None, max_length=MAX_CONTEXT_LENGTH)

    # Private attribute for O(1) lookups
    _question_index: dict[str, ClarificationQuestion] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Build index after model initialization."""
        super().model_post_init(__context)
        self._question_index = {q.id: q for q in self.questions}

    def get_question_by_id(self, question_id: str) -> ClarificationQuestion | None:
        """Retrieve a question by its ID in O(1) time."""
        return self._question_index.get(question_id)

    def get_required_questions(self) -> list[ClarificationQuestion]:
        """Get only required questions."""
        return [q for q in self.questions if q.is_required]

    def get_sorted_questions(self) -> list[ClarificationQuestion]:
        """Get questions sorted by order."""
        return sorted(self.questions, key=lambda q: q.order)

    @model_validator(mode='after')
    def validate_questions(self) -> Self:
        """Validate that request has at least one question."""
        if not self.questions:
            raise ValueError("Request must have at least one question")
        return self


class ClarificationResponse(BaseModel):
    """Complete response containing all answers."""

    request_id: str
    answers: list[ClarificationAnswer]
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Private attribute for O(1) lookups
    _answer_index: dict[str, ClarificationAnswer] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Build index after model initialization."""
        super().model_post_init(__context)
        self._answer_index = {a.question_id: a for a in self.answers}

    def get_answer_for_question(self, question_id: str) -> ClarificationAnswer | None:
        """Get answer for a specific question ID in O(1) time."""
        return self._answer_index.get(question_id)

    def validate_against_request(self, request: ClarificationRequest) -> list[str]:
        """Validate this response against the original request."""
        errors: list[str] = []

        # Check all required questions are answered
        for question in request.get_required_questions():
            answer = self.get_answer_for_question(question.id)
            if not answer or (answer.skipped and question.is_required):
                errors.append(f"Required question '{question.id}' not answered")

        # Check no unknown question IDs and validate choices
        valid_ids = {q.id for q in request.questions}
        for answer in self.answers:
            if answer.question_id not in valid_ids:
                errors.append(f"Unknown question ID: {answer.question_id}")
                continue
            
            # Validate choice answers
            question = request.get_question_by_id(answer.question_id)
            if question and not answer.skipped and answer.answer:
                if question.question_type == "choice" and question.choices:
                    if answer.answer not in question.choices:
                        errors.append(
                            f"Invalid choice '{answer.answer}' for question '{question.id}'. "
                            f"Valid choices: {', '.join(question.choices)}"
                        )
                elif question.question_type == "multi_choice" and question.choices:
                    # For multi-choice, answer might be comma-separated values
                    selected_choices = [c.strip() for c in answer.answer.split(',')]
                    invalid_choices = [c for c in selected_choices if c not in question.choices]
                    if invalid_choices:
                        errors.append(
                            f"Invalid choices {invalid_choices} for question '{question.id}'. "
                            f"Valid choices: {', '.join(question.choices)}"
                        )

        return errors

    @model_validator(mode='after')
    def validate_answers(self) -> Self:
        """Validate that response has at least one answer."""
        if not self.answers:
            raise ValueError("Response must have at least one answer")
        return self


