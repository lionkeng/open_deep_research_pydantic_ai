"""Data models for the clarification system with UUID-based identification.

Breaking change: Structured choices and answers (no legacy string parsing for non-text).
"""

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
MIN_QUESTION_LENGTH: Final[int] = 5
MAX_CONCURRENT_QUESTIONS: Final[int] = 10


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
        # Store immutable copy to prevent external modification
        self.valid_choices = valid_choices.copy()
        super().__init__(
            f"Invalid choice '{invalid_choice}'. Valid choices: {', '.join(valid_choices)}"
        )


class MissingRequiredAnswerError(ValidationError):
    """Raised when a required question is not answered."""

    def __init__(self, question_id: str):
        self.question_id = question_id
        super().__init__(f"Required question '{question_id}' not answered")


class ClarificationChoice(BaseModel):
    """Single selectable option for choice questions."""

    model_config = {"frozen": True}

    id: str = Field(description="Stable ID for this choice (unique within question)")
    label: str = Field(description="Human-readable label presented to the user")
    requires_details: bool = Field(
        default=False, description="If True, user must enter details when selecting"
    )
    is_other: bool = Field(
        default=False, description="If True, represents an 'Other' free-text option"
    )
    details_prompt: str | None = Field(
        default=None, description="Optional custom prompt when details are required"
    )


class ClarificationQuestion(BaseModel):
    """Individual clarification question with metadata."""

    model_config = {"frozen": True}

    id: str = Field(default_factory=lambda: str(uuid4()))
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH)
    is_required: bool = True
    question_type: Literal["text", "choice", "multi_choice"] = "text"
    choices: list[ClarificationChoice] | None = Field(default=None, max_length=MAX_CHOICES)
    context: str | None = Field(default=None, max_length=MAX_CONTEXT_LENGTH)
    order: int = 0

    @field_validator("question", "context")
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

    @field_validator("choices")
    @classmethod
    def validate_choices(
        cls, v: list[ClarificationChoice] | None
    ) -> list[ClarificationChoice] | None:
        """Validate choice options (structured only)."""
        if v is None:
            return None
        # Validate each choice
        validated: list[ClarificationChoice] = []
        seen_ids: set[str] = set()
        for choice in v:
            # Auto-generate ID if missing/blank
            cid = choice.id.strip() if isinstance(choice.id, str) else ""
            if not cid:
                choice = ClarificationChoice(
                    id=str(uuid4()),
                    label=choice.label,
                    requires_details=choice.requires_details,
                    is_other=choice.is_other,
                    details_prompt=choice.details_prompt,
                )
                cid = choice.id
            if choice.id in seen_ids:
                # Regenerate clashing ID to ensure uniqueness
                choice = ClarificationChoice(
                    id=str(uuid4()),
                    label=choice.label,
                    requires_details=choice.requires_details,
                    is_other=choice.is_other,
                    details_prompt=choice.details_prompt,
                )
            if not choice.label or not choice.label.strip():
                raise ValueError("Choice label cannot be empty")
            if len(choice.label) > MAX_CHOICE_LENGTH:
                raise ValueError(f"Choice label too long (max {MAX_CHOICE_LENGTH} chars)")
            seen_ids.add(choice.id)
            validated.append(choice)
        return validated

    @model_validator(mode="after")
    def validate_choices_consistency(self) -> Self:
        """Validate that choices are provided for choice questions."""
        if self.question_type in ("choice", "multi_choice") and not self.choices:
            raise ValueError(f"Choices must be provided for {self.question_type} questions")
        if self.question_type == "text" and self.choices:
            raise ValueError("Choices should not be provided for text questions")
        return self


class ChoiceSelection(BaseModel):
    """Structured selection for a choice option."""

    id: str = Field(description="Choice id selected")
    details: str | None = Field(default=None, description="Optional details when required")


class ClarificationAnswer(BaseModel):
    """Answer to a clarification question (structured)."""

    question_id: str
    # Exactly one of the following content fields is used depending on question_type
    text: str | None = Field(default=None, max_length=MAX_ANSWER_LENGTH)
    selection: ChoiceSelection | None = None
    selections: list[ChoiceSelection] | None = None
    skipped: bool = False
    answered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        return v if v else None

    @model_validator(mode="after")
    def validate_content_exclusive(self) -> Self:
        """Ensure mutually exclusive content fields and consistency with skipped."""
        content_fields = [
            ("text", self.text is not None and self.text.strip() != ""),
            ("selection", self.selection is not None),
            ("selections", bool(self.selections)),
        ]
        provided = [name for name, present in content_fields if present]
        if self.skipped:
            if provided:
                raise ValueError("Cannot have both content and skipped=True")
            return self
        # Not skipped
        if not provided:
            raise ValueError("Content must be provided if not skipped")
        if len(provided) > 1:
            raise ValueError("Only one content field can be set per answer")
        return self


class ClarificationRequest(BaseModel):
    """Collection of clarification questions to ask the user."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    questions: list[ClarificationQuestion] = Field(
        ...,
        min_length=1,
        max_length=MAX_CONCURRENT_QUESTIONS,
        description="List of clarification questions (max 10)",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    context: str | None = Field(default=None, max_length=MAX_CONTEXT_LENGTH)

    # Private attribute for O(1) lookups
    _question_index: dict[str, ClarificationQuestion] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Build index after model initialization."""
        super().model_post_init(__context)
        # Always rebuild index to handle dynamic updates
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

    @model_validator(mode="after")
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
        # Always rebuild index to handle dynamic updates
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
        valid_question_ids = {q.id for q in request.questions}
        for answer in self.answers:
            if answer.question_id not in valid_question_ids:
                errors.append(f"Unknown question ID: {answer.question_id}")
                continue

            # Validate answers by type using structured fields
            question = request.get_question_by_id(answer.question_id)
            if not question:
                continue
            if answer.skipped:
                continue
            if question.question_type == "text":
                if not (answer.text and answer.text.strip()):
                    errors.append(f"Missing text for question '{question.id}'")
            elif question.question_type == "choice":
                if not answer.selection:
                    errors.append(f"Missing selection for question '{question.id}'")
                    continue
                choice = next(
                    (c for c in (question.choices or []) if c.id == answer.selection.id), None
                )
                if not choice:
                    errors.append(
                        f"Invalid choice id '{answer.selection.id}' for question '{question.id}'"
                    )
                    continue
                if (choice.requires_details or choice.is_other) and not (
                    answer.selection.details and answer.selection.details.strip()
                ):
                    errors.append(
                        f"Missing details for selection '{choice.label}' in "
                        f"question '{question.id}'"
                    )
            elif question.question_type == "multi_choice":
                if not answer.selections or not isinstance(answer.selections, list):
                    errors.append(f"Missing selections for question '{question.id}'")
                    continue
                valid_choice_ids = {c.id: c for c in (question.choices or [])}
                invalid: list[str] = []
                for sel in answer.selections:
                    ch = valid_choice_ids.get(sel.id)
                    if not ch:
                        invalid.append(sel.id)
                        continue
                    if (ch.requires_details or ch.is_other) and not (
                        sel.details and sel.details.strip()
                    ):
                        errors.append(
                            f"Missing details for selection '{ch.label}' in "
                            f"question '{question.id}'"
                        )
                if invalid:
                    errors.append(f"Invalid choice ids {invalid} for question '{question.id}'")

        return errors

    @model_validator(mode="after")
    def validate_answers(self) -> Self:
        """Validate that response has at least one answer."""
        if not self.answers:
            raise ValueError("Response must have at least one answer")
        return self
