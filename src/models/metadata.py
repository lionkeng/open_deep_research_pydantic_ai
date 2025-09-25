"""Clean composed metadata structure for research workflow.

This module provides a composed architecture for managing agent-specific
metadata with clear ownership boundaries and type safety.
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .api_models import ConversationMessage
from .clarification import (
    ClarificationAnswer,
    ClarificationQuestion,
    ClarificationRequest,
    ClarificationResponse,
)
from .research_plan_models import TransformedQuery


class ClarificationMetadata(BaseModel):
    """Metadata specific to the clarification agent."""

    model_config = ConfigDict(validate_assignment=True)

    # Core clarification fields
    assessment: dict[str, Any] | None = Field(
        default=None, description="Clarification agent's assessment results"
    )
    request: ClarificationRequest | None = Field(
        default=None, description="Current clarification request with questions"
    )
    response: ClarificationResponse | None = Field(
        default=None, description="User's responses to clarification questions"
    )
    question: str | None = Field(
        default=None,
        description="Current clarification question awaiting response (formatted for display)",
    )
    awaiting_clarification: bool = Field(
        default=False, description="Whether system is waiting for clarification response"
    )

    @model_validator(mode="after")
    def validate_clarification_state(self) -> Self:
        """Ensure clarification state consistency.

        Rules:
        - If awaiting_clarification is True, there must be a request
        - If response exists, there must be a corresponding request
        """
        if self.awaiting_clarification and not self.request:
            raise ValueError("Cannot be awaiting clarification without a request")

        if self.response and not self.request:
            raise ValueError("Cannot have a response without a corresponding request")

        return self

    def get_pending_questions(self) -> list[ClarificationQuestion]:
        """Get list of unanswered required questions."""
        if not self.request:
            return []

        pending = []
        for question in self.request.get_required_questions():
            if self.response:
                answer = self.response.get_answer_for_question(question.id)
                if answer and not answer.skipped:
                    continue
            pending.append(question)

        return pending

    def add_clarification_response(self, question_id: str, answer: str) -> None:
        """Add or update an answer for a specific question.

        Raises:
            ValueError: If no clarification request exists.
        """
        if not self.request:
            raise ValueError("Cannot add clarification response without an existing request")

        new_answer = ClarificationAnswer(question_id=question_id, answer=answer, skipped=False)

        if not self.response:
            self.response = ClarificationResponse(
                request_id=self.request.id,
                answers=[new_answer],
            )
        else:
            # Remove existing answer for this question if any
            self.response.answers = [
                a for a in self.response.answers if a.question_id != question_id
            ]
            self.response.answers.append(new_answer)
            self.response.model_post_init(None)

    def is_clarification_complete(self) -> bool:
        """Check if all required questions have been answered."""
        if not self.request:
            return True

        if not self.response:
            return False

        errors = self.response.validate_against_request(self.request)
        return len(errors) == 0


class QueryMetadata(BaseModel):
    """Metadata specific to the query transformation agent."""

    model_config = ConfigDict(validate_assignment=True)

    transformed_query: TransformedQuery | None = Field(
        default=None, description="Query transformation results"
    )


class BriefMetadata(BaseModel):
    """Metadata specific to the research brief generator agent."""

    model_config = ConfigDict(validate_assignment=True)

    text: str | None = Field(default=None, description="Generated research brief text")
    full: dict[str, Any] | None = Field(default=None, description="Complete research brief object")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence score for research brief"
    )


class ExecutionMetadata(BaseModel):
    """Metadata specific to the research executor agent."""

    model_config = ConfigDict(validate_assignment=True)

    # Add execution-specific fields as needed
    status: str | None = Field(default=None, description="Execution status")
    results: list[dict[str, Any]] = Field(default_factory=list, description="Execution results")


class ReportSectionPlan(BaseModel):
    """Deterministic outline entry for the report generator."""

    title: str = Field(description="Proposed section heading derived from research content")
    bullets: list[str] = Field(
        default_factory=list,
        description="Key talking points to weave into the section body",
    )
    salient_evidence_ids: list[str] = Field(
        default_factory=list,
        description="Source identifiers most relevant to this section",
    )


class ReportMetadata(BaseModel):
    """Metadata specific to the report generator agent."""

    model_config = ConfigDict(validate_assignment=True)

    # Add report-specific fields as needed
    sections: list[dict[str, Any]] = Field(default_factory=list, description="Report sections")
    final: str | None = Field(default=None, description="Final report text")
    section_outline: list[ReportSectionPlan] = Field(
        default_factory=list,
        description="Deterministic outline passed to the report generator",
    )


class ResearchMetadata(BaseModel):
    """Composed research metadata with agent-specific namespaces.

    This is the clean, final structure without any backward compatibility.
    Each agent has its own namespace for metadata isolation.
    """

    model_config = ConfigDict(validate_assignment=True, extra="allow")

    # Agent-specific metadata namespaces
    clarification: ClarificationMetadata = Field(default_factory=ClarificationMetadata)
    query: QueryMetadata = Field(default_factory=QueryMetadata)
    brief: BriefMetadata = Field(default_factory=BriefMetadata)
    execution: ExecutionMetadata = Field(default_factory=ExecutionMetadata)
    report: ReportMetadata = Field(default_factory=ReportMetadata)

    # Shared metadata fields (used by multiple agents)
    conversation_messages: list[ConversationMessage] = Field(
        default_factory=list, description="Conversation history between user and system"
    )
    sources_consulted: int = Field(default=0, ge=0, description="Number of sources consulted")
    total_tokens_used: int = Field(default=0, ge=0, description="Total LLM tokens consumed")
    processing_time: float = Field(
        default=0.0, ge=0.0, description="Total processing time in seconds"
    )
    confidence_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall research confidence score"
    )
    tags: list[str] = Field(default_factory=list, description="Research topic tags")

    # Delegated properties for convenient access
    @property
    def pending_questions(self) -> list[ClarificationQuestion]:
        """Get list of unanswered required questions."""
        return self.clarification.get_pending_questions()

    # Keep method versions for backward compatibility
    def get_pending_questions(self) -> list[ClarificationQuestion]:
        """Get list of unanswered required questions.

        Deprecated: Use the pending_questions property instead.
        """
        return self.pending_questions

    def add_clarification_response(self, question_id: str, answer: str) -> None:
        """Add or update an answer for a specific question."""
        self.clarification.add_clarification_response(question_id, answer)

    def is_clarification_complete(self) -> bool:
        """Check if all required questions have been answered."""
        return self.clarification.is_clarification_complete()
