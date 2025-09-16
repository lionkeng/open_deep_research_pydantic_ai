"""Core models for the deep research workflow."""

import uuid
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

# Import Phase 2 models
from .compression import CompressedContent
from .metadata import ResearchMetadata
from .report_generator import ReportSection as ResearchSection
from .report_generator import ResearchReport
from .research_executor import ResearchFinding, ResearchResults

if TYPE_CHECKING:
    pass  # For future type checking imports


class ResearchStage(str, Enum):
    """Stages of the research workflow."""

    PENDING = "pending"  # Initial state before research starts
    CLARIFICATION = "clarification"
    QUERY_TRANSFORMATION = "query_transformation"  # Transform query based on clarification
    RESEARCH_EXECUTION = "research_execution"
    COMPRESSION = "compression"
    REPORT_GENERATION = "report_generation"
    COMPLETED = "completed"
    FAILED = "failed"  # Terminal state for failed research


class ResearchPriority(str, Enum):
    """Priority levels for research tasks."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ClarificationResult(BaseModel):
    """Structured output for the Clarification Agent following Pydantic-AI patterns."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    needs_clarification: bool = Field(description="Whether the query requires clarification")
    question: str = Field("", description="Clarifying question if needed")
    verification: str = Field("", description="Verification when no clarification needed")
    confidence_score: Annotated[
        float, Field(ge=0.0, le=1.0, description="Confidence in assessment")
    ]
    missing_dimensions: list[str] = Field(
        default_factory=list, description="Specific dimensions missing from query"
    )
    breadth_score: Annotated[
        float,
        Field(ge=0.0, le=1.0, description="Score indicating query breadth (higher = more broad)"),
    ]
    assessment_reasoning: str = Field(
        default="", description="Explanation of why clarification is or isn't needed"
    )
    suggested_clarifications: list[str] = Field(
        default_factory=list, description="Specific areas that would benefit from clarification"
    )

    @field_validator("question", "verification")
    @classmethod
    def validate_conditional_fields(cls, v: str, info: ValidationInfo) -> str:
        """Validate conditional field population based on needs_clarification."""
        if info.data and "needs_clarification" in info.data:
            needs_clarification = info.data["needs_clarification"]
            field_name = info.field_name

            if needs_clarification:
                if field_name == "question" and not v.strip():
                    raise ValueError("Question required when clarification needed")
                if field_name == "verification" and v.strip():
                    raise ValueError("Verification must be empty when clarification needed")
            else:
                if field_name == "verification" and not v.strip():
                    raise ValueError("Verification required when no clarification needed")
                if field_name == "question" and v.strip():
                    raise ValueError("Question must be empty when no clarification needed")
        return v

    @model_validator(mode="after")
    def validate_consistency(self) -> "ClarificationResult":
        """Ensure internal consistency of clarification result."""
        if self.needs_clarification and self.confidence_score > 0.8:
            raise ValueError("High confidence inconsistent with need for clarification")
        if not self.needs_clarification and self.confidence_score < 0.3:
            raise ValueError("Low confidence inconsistent with no clarification needed")
        return self


class ResearchState(BaseModel):
    """Current state of the research workflow."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="allow",  # Allow extra fields for flexibility
    )

    request_id: Annotated[str, Field(min_length=1, description="Unique request identifier")]
    user_id: Annotated[
        str, Field(default="default", min_length=1, description="User identifier for isolation")
    ]
    session_id: Annotated[
        str | None, Field(default=None, description="Optional session identifier for user")
    ]
    user_query: Annotated[str, Field(min_length=1, description="Original user query")]
    current_stage: ResearchStage = Field(default=ResearchStage.PENDING)
    clarified_query: str | None = Field(
        default=None, description="Clarified query after validation"
    )
    research_plan: dict[str, Any] | None = Field(
        default=None, description="Research plan from query transformation"
    )
    findings: list[ResearchFinding] = Field(default_factory=list, description="All findings")
    research_results: ResearchResults | None = Field(
        default=None, description="Structured research results from the executor"
    )
    compressed_findings: CompressedContent | None = Field(
        default=None, description="Compressed and synthesized findings"
    )
    final_report: ResearchReport | None = Field(default=None, description="Final report")
    error_message: str | None = Field(default=None, description="Error message if any")
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = Field(default=None)
    metadata: ResearchMetadata = Field(
        default_factory=ResearchMetadata, description="Typed metadata for research workflow"
    )

    @model_validator(mode="before")
    @classmethod
    def migrate_metadata(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Create new ResearchMetadata if needed."""
        if "metadata" in values and isinstance(values["metadata"], dict):
            # For now, just create a new ResearchMetadata
            # Migration from old structure will be handled in cleanup phase
            values["metadata"] = ResearchMetadata()
        return values

    def advance_stage(self) -> None:
        """Advance to the next research stage."""
        stage_order = list(ResearchStage)
        current_index = stage_order.index(self.current_stage)
        if current_index < len(stage_order) - 1:
            self.current_stage = stage_order[current_index + 1]

    def start_research(self) -> None:
        """Move from PENDING to CLARIFICATION stage."""
        if self.current_stage == ResearchStage.PENDING:
            self.current_stage = ResearchStage.CLARIFICATION

    def is_completed(self) -> bool:
        """Check if research is completed (either successfully or with failure)."""
        return self.current_stage in (ResearchStage.COMPLETED, ResearchStage.FAILED)

    def add_finding(self, finding: ResearchFinding) -> None:
        """Add a research finding."""
        self.findings.append(finding)

    def set_error(self, message: str) -> None:
        """Set error message and mark as failed."""
        self.error_message = message
        self.current_stage = ResearchStage.FAILED
        self.completed_at = datetime.now()

    @classmethod
    def generate_request_id(cls, user_id: str = "default", session_id: str | None = None) -> str:
        """Generate a scoped request ID that includes user context.

        Format: {user_id}:{session_id}:{uuid} or {user_id}:{uuid}

        Args:
            user_id: User identifier (defaults to "default" for CLI usage)
            session_id: Optional session identifier

        Returns:
            Scoped request identifier
        """
        request_uuid = str(uuid.uuid4())
        if session_id:
            return f"{user_id}:{session_id}:{request_uuid}"
        return f"{user_id}:{request_uuid}"


# ResearchSection needs model rebuild since it has self-reference
_ = ResearchSection.model_rebuild()

# Export all public models
__all__ = [
    "ResearchStage",
    "ResearchPriority",
    "ClarificationResult",
    "ResearchState",
    "ResearchMetadata",
    "ResearchFinding",
    "ResearchResults",
    "ResearchSection",
    "ResearchReport",
    "CompressedContent",
]
