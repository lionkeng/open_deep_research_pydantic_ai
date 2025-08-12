"""Pydantic models for the deep research workflow."""

from datetime import datetime
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field


class ResearchStage(str, Enum):
    """Stages of the research workflow."""

    PENDING = "pending"  # Initial state before research starts
    CLARIFICATION = "clarification"
    BRIEF_GENERATION = "brief_generation"
    RESEARCH_EXECUTION = "research_execution"
    COMPRESSION = "compression"
    REPORT_GENERATION = "report_generation"
    COMPLETED = "completed"


class ResearchPriority(str, Enum):
    """Priority levels for research tasks."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ResearchBrief(BaseModel):
    """Structured research plan generated from user request."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    topic: Annotated[str, Field(min_length=1, max_length=500, description="Main research topic")]
    objectives: Annotated[
        list[str], Field(min_length=1, max_length=10, description="Research objectives")
    ]
    key_questions: Annotated[
        list[str], Field(min_length=1, max_length=20, description="Key questions to answer")
    ]
    scope: Annotated[
        str, Field(min_length=1, max_length=1000, description="Scope and boundaries of research")
    ]
    priority_areas: list[str] = Field(default_factory=list, description="Priority research areas")
    constraints: list[str] = Field(default_factory=list, description="Research constraints")
    expected_deliverables: list[str] = Field(
        default_factory=list, description="Expected deliverables"
    )
    created_at: datetime = Field(default_factory=datetime.now)


class ResearchFinding(BaseModel):
    """Individual research finding with source attribution."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    content: Annotated[str, Field(min_length=1, description="Finding content")]
    source: Annotated[str, Field(min_length=1, description="Source URL or reference")]
    relevance_score: Annotated[float, Field(ge=0.0, le=1.0, description="Relevance score")] = 0.0
    confidence: Annotated[float, Field(ge=0.0, le=1.0, description="Confidence level")] = 0.0
    summary: str | None = Field(default=None, description="Brief summary")
    extracted_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ResearchSection(BaseModel):
    """Section of the final research report."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    title: Annotated[str, Field(min_length=1, max_length=200, description="Section title")]
    content: Annotated[str, Field(min_length=1, description="Section content")]
    findings: list[ResearchFinding] = Field(default_factory=list, description="Supporting findings")
    subsections: list["ResearchSection"] = Field(
        default_factory=list, description="Nested subsections"
    )
    order: int = Field(default=0, description="Display order")


class ResearchReport(BaseModel):
    """Final comprehensive research report."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    title: Annotated[str, Field(min_length=1, max_length=300, description="Report title")]
    executive_summary: Annotated[str, Field(min_length=1, description="Executive summary")]
    introduction: Annotated[str, Field(min_length=1, description="Introduction")]
    methodology: Annotated[str, Field(min_length=1, description="Research methodology")]
    sections: list[ResearchSection] = Field(description="Report sections")
    conclusion: str = Field(description="Conclusion")
    recommendations: list[str] = Field(default_factory=list, description="Recommendations")
    citations: list[str] = Field(default_factory=list, description="All citations")
    appendices: dict[str, Any] = Field(default_factory=dict, description="Additional materials")
    generated_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict, description="Report metadata")


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
    research_brief: ResearchBrief | None = Field(default=None, description="Research plan")
    findings: list[ResearchFinding] = Field(default_factory=list, description="All findings")
    compressed_findings: str | None = Field(default=None, description="Synthesized findings")
    final_report: ResearchReport | None = Field(default=None, description="Final report")
    error_message: str | None = Field(default=None, description="Error message if any")
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional state data")

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
        """Check if research is completed."""
        return self.current_stage == ResearchStage.COMPLETED

    def add_finding(self, finding: ResearchFinding) -> None:
        """Add a research finding."""
        self.findings.append(finding)

    def set_error(self, message: str) -> None:
        """Set error message and mark as completed."""
        self.error_message = message
        self.current_stage = ResearchStage.COMPLETED
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
        import uuid

        request_uuid = str(uuid.uuid4())
        if session_id:
            return f"{user_id}:{session_id}:{request_uuid}"
        return f"{user_id}:{request_uuid}"


ResearchSection.model_rebuild()
