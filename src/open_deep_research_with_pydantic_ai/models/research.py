"""Pydantic models for the deep research workflow."""

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

if TYPE_CHECKING:
    pass  # For future type checking imports


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


# Pydantic-AI Agent Output Models
# These models define the structured outputs for each phase following Pydantic-AI best practices


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


class TransformedQueryResult(BaseModel):
    """Enhanced structured output for Query Transformation Agent."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    original_query: str = Field(description="Original user query")
    transformed_query: Annotated[
        str, Field(min_length=10, description="Transformed research question")
    ]
    transformation_rationale: Annotated[
        str, Field(min_length=20, description="Detailed explanation of transformation")
    ]
    specificity_score: Annotated[
        float, Field(ge=0.0, le=1.0, description="Score indicating query specificity improvement")
    ]
    supporting_questions: list[str] = Field(
        default_factory=list, description="Additional supporting research questions"
    )
    clarification_responses: dict[str, str] = Field(
        default_factory=dict, description="Responses that informed the transformation"
    )
    domain_indicators: list[str] = Field(
        default_factory=list, description="Domain or subject area indicators identified"
    )
    complexity_assessment: Literal["low", "medium", "high"] = Field(
        default="medium", description="Assessed complexity of research required"
    )
    estimated_scope: Literal["narrow", "moderate", "broad"] = Field(
        default="moderate", description="Estimated scope of research needed"
    )

    @model_validator(mode="after")
    def validate_transformation_quality(self) -> "TransformedQueryResult":
        """Ensure transformation adds value and maintains quality."""
        if self.original_query.strip().lower() == self.transformed_query.strip().lower():
            raise ValueError("Transformed query must meaningfully differ from original")

        if self.specificity_score < 0.1:
            raise ValueError("Transformation must provide some improvement in specificity")

        # Check for minimum length improvement for very short queries
        if len(self.original_query.split()) < 5 and len(self.transformed_query.split()) < 8:
            raise ValueError("Short queries should be expanded significantly during transformation")

        return self


class BriefGenerationResult(BaseModel):
    """Structured output for Brief Generation Agent."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    brief_text: Annotated[str, Field(min_length=100, description="Generated research brief")]
    confidence_score: Annotated[
        float, Field(ge=0.0, le=1.0, description="Confidence in brief quality")
    ]
    key_research_areas: Annotated[
        list[str], Field(min_length=1, description="Primary research areas identified")
    ]
    research_objectives: list[str] = Field(
        default_factory=list, description="Specific research objectives"
    )
    methodology_suggestions: list[str] = Field(
        default_factory=list, description="Suggested research methodologies"
    )
    estimated_complexity: Literal["low", "medium", "high"] = Field(
        default="medium", description="Estimated research complexity"
    )
    estimated_duration: str = Field(default="", description="Estimated time to complete research")
    suggested_sources: list[str] = Field(
        default_factory=list, description="Suggested source types or specific sources"
    )
    potential_challenges: list[str] = Field(
        default_factory=list, description="Anticipated research challenges"
    )
    success_criteria: list[str] = Field(
        default_factory=list, description="Criteria for successful research completion"
    )

    @field_validator("brief_text")
    @classmethod
    def validate_brief_completeness(cls, v: str) -> str:
        """Ensure brief contains essential elements."""
        v = v.strip()

        # Check for minimum content indicators
        required_indicators = ["research", "objective", "scope"]
        content_lower = v.lower()

        missing_indicators = [
            indicator for indicator in required_indicators if indicator not in content_lower
        ]

        if missing_indicators:
            raise ValueError(f"Brief missing key elements: {', '.join(missing_indicators)}")

        return v

    @model_validator(mode="after")
    def validate_brief_consistency(self) -> "BriefGenerationResult":
        """Ensure consistency across brief elements."""
        if not self.key_research_areas:
            raise ValueError("At least one key research area must be identified")

        if self.confidence_score > 0.9 and self.estimated_complexity == "high":
            raise ValueError("High confidence inconsistent with high complexity assessment")

        return self


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


class ResearchResults(BaseModel):
    """Wrapper for a list of research findings."""

    findings: list[ResearchFinding] = Field(default_factory=list, description="Research findings")


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


class TransformedQuery(BaseModel):
    """Model for a query that has been transformed from broad to specific."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    original_query: Annotated[
        str, Field(min_length=1, description="The original user query before transformation")
    ]
    transformed_query: Annotated[
        str, Field(min_length=1, description="The primary transformed research question")
    ]
    supporting_questions: list[str] = Field(
        default_factory=list, description="Additional supporting research questions"
    )
    transformation_rationale: Annotated[
        str, Field(min_length=1, description="Explanation of how and why the query was transformed")
    ]
    specificity_score: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Score indicating how specific the transformed query is (0-1)",
        ),
    ]
    missing_dimensions: list[str] = Field(
        default_factory=list,
        description="Dimensions that are still missing or could be further refined",
    )
    clarification_responses: dict[str, str] = Field(
        default_factory=dict,
        description="User responses to clarification questions that informed transformation",
    )
    transformation_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the transformation process"
    )
    created_at: datetime = Field(default_factory=datetime.now)


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
