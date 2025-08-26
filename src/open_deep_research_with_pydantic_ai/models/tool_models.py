"""Pydantic models for agent tool return types."""

from datetime import UTC, datetime
from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ToolStatus(str, Enum):
    """Tool execution status."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    SKIPPED = "skipped"


class ToolResult[T](BaseModel):
    """Generic tool result wrapper."""

    model_config = ConfigDict(extra="allow")

    status: ToolStatus = Field(description="Execution status")
    data: T | None = Field(default=None, description="Result data if successful")
    error: str | None = Field(default=None, description="Error message if failed")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    execution_time: float = Field(default=0.0, description="Execution time in seconds")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Execution timestamp"
    )


class ValidationResult(BaseModel):
    """Result from scope validation tool."""

    has_issues: bool = Field(description="Whether validation issues were found")
    issues: list[str] = Field(default_factory=list, description="List of validation issues")
    suggestions: list[str] = Field(default_factory=list, description="Improvement suggestions")
    severity: str = Field(default="info", description="Issue severity: info, warning, error")


class ComplexityAssessment(BaseModel):
    """Result from complexity assessment tool."""

    level: str = Field(description="Complexity level: simple, medium, complex")
    factors: list[str] = Field(default_factory=list, description="Contributing complexity factors")
    estimated_duration: float = Field(
        default=0.0, description="Estimated research duration in minutes"
    )
    recommended_approach: str = Field(
        default="standard", description="Recommended research approach"
    )


class SearchToolResult(BaseModel):
    """Result from search tool execution."""

    query: str = Field(description="Search query executed")
    results_count: int = Field(description="Number of results found")
    relevant_count: int = Field(description="Number of relevant results")
    top_sources: list[str] = Field(default_factory=list, description="Top source URLs")
    key_findings: list[str] = Field(default_factory=list, description="Key findings from search")


class AnalysisToolResult(BaseModel):
    """Result from analysis tool."""

    topic: str = Field(description="Analysis topic")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    key_points: list[str] = Field(default_factory=list, description="Key analysis points")
    evidence: list[str] = Field(default_factory=list, description="Supporting evidence")
    contradictions: list[str] = Field(default_factory=list, description="Contradictory findings")
    gaps: list[str] = Field(default_factory=list, description="Information gaps")


class SynthesisToolResult(BaseModel):
    """Result from synthesis tool."""

    sections_generated: int = Field(description="Number of sections generated")
    word_count: int = Field(description="Total word count")
    sources_cited: int = Field(description="Number of sources cited")
    coherence_score: float = Field(ge=0.0, le=1.0, description="Content coherence score")
    completeness_score: float = Field(ge=0.0, le=1.0, description="Topic completeness score")


class DelegationResult(BaseModel):
    """Result from agent delegation."""

    target_agent: str = Field(description="Agent delegated to")
    task: str = Field(description="Delegated task description")
    success: bool = Field(description="Whether delegation succeeded")
    result_summary: str = Field(description="Summary of delegation result")
    tokens_used: int = Field(default=0, description="Tokens consumed by delegation")


class ToolExecutionSummary(BaseModel):
    """Summary of all tool executions in a session."""

    total_executions: int = Field(description="Total tool executions")
    successful: int = Field(description="Successful executions")
    failed: int = Field(description="Failed executions")
    total_time: float = Field(description="Total execution time in seconds")
    tools_used: dict[str, int] = Field(default_factory=dict, description="Count by tool name")
    errors: list[str] = Field(default_factory=list, description="Error messages from failures")
