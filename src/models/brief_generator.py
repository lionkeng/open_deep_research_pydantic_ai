"""Research brief models for the research system."""


from pydantic import BaseModel, Field


class ResearchObjective(BaseModel):
    """Individual research objective within a brief."""

    objective: str = Field(description="Specific research objective")
    priority: int = Field(
        ge=1, le=5, description="Priority level of this objective (1=lowest, 5=highest)"
    )
    success_criteria: str = Field(description="Criteria for determining if this objective is met")


class ResearchMethodology(BaseModel):
    """Research methodology specification."""

    approach: str = Field(description="Overall research approach")
    data_sources: list[str] = Field(default_factory=list, description="Recommended data sources")
    analysis_methods: list[str] = Field(
        default_factory=list, description="Methods for analyzing findings"
    )
    quality_checks: list[str] = Field(
        default_factory=list, description="Quality assurance checks to perform"
    )


class ResearchBrief(BaseModel):
    """Output model for brief generator agent."""

    title: str = Field(description="Research brief title")
    executive_summary: str = Field(description="High-level summary of the research requirements")
    objectives: list[ResearchObjective] = Field(
        default_factory=list, description="List of research objectives"
    )
    methodology: ResearchMethodology = Field(description="Proposed research methodology")
    scope: str = Field(description="Detailed scope of the research")
    constraints: list[str] = Field(default_factory=list, description="Constraints and limitations")
    deliverables: list[str] = Field(default_factory=list, description="Expected deliverables")
    timeline_estimate: str | None = Field(
        default=None, description="Estimated timeline for research completion"
    )
    success_metrics: list[str] = Field(
        default_factory=list, description="Metrics for measuring research success"
    )
