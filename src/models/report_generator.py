"""Report generator models for the research system."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ReportSection(BaseModel):
    """A section within a research report."""

    title: str = Field(description="Section title")
    content: str = Field(description="Section content")
    subsections: list[ReportSection] = Field(default_factory=list, description="Nested subsections")
    figures: list[str] = Field(default_factory=list, description="References to figures or charts")
    citations: list[str] = Field(default_factory=list, description="Citations used in this section")


class ReportMetadata(BaseModel):
    """Metadata for a research report."""

    created_at: datetime = Field(
        default_factory=datetime.now, description="Report creation timestamp"
    )
    version: str = Field(default="1.0", description="Report version")
    authors: list[str] = Field(default_factory=list, description="Report authors or contributors")
    keywords: list[str] = Field(default_factory=list, description="Report keywords")
    classification: str | None = Field(
        default=None, description="Report classification or category"
    )


class ResearchReport(BaseModel):
    """Output model for report generator agent."""

    title: str = Field(description="Report title")
    executive_summary: str = Field(description="Executive summary of the report")
    introduction: str = Field(description="Report introduction")
    sections: list[ReportSection] = Field(default_factory=list, description="Main report sections")
    conclusions: str = Field(description="Report conclusions")
    recommendations: list[str] = Field(default_factory=list, description="Report recommendations")
    references: list[str] = Field(default_factory=list, description="List of references")
    appendices: dict[str, str] = Field(default_factory=dict, description="Report appendices")
    metadata: ReportMetadata = Field(default_factory=ReportMetadata, description="Report metadata")
    quality_score: float = Field(ge=0.0, le=1.0, description="Overall quality score of the report")


# Update forward references
ReportSection.model_rebuild()
