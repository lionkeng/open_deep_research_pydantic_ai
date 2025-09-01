"""Research executor models for the research system."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ResearchSource(BaseModel):
    """Information about a research source."""

    url: str | None = Field(default=None, description="URL of the source")
    title: str = Field(description="Title of the source")
    author: str | None = Field(default=None, description="Author of the source")
    date: datetime | None = Field(default=None, description="Publication date of the source")
    relevance_score: float = Field(ge=0.0, le=1.0, description="Relevance score of the source")


class ResearchFinding(BaseModel):
    """Individual research finding."""

    finding: str = Field(description="The research finding")
    supporting_evidence: list[str] = Field(
        default_factory=list, description="Evidence supporting this finding"
    )
    confidence_level: float = Field(ge=0.0, le=1.0, description="Confidence level in this finding")
    source: ResearchSource | None = Field(default=None, description="Source of this finding")
    category: str | None = Field(default=None, description="Category or topic of this finding")


class ResearchResults(BaseModel):
    """Output model for research executor agent."""

    query: str = Field(description="The research query that was executed")
    execution_time: datetime = Field(
        default_factory=datetime.now, description="Time of research execution"
    )
    findings: list[ResearchFinding] = Field(
        default_factory=list, description="List of research findings"
    )
    sources: list[ResearchSource] = Field(default_factory=list, description="All sources consulted")
    key_insights: list[str] = Field(
        default_factory=list, description="Key insights from the research"
    )
    data_gaps: list[str] = Field(
        default_factory=list, description="Identified gaps in available data"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the research"
    )
    quality_score: float = Field(
        ge=0.0, le=1.0, description="Overall quality score of the research"
    )
