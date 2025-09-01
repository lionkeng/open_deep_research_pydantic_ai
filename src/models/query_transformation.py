"""Query transformation models for the research system."""


from pydantic import BaseModel, Field


class TransformedQuery(BaseModel):
    """Output model for query transformation agent."""

    original_query: str = Field(description="The original user query before transformation")
    transformed_query: str = Field(description="The transformed and optimized research query")
    search_keywords: list[str] = Field(
        default_factory=list,
        description="Key search terms extracted from the query",
    )
    research_scope: str = Field(description="The scope and boundaries of the research")
    expected_output_type: str = Field(
        description="The type of output expected (report, summary, analysis, etc.)"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Confidence in the transformation quality"
    )
    transformation_rationale: str | None = Field(
        default=None,
        description="Explanation of why and how the query was transformed",
    )
