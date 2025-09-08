"""Query transformation models for the research system."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class TransformedQuery(BaseModel):
    """Enhanced output model for query transformation agent with clarification integration."""

    # Core Transformation
    original_query: str = Field(description="The original user query before transformation")
    transformed_query: str = Field(description="The transformed and optimized research query")

    # Search Strategy
    search_keywords: list[str] = Field(
        default_factory=list,
        description="3-10 key search terms extracted from the query",
    )
    supporting_questions: list[str] = Field(
        default_factory=list, description="3-5 specific sub-questions that decompose the main query"
    )
    excluded_terms: list[str] = Field(
        default_factory=list, description="Terms to exclude from search to avoid irrelevant results"
    )

    # Scope Definition
    research_scope: str = Field(description="The scope and boundaries of the research")
    temporal_scope: str | None = Field(
        default=None, description="Time period if relevant (e.g., '2020-2024', 'last decade')"
    )
    geographic_scope: str | None = Field(
        default=None, description="Geographic boundaries if relevant"
    )
    domain_scope: str | None = Field(default=None, description="Specific domain/field focus")

    # Output Requirements
    expected_output_type: str = Field(
        description="The type of output expected (report, summary, analysis, comparison, etc.)"
    )
    expected_depth: Literal["overview", "detailed", "expert"] = Field(
        default="detailed", description="Depth level for research"
    )

    # Clarification Integration
    clarification_insights: dict[str, Any] = Field(
        default_factory=dict,
        description="Key insights from clarification phase used in transformation",
    )
    assumptions_made: list[str] = Field(
        default_factory=list, description="Explicit assumptions for unresolved ambiguities"
    )
    ambiguities_resolved: list[str] = Field(
        default_factory=list, description="Ambiguities successfully addressed through clarification"
    )
    ambiguities_remaining: list[str] = Field(
        default_factory=list, description="Ambiguities still present after transformation"
    )

    # Quality Metrics
    specificity_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How specific the transformed query is (0=vague, 1=very specific)",
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Confidence in transformation quality"
    )
    clarification_coverage: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Percentage of clarification questions answered"
    )

    # Transformation Metadata
    transformation_rationale: str = Field(
        description="Explanation of why and how the query was transformed",
    )
    transformation_strategy: str = Field(
        default="specification",
        description="Strategy: decomposition, specification, scoping, or assumption-based",
    )

    @field_validator("supporting_questions")
    @classmethod
    def validate_supporting_questions(cls, v: list[str]) -> list[str]:
        """Ensure 3-5 specific questions."""
        if v and len(v) < 3:
            # Pad with generic questions if needed
            while len(v) < 3:
                v.append(f"Additional aspect {len(v) + 1} to explore")
        if len(v) > 5:
            v = v[:5]  # Truncate to 5
        return v

    @field_validator("search_keywords")
    @classmethod
    def validate_keywords(cls, v: list[str]) -> list[str]:
        """Ensure 3-10 relevant keywords."""
        if v and len(v) < 3:
            # Don't raise error, just note it
            pass
        if len(v) > 10:
            v = v[:10]  # Keep most relevant 10
        return v

    @model_validator(mode="after")
    def validate_coherence(self) -> "TransformedQuery":
        """Ensure output is internally consistent."""
        # Confidence should reflect clarification coverage
        if self.clarification_coverage < 0.3 and self.confidence_score > 0.8:
            self.confidence_score = min(0.7, self.confidence_score)

        # Specificity should align with ambiguities
        if len(self.ambiguities_remaining) > 3 and self.specificity_score > 0.8:
            self.specificity_score = min(0.7, self.specificity_score)

        # Ensure transformed query is different from original
        if self.transformed_query == self.original_query and self.confidence_score > 0.5:
            self.confidence_score = 0.5  # Low confidence if no transformation occurred

        return self
