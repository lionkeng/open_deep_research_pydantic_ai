"""Compression models for the research system."""


from pydantic import BaseModel, Field


class CompressedSection(BaseModel):
    """A compressed section of content."""

    original_length: int = Field(description="Original length in characters")
    compressed_length: int = Field(description="Compressed length in characters")
    compression_ratio: float = Field(ge=0.0, le=1.0, description="Compression ratio achieved")
    content: str = Field(description="The compressed content")
    key_points: list[str] = Field(
        default_factory=list, description="Key points preserved in compression"
    )


class CompressedContent(BaseModel):
    """Output model for compression agent."""

    original_content: str = Field(description="The original content before compression")
    compressed_summary: str = Field(description="The main compressed summary")
    sections: list[CompressedSection] = Field(
        default_factory=list, description="Compressed sections if content was segmented"
    )
    key_facts: list[str] = Field(
        default_factory=list, description="Key facts extracted and preserved"
    )
    removed_redundancies: list[str] = Field(
        default_factory=list, description="Types of redundancies removed"
    )
    compression_strategy: str = Field(description="Strategy used for compression")
    quality_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Quality metrics (e.g., information_retention, readability)",
    )
    total_compression_ratio: float = Field(ge=0.0, le=1.0, description="Overall compression ratio")
