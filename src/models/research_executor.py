"""Research executor models for the research system."""

from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ConfidenceLevel(str, Enum):
    """Confidence level categories for research findings."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"

    def to_score(self) -> float:
        """Convert confidence level to numeric score."""
        mapping = {
            self.HIGH: 0.9,
            self.MEDIUM: 0.7,
            self.LOW: 0.4,
            self.UNCERTAIN: 0.2,
        }
        return mapping[self]

    @classmethod
    def from_score(cls, score: float) -> "ConfidenceLevel":
        """Convert numeric score to confidence level."""
        if score >= 0.8:
            return cls.HIGH
        elif score >= 0.6:
            return cls.MEDIUM
        elif score >= 0.3:
            return cls.LOW
        else:
            return cls.UNCERTAIN


class ImportanceLevel(str, Enum):
    """Importance level categories for findings."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    def to_score(self) -> float:
        """Convert importance level to numeric score."""
        mapping = {
            self.CRITICAL: 1.0,
            self.HIGH: 0.8,
            self.MEDIUM: 0.5,
            self.LOW: 0.2,
        }
        return mapping[self]

    @classmethod
    def from_score(cls, score: float) -> "ImportanceLevel":
        """Convert numeric score to importance level."""
        if score >= 0.9:
            return cls.CRITICAL
        elif score >= 0.7:
            return cls.HIGH
        elif score >= 0.4:
            return cls.MEDIUM
        else:
            return cls.LOW


class ResearchSource(BaseModel):
    """Information about a research source."""

    url: str | None = Field(default=None, description="URL of the source")
    title: str = Field(description="Title of the source")
    author: str | None = Field(default=None, description="Author of the source")
    date: datetime | None = Field(default=None, description="Publication date of the source")
    source_type: str | None = Field(
        default=None, description="Type of source (academic, news, blog, etc.)"
    )
    credibility_score: float = Field(
        ge=0.0, le=1.0, default=0.5, description="Credibility score of the source"
    )
    relevance_score: float = Field(
        ge=0.0, le=1.0, default=0.5, description="Relevance score to the research query"
    )
    citations: list[str] = Field(
        default_factory=list, description="Citations or references from this source"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the source"
    )

    @field_validator("credibility_score", "relevance_score")
    @classmethod
    def validate_scores(cls, v: float) -> float:
        """Ensure scores are within valid range."""
        return max(0.0, min(1.0, v))

    def overall_quality(self) -> float:
        """Calculate overall source quality score."""
        return self.credibility_score * 0.6 + self.relevance_score * 0.4


class HierarchicalFinding(BaseModel):
    """A finding with hierarchical importance scoring.

    Simplified version without relationships for MVP.
    """

    finding: str = Field(description="The core finding or insight")
    supporting_evidence: list[str] = Field(
        default_factory=list, description="Evidence supporting this finding"
    )
    confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM, description="Confidence level in this finding"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0, default=0.5, description="Numeric confidence score"
    )
    importance: ImportanceLevel = Field(
        default=ImportanceLevel.MEDIUM, description="Importance level of this finding"
    )
    importance_score: float = Field(
        ge=0.0, le=1.0, default=0.5, description="Numeric importance score"
    )
    source: ResearchSource | None = Field(default=None, description="Source of this finding")
    category: str | None = Field(default=None, description="Category or topic of this finding")
    temporal_relevance: str | None = Field(
        default=None, description="Time period or temporal context"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def __init__(self, **data: Any):
        """Initialize with score-level synchronization."""
        super().__init__(**data)
        # Sync scores with levels if not provided
        if "confidence_score" not in data:
            self.confidence_score = self.confidence.to_score()
        if "importance_score" not in data:
            self.importance_score = self.importance.to_score()

    @field_validator("confidence_score", mode="after")
    @classmethod
    def sync_confidence_level(cls, v: float) -> float:
        """Validate confidence score is in range."""
        return max(0.0, min(1.0, v))

    @field_validator("importance_score", mode="after")
    @classmethod
    def sync_importance_level(cls, v: float) -> float:
        """Validate importance score is in range."""
        return max(0.0, min(1.0, v))

    def hierarchical_score(self) -> float:
        """Calculate combined hierarchical score for ranking."""
        # Weight importance more heavily than confidence
        return (self.importance_score * 0.7) + (self.confidence_score * 0.3)


class ThemeCluster(BaseModel):
    """A cluster of related findings forming a theme."""

    theme_name: str = Field(description="Name or title of the theme")
    description: str = Field(description="Description of what this theme represents")
    findings: list[HierarchicalFinding] = Field(
        default_factory=list, description="Findings in this cluster"
    )
    coherence_score: float = Field(
        ge=0.0, le=1.0, default=0.5, description="How coherent/related the findings are"
    )
    importance_score: float = Field(
        ge=0.0, le=1.0, default=0.5, description="Overall importance of this theme"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional theme metadata")

    def average_confidence(self) -> float:
        """Calculate average confidence across findings."""
        if not self.findings:
            return 0.0
        return sum(f.confidence_score for f in self.findings) / len(self.findings)

    def critical_findings_count(self) -> int:
        """Count critical importance findings."""
        return sum(1 for f in self.findings if f.importance == ImportanceLevel.CRITICAL)


class Contradiction(BaseModel):
    """Represents a contradiction between findings."""

    finding_1_id: str = Field(description="ID/index of first finding")
    finding_2_id: str = Field(description="ID/index of second finding")
    contradiction_type: str = Field(
        description="Type of contradiction (direct, partial, temporal, etc.)"
    )
    explanation: str = Field(description="Explanation of the contradiction")
    resolution_hint: str | None = Field(
        default=None, description="Hint for resolving the contradiction"
    )
    severity: float = Field(
        ge=0.0, le=1.0, default=0.5, description="Severity of the contradiction"
    )

    def needs_immediate_resolution(self) -> bool:
        """Check if contradiction needs immediate resolution."""
        return self.severity > 0.7 or self.contradiction_type == "direct"


class ExecutiveSummary(BaseModel):
    """Executive summary of research results."""

    key_findings: list[str] = Field(
        description="Top 3-5 most important findings", default_factory=list
    )
    confidence_assessment: str = Field(description="Overall confidence in the research", default="")
    critical_gaps: list[str] = Field(
        description="Critical information gaps identified", default_factory=list
    )
    recommended_actions: list[str] = Field(
        description="Recommended next steps or actions", default_factory=list
    )
    risk_factors: list[str] = Field(
        description="Identified risks or concerns", default_factory=list
    )

    def to_markdown(self) -> str:
        """Convert summary to markdown format."""
        sections = []

        if self.key_findings:
            sections.append("## Key Findings\n" + "\n".join(f"- {f}" for f in self.key_findings))

        if self.confidence_assessment:
            sections.append(f"## Confidence Assessment\n{self.confidence_assessment}")

        if self.critical_gaps:
            sections.append("## Critical Gaps\n" + "\n".join(f"- {g}" for g in self.critical_gaps))

        if self.recommended_actions:
            sections.append(
                "## Recommended Actions\n" + "\n".join(f"- {a}" for a in self.recommended_actions)
            )

        if self.risk_factors:
            sections.append("## Risk Factors\n" + "\n".join(f"- {r}" for r in self.risk_factors))

        return "\n\n".join(sections)


class SynthesisMetadata(BaseModel):
    """Metadata about the synthesis process."""

    synthesis_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="When synthesis was performed"
    )
    synthesis_method: str = Field(default="ml_clustering", description="Method used for synthesis")
    total_sources_analyzed: int = Field(default=0, description="Number of sources analyzed")
    total_findings_extracted: int = Field(default=0, description="Number of findings extracted")
    clustering_parameters: dict[str, Any] = Field(
        default_factory=dict, description="Parameters used for clustering"
    )
    quality_metrics: dict[str, float] = Field(
        default_factory=dict, description="Quality metrics of the synthesis"
    )
    processing_time_seconds: float | None = Field(
        default=None, description="Time taken for synthesis"
    )

    def synthesis_quality_score(self) -> float:
        """Calculate overall synthesis quality score."""
        if not self.quality_metrics:
            return 0.0
        return sum(self.quality_metrics.values()) / len(self.quality_metrics)


class ResearchResults(BaseModel):
    """Comprehensive output model for research executor agent."""

    # Core information
    query: str = Field(description="The original research query")
    execution_time: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Time of research execution"
    )

    # Findings and themes
    findings: list[HierarchicalFinding] = Field(
        default_factory=list, description="All research findings with hierarchy"
    )
    theme_clusters: list[ThemeCluster] = Field(
        default_factory=list, description="Findings organized into thematic clusters"
    )

    # Analysis results
    contradictions: list[Contradiction] = Field(
        default_factory=list, description="Identified contradictions"
    )
    executive_summary: ExecutiveSummary | None = Field(
        default=None, description="Executive summary of the research"
    )

    # Source tracking
    sources: list[ResearchSource] = Field(default_factory=list, description="All sources consulted")

    # Quality and metadata
    synthesis_metadata: SynthesisMetadata | None = Field(
        default=None, description="Metadata about synthesis process"
    )
    overall_quality_score: float = Field(
        ge=0.0, le=1.0, default=0.0, description="Overall quality score of the research"
    )

    # Additional insights
    key_insights: list[str] = Field(
        default_factory=list, description="Key insights from the research"
    )
    data_gaps: list[str] = Field(
        default_factory=list, description="Identified gaps in available data"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the research"
    )

    def get_critical_findings(self) -> list[HierarchicalFinding]:
        """Get all critical importance findings."""
        return [f for f in self.findings if f.importance == ImportanceLevel.CRITICAL]

    def get_high_confidence_findings(self) -> list[HierarchicalFinding]:
        """Get all high confidence findings."""
        return [f for f in self.findings if f.confidence == ConfidenceLevel.HIGH]

    def has_contradictions(self) -> bool:
        """Check if there are any contradictions."""
        return len(self.contradictions) > 0

    def needs_further_research(self) -> bool:
        """Determine if further research is needed."""
        return (
            len(self.data_gaps) > 0
            or self.overall_quality_score < 0.7
            or any(c.needs_immediate_resolution() for c in self.contradictions)
        )

    def to_report(self) -> str:
        """Generate a text report of the research results."""
        sections = []

        # Header
        sections.append(f"# Research Report: {self.query}")
        sections.append(f"Executed: {self.execution_time.isoformat()}")
        sections.append(f"Quality Score: {self.overall_quality_score:.2f}")

        # Executive Summary
        if self.executive_summary:
            sections.append("\n" + self.executive_summary.to_markdown())

        # Theme Clusters
        if self.theme_clusters:
            sections.append("\n## Thematic Analysis")
            for cluster in self.theme_clusters:
                sections.append(f"\n### {cluster.theme_name}")
                sections.append(f"{cluster.description}")
                sections.append(f"Findings: {len(cluster.findings)}")
                sections.append(f"Coherence: {cluster.coherence_score:.2f}")

        # Contradictions
        if self.contradictions:
            sections.append("\n## Contradictions Identified")
            for cont in self.contradictions:
                sections.append(f"- {cont.explanation}")
                if cont.resolution_hint:
                    sections.append(f"  Resolution: {cont.resolution_hint}")

        # Data Gaps
        if self.data_gaps:
            sections.append("\n## Data Gaps")
            for gap in self.data_gaps:
                sections.append(f"- {gap}")

        return "\n".join(sections)


# For backward compatibility - simple models
class ResearchFinding(BaseModel):
    """Individual research finding (simple version for compatibility)."""

    finding: str = Field(description="The research finding")
    supporting_evidence: list[str] = Field(
        default_factory=list, description="Evidence supporting this finding"
    )
    confidence_level: float = Field(ge=0.0, le=1.0, description="Confidence level in this finding")
    source: ResearchSource | None = Field(default=None, description="Source of this finding")
    category: str | None = Field(default=None, description="Category or topic of this finding")

    @classmethod
    def from_hierarchical(cls, hf: HierarchicalFinding) -> "ResearchFinding":
        """Convert from hierarchical finding."""
        return cls(
            finding=hf.finding,
            supporting_evidence=hf.supporting_evidence,
            confidence_level=hf.confidence_score,
            source=hf.source,
            category=hf.category,
        )


# Phase 3 Models for Performance Optimization


class PerformanceMetrics(BaseModel):
    """Performance metrics for research execution."""

    total_execution_time: float = Field(default=0.0, description="Total execution time in seconds")
    synthesis_time: float = Field(default=0.0, description="Time spent on synthesis")
    pattern_detection_time: float = Field(
        default=0.0, description="Time spent on pattern detection"
    )
    contradiction_check_time: float = Field(
        default=0.0, description="Time spent on contradiction checking"
    )
    clustering_time: float = Field(default=0.0, description="Time spent on clustering")
    confidence_analysis_time: float = Field(
        default=0.0, description="Time spent on confidence analysis"
    )
    memory_usage_mb: float = Field(default=0.0, description="Peak memory usage in MB")
    findings_processed: int = Field(default=0, description="Number of findings processed")
    patterns_detected: int = Field(default=0, description="Number of patterns detected")
    contradictions_found: int = Field(default=0, description="Number of contradictions found")
    clusters_formed: int = Field(default=0, description="Number of clusters formed")
    cache_hits: int = Field(default=0, description="Number of cache hits")
    cache_misses: int = Field(default=0, description="Number of cache misses")
    parallel_tasks_executed: int = Field(default=0, description="Number of parallel tasks executed")
    api_calls_made: int = Field(default=0, description="Number of API calls made")
    estimated_tokens_used: int = Field(default=0, description="Estimated tokens used")


class CacheMetadata(BaseModel):
    """Metadata for cached items."""

    key: str = Field(description="Cache key")
    content_hash: str = Field(description="Hash of the cached content")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Creation timestamp"
    )
    expires_at: datetime = Field(description="Expiration timestamp")
    access_count: int = Field(default=0, description="Number of times accessed")
    last_accessed: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Last access timestamp"
    )
    size_bytes: int = Field(default=0, description="Size of cached data in bytes")
    cache_type: str = Field(description="Type of cached data (synthesis, vectorization, etc.)")

    @property
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return datetime.now(UTC) > self.expires_at

    @property
    def ttl_remaining(self) -> timedelta:
        """Get remaining time-to-live."""
        return max(self.expires_at - datetime.now(UTC), timedelta(seconds=0))


class OptimizationConfig(BaseModel):
    """Configuration for optimization settings."""

    enable_caching: bool = Field(default=True, description="Enable caching")
    cache_ttl_seconds: int = Field(default=3600, ge=0, description="Cache TTL in seconds")
    max_cache_size_mb: float = Field(default=100.0, ge=0, description="Maximum cache size in MB")
    enable_parallel_execution: bool = Field(default=True, description="Enable parallel execution")
    max_concurrent_tasks: int = Field(default=4, ge=1, description="Maximum concurrent tasks")
    batch_size: int = Field(default=10, ge=1, description="Batch size for processing")
    enable_adaptive_thresholds: bool = Field(
        default=True, description="Enable adaptive quality thresholds"
    )
    memory_limit_mb: float = Field(default=500.0, ge=0, description="Memory limit in MB")
    enable_circuit_breaker: bool = Field(default=True, description="Enable circuit breaker pattern")
    circuit_breaker_threshold: int = Field(
        default=5, ge=1, description="Failure threshold for circuit breaker"
    )
    circuit_breaker_timeout: int = Field(
        default=60, ge=1, description="Circuit breaker timeout in seconds"
    )
    enable_graceful_degradation: bool = Field(
        default=True, description="Enable graceful degradation under load"
    )
    connection_pool_size: int = Field(default=10, ge=1, description="Connection pool size")
    request_timeout_seconds: int = Field(default=30, ge=1, description="Request timeout in seconds")
    enable_metrics_collection: bool = Field(default=True, description="Enable metrics collection")
    metrics_export_format: str = Field(
        default="json", description="Metrics export format (json, csv)"
    )


# Phase 2 Models for Pattern Analysis


class PatternType(str, Enum):
    """Types of patterns that can be detected in research findings."""

    CONVERGENCE = "convergence"  # Multiple findings pointing to same conclusion
    DIVERGENCE = "divergence"  # Findings branching into different areas
    EMERGENCE = "emergence"  # New patterns emerging from synthesis
    TEMPORAL = "temporal"  # Time-based patterns or trends
    CAUSAL = "causal"  # Cause-effect relationships
    CORRELATION = "correlation"  # Statistical correlations
    CYCLE = "cycle"  # Repeating patterns or cycles
    ANOMALY = "anomaly"  # Outliers or unexpected patterns


class PatternAnalysis(BaseModel):
    """Analysis of patterns detected in research findings."""

    pattern_type: PatternType = Field(description="Type of pattern detected")
    pattern_name: str = Field(description="Descriptive name for the pattern")
    description: str = Field(description="Detailed description of the pattern")

    # Pattern characteristics
    strength: float = Field(
        ge=0.0, le=1.0, description="Strength or confidence in the pattern (0-1)"
    )
    finding_ids: list[str] = Field(
        default_factory=list, description="IDs of findings involved in this pattern"
    )

    # Pattern metadata
    temporal_span: str | None = Field(
        default=None, description="Time span over which pattern occurs"
    )
    confidence_factors: dict[str, float] = Field(
        default_factory=dict, description="Factors contributing to pattern confidence"
    )

    # Implications
    implications: list[str] = Field(
        default_factory=list, description="Implications or predictions from this pattern"
    )
    related_patterns: list[str] = Field(
        default_factory=list, description="IDs of related or connected patterns"
    )

    def is_significant(self, threshold: float = 0.7) -> bool:
        """Check if pattern is significant based on strength threshold."""
        return self.strength >= threshold

    def average_confidence_factor(self) -> float:
        """Calculate average confidence across all factors."""
        if not self.confidence_factors:
            return 0.0
        return sum(self.confidence_factors.values()) / len(self.confidence_factors)


class ConfidenceAnalysis(BaseModel):
    """Detailed analysis of confidence levels across research findings."""

    overall_confidence: float = Field(
        ge=0.0, le=1.0, description="Overall confidence in the research results"
    )
    confidence_distribution: dict[str, int] = Field(
        default_factory=dict, description="Distribution of findings by confidence level"
    )

    # Confidence factors
    source_reliability: float = Field(
        ge=0.0, le=1.0, default=0.5, description="Average reliability of sources"
    )
    consistency_score: float = Field(
        ge=0.0, le=1.0, default=0.5, description="Consistency across different findings"
    )
    evidence_strength: float = Field(
        ge=0.0, le=1.0, default=0.5, description="Strength of supporting evidence"
    )

    # Confidence breakdown by category
    category_confidence: dict[str, float] = Field(
        default_factory=dict, description="Confidence scores by finding category"
    )

    # Areas of uncertainty
    uncertainty_areas: list[str] = Field(
        default_factory=list, description="Areas with low confidence or high uncertainty"
    )
    confidence_gaps: list[str] = Field(
        default_factory=list, description="Gaps in confidence that need addressing"
    )

    # Recommendations
    confidence_improvements: list[str] = Field(
        default_factory=list, description="Suggestions for improving confidence"
    )

    def needs_validation(self) -> bool:
        """Check if results need additional validation."""
        return self.overall_confidence < 0.6 or len(self.uncertainty_areas) > 3

    def get_weak_areas(self, threshold: float = 0.5) -> list[str]:
        """Get categories with confidence below threshold."""
        return [cat for cat, conf in self.category_confidence.items() if conf < threshold]
