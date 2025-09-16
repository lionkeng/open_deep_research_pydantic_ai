"""Research executor models for the research system."""

import json
import math
from collections import Counter
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from utils.validation import (
    ContradictionSeverityCalculator,
    RobustScoreValidator,
    SeverityResult,
)


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
        if score >= 0.6:
            return cls.MEDIUM
        if score >= 0.3:
            return cls.LOW
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
        if score >= 0.7:
            return cls.HIGH
        if score >= 0.4:
            return cls.MEDIUM
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
    def validate_scores(cls, v: Any, info: Any) -> float:
        """Ensure scores are within valid range using robust validation."""
        try:
            from utils.validation import RobustScoreValidator

            field_name = info.field_name if info else "score"
            return RobustScoreValidator.validate_probability_score(
                value=v, field_name=field_name, default_value=0.5
            )
        except ImportError:
            # Fallback validation if utils not available
            if v is None or math.isnan(v) or math.isinf(v):
                return 0.5
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
    def sync_confidence_level(cls, v: Any) -> float:
        """Validate confidence score is in range using robust validation."""
        try:
            from utils.validation import RobustScoreValidator

            return RobustScoreValidator.validate_probability_score(
                value=v, field_name="confidence_score", default_value=0.5
            )
        except ImportError:
            if v is None or math.isnan(v) or math.isinf(v):
                return 0.5
            return max(0.0, min(1.0, v))

    @field_validator("importance_score", mode="after")
    @classmethod
    def sync_importance_level(cls, v: Any) -> float:
        """Validate importance score is in range using robust validation."""
        try:
            from utils.validation import RobustScoreValidator

            return RobustScoreValidator.validate_probability_score(
                value=v, field_name="importance_score", default_value=0.5
            )
        except ImportError:
            if v is None or math.isnan(v) or math.isinf(v):
                return 0.5
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
    """Represents a contradiction between findings with comprehensive severity analysis."""

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
    severity_components: SeverityResult | None = Field(
        default=None, description="Detailed breakdown of severity calculation"
    )
    resolution_strategy: dict[str, Any] = Field(
        default_factory=dict, description="Suggested resolution strategy and actions"
    )
    priority: int = Field(
        ge=1, le=5, default=3, description="Resolution priority (1=highest, 5=lowest)"
    )
    severity_level: str = Field(default="medium", description="Categorical severity level")
    domain_overlap: float = Field(
        ge=0.0, le=1.0, default=1.0, description="Domain overlap between findings"
    )
    temporal_distance_days: float = Field(ge=0.0, default=0.0, description="Days between findings")

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: Any) -> float:
        """Validate severity score with robust handling."""
        return RobustScoreValidator.validate_probability_score(
            v, "contradiction_severity", default_value=0.5
        )

    @field_validator("domain_overlap")
    @classmethod
    def validate_domain_overlap(cls, v: Any) -> float:
        """Validate domain overlap score."""
        return RobustScoreValidator.validate_probability_score(
            v, "domain_overlap", default_value=1.0
        )

    def calculate_comprehensive_severity(
        self,
        finding_1_confidence: float,
        finding_2_confidence: float,
        source_1_credibility: float,
        source_2_credibility: float,
        source_1_type: str | None = None,
        source_2_type: str | None = None,
        importance_1: float = 0.5,
        importance_2: float = 0.5,
    ) -> None:
        """
        Calculate comprehensive severity using all available factors.

        This method updates the severity, severity_components, resolution_strategy,
        priority, and severity_level fields based on comprehensive analysis.

        Args:
            finding_1_confidence: Confidence score of first finding
            finding_2_confidence: Confidence score of second finding
            source_1_credibility: Credibility score of first source
            source_2_credibility: Credibility score of second source
            source_1_type: Type of first source (academic, news, etc.)
            source_2_type: Type of second source
            importance_1: Importance score of first finding
            importance_2: Importance score of second finding
        """
        # Calculate severity using comprehensive calculator
        severity_result = ContradictionSeverityCalculator.calculate_contradiction_severity(
            contradiction_type=self.contradiction_type,
            finding_1_confidence=finding_1_confidence,
            finding_2_confidence=finding_2_confidence,
            source_1_credibility=source_1_credibility,
            source_2_credibility=source_2_credibility,
            source_1_type=source_1_type,
            source_2_type=source_2_type,
            domain_overlap=self.domain_overlap,
            temporal_distance_days=self.temporal_distance_days,
            importance_1=importance_1,
            importance_2=importance_2,
        )

        # Update fields based on calculation
        self.severity = severity_result["overall_severity"]
        self.severity_components = severity_result
        self.severity_level = ContradictionSeverityCalculator.classify_severity_level(self.severity)
        self.priority = ContradictionSeverityCalculator.get_resolution_priority(
            self.severity, self.contradiction_type
        )
        metadata = severity_result.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        self.resolution_strategy = ContradictionSeverityCalculator.suggest_resolution_strategy(
            self.contradiction_type, self.severity, metadata
        )

    def needs_immediate_resolution(self) -> bool:
        """Check if contradiction needs immediate resolution."""
        return (
            self.severity > 0.7
            or self.contradiction_type.lower() == "direct"
            or self.priority <= 2
            or self.severity_level in ["critical", "high"]
        )

    def get_severity_summary(self) -> dict[str, Any]:
        """Get a comprehensive summary of the contradiction severity."""
        return {
            "overall_severity": self.severity,
            "severity_level": self.severity_level,
            "priority": self.priority,
            "contradiction_type": self.contradiction_type,
            "needs_immediate_resolution": self.needs_immediate_resolution(),
            "domain_overlap": self.domain_overlap,
            "temporal_distance_days": self.temporal_distance_days,
            "resolution_timeline": self.resolution_strategy.get("timeline", "Unknown"),
            "estimated_effort": self.resolution_strategy.get("estimated_effort", "Unknown"),
        }

    def get_resolution_actions(self) -> list[str]:
        """Get prioritized list of resolution actions."""
        actions = []

        # Add strategy-specific actions
        if "actions" in self.resolution_strategy:
            actions.extend(self.resolution_strategy["actions"])

        # Add type-specific actions
        if "type_specific_actions" in self.resolution_strategy:
            actions.extend(self.resolution_strategy["type_specific_actions"])

        return actions

    def to_markdown(self) -> str:
        """Convert contradiction to markdown format for reporting."""
        lines = [
            f"### {self.contradiction_type.title()} Contradiction",
            f"**Severity:** {self.severity:.2f} ({self.severity_level.title()})",
            f"**Priority:** {self.priority}/5",
            f"**Findings:** {self.finding_1_id} ↔ {self.finding_2_id}",
            "",
            f"**Explanation:** {self.explanation}",
        ]

        if self.resolution_hint:
            lines.extend(["", f"**Resolution Hint:** {self.resolution_hint}"])

        if self.resolution_strategy:
            lines.extend(
                [
                    "",
                    "**Resolution Strategy:**",
                    f"- **Approach:** {self.resolution_strategy.get('strategy', 'Unknown')}",
                    f"- **Timeline:** {self.resolution_strategy.get('timeline', 'Unknown')}",
                    f"- **Effort:** {self.resolution_strategy.get('estimated_effort', 'Unknown')}",
                ]
            )

            actions = self.get_resolution_actions()
            if actions:
                lines.extend(["", "**Recommended Actions:**"])
                lines.extend([f"- {action}" for action in actions[:5]])  # Limit to top 5

        # Add severity breakdown if available
        if self.severity_components and "components" in self.severity_components:
            lines.extend(["", "**Severity Breakdown:**"])
            components = self.severity_components["components"]
            for key, value in components.items():
                if isinstance(value, float):
                    lines.append(f"- {key.replace('_', ' ').title()}: {value:.2f}")

        return "\n".join(lines)


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

    # Pattern analysis integration (Phase 4 enhancement)
    patterns: list[PatternAnalysis] = Field(
        default_factory=list, description="Detected patterns in research findings"
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

    # Content hierarchy (Phase 4 enhancement)
    content_hierarchy: dict[str, Any] = Field(
        default_factory=dict, description="Hierarchical organization of content by priority levels"
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

    def get_hierarchical_structure(self) -> dict[str, list[HierarchicalFinding]]:
        """Returns organized content by importance level."""
        hierarchy = {"Critical": [], "Important": [], "Supplementary": [], "Contextual": []}

        for finding in self.findings:
            # Map ImportanceLevel enum to string keys
            if finding.importance == ImportanceLevel.CRITICAL:
                hierarchy["Critical"].append(finding)
            elif finding.importance == ImportanceLevel.HIGH:
                hierarchy["Important"].append(finding)
            elif finding.importance == ImportanceLevel.MEDIUM:
                hierarchy["Supplementary"].append(finding)
            elif finding.importance == ImportanceLevel.LOW:
                hierarchy["Contextual"].append(finding)

        return hierarchy

    def get_convergence_rate(self) -> float:
        """Calculate percentage of findings with multiple source agreement."""
        if not self.findings:
            return 0.0

        # Count findings with multiple supporting evidence items
        multi_source_findings = sum(1 for f in self.findings if len(f.supporting_evidence) > 1)
        return round(multi_source_findings / len(self.findings), 3)

    def get_contradiction_rate(self) -> float:
        """
        Calculate weighted contradiction rate based on severity.

        Returns:
            Weighted contradiction rate (0.0-1.0) where severity influences the impact
        """
        if not self.findings:
            return 0.0

        if not self.contradictions:
            # Check if any findings mention contradictions in their metadata
            metadata_contradictions = sum(
                1 for f in self.findings if f.metadata.get("has_contradictions", False)
            )
            return round(metadata_contradictions / len(self.findings), 3)

        # Calculate weighted contradiction rate based on severity
        total_severity_impact = 0.0
        for contradiction in self.contradictions:
            # Severity acts as a multiplier for impact
            severity_weight = contradiction.severity
            total_severity_impact += severity_weight

        # Normalize by maximum possible impact (if all findings had critical contradictions)
        max_possible_impact = len(self.findings)
        if max_possible_impact > 0:
            weighted_rate = total_severity_impact / max_possible_impact
        else:
            weighted_rate = 0.0

        return round(min(weighted_rate, 1.0), 3)

    def calculate_all_contradiction_severities(self) -> None:
        """
        Calculate comprehensive severity for all contradictions using available data.

        This method updates all contradictions with proper severity calculations
        based on the findings and sources they reference.
        """
        if not self.contradictions or not self.findings:
            return

        # Create lookup dictionary for efficient access
        findings_by_id = {str(i): finding for i, finding in enumerate(self.findings)}

        for contradiction in self.contradictions:
            try:
                # Get the referenced findings
                finding_1 = findings_by_id.get(contradiction.finding_1_id)
                finding_2 = findings_by_id.get(contradiction.finding_2_id)

                if not finding_1 or not finding_2:
                    continue

                # Get source information
                source_1 = finding_1.source if hasattr(finding_1, "source") else None
                source_2 = finding_2.source if hasattr(finding_2, "source") else None

                # Extract parameters for severity calculation
                finding_1_confidence = getattr(finding_1, "confidence_score", 0.5)
                finding_2_confidence = getattr(finding_2, "confidence_score", 0.5)

                source_1_credibility = source_1.credibility_score if source_1 else 0.5
                source_2_credibility = source_2.credibility_score if source_2 else 0.5

                source_1_type = source_1.source_type if source_1 else None
                source_2_type = source_2.source_type if source_2 else None

                importance_1 = getattr(finding_1, "importance_score", 0.5)
                importance_2 = getattr(finding_2, "importance_score", 0.5)

                # Calculate temporal distance if possible
                temporal_distance = 0.0
                if source_1 and source_1.date and source_2 and source_2.date:
                    time_diff = abs((source_1.date - source_2.date).days)
                    temporal_distance = float(time_diff)

                # Calculate comprehensive severity
                contradiction.temporal_distance_days = temporal_distance
                contradiction.calculate_comprehensive_severity(
                    finding_1_confidence=finding_1_confidence,
                    finding_2_confidence=finding_2_confidence,
                    source_1_credibility=source_1_credibility,
                    source_2_credibility=source_2_credibility,
                    source_1_type=source_1_type,
                    source_2_type=source_2_type,
                    importance_1=importance_1,
                    importance_2=importance_2,
                )

            except Exception as e:
                # Log error but continue processing other contradictions
                import logfire

                logfire.warning(f"Failed to calculate severity for contradiction: {e}")
                continue

    def get_contradiction_summary(self) -> dict[str, Any]:
        """
        Get comprehensive summary of all contradictions with severity analysis.

        Returns:
            Dictionary with contradiction statistics and severity breakdown
        """
        if not self.contradictions:
            return {
                "total_contradictions": 0,
                "weighted_contradiction_rate": 0.0,
                "severity_distribution": {},
                "high_priority_count": 0,
                "immediate_resolution_needed": 0,
            }

        # Calculate severity statistics
        severity_levels = {}
        priority_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        immediate_resolution_count = 0
        total_severity = 0.0

        for contradiction in self.contradictions:
            # Count by severity level
            level = contradiction.severity_level
            severity_levels[level] = severity_levels.get(level, 0) + 1

            # Count by priority
            priority = getattr(contradiction, "priority", 3)
            priority_counts[priority] = priority_counts.get(priority, 0) + 1

            # Count immediate resolution needs
            if contradiction.needs_immediate_resolution():
                immediate_resolution_count += 1

            # Sum total severity
            total_severity += contradiction.severity

        return {
            "total_contradictions": len(self.contradictions),
            "weighted_contradiction_rate": self.get_contradiction_rate(),
            "average_severity": round(total_severity / len(self.contradictions), 3),
            "severity_distribution": severity_levels,
            "priority_distribution": priority_counts,
            "high_priority_count": priority_counts[1] + priority_counts[2],
            "immediate_resolution_needed": immediate_resolution_count,
            "most_severe_contradiction": max(self.contradictions, key=lambda c: c.severity).severity
            if self.contradictions
            else 0.0,
        }

    def get_source_diversity(self) -> float:
        """
        Calculate normalized Shannon entropy of source domains.

        This method now returns properly normalized Shannon entropy (0.0 to 1.0)
        instead of raw entropy values that could exceed expected bounds.

        Returns:
            Normalized Shannon entropy (0.0 to 1.0):
            - 0.0: All sources from same domain (no diversity)
            - 1.0: Maximum possible diversity given the number of unique domains
        """
        if not self.sources:
            return 0.0

        # Extract domains/types from sources
        domains = []
        for source in self.sources:
            if source.url and "://" in source.url:
                domain = source.url.split("://")[1].split("/")[0]
                domains.append(domain)
            elif source.source_type:
                domains.append(source.source_type)
            else:
                domains.append("unknown")

        if not domains:
            return 0.0

        # Use the robust mathematical validator for proper normalization
        try:
            from utils.validation import MathematicalValidator

            return MathematicalValidator.calculate_normalized_shannon_entropy(domains)
        except ImportError:
            # Fallback to local implementation if utils not available
            return self._calculate_shannon_entropy_fallback(domains)

    def _calculate_shannon_entropy_fallback(self, domains: list[str]) -> float:
        """Fallback Shannon entropy calculation with proper normalization."""
        domain_counts = Counter(domains)
        total_sources = len(domains)
        unique_domains = len(domain_counts)

        if unique_domains <= 1:
            return 0.0

        # Calculate Shannon entropy
        entropy = 0.0
        for count in domain_counts.values():
            if count > 0:
                probability = count / total_sources
                entropy -= probability * math.log2(probability)

        # Theoretical maximum entropy for this dataset
        max_entropy = math.log2(unique_domains)

        # Normalize to [0, 1] range - THIS IS THE KEY FIX
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return round(normalized_entropy, 3)

    def calculate_comprehensive_quality(self) -> float:
        """
        Calculate comprehensive quality score considering multiple factors.

        Uses robust validation to handle edge cases and ensure all components
        are properly validated before calculation.

        Returns:
            Quality score between 0.0 and 1.0
        """
        scores: list[float] = []
        weights: list[float] = []

        # Source diversity (Shannon entropy) - now properly normalized 0-1
        diversity_score = self.get_source_diversity()
        convergence_rate = self.get_convergence_rate()
        contradiction_rate = self.get_contradiction_rate()

        # Calculate pattern strength average
        pattern_strength = None
        if self.patterns:
            pattern_strength = sum(p.strength for p in self.patterns) / len(self.patterns)

        # Calculate confidence average
        confidence_avg = None
        if self.findings:
            confidence_avg = sum(f.confidence_score for f in self.findings) / len(self.findings)

        # Use robust validation for all components
        try:
            from utils.validation import MathematicalValidator

            validated_components = MathematicalValidator.validate_quality_score_components(
                diversity_score=diversity_score,
                convergence_rate=convergence_rate,
                contradiction_rate=contradiction_rate,
                pattern_strength=pattern_strength,
                confidence_avg=confidence_avg,
            )
        except ImportError:
            # Fallback validation if utils not available
            validated_components = self._validate_components_fallback(
                diversity_score,
                convergence_rate,
                contradiction_rate,
                pattern_strength,
                confidence_avg,
            )

        # Build weighted score components
        if validated_components["diversity"] > 0:
            scores.append(validated_components["diversity"])
            weights.append(0.2)

        scores.append(validated_components["convergence"])
        weights.append(0.25)

        # Resolution score (lower contradiction rate is better)
        resolution_score = 1.0 - validated_components["contradiction"]
        scores.append(resolution_score)
        weights.append(0.15)

        if "pattern_strength" in validated_components:
            scores.append(validated_components["pattern_strength"])
            weights.append(0.2)

        if "confidence" in validated_components:
            scores.append(validated_components["confidence"])
            weights.append(0.2)

        # Calculate weighted average
        if scores and weights:
            total_weight = sum(weights[: len(scores)])
            weighted_score = (
                sum(s * w for s, w in zip(scores, weights, strict=False)) / total_weight
            )
            return round(max(0.0, min(1.0, weighted_score)), 3)

        return 0.0

    def _validate_components_fallback(
        self,
        diversity_score: float,
        convergence_rate: float,
        contradiction_rate: float,
        pattern_strength: float | None = None,
        confidence_avg: float | None = None,
    ) -> dict[str, float]:
        """Fallback validation for quality score components."""
        components = {}

        # Basic validation with clamping
        components["diversity"] = (
            max(0.0, min(1.0, diversity_score))
            if not math.isnan(diversity_score) and not math.isinf(diversity_score)
            else 0.0
        )
        components["convergence"] = (
            max(0.0, min(1.0, convergence_rate))
            if not math.isnan(convergence_rate) and not math.isinf(convergence_rate)
            else 0.0
        )
        components["contradiction"] = (
            max(0.0, min(1.0, contradiction_rate))
            if not math.isnan(contradiction_rate) and not math.isinf(contradiction_rate)
            else 0.0
        )

        if pattern_strength is not None:
            components["pattern_strength"] = (
                max(0.0, min(1.0, pattern_strength))
                if not math.isnan(pattern_strength) and not math.isinf(pattern_strength)
                else 0.0
            )

        if confidence_avg is not None:
            components["confidence"] = (
                max(0.0, min(1.0, confidence_avg))
                if not math.isnan(confidence_avg) and not math.isinf(confidence_avg)
                else 0.0
            )

        return components

    def export_to_json(self) -> str:
        """Export results in structured JSON format."""
        hierarchy = self.get_hierarchical_structure()

        export_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "query": self.query,
            "execution_time": self.execution_time.isoformat(),
            "overall_quality_score": self.calculate_comprehensive_quality(),
            "metrics": {
                "convergence_rate": self.get_convergence_rate(),
                "contradiction_rate": self.get_contradiction_rate(),
                "source_diversity": self.get_source_diversity(),
                "total_findings": len(self.findings),
                "total_sources": len(self.sources),
                "total_patterns": len(self.patterns),
                "total_themes": len(self.theme_clusters),
            },
            "hierarchy": {
                level: [
                    {
                        "finding": f.finding,
                        "confidence": f.confidence_score,
                        "importance": f.importance_score,
                        "evidence_count": len(f.supporting_evidence),
                    }
                    for f in findings
                ]
                for level, findings in hierarchy.items()
            },
            "patterns": [
                {
                    "type": p.pattern_type.value
                    if hasattr(p.pattern_type, "value")
                    else str(p.pattern_type),
                    "name": p.pattern_name,
                    "description": p.description,
                    "strength": p.strength,
                }
                for p in self.patterns
            ],
            "themes": [
                {
                    "name": t.theme_name,
                    "description": t.description,
                    "coherence": t.coherence_score,
                    "importance": t.importance_score,
                    "finding_count": len(t.findings),
                }
                for t in self.theme_clusters
            ],
            "data_gaps": self.data_gaps,
            "key_insights": self.key_insights,
            "content_hierarchy": self.content_hierarchy,
        }

        return json.dumps(export_data, indent=2, default=str)

    def to_report(self) -> str:
        """Generate comprehensive markdown report with all sections."""
        sections = []
        timestamp = self.execution_time.strftime("%Y-%m-%d %H:%M:%S")

        # Header
        sections.append(f"# Research Report: {self.query}")
        sections.append(f"\n*Generated: {timestamp}*\n")

        # Executive Summary
        sections.append("## Executive Summary\n")

        # Calculate quality metrics
        quality_score = self.calculate_comprehensive_quality()
        convergence_rate = self.get_convergence_rate()
        contradiction_rate = self.get_contradiction_rate()
        source_diversity = self.get_source_diversity()

        # Quality metrics summary
        sections.append("### Research Quality Metrics\n")
        sections.append(f"- **Overall Quality Score**: {quality_score:.1%}")
        sections.append(f"- **Source Diversity**: {source_diversity:.2f} (Shannon entropy)")
        sections.append(f"- **Finding Convergence**: {convergence_rate:.1%}")
        sections.append(f"- **Contradiction Rate**: {contradiction_rate:.1%}")
        sections.append(f"- **Total Findings**: {len(self.findings)}")
        sections.append(f"- **Total Sources**: {len(self.sources)}\n")

        # Key insights
        if self.key_insights:
            sections.append("### Key Insights\n")
            for insight in self.key_insights:
                sections.append(f"- {insight}")
            sections.append("")

        # Executive summary content
        if self.executive_summary:
            sections.append(self.executive_summary.to_markdown())
            sections.append("")

        # Hierarchical Findings
        sections.append("## Hierarchical Findings\n")
        hierarchy = self.get_hierarchical_structure()

        for level in ["Critical", "Important", "Supplementary", "Contextual"]:
            findings = hierarchy.get(level, [])
            if findings:
                sections.append(f"### {level} Findings\n")
                for i, finding in enumerate(findings, 1):
                    sections.append(f"**Finding {i}** (Confidence: {finding.confidence_score:.1%})")
                    sections.append(f"\n{finding.finding}\n")

                    if finding.supporting_evidence:
                        sections.append("*Supporting Evidence:*")
                        for evidence in finding.supporting_evidence[:3]:
                            sections.append(f"- {evidence}")
                        if len(finding.supporting_evidence) > 3:
                            sections.append(
                                f"- *... and {len(finding.supporting_evidence) - 3} more*"
                            )
                        sections.append("")

                    if finding.source:
                        sections.append(f"*Source*: {finding.source.title}")
                        if finding.source.url:
                            sections.append(f"  URL: {finding.source.url}")
                        sections.append("")

        # Theme Clusters
        if self.theme_clusters:
            sections.append("## Theme Clusters\n")
            for theme in sorted(
                self.theme_clusters, key=lambda t: t.importance_score, reverse=True
            ):
                sections.append(f"### {theme.theme_name}")
                coherence_text = f"Coherence: {theme.coherence_score:.1%}"
                importance_text = f"Importance: {theme.importance_score:.1%}"
                sections.append(f"\n*{coherence_text} | {importance_text}*\n")
                sections.append(f"{theme.description}\n")
                sections.append(f"**Related Findings**: {len(theme.findings)} items")
                sections.append(f"**Average Confidence**: {theme.average_confidence():.1%}\n")

        # Pattern Analysis
        if self.patterns:
            sections.append("## Pattern Analysis\n")

            # Group patterns by type
            pattern_groups = {}
            for pattern in self.patterns:
                ptype = str(pattern.pattern_type)
                if ptype not in pattern_groups:
                    pattern_groups[ptype] = []
                pattern_groups[ptype].append(pattern)

            for ptype, patterns_list in pattern_groups.items():
                sections.append(f"### {ptype.replace('_', ' ').title()} Patterns\n")
                for pattern in sorted(patterns_list, key=lambda p: p.strength, reverse=True):
                    sections.append(f"**{pattern.pattern_name}**")
                    sections.append(f"{pattern.description}")
                    sections.append(f"- Strength: {pattern.strength:.1%}")

                    if pattern.implications:
                        sections.append("- Implications:")
                        for implication in pattern.implications[:2]:
                            sections.append(f"  - {implication}")
                    sections.append("")

        # Contradictions and Resolutions
        if self.contradictions:
            sections.append("## Contradictions and Resolutions\n")
            for cont in self.contradictions:
                sections.append("### Contradiction")
                sections.append(f"**Type**: {cont.contradiction_type}")
                sections.append(f"**Severity**: {cont.severity:.1%}\n")
                sections.append(f"{cont.explanation}\n")
                if cont.resolution_hint:
                    sections.append(f"*Resolution Hint*: {cont.resolution_hint}\n")

        # Data Gaps and Limitations
        if self.data_gaps:
            sections.append("## Data Gaps and Limitations\n")
            for gap in self.data_gaps:
                sections.append(f"- {gap}")
            sections.append("")

        # Sources and Citations
        if self.sources:
            sections.append("## Sources and Citations\n")
            sections.append(f"*Total sources consulted: {len(self.sources)}*\n")

            # Group by source type
            type_groups = {}
            for source in self.sources:
                stype = source.source_type or "Other"
                if stype not in type_groups:
                    type_groups[stype] = []
                type_groups[stype].append(source)

            for stype, sources_list in sorted(type_groups.items()):
                sections.append(f"### {stype.title()}\n")
                for source in sources_list[:5]:
                    sections.append(f"- **{source.title}**")
                    if source.url:
                        sections.append(f"  - URL: {source.url}")
                    if source.credibility_score != 0.5:
                        sections.append(f"  - Credibility: {source.credibility_score:.1%}")
                if len(sources_list) > 5:
                    sections.append(f"- *... and {len(sources_list) - 5} more*")
                sections.append("")

        # Quality Metrics Summary Table
        sections.append("## Quality Metrics Summary\n")
        sections.append("| Metric | Value | Status |")
        sections.append("|--------|-------|--------|")
        if quality_score > 0.7:
            quality_status = "✅ Good"
        elif quality_score > 0.4:
            quality_status = "⚠️ Moderate"
        else:
            quality_status = "❌ Low"
        sections.append(f"| Overall Quality | {quality_score:.1%} | {quality_status} |")
        if source_diversity > 1.5:
            diversity_status = "✅ High"
        elif source_diversity > 0.5:
            diversity_status = "⚠️ Medium"
        else:
            diversity_status = "❌ Low"
        sections.append(f"| Source Diversity | {source_diversity:.2f} | {diversity_status} |")
        if convergence_rate > 0.6:
            convergence_status = "✅ Strong"
        elif convergence_rate > 0.3:
            convergence_status = "⚠️ Moderate"
        else:
            convergence_status = "❌ Weak"
        sections.append(f"| Convergence Rate | {convergence_rate:.1%} | {convergence_status} |")
        if contradiction_rate < 0.2:
            contradiction_status = "✅ Low"
        elif contradiction_rate < 0.4:
            contradiction_status = "⚠️ Medium"
        else:
            contradiction_status = "❌ High"
        contradiction_line = (
            f"| Contradiction Rate | {contradiction_rate:.1%} | {contradiction_status} |"
        )
        sections.append(contradiction_line)
        findings_count = len(self.findings)
        if findings_count > 10:
            findings_status = "✅ Comprehensive"
        elif findings_count > 5:
            findings_status = "⚠️ Adequate"
        else:
            findings_status = "❌ Limited"
        sections.append(f"| Total Findings | {findings_count} | {findings_status} |")
        patterns_count = len(self.patterns)
        if patterns_count > 5:
            patterns_status = "✅ Rich"
        elif patterns_count > 2:
            patterns_status = "⚠️ Some"
        else:
            patterns_status = "❌ Few"
        sections.append(f"| Pattern Count | {patterns_count} | {patterns_status} |")
        sections.append("")

        # Metadata Footer
        sections.append("---\n")
        sections.append("## Metadata\n")
        sections.append(f"- **Report Generated**: {timestamp}")
        sections.append(f"- **Query**: {self.query}")
        sections.append(f"- **Total Sources**: {len(self.sources)}")
        sections.append(f"- **Total Findings**: {len(self.findings)}")
        sections.append(f"- **Total Themes**: {len(self.theme_clusters)}")
        sections.append(f"- **Total Patterns**: {len(self.patterns)}")
        sections.append(f"- **Quality Score**: {quality_score:.3f}")

        if self.synthesis_metadata:
            sections.append(f"- **Synthesis Method**: {self.synthesis_metadata.synthesis_method}")
            sections.append(
                f"- **Processing Time**: {self.synthesis_metadata.processing_time_seconds:.1f}s"
            )

        if self.metadata:
            sections.append("\n### Additional Metadata")
            for key, value in self.metadata.items():
                sections.append(f"- **{key}**: {value}")

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


# Enhanced Models for Service Integration


class EvidenceItem(BaseModel):
    """Individual piece of evidence for research synthesis."""

    content: str = Field(description="Content of the evidence")
    source: ResearchSource | None = Field(default=None, description="Source of the evidence")
    timestamp: float | None = Field(
        default=None, description="Timestamp when evidence was collected"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0, default=0.5, description="Confidence in this evidence"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ResearchQuery(BaseModel):
    """Research query with parameters."""

    query: str = Field(description="The research query text")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Query parameters")
    context: str | None = Field(default=None, description="Additional context for the query")


class Pattern(BaseModel):
    """Detected pattern in research data."""

    id: str = Field(description="Unique pattern identifier")
    type: PatternType = Field(description="Type of pattern")
    description: str = Field(description="Description of the pattern")
    evidence_indices: list[int] = Field(description="Indices of evidence items involved")
    strength: float = Field(ge=0.0, le=1.0, description="Strength of the pattern")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Pattern metadata")


class ContradictionType(str, Enum):
    """Types of contradictions that can be detected."""

    SEMANTIC = "semantic"
    FACTUAL = "factual"
    STATISTICAL = "statistical"
    TEMPORAL = "temporal"


class Contradiction(BaseModel):
    """Detected contradiction between evidence items."""

    id: str = Field(description="Unique contradiction identifier")
    type: ContradictionType = Field(description="Type of contradiction")
    evidence_indices: list[int] = Field(description="Indices of conflicting evidence")
    description: str = Field(description="Description of the contradiction")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in this contradiction")
    resolution_suggestion: str = Field(description="Suggestion for resolving the contradiction")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Contradiction metadata")


class ContradictionAnalysis(BaseModel):
    """Analysis of contradictions in evidence."""

    evidence_items: list[EvidenceItem] = Field(description="Evidence items analyzed")
    contradictions: list[Contradiction] = Field(description="Detected contradictions")
    resolution_strategy: str = Field(description="Strategy for resolving contradictions")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Overall confidence in analysis")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Analysis metadata")


class PatternRecognitionRequest(BaseModel):
    """Request for pattern recognition analysis."""

    evidence_items: list[EvidenceItem] = Field(description="Evidence items to analyze")
    patterns: list[Pattern] = Field(description="Detected patterns")
    pattern_types: list[str] = Field(description="Types of patterns to detect")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Request metadata")


class ResearchSynthesis(BaseModel):
    """Result of research synthesis."""

    summary: str = Field(description="Summary of the synthesis")
    key_themes: list[dict[str, Any]] = Field(description="Key themes identified")
    evidence_items: list[EvidenceItem] = Field(description="Evidence items used")
    confidence_metrics: dict[str, float] = Field(description="Confidence metrics")


class SynthesisContext(BaseModel):
    """Context for synthesis operations."""

    evidence_items: list[EvidenceItem] = Field(description="Evidence items to synthesize")
    research_query: ResearchQuery | None = Field(default=None, description="Research query")
    config: "SynthesisConfig | None" = Field(default=None, description="Synthesis configuration")


class SynthesisResult(BaseModel):
    """Result of synthesis operation."""

    synthesis: ResearchSynthesis = Field(description="The synthesis result")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in synthesis")
    processing_time: float = Field(description="Time taken for synthesis")
    evidence_coverage: float = Field(ge=0.0, le=1.0, description="Coverage of evidence")


class SynthesisConfig(BaseModel):
    """Configuration for synthesis engine."""

    max_features: int = Field(default=5000, description="Maximum TF-IDF features")
    similarity_threshold: float = Field(default=0.7, description="Similarity threshold")
    max_themes: int = Field(default=10, description="Maximum themes to extract")
    max_evidence_items: int = Field(default=100, description="Maximum evidence items to process")


class CacheConfig(BaseModel):
    """Configuration for cache manager."""

    max_memory_mb: int = Field(default=100, description="Maximum memory usage in MB")
    default_ttl: float = Field(default=3600, description="Default TTL in seconds")
    cleanup_interval: int = Field(default=300, description="Cleanup interval in seconds")


class CacheStats(BaseModel):
    """Cache statistics."""

    hits: int = Field(default=0, description="Cache hits")
    misses: int = Field(default=0, description="Cache misses")
    sets: int = Field(default=0, description="Cache sets")
    evictions: int = Field(default=0, description="Cache evictions")
    expired_entries: int = Field(default=0, description="Expired entries")
    deletions: int = Field(default=0, description="Deletions")
    cleared_entries: int = Field(default=0, description="Cleared entries")


class CacheMetrics(BaseModel):
    """Cache performance metrics."""

    hit_rate: float = Field(description="Cache hit rate")
    total_entries: int = Field(description="Total cache entries")
    memory_usage_mb: float = Field(description="Memory usage in MB")
    total_hits: int = Field(description="Total hits")
    total_misses: int = Field(description="Total misses")
    total_sets: int = Field(description="Total sets")
    total_evictions: int = Field(description="Total evictions")


class ResearchExecutorConfig(BaseModel):
    """Configuration for research executor."""

    synthesis_config: SynthesisConfig = Field(default_factory=SynthesisConfig)
    cache_config: CacheConfig = Field(default_factory=CacheConfig)
    enable_parallel_processing: bool = Field(default=True)
    max_concurrent_tasks: int = Field(default=4)


class ResearchExecutorDeps(BaseModel):
    """Dependencies for research executor."""

    config: ResearchExecutorConfig = Field(description="Executor configuration")
    cache_manager: Any = Field(description="Cache manager instance")
    synthesis_engine: Any = Field(description="Synthesis engine instance")
    pattern_recognizer: Any = Field(description="Pattern recognizer instance")
    contradiction_detector: Any = Field(description="Contradiction detector instance")


# Phase 2 Models for Pattern Analysis


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
