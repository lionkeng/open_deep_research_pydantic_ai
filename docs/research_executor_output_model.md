# Research Executor Enhanced Output Model

## Overview
This document defines the validated output shape from the Research Executor Agent that will be passed directly to the Report Generator in the 4-agent architecture.

## Enhanced ResearchResults Model

```python
from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field

class ResearchSource(BaseModel):
    """Enhanced source with credibility assessment."""
    url: str | None = Field(default=None)
    title: str
    author: str | None = Field(default=None)
    date: datetime | None = Field(default=None)
    relevance_score: float = Field(ge=0.0, le=1.0)
    credibility_tier: int = Field(ge=1, le=4, description="1=highest, 4=lowest credibility")
    source_type: str  # "academic", "industry", "news", "documentation", etc.

class HierarchicalFinding(BaseModel):
    """Finding with hierarchical classification."""
    finding: str
    supporting_evidence: list[str]
    confidence_level: float = Field(ge=0.0, le=1.0)
    source: ResearchSource | None
    category: str | None
    # New hierarchical fields
    importance_level: Literal["critical", "important", "supplementary", "contextual"]
    theme_cluster: str  # Which theme this belongs to
    sub_findings: list[str] = Field(default_factory=list)  # Related sub-points

class ThemeCluster(BaseModel):
    """Organized theme with related findings."""
    theme_name: str
    description: str
    finding_ids: list[int]  # Indices into findings list
    confidence: float = Field(ge=0.0, le=1.0)
    consensus_level: Literal["strong", "moderate", "weak", "conflicting"]
    key_insights: list[str]

class PatternAnalysis(BaseModel):
    """Identified patterns across research."""
    pattern_type: Literal["convergence", "divergence", "emergence", "temporal"]
    description: str
    supporting_findings: list[int]  # Indices into findings list
    confidence: float = Field(ge=0.0, le=1.0)
    implications: list[str]

class SynthesisMetadata(BaseModel):
    """Metadata from synthesis process."""
    compression_ratio: float
    information_retention: float
    synthesis_approach: str  # "technical", "business", "narrative"
    removed_content_categories: list[str]
    verification_checklist: dict[str, bool]
    synthesis_timestamp: datetime

class ExecutiveSummary(BaseModel):
    """Structured executive summary."""
    key_findings: list[str]  # 3-5 bullet points
    overall_confidence: float = Field(ge=0.0, le=1.0)
    critical_gaps: list[str]
    immediate_insights: list[str]
    strategic_implications: list[str]

class ResearchResults(BaseModel):
    """Enhanced output from Research Executor with synthesis."""

    # Original fields (maintained for compatibility)
    query: str
    execution_time: datetime
    sources: list[ResearchSource]
    metadata: dict[str, Any]

    # Enhanced hierarchical findings
    findings: list[HierarchicalFinding]

    # New synthesis outputs
    executive_summary: ExecutiveSummary
    theme_clusters: list[ThemeCluster]
    pattern_analysis: list[PatternAnalysis]

    # Aggregated insights
    key_insights: list[str]  # Top-level insights from synthesis
    actionable_recommendations: list[str]  # Direct recommendations
    data_gaps: list[str]  # What couldn't be determined

    # Quality and confidence metrics
    quality_score: float = Field(ge=0.0, le=1.0)
    confidence_metrics: dict[str, float] = Field(
        description="Confidence by category/theme"
    )
    coverage_assessment: dict[str, float] = Field(
        description="How well each aspect was covered"
    )

    # Synthesis metadata
    synthesis_metadata: SynthesisMetadata

    # For Report Generator navigation
    content_hierarchy: dict[str, Any] = Field(
        description="Hierarchical structure for report sections"
    )
```

## Key Enhancements

### 1. Hierarchical Organization
- Findings are classified by importance level (critical → contextual)
- Theme clusters group related findings
- Content hierarchy provides structure for report generation

### 2. Pattern Recognition
- Convergence: Where sources agree
- Divergence: Where sources conflict
- Emergence: New trends identified
- Temporal: Changes over time

### 3. Synthesis Outputs
- Executive summary with structured components
- Actionable recommendations separated from insights
- Confidence metrics by category/theme
- Coverage assessment for completeness

### 4. Quality Metrics
- Overall quality score
- Information retention ratio
- Verification checklist results
- Confidence levels at multiple granularities

## Data Flow to Report Generator

The Report Generator receives this enhanced structure and:

1. **Uses ExecutiveSummary** → Populates report's executive_summary field
2. **Uses ThemeClusters** → Creates main report sections
3. **Uses HierarchicalFindings** → Populates section content with proper importance weighting
4. **Uses PatternAnalysis** → Enriches conclusions section
5. **Uses ActionableRecommendations** → Populates recommendations section
6. **Uses Sources** → Builds references section
7. **Uses ContentHierarchy** → Structures nested subsections
8. **Uses QualityMetrics** → Provides confidence indicators throughout

## Example Output Structure

```python
{
    "query": "AI adoption in healthcare 2024",
    "execution_time": "2024-01-15T10:30:00Z",

    "executive_summary": {
        "key_findings": [
            "70% of hospitals have adopted AI for diagnostics",
            "ROI averages 3.2x within 18 months",
            "Regulatory compliance remains primary barrier"
        ],
        "overall_confidence": 0.85,
        "critical_gaps": ["Limited data on small clinics"],
        "immediate_insights": ["AI adoption accelerating post-2023"],
        "strategic_implications": ["First-mover advantage closing"]
    },

    "theme_clusters": [
        {
            "theme_name": "Diagnostic Applications",
            "description": "AI use in medical imaging and diagnosis",
            "finding_ids": [0, 2, 5, 8],
            "confidence": 0.92,
            "consensus_level": "strong",
            "key_insights": ["Radiology leads adoption at 85%"]
        }
    ],

    "findings": [
        {
            "finding": "AI reduces diagnostic errors by 35%",
            "supporting_evidence": ["Study of 10,000 cases", "FDA report 2024"],
            "confidence_level": 0.88,
            "importance_level": "critical",
            "theme_cluster": "Diagnostic Applications"
        }
    ],

    "pattern_analysis": [
        {
            "pattern_type": "emergence",
            "description": "Shift from rule-based to ML systems",
            "supporting_findings": [1, 3, 7],
            "confidence": 0.79,
            "implications": ["Need for ML expertise growing"]
        }
    ],

    "actionable_recommendations": [
        "Prioritize AI training for radiologists",
        "Establish data governance framework",
        "Partner with AI vendors for pilot programs"
    ],

    "confidence_metrics": {
        "diagnostic_applications": 0.92,
        "administrative_efficiency": 0.78,
        "patient_outcomes": 0.81
    },

    "synthesis_metadata": {
        "compression_ratio": 0.68,
        "information_retention": 0.95,
        "synthesis_approach": "technical",
        "verification_checklist": {
            "facts_preserved": true,
            "logic_maintained": true,
            "ambiguity_check": true
        }
    }
}
```

## Validation Requirements

The Research Executor must ensure:

1. **Completeness**: All required fields populated
2. **Consistency**: Finding IDs correctly reference list indices
3. **Quality Thresholds**:
   - Minimum 3 findings per theme cluster
   - Quality score > 0.6 for valid output
   - At least 1 finding per importance level
4. **Relationship Integrity**:
   - Theme clusters reference valid finding indices
   - Pattern analysis references valid findings
   - All sources properly attributed

## Benefits for Report Generator

1. **Direct Section Mapping**: Theme clusters → Report sections
2. **Pre-Organized Content**: Hierarchical findings reduce processing
3. **Quality Indicators**: Confidence metrics guide emphasis
4. **Complete Context**: All synthesis work already done
5. **Flexible Structure**: Can generate various report formats

This enhanced output structure eliminates the need for a separate Compression Agent while providing richer, more organized data to the Report Generator.
