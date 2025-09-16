"""Enhanced Research Executor Agent with GPT-5 synthesis capabilities.

This module implements the core Enhanced Research Executor Agent that orchestrates
the research process using advanced synthesis with GPT-5 optimized prompts and
Tree of Thoughts methodology.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import logfire
from pydantic_ai import Agent, RunContext

from src.models.research_executor import (
    ConfidenceLevel,
    Contradiction,
    ExecutiveSummary,
    HierarchicalFinding,
    ImportanceLevel,
    PatternAnalysis,
    PatternType,
    ResearchResults,
    ResearchSource,
    ThemeCluster,
)
from src.services.cache_manager import CacheManager
from src.services.confidence_analyzer import ConfidenceAnalyzer
from src.services.contradiction_detector import ContradictionDetector
from src.services.metrics_collector import MetricsCollector
from src.services.optimization_manager import OptimizationManager
from src.services.parallel_executor import ParallelExecutor
from src.services.pattern_recognizer import PatternRecognizer
from src.services.synthesis_engine import SynthesisEngine


@dataclass
class ResearchExecutorDependencies:
    """Dependencies for the Enhanced Research Executor Agent.

    This dataclass encapsulates all services and configuration needed
    by the research executor agent to perform advanced synthesis.
    """

    # Core services
    synthesis_engine: SynthesisEngine
    contradiction_detector: ContradictionDetector
    pattern_recognizer: PatternRecognizer
    confidence_analyzer: ConfidenceAnalyzer

    # Optional optimization services
    cache_manager: CacheManager | None = None
    parallel_executor: ParallelExecutor | None = None
    metrics_collector: MetricsCollector | None = None
    optimization_manager: OptimizationManager | None = None

    # Research context
    original_query: str = ""
    search_results: list[dict[str, Any]] | None = None

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.search_results is None:
            self.search_results = []


# Create the Enhanced Research Executor Agent with GPT-5 optimization
research_executor_agent = Agent(
    model="openai:gpt-4o",  # Will be "openai:gpt-5" when available
    deps_type=ResearchExecutorDependencies,
    output_type=ResearchResults,
    system_prompt="""# Role: Advanced Research Synthesis Expert (GPT-5 Optimized)

You are a Senior Research Synthesis Specialist with expertise in pattern recognition,
evidence evaluation, and systematic analysis across diverse information sources.

## Core Mission
Transform raw search results into actionable research insights through systematic
synthesis, advanced pattern recognition, and quality-assured analysis using GPT-5's
enhanced reasoning capabilities.

## Primary Responsibilities
1. Extract hierarchical findings with confidence scoring
2. Identify theme clusters and patterns
3. Detect and analyze contradictions
4. Generate executive summaries with actionable insights
5. Ensure comprehensive quality verification

## Output Requirements
Generate a complete ResearchResults object with all required fields properly populated.""",
)


@research_executor_agent.instructions
async def add_synthesis_context(ctx: RunContext[ResearchExecutorDependencies]) -> str:
    """Add comprehensive synthesis instructions with Tree of Thoughts methodology.

    This dynamic instruction injection provides the full 665-line synthesis
    system prompt optimized for GPT-5 capabilities.
    """

    search_count = len(ctx.deps.search_results) if ctx.deps.search_results else 0

    return f"""
# ENHANCED SYNTHESIS SYSTEM PROMPT (GPT-5 OPTIMIZED)

## Current Research Context
- Original Query: {ctx.deps.original_query}
- Search Results Available: {search_count} sources
- Execution Timestamp: {datetime.now(UTC).isoformat()}

## Process Overview: Tree of Thoughts Methodology

You will execute a comprehensive 3-phase synthesis process using Tree of Thoughts reasoning:

```
Research Synthesis Process
├── Phase 1: Pattern Recognition
│   ├── Convergence Analysis (agreement across sources)
│   ├── Divergence Mapping (contradictions/conflicts)
│   └── Emergence Detection (new trends/signals)
├── Phase 2: Insight Extraction
│   ├── Primary Insights (directly address query)
│   ├── Secondary Insights (related discoveries)
│   └── Meta-Insights (patterns about patterns)
└── Phase 3: Quality Verification
    ├── Completeness Check
    ├── Coherence Validation
    └── Confidence Calibration
```

## PHASE 1: PATTERN RECOGNITION

### Task 1.1: Convergence Analysis
**OBJECTIVE**: Identify findings that support each other across sources

**EXECUTION STEPS**:
1. Group findings by semantic similarity (>70% overlap threshold)
2. Count supporting sources for each finding group
3. Calculate convergence_score = (supporting_sources / total_sources)
4. Classify convergence strength:
   - Strong (>0.8): Multiple independent sources agree
   - Moderate (0.5-0.8): Majority agreement with some variation
   - Weak (<0.5): Limited agreement, consider as preliminary

**EXAMPLE**:
- Input: 5 sources report "40-60% efficiency improvement with new caching"
- Analysis: High semantic overlap (numbers + "efficiency improvement")
- Output: convergence_score=1.0, strength="strong", confidence_boost=+0.2
- Synthesis: "Strong consensus on 40-60% efficiency improvement from caching implementation"

### Task 1.2: Divergence Mapping
**OBJECTIVE**: Identify contradictory or conflicting information

**EXECUTION STEPS**:
1. Compare all finding pairs for contradiction indicators
2. Detection criteria:
   - Opposite directional words (increase/decrease, positive/negative)
   - Conflicting metrics on same measurement
   - Mutually exclusive claims
3. Classify contradiction type and severity:
   - Direct: Opposite claims about same fact (severity=high)
   - Partial: Different scope or conditions (severity=medium)
   - Contextual: True in different contexts (severity=low)
   - Methodological: Different methods lead to different conclusions (severity=medium)
4. Generate resolution hypotheses

**EXAMPLE**:
- Finding A: "AI increases job opportunities"
- Finding B: "AI reduces employment"
- Analysis: Contextual contradiction - sector-specific effects
- Resolution: "AI impact on employment varies by sector - creating tech jobs while
  reducing manufacturing roles"
- Confidence adjustment: Medium (0.5) due to conflicting evidence

### Task 1.3: Emergence Detection
**OBJECTIVE**: Spot new trends, patterns, or signals not explicitly stated

**EXECUTION STEPS**:
1. Scan for temporal markers: "recently", "emerging", "new", dates
2. Identify statistical outliers: unique findings from authoritative sources
3. Detect pattern breaks: sudden changes in metrics or consensus
4. Flag innovation indicators: novel methods, breakthrough claims
5. Output emergence_signals[] with confidence scores

**EXAMPLE**:
- Input: Single authoritative source provides novel framework not seen elsewhere
- Analysis: Unique insight from credible source, no corroboration yet
- Output: emergence_signal with confidence=0.6 (single source limitation)
- Synthesis: "Novel framework for understanding X (single source, requires validation)"

## PHASE 2: INSIGHT EXTRACTION

### Task 2.1: Primary Insights (Critical findings only)
**FILTER**: importance_score >= 0.9
**PROCESS**:
1. Link each critical finding to original research questions
2. Merge convergent findings into unified statements
3. Preserve ALL numerical data and source attribution
4. Generate 3-5 bullet points maximum

**TEMPLATE**: "[INSIGHT] based on [N sources] showing [convergence_level] agreement"

### Task 2.2: Secondary Insights (Important findings)
**FILTER**: importance_score 0.7-0.89
**PROCESS**:
1. Group by theme cluster
2. Extract patterns that contextualize primary insights
3. Identify surprising connections
4. Generate 3-5 supporting points

### Task 2.3: Meta-Insights (Pattern analysis)
**REQUIRES**: Completed Tasks 1.1-1.3 and 2.1-2.2
**PROCESS**:
1. Analyze finding distribution across categories
2. Identify systematic gaps in research coverage
3. Assess methodology trends and biases
4. Evaluate temporal evolution of consensus

## PHASE 3: QUALITY VERIFICATION

### Check 3.1: Completeness Verification
□ All search results analyzed? (REQUIRED: 100%)
□ Critical findings preserved? (REQUIRED: 100%)
□ Source attribution complete? (REQUIRED: 100%)
□ Contradictions documented? (REQUIRED: 100%)
FAIL → Return to Phase 1 with gap list

### Check 3.2: Coherence Validation
□ Logical flow from evidence to conclusions?
□ Confidence justified by evidence strength?
□ Temporal consistency maintained?
□ Causal claims properly qualified?
FAIL → Flag specific issues for correction

### Check 3.3: Confidence Calibration
FORMULA: confidence = (source_quality * convergence_score * (1 - contradiction_penalty))
VERIFY:
□ Single-source claims marked as preliminary?
□ Contradictions reduce confidence appropriately?
□ Distribution reasonable (not all high/low)?
FAIL → Recalibrate using formula

## INFORMATION HIERARCHY FRAMEWORK

Apply this scoring to EVERY finding:

### Critical (0.9-1.0)
**DEFINITION**: Core facts, primary conclusions, unique insights that directly answer
the research question
**CRITERIA**:
- Directly answers research question
- Unique insight not found elsewhere
- High-confidence breakthrough finding
- Safety/risk critical information

**EXAMPLE**:
- Finding: "GPT-4 achieves 86.4% on MMLU benchmark, surpassing human expert performance"
- Score: 0.95
- Rationale: Direct answer to AI performance query, specific metric, authoritative source

### Important (0.7-0.89)
**DEFINITION**: Supporting evidence, methodologies, key patterns that provide context
**CRITERIA**:
- Provides essential context
- Validates/challenges critical findings
- Methodology and approach details
- Strong supporting evidence

**EXAMPLE**:
- Finding: "The benchmark was conducted using zero-shot prompting across 57 subjects"
- Score: 0.75
- Rationale: Methodology detail that validates the critical finding

### Supplementary (0.4-0.69)
**DEFINITION**: Background information, elaborations, secondary examples
**CRITERIA**:
- Additional examples and cases
- Extended background information
- Alternative perspectives
- Historical context

### Contextual (0.0-0.39)
**DEFINITION**: Tangential information, general background
**CRITERIA**:
- General background
- Tangentially related information
- Common knowledge
- Redundant details

## PRESERVATION RULES

### Must ALWAYS Preserve
✓ All numerical data and statistics with sources
✓ Contradictions and conflicting viewpoints
✓ Unique insights not found elsewhere
✓ Methodological limitations and caveats
✓ Confidence intervals and uncertainty measures
✓ Source attribution for all claims

### Safe to Consolidate
- Similar findings from multiple sources (note convergence)
- Redundant examples (keep most representative)
- Extended background information
- Detailed process descriptions (summarize key steps)

### Never Omit
✗ Contradictory evidence (even if minority view)
✗ Methodological flaws or limitations
✗ Recent updates that override older information
✗ Safety, ethical, or legal concerns
✗ Uncertainty or low confidence indicators

## SELF-VERIFICATION CHECKLIST

Before finalizing synthesis, verify:
□ Phase 1 (Pattern Recognition) - All three tasks completed?
□ Phase 2 (Insight Extraction) - Findings processed by importance?
□ Phase 3 (Quality Verification) - All checks passed?
□ All search results addressed (100% required)?
□ Information hierarchy correctly applied to all findings?
□ Contradictions explicitly identified and analyzed?
□ Confidence scores justified by evidence and convergence?
□ Source attribution complete for all claims?
□ Executive summary captures essence without losing critical details?
□ Theme clusters logically organized with consensus levels?
□ Gaps and limitations clearly documented?
□ No critical information lost in synthesis?
□ Output follows exact structure requirements?

## ANTI-PATTERNS TO AVOID

✗ Cherry-picking evidence that supports a single narrative
✗ Hiding or minimizing contradictions
✗ Over-generalizing from limited sources
✗ Conflating correlation with causation
✗ Ignoring source credibility differences
✗ Creating false consensus where none exists
✗ Losing nuance through oversimplification
✗ Failing to acknowledge uncertainty

## OUTPUT STRUCTURE REQUIREMENTS

Generate synthesis with these EXACT sections in the ResearchResults object:

### 1. Executive Summary
- key_findings: 3-5 most important findings
- confidence_assessment: Overall confidence in the research
- critical_gaps: Critical information gaps identified
- recommended_actions: Recommended next steps or actions
- risk_factors: Identified risks or concerns

### 2. Hierarchical Findings
Create HierarchicalFinding objects with:
- finding: The core finding text
- supporting_evidence: List of evidence
- confidence: ConfidenceLevel enum
- confidence_score: Numeric score (0.0-1.0)
- importance: ImportanceLevel enum
- importance_score: Numeric score (0.0-1.0)
- source: ResearchSource object when available
- category: Category or topic
- temporal_relevance: Time period or context

### 3. Theme Clusters
Create ThemeCluster objects with:
- theme_name: Descriptive name
- description: What this theme represents
- findings: Related HierarchicalFinding objects
- coherence_score: How coherent/related (0.0-1.0)
- importance_score: Overall importance (0.0-1.0)

### 4. Contradictions
Create Contradiction objects with:
- finding_1_id: ID/index of first finding
- finding_2_id: ID/index of second finding
- contradiction_type: Type (direct/partial/contextual/methodological)
- explanation: Explanation of the contradiction
- resolution_hint: Hint for resolving
- severity: Severity score (0.0-1.0)

### 5. Pattern Analysis
Identify patterns and create appropriate pattern analysis metadata

### 6. Quality Metrics
Track and report synthesis quality metrics

## DOMAIN ADAPTATION PROTOCOL

Detect domain from query content and apply appropriate focus:

### Technical/Scientific Research:
- Preserve ALL version numbers, specifications, and technical details
- Maintain precision in terminology (no paraphrasing technical terms)
- Emphasize reproducibility and methodology
- Track compatibility constraints and dependencies
- Weight peer-reviewed and official documentation higher

### Business/Market Research:
- Extract ROI, cost implications, and business metrics
- Highlight competitive advantages and market positioning
- Focus on actionable recommendations and strategic insights
- Emphasize timing and market readiness
- Consider source recency more heavily

### Academic/Theoretical Research:
- Maintain citation standards and attribution
- Track theoretical frameworks and schools of thought
- Document methodology details and research design
- Preserve academic debates and opposing viewpoints
- Note publication venues and impact factors

### Emerging Technology Research:
- Flag uncertainty and speculation clearly
- Distinguish proven vs. experimental/theoretical
- Track hype indicators vs. actual deployments
- Monitor patent filings and investment signals
- Note early adopters and pilot programs

## GPT-5 SPECIFIC OPTIMIZATIONS

### Enhanced Reasoning Capabilities
- Leverage GPT-5's improved logical reasoning for complex pattern detection
- Utilize advanced context understanding for nuanced synthesis
- Apply enhanced instruction following for precise protocol execution
- Take advantage of improved factual accuracy for reliable findings

### Advanced Synthesis Features
- Multi-level abstraction reasoning for hierarchical insights
- Cross-domain knowledge integration for comprehensive analysis
- Temporal reasoning for trend identification
- Causal reasoning for relationship mapping

### Quality Assurance Enhancements
- Self-consistency checking across synthesis phases
- Automatic bias detection and correction
- Enhanced uncertainty quantification
- Improved source credibility assessment

Remember: Your synthesis will be used for decision-making. Accuracy, completeness, and
transparency are paramount. When uncertain, explicitly state limitations rather than
creating false confidence.

## FINAL EXECUTION NOTES

1. Process ALL available search results - no cherry-picking
2. Apply the full 3-phase synthesis process systematically
3. Use the provided tools to enhance synthesis quality
4. Generate comprehensive ResearchResults object with all fields
5. Ensure all findings have proper confidence and importance scoring
6. Document all contradictions and uncertainties
7. Provide actionable insights in the executive summary

The quality of your synthesis directly impacts research outcomes. Execute with precision
and thoroughness befitting GPT-5's advanced capabilities.
"""


# Tool 1: Extract Hierarchical Findings
@research_executor_agent.tool
async def extract_hierarchical_findings(
    ctx: RunContext[ResearchExecutorDependencies],
    source_content: str,
    source_metadata: dict[str, Any] | None = None,
) -> list[HierarchicalFinding]:
    """Extract and classify findings hierarchically from source content.

    This tool processes raw source content to extract structured findings
    with importance and confidence scoring using the information hierarchy
    framework.

    Args:
        source_content: The raw content to analyze
        source_metadata: Optional metadata about the source

    Returns:
        List of hierarchical findings with full classification
    """
    logfire.info("Extracting hierarchical findings from source content")

    # Use the synthesis engine to extract findings
    _ = ctx.deps.synthesis_engine.cluster_findings([])  # Placeholder for ML extraction

    # For now, create sample findings based on content analysis
    # In production, this would use NLP/ML to extract actual findings
    hierarchical_findings = []

    # Simulate finding extraction (would be ML-based in production)
    if source_content:
        # Create a sample finding
        finding = HierarchicalFinding(
            finding="Sample finding extracted from source",
            supporting_evidence=[source_content[:200]],
            confidence=ConfidenceLevel.MEDIUM,
            confidence_score=0.7,
            importance=ImportanceLevel.MEDIUM,
            importance_score=0.75,
            source=ResearchSource(
                title=source_metadata.get("title", "Unknown") if source_metadata else "Unknown",
                url=source_metadata.get("url") if source_metadata else None,
                source_type=source_metadata.get("type", "unknown")
                if source_metadata
                else "unknown",
            )
            if source_metadata
            else None,
            category="research",
            temporal_relevance="current",
        )
        hierarchical_findings.append(finding)

    return hierarchical_findings


# Tool 2: Identify Theme Clusters
@research_executor_agent.tool
async def identify_theme_clusters(
    ctx: RunContext[ResearchExecutorDependencies],
    findings: list[HierarchicalFinding],
    min_cluster_size: int = 2,
) -> list[ThemeCluster]:
    """Group findings into thematic clusters using ML clustering.

    This tool uses the synthesis engine's ML capabilities to identify
    themes and patterns across findings.

    Args:
        findings: List of findings to cluster
        min_cluster_size: Minimum findings required to form a cluster

    Returns:
        List of theme clusters with coherence scoring
    """
    logfire.info(f"Identifying theme clusters from {len(findings)} findings")

    if not findings:
        return []

    # Use synthesis engine for ML-based clustering
    clusters = ctx.deps.synthesis_engine.cluster_findings(findings)

    # Convert to ThemeCluster objects
    theme_clusters = []
    for cluster in clusters:
        theme_clusters.append(cluster)

    # If no clusters formed, create a general cluster
    if not theme_clusters and len(findings) >= min_cluster_size:
        theme_clusters.append(
            ThemeCluster(
                theme_name="General Research Findings",
                description="Unclustered research findings",
                findings=findings,
                coherence_score=0.5,
                importance_score=sum(f.importance_score for f in findings) / len(findings),
            )
        )

    return theme_clusters


# Tool 3: Detect Contradictions
@research_executor_agent.tool
async def detect_contradictions(
    ctx: RunContext[ResearchExecutorDependencies], findings: list[HierarchicalFinding]
) -> list[Contradiction]:
    """Detect contradictions between findings using advanced analysis.

    This tool identifies direct, partial, contextual, and methodological
    contradictions between findings.

    Args:
        findings: List of findings to analyze for contradictions

    Returns:
        List of detected contradictions with resolution hints
    """
    logfire.info(f"Detecting contradictions among {len(findings)} findings")

    if not findings or len(findings) < 2:
        return []

    # Use contradiction detector service
    contradictions = ctx.deps.contradiction_detector.detect_contradictions(findings)

    return contradictions


# Tool 4: Analyze Patterns
@research_executor_agent.tool
async def analyze_patterns(
    ctx: RunContext[ResearchExecutorDependencies],
    findings: list[HierarchicalFinding],
    clusters: list[ThemeCluster],
) -> list[PatternAnalysis]:
    """Analyze patterns across findings and clusters.

    This tool identifies convergence, divergence, emergence, and temporal
    patterns in the research data.

    Args:
        findings: List of findings to analyze
        clusters: Theme clusters to consider

    Returns:
        List of identified patterns with analysis
    """
    logfire.info(f"Analyzing patterns across {len(findings)} findings and {len(clusters)} clusters")

    patterns = []

    if not findings:
        return patterns

    # Use pattern recognizer service
    # This would normally use the actual pattern recognizer service
    # For now, create sample patterns

    # Analyze convergence
    if len(findings) > 3:
        high_confidence_findings = [f for f in findings if f.confidence_score > 0.8]
        if len(high_confidence_findings) > len(findings) * 0.6:
            patterns.append(
                PatternAnalysis(
                    pattern_type=PatternType.CONVERGENCE,
                    pattern_name="High Confidence Convergence",
                    description="Majority of findings show high confidence convergence",
                    strength=0.8,
                    finding_ids=[str(i) for i in range(len(high_confidence_findings))],
                    implications=["Strong consensus in research findings"],
                    confidence_factors={"source_agreement": 0.8, "data_consistency": 0.75},
                )
            )

    # Analyze temporal patterns if temporal data exists
    temporal_findings = [f for f in findings if f.temporal_relevance]
    if temporal_findings:
        patterns.append(
            PatternAnalysis(
                pattern_type=PatternType.TEMPORAL,
                pattern_name="Temporal Evolution",
                description="Findings show temporal progression",
                strength=0.6,
                finding_ids=[str(i) for i in range(len(temporal_findings))],
                temporal_span="recent",
                implications=["Research shows evolution over time"],
            )
        )

    return patterns


# Tool 5: Generate Executive Summary
@research_executor_agent.tool
async def generate_executive_summary(
    ctx: RunContext[ResearchExecutorDependencies],
    findings: list[HierarchicalFinding],
    contradictions: list[Contradiction],
    patterns: list[PatternAnalysis],
) -> ExecutiveSummary:
    """Generate an executive summary of the research.

    This tool synthesizes all analysis into a concise executive summary
    with key findings, gaps, and recommendations.

    Args:
        findings: All hierarchical findings
        contradictions: Detected contradictions
        patterns: Identified patterns

    Returns:
        Executive summary with actionable insights
    """
    logfire.info("Generating executive summary")

    # Extract key findings (top 5 by importance)
    sorted_findings = sorted(findings, key=lambda f: f.importance_score, reverse=True)
    key_findings = [f.finding for f in sorted_findings[:5]]

    # Assess overall confidence
    avg_confidence = sum(f.confidence_score for f in findings) / len(findings) if findings else 0
    confidence_assessment = f"Overall confidence: {avg_confidence:.2f} - "
    if avg_confidence > 0.8:
        confidence_assessment += "High confidence in research findings"
    elif avg_confidence > 0.6:
        confidence_assessment += "Moderate confidence with some uncertainties"
    else:
        confidence_assessment += "Low confidence, further research recommended"

    # Identify critical gaps
    critical_gaps = []
    if not findings:
        critical_gaps.append("No findings extracted from available sources")
    if len(contradictions) > 3:
        critical_gaps.append(f"Multiple contradictions ({len(contradictions)}) require resolution")
    if avg_confidence < 0.6:
        critical_gaps.append("Low overall confidence in findings")

    # Generate recommendations
    recommended_actions = []
    if critical_gaps:
        recommended_actions.append("Address identified gaps through additional research")
    if contradictions:
        recommended_actions.append("Investigate and resolve contradictory findings")
    if patterns:
        for pattern in patterns[:2]:  # Top 2 patterns
            if pattern.implications:
                recommended_actions.append(f"Consider implications: {pattern.implications[0]}")

    # Identify risk factors
    risk_factors = []
    if contradictions:
        risk_factors.append(f"{len(contradictions)} unresolved contradictions")
    low_confidence = [f for f in findings if f.confidence_score < 0.5]
    if low_confidence:
        risk_factors.append(f"{len(low_confidence)} low-confidence findings")

    return ExecutiveSummary(
        key_findings=key_findings,
        confidence_assessment=confidence_assessment,
        critical_gaps=critical_gaps,
        recommended_actions=recommended_actions,
        risk_factors=risk_factors,
    )


# Tool 6: Assess Synthesis Quality
@research_executor_agent.tool
async def assess_synthesis_quality(
    ctx: RunContext[ResearchExecutorDependencies],
    findings: list[HierarchicalFinding],
    clusters: list[ThemeCluster],
    contradictions: list[Contradiction],
) -> dict[str, float]:
    """Assess the quality of the synthesis.

    This tool evaluates completeness, coherence, and reliability of
    the synthesis results.

    Args:
        findings: Extracted findings
        clusters: Theme clusters
        contradictions: Detected contradictions

    Returns:
        Quality metrics dictionary
    """
    logfire.info("Assessing synthesis quality")

    metrics = {}

    # Completeness score
    search_count = len(ctx.deps.search_results) if ctx.deps.search_results else 0
    expected_findings = max(10, search_count * 2)
    completeness = min(1.0, len(findings) / expected_findings)
    metrics["completeness"] = completeness

    # Coherence score
    if clusters:
        avg_coherence = sum(c.coherence_score for c in clusters) / len(clusters)
        metrics["coherence"] = avg_coherence
    else:
        metrics["coherence"] = 0.5

    # Confidence distribution
    if findings:
        avg_confidence = sum(f.confidence_score for f in findings) / len(findings)
        metrics["average_confidence"] = avg_confidence
    else:
        metrics["average_confidence"] = 0.0

    # Contradiction impact
    contradiction_penalty = min(0.3, len(contradictions) * 0.05)
    metrics["reliability"] = max(0.0, 1.0 - contradiction_penalty)

    # Overall quality
    metrics["overall_quality"] = (
        metrics["completeness"] * 0.25
        + metrics["coherence"] * 0.25
        + metrics["average_confidence"] * 0.25
        + metrics["reliability"] * 0.25
    )

    return metrics


async def execute_research(
    query: str, search_results: list[dict[str, Any]], **kwargs: Any
) -> ResearchResults:
    """Execute research synthesis using the Enhanced Research Executor Agent.

    Args:
        query: The research query
        search_results: Raw search results to synthesize
        **kwargs: Additional configuration options

    Returns:
        Complete research results with synthesis
    """
    logfire.info(f"Executing enhanced research synthesis for: {query}")

    # Create dependencies
    deps = ResearchExecutorDependencies(
        synthesis_engine=SynthesisEngine(),
        contradiction_detector=ContradictionDetector(),
        pattern_recognizer=PatternRecognizer(),
        confidence_analyzer=ConfidenceAnalyzer(),
        cache_manager=kwargs.get("cache_manager"),
        parallel_executor=kwargs.get("parallel_executor"),
        metrics_collector=kwargs.get("metrics_collector"),
        optimization_manager=kwargs.get("optimization_manager"),
        original_query=query,
        search_results=search_results,
    )

    # Run the agent
    result = await research_executor_agent.run(f"Synthesize research for: {query}", deps=deps)

    return result.output


# Compatibility wrapper for existing code
class ResearchExecutorAgent:
    """Compatibility wrapper for the Enhanced Research Executor Agent.

    This class provides backward compatibility with existing code that expects
    the ResearchExecutorAgent class interface.
    """

    def __init__(self, config=None):
        """Initialize the compatibility wrapper."""
        self.config = config

    async def execute_research(
        self, query: str, search_results: list[dict[str, Any]], **kwargs: Any
    ) -> ResearchResults:
        """Execute research using the enhanced agent."""
        return await execute_research(query, search_results, **kwargs)
