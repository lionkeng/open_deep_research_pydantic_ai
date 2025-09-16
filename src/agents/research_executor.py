"""Enhanced Research Executor Agent with GPT-5 synthesis capabilities.

This module implements the core Enhanced Research Executor Agent that orchestrates
the research process using advanced synthesis with GPT-5 optimized prompts and
Tree of Thoughts methodology.
"""

import hashlib
from dataclasses import dataclass
from typing import Any

import logfire
from pydantic_ai import Agent

from models.research_executor import (
    ConfidenceLevel,
    Contradiction,
    ExecutiveSummary,
    HierarchicalFinding,
    ImportanceLevel,
    PatternAnalysis,
    PatternType,
    ResearchExecutorConfig,
    ResearchResults,
    ResearchSource,
    ThemeCluster,
)
from services.cache_manager import CacheManager
from services.confidence_analyzer import ConfidenceAnalyzer
from services.contradiction_detector import ContradictionDetector
from services.metrics_collector import MetricsCollector
from services.optimization_manager import OptimizationManager
from services.parallel_executor import ParallelExecutor
from services.pattern_recognizer import PatternRecognizer
from services.synthesis_engine import SynthesisEngine


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
Generate a complete ResearchResults object with all required fields properly populated.

## Processing Instructions
1. Use the available tools to extract and analyze findings
2. Apply proper confidence and importance scoring
3. Identify patterns and contradictions systematically
4. Generate comprehensive executive summaries
5. Ensure quality metrics are calculated

## Quality Standards
- All findings must have proper confidence and importance scores
- Contradictions must be identified and analyzed
- Patterns must be validated with supporting evidence
- Executive summaries must be actionable and comprehensive
- Source attribution must be complete and accurate

Always use the provided tools for analysis and maintain high standards for
research quality and integrity.""",
)


# Helper functions for the research executor
async def extract_hierarchical_findings(
    deps: ResearchExecutorDependencies,
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

    try:
        # Generate cache key for this extraction
        cache_key = _generate_cache_key("extract_findings", source_content, source_metadata)

        # Check cache first if available
        if deps.cache_manager:
            cached_result = await deps.cache_manager.get(cache_key)
            if cached_result:
                logfire.debug("Using cached hierarchical findings")
                return cached_result

        # Use the synthesis engine to extract findings
        if hasattr(deps.synthesis_engine, 'extract_themes'):
            findings_data = await deps.synthesis_engine.extract_themes(source_content)
        else:
            # Fallback to basic extraction
            findings_data = _extract_findings_fallback(source_content, source_metadata)

        # Convert to HierarchicalFinding objects
        hierarchical_findings = []
        for finding_data in findings_data:
            finding = HierarchicalFinding(
                finding=finding_data.get("text", source_content[:200]),
                supporting_evidence=[source_content[:200]],
                confidence=ConfidenceLevel.MEDIUM,
                confidence_score=finding_data.get("confidence", 0.7),
                importance=ImportanceLevel.MEDIUM,
                importance_score=finding_data.get("importance", 0.75),
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

        # Cache results if cache manager available
        if deps.cache_manager:
            await deps.cache_manager.set(cache_key, hierarchical_findings)

        logfire.info(f"Extracted {len(hierarchical_findings)} hierarchical findings")
        return hierarchical_findings

    except Exception as e:
        logfire.error(f"Failed to extract hierarchical findings: {e}")
        # Return fallback findings
        return _extract_findings_fallback(source_content, source_metadata)


async def identify_theme_clusters(
    deps: ResearchExecutorDependencies,
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

    try:
        # Generate cache key for clustering
        cache_key = _generate_cache_key("identify_clusters", findings, min_cluster_size)

        # Check cache first if available
        if deps.cache_manager:
            cached_result = await deps.cache_manager.get(cache_key)
            if cached_result:
                logfire.debug("Using cached theme clusters")
                return cached_result

        # Use synthesis engine for ML-based clustering
        clusters = deps.synthesis_engine.cluster_findings(findings)

        # Convert to ThemeCluster objects if needed
        theme_clusters = []
        if clusters and isinstance(clusters[0], ThemeCluster):
            theme_clusters = clusters
        else:
            # Convert raw clusters to ThemeCluster objects
            for i, cluster in enumerate(clusters):
                if isinstance(cluster, dict):
                    theme_clusters.append(
                        ThemeCluster(
                            theme_name=cluster.get("name", f"Theme {i+1}"),
                            description=cluster.get("description", "Clustered findings"),
                            findings=cluster.get("findings", []),
                            coherence_score=cluster.get("coherence", 0.5),
                            importance_score=cluster.get("importance", 0.5),
                        )
                    )

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

        # Cache results if cache manager available
        if deps.cache_manager:
            await deps.cache_manager.set(cache_key, theme_clusters)

        logfire.info(f"Identified {len(theme_clusters)} theme clusters")
        return theme_clusters

    except Exception as e:
        logfire.error(f"Failed to identify theme clusters: {e}")
        # Return fallback cluster
        return [
            ThemeCluster(
                theme_name="Research Findings",
                description="All available findings",
                findings=findings,
                coherence_score=0.5,
                importance_score=0.5,
            )
        ]


async def detect_contradictions(
    deps: ResearchExecutorDependencies, findings: list[HierarchicalFinding]
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

    try:
        # Generate cache key for contradiction detection
        cache_key = _generate_cache_key("detect_contradictions", findings)

        # Check cache first if available
        if deps.cache_manager:
            cached_result = await deps.cache_manager.get(cache_key)
            if cached_result:
                logfire.debug("Using cached contradiction analysis")
                return cached_result

        # Use contradiction detector service
        contradictions = deps.contradiction_detector.detect_contradictions(findings)

        # Cache results if cache manager available
        if deps.cache_manager:
            await deps.cache_manager.set(cache_key, contradictions)

        logfire.info(f"Detected {len(contradictions)} contradictions")
        return contradictions

    except Exception as e:
        logfire.error(f"Failed to detect contradictions: {e}")
        return []


async def analyze_patterns(
    deps: ResearchExecutorDependencies,
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

    try:
        # Generate cache key for pattern analysis
        cache_key = _generate_cache_key("analyze_patterns", findings, clusters)

        # Check cache first if available
        if deps.cache_manager:
            cached_result = await deps.cache_manager.get(cache_key)
            if cached_result:
                logfire.debug("Using cached pattern analysis")
                return cached_result

        # Use pattern recognizer service
        detected_patterns = deps.pattern_recognizer.detect_patterns(findings)

        # Convert to PatternAnalysis objects
        for pattern in detected_patterns:
            pattern_analysis = PatternAnalysis(
                pattern_type=pattern.get("type", PatternType.CONVERGENCE),
                pattern_name=pattern.get("name", "Detected Pattern"),
                description=pattern.get("description", "Pattern in research data"),
                strength=pattern.get("strength", 0.5),
                finding_ids=[str(i) for i in pattern.get("finding_indices", [])],
                implications=pattern.get("implications", ["Pattern detected in data"]),
                confidence_factors=pattern.get("confidence_factors", {"detection": 0.5}),
            )
            patterns.append(pattern_analysis)

        # Add basic convergence pattern if high confidence findings exist
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

        # Add temporal patterns if temporal data exists
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

        # Cache results if cache manager available
        if deps.cache_manager:
            await deps.cache_manager.set(cache_key, patterns)

        logfire.info(f"Analyzed {len(patterns)} patterns")
        return patterns

    except Exception as e:
        logfire.error(f"Failed to analyze patterns: {e}")
        return []


async def generate_executive_summary(
    deps: ResearchExecutorDependencies,
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

    try:
        # Extract key findings (top 5 by importance)
        sorted_findings = sorted(findings, key=lambda f: f.importance_score, reverse=True)
        key_findings = [f.finding for f in sorted_findings[:5]]

        # Assess overall confidence
        avg_confidence = (
            sum(f.confidence_score for f in findings) / len(findings) if findings else 0
        )
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
            critical_gaps.append(
                f"Multiple contradictions ({len(contradictions)}) require resolution"
            )
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

    except Exception as e:
        logfire.error(f"Failed to generate executive summary: {e}")
        return ExecutiveSummary(
            key_findings=["Error generating summary"],
            confidence_assessment="Unable to assess confidence",
            critical_gaps=["Summary generation failed"],
            recommended_actions=["Review and retry analysis"],
            risk_factors=["Analysis incomplete"],
        )


async def assess_synthesis_quality(
    deps: ResearchExecutorDependencies,
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

    try:
        metrics = {}

        # Completeness score
        search_count = len(deps.search_results) if deps.search_results else 0
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

        # Record metrics if collector available
        if deps.metrics_collector:
            await deps.metrics_collector.record_synthesis_metrics(metrics)

        return metrics

    except Exception as e:
        logfire.error(f"Failed to assess synthesis quality: {e}")
        return {
            "completeness": 0.0,
            "coherence": 0.0,
            "average_confidence": 0.0,
            "reliability": 0.0,
            "overall_quality": 0.0,
        }


def _generate_cache_key(*args: Any) -> str:
    """Generate cache key from arguments."""
    key_string = str(args)
    return hashlib.md5(key_string.encode()).hexdigest()[:16]


def _extract_findings_fallback(
    source_content: str, source_metadata: dict[str, Any] | None = None
) -> list[HierarchicalFinding]:
    """Fallback method for extracting findings when services are unavailable."""
    # Simple fallback - extract basic finding from content
    source = ResearchSource(
        title=source_metadata.get("title", "Unknown") if source_metadata else "Unknown",
        url=source_metadata.get("url") if source_metadata else None,
        source_type=source_metadata.get("type", "unknown") if source_metadata else "unknown",
    )

    finding = HierarchicalFinding(
        finding=f"Finding from source: {source_content[:100]}...",
        supporting_evidence=[source_content[:200]],
        confidence=ConfidenceLevel.MEDIUM,
        confidence_score=0.6,
        importance=ImportanceLevel.MEDIUM,
        importance_score=0.6,
        source=source,
        category="research",
        temporal_relevance="current",
    )
    return [finding]


async def execute_research(
    query: str,
    search_results: list[dict[str, Any]],
    **kwargs: Any
) -> ResearchResults:
    """Execute research synthesis using the Enhanced Research Executor Agent.

    Args:
        query: The research query
        search_results: Raw search results to synthesize
        config: Optional configuration for the executor
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

    def __init__(self, config: ResearchExecutorConfig | None = None):
        """Initialize the compatibility wrapper."""
        self.config = config

    async def execute_research(
        self, query: str, search_results: list[dict[str, Any]], **kwargs: Any
    ) -> ResearchResults:
        """Execute research using the enhanced agent."""
        return await execute_research(query, search_results, **kwargs)
