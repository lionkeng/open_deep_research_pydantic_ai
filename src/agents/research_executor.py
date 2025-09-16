"""Research executor agent implementation."""

import hashlib
from dataclasses import dataclass, field
from typing import Any

import httpx
import logfire

from agents.base import (
    AgentConfiguration,
    AgentConfigurationError,
    AgentExecutionError,
    AgentStatus,
    BaseResearchAgent,
    ResearchDependencies,
)
from models.api_models import APIKeys
from models.core import ResearchMetadata, ResearchStage, ResearchState
from models.research_executor import (
    ConfidenceLevel,
    Contradiction,
    ExecutiveSummary,
    HierarchicalFinding,
    ImportanceLevel,
    PatternAnalysis,
    PatternType,
    ResearchFinding,
    ResearchResults,
    ResearchSource,
    SynthesisMetadata,
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
    synthesis_engine: SynthesisEngine = field(default_factory=SynthesisEngine)
    contradiction_detector: ContradictionDetector = field(default_factory=ContradictionDetector)
    pattern_recognizer: PatternRecognizer = field(default_factory=PatternRecognizer)
    confidence_analyzer: ConfidenceAnalyzer = field(default_factory=ConfidenceAnalyzer)

    # Optional optimization services
    cache_manager: CacheManager | None = None
    parallel_executor: ParallelExecutor | None = None
    metrics_collector: MetricsCollector | None = None
    optimization_manager: OptimizationManager | None = None

    # Research context
    original_query: str = ""
    search_results: list[dict[str, Any]] = field(default_factory=list)


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
        if hasattr(deps.synthesis_engine, "extract_themes"):
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
                            theme_name=cluster.get("name", f"Theme {i + 1}"),
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


class ResearchExecutorAgent(BaseResearchAgent[ResearchDependencies, ResearchResults]):
    """Base research agent wrapper around the synthesis helpers."""

    def __init__(
        self,
        config: AgentConfiguration | None = None,
        dependencies: ResearchDependencies | None = None,
    ):
        if config is None:
            config = AgentConfiguration(
                agent_name="research_executor",
                agent_type="synthesis",
            )
        super().__init__(config=config, dependencies=dependencies)

    def _get_default_system_prompt(self) -> str:
        """Return a concise system prompt describing the agent role."""
        return (
            "You are a hybrid research synthesis agent that combines deterministic "
            "query execution with structured analysis. Use available findings to "
            "produce organized outputs that the downstream report generator can consume."
        )

    def _get_output_type(self) -> type[ResearchResults]:
        return ResearchResults

    def _register_tools(self) -> None:
        """No LLM tool registration in the transitional implementation."""
        return None

    async def run(
        self,
        deps: ResearchDependencies | None = None,
        message_history: list[Any] | None = None,
        stream: bool = False,
    ) -> ResearchResults:
        actual_deps = deps or self.dependencies
        if not actual_deps:
            raise AgentConfigurationError("Research executor requires dependencies")

        self.status = AgentStatus.RUNNING
        self.start_execution_timer()

        try:
            executor_deps = self._build_executor_dependencies(actual_deps)
            result = await self._generate_structured_result(actual_deps, executor_deps)
            self.metrics.record_success()
            self.status = AgentStatus.COMPLETED
            return result
        except Exception as exc:  # pragma: no cover - unexpected runtime errors
            self.metrics.record_failure()
            self.status = AgentStatus.FAILED
            raise AgentExecutionError("Research executor failed", agent_name=self.name) from exc
        finally:
            self.end_execution_timer()

    def _build_executor_dependencies(
        self, deps: ResearchDependencies
    ) -> ResearchExecutorDependencies:
        search_results = getattr(deps, "search_results", None) or []
        return ResearchExecutorDependencies(
            cache_manager=getattr(deps, "cache_manager", None),
            parallel_executor=getattr(deps, "parallel_executor", None),
            metrics_collector=getattr(deps, "metrics_collector", None),
            optimization_manager=getattr(deps, "optimization_manager", None),
            original_query=deps.research_state.user_query,
            search_results=search_results,
        )

    async def _generate_structured_result(
        self,
        deps: ResearchDependencies,
        executor_deps: ResearchExecutorDependencies,
    ) -> ResearchResults:
        query = deps.research_state.user_query
        findings: list[HierarchicalFinding] = []

        # Convert raw search results into hierarchical findings using fallbacks when needed
        for result in executor_deps.search_results:
            content = result.get("content") or result.get("snippet") or ""
            if not content.strip():
                continue
            metadata = {
                "title": result.get("title", "Unknown"),
                "url": result.get("url"),
                "type": result.get("type") or result.get("source_type", "unknown"),
            }
            extracted = await extract_hierarchical_findings(executor_deps, content, metadata)
            findings.extend(extracted)

        clusters = await identify_theme_clusters(executor_deps, findings)
        contradictions = await detect_contradictions(executor_deps, findings)
        patterns = await analyze_patterns(executor_deps, findings, clusters)
        summary = await generate_executive_summary(
            executor_deps, findings, contradictions, patterns
        )
        quality_metrics = await assess_synthesis_quality(
            executor_deps,
            findings,
            clusters,
            contradictions,
        )

        overall_quality = quality_metrics.get("overall_quality", 0.0) if quality_metrics else 0.0

        sources: list[ResearchSource] = [
            finding.source for finding in findings if finding.source is not None
        ]

        synthesis_metadata = SynthesisMetadata(
            synthesis_method="fallback_pipeline",
            total_sources_analyzed=len(executor_deps.search_results),
            total_findings_extracted=len(findings),
            quality_metrics=quality_metrics,
        )

        key_insights = list((summary.key_findings if summary else [])[:5])
        data_gaps: list[str] = []
        if not findings:
            data_gaps.append("No findings were generated from the available search results.")
        if contradictions:
            data_gaps.append("Detected contradictions require follow-up analysis.")

        content_hierarchy = {
            "critical": [f.finding for f in findings if f.importance == ImportanceLevel.CRITICAL],
            "important": [f.finding for f in findings if f.importance == ImportanceLevel.HIGH],
            "supplementary": [
                f.finding for f in findings if f.importance == ImportanceLevel.MEDIUM
            ],
            "contextual": [f.finding for f in findings if f.importance == ImportanceLevel.LOW],
        }

        metadata = {
            "search_query_count": len(getattr(deps, "search_queries", []) or []),
            "generation_mode": "structured_fallback",
        }

        return ResearchResults(
            query=query,
            findings=findings,
            theme_clusters=clusters,
            contradictions=contradictions,
            executive_summary=summary,
            patterns=patterns,
            sources=[source for source in sources if source],
            synthesis_metadata=synthesis_metadata,
            overall_quality_score=overall_quality,
            content_hierarchy=content_hierarchy,
            key_insights=key_insights,
            data_gaps=data_gaps,
            metadata=metadata,
        )


_research_executor_agent_instance: ResearchExecutorAgent | None = None


def get_research_executor_agent() -> ResearchExecutorAgent:
    """Return a lazily instantiated research executor agent."""
    global _research_executor_agent_instance
    if _research_executor_agent_instance is None:
        _research_executor_agent_instance = ResearchExecutorAgent()
        logfire.info("Initialized research executor agent")
    return _research_executor_agent_instance


research_executor_agent = get_research_executor_agent()


async def execute_research(
    query: str,
    search_results: list[dict[str, Any]] | None = None,
    *,
    agent: ResearchExecutorAgent | None = None,
) -> ResearchResults:
    """Convenience entry point for executing research outside the workflow."""
    logfire.info("Executing research for query", query=query)

    agent_instance = agent or get_research_executor_agent()

    research_state = ResearchState(
        request_id=ResearchState.generate_request_id(),
        user_query=query,
        current_stage=ResearchStage.RESEARCH_EXECUTION,
        metadata=ResearchMetadata(),
    )

    async with httpx.AsyncClient() as http_client:
        deps = ResearchDependencies(
            http_client=http_client,
            api_keys=APIKeys(),
            research_state=research_state,
        )
        deps.search_results = search_results or []
        result = await agent_instance.run(deps)
        research_state.research_results = result
        research_state.findings = [
            ResearchFinding.from_hierarchical(finding) for finding in result.findings
        ]
        return result
