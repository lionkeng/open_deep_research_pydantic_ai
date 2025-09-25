"""Tool helpers and dependencies for the research executor agent."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import logfire

from models.research_executor import (
    ConfidenceLevel,
    Contradiction,
    ExecutiveSummary,
    HierarchicalFinding,
    ImportanceLevel,
    PatternAnalysis,
    PatternType,
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
from services.source_repository import AbstractSourceRepository
from services.synthesis_engine import SynthesisEngine
from utils.cache_serialization import dumps_for_cache


@dataclass
class ResearchExecutorDependencies:
    """Concrete dependencies bundle for research executor tools."""

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
    embedding_service: Any | None = None  # Optional semantic helper

    # Research context
    original_query: str = ""
    search_results: list[dict[str, Any]] = field(default_factory=list)
    source_repository: AbstractSourceRepository | None = None


async def extract_hierarchical_findings(
    deps: ResearchExecutorDependencies,
    source_content: str,
    source_metadata: dict[str, Any] | None = None,
) -> list[HierarchicalFinding]:
    """Extract and classify findings hierarchically from source content."""

    logfire.trace("Extracting hierarchical findings from source content")

    try:
        cache_content_key = _generate_cache_key("extract_findings", source_content, source_metadata)

        if deps.cache_manager:
            cached_result = deps.cache_manager.get("extract_findings", cache_content_key)
            if cached_result:
                logfire.debug("Using cached hierarchical findings")
                return cached_result

        findings_data = (
            await deps.synthesis_engine.extract_themes(source_content)
            if hasattr(deps.synthesis_engine, "extract_themes")
            else _extract_findings_fallback(source_content, source_metadata)
        )

        async def _resolve_source(
            existing: ResearchSource | None,
        ) -> tuple[ResearchSource | None, str | None]:
            source_id: str | None = None
            if existing and existing.source_id:
                source_id = existing.source_id
            if source_metadata and source_metadata.get("source_id"):
                source_id = str(source_metadata["source_id"])

            source_obj: ResearchSource | None = None
            if source_id and deps.source_repository:
                source_obj = await deps.source_repository.get(source_id)

            if source_obj is None:
                source_obj = existing

            if source_obj is None and source_metadata:
                source_obj = ResearchSource(
                    source_id=source_id,
                    title=source_metadata.get("title", "Unknown"),
                    url=source_metadata.get("url"),
                    source_type=source_metadata.get("type", "unknown"),
                    author=source_metadata.get("author"),
                    publisher=source_metadata.get("publisher"),
                    metadata={
                        k: v
                        for k, v in source_metadata.items()
                        if k not in {"title", "url", "type", "author", "publisher", "source_id"}
                    },
                )

            if source_obj and source_id and not source_obj.source_id:
                source_obj = source_obj.model_copy(update={"source_id": source_id})

            return source_obj, source_id

        hierarchical_findings: list[HierarchicalFinding] = []
        for finding_data in findings_data:
            if isinstance(finding_data, HierarchicalFinding):
                finding = finding_data.model_copy(deep=True)
                resolved_source, resolved_source_id = await _resolve_source(finding.source)
                finding.source = resolved_source
                if resolved_source_id and resolved_source_id not in finding.source_ids:
                    finding.source_ids.insert(0, resolved_source_id)
                hierarchical_findings.append(finding)
                continue

            if isinstance(finding_data, dict):
                text = finding_data.get("text", source_content[:200])
                confidence = finding_data.get("confidence", 0.7)
                importance = finding_data.get("importance", 0.75)
            else:
                text = str(finding_data)
                confidence = 0.7
                importance = 0.75

            resolved_source, resolved_source_id = await _resolve_source(None)

            finding = HierarchicalFinding(
                finding=text,
                supporting_evidence=[source_content[:200]],
                confidence=ConfidenceLevel.MEDIUM,
                confidence_score=confidence,
                importance=ImportanceLevel.MEDIUM,
                importance_score=importance,
                source=resolved_source,
                category="research",
                temporal_relevance="current",
            )
            if resolved_source_id and resolved_source_id not in finding.source_ids:
                finding.source_ids.insert(0, resolved_source_id)
            hierarchical_findings.append(finding)

        if source_metadata and source_metadata.get("source_id"):
            source_id = str(source_metadata["source_id"])
            for finding in hierarchical_findings:
                if source_id not in finding.source_ids:
                    finding.source_ids.insert(0, source_id)

        if deps.cache_manager:
            _ = deps.cache_manager.set(
                "extract_findings",
                cache_content_key,
                hierarchical_findings,
            )

        logfire.trace("Extracted hierarchical findings", count=len(hierarchical_findings))
        return hierarchical_findings

    except Exception as exc:  # pragma: no cover - when logging fallback
        logfire.error("Failed to extract hierarchical findings", error=str(exc))
        return _extract_findings_fallback(source_content, source_metadata)


async def identify_theme_clusters(
    deps: ResearchExecutorDependencies,
    findings: list[HierarchicalFinding],
    min_cluster_size: int = 2,
) -> list[ThemeCluster]:
    """Group findings into thematic clusters using ML clustering."""

    logfire.info("Identifying theme clusters", findings=len(findings))

    if not findings:
        return []

    try:
        cache_content_key = _generate_cache_key("identify_clusters", findings, min_cluster_size)

        if deps.cache_manager:
            cached_result = deps.cache_manager.get("identify_theme_clusters", cache_content_key)
            if cached_result:
                logfire.debug("Using cached theme clusters")
                return cached_result

        clusters = deps.synthesis_engine.cluster_findings(findings)

        # If embeddings are available, attempt embedding-aware clustering
        try:
            if getattr(deps, "embedding_service", None) is not None and hasattr(
                deps.synthesis_engine, "cluster_findings_from_vectors"
            ):
                texts = []
                for f in findings:
                    parts = [f.finding]
                    if f.category:
                        parts.append(f"Category: {f.category}")
                    if f.supporting_evidence:
                        parts.extend(f.supporting_evidence[:2])
                    texts.append(" ".join(parts))
                vectors = await deps.embedding_service.embed_batch(texts)  # type: ignore[union-attr]
                if vectors:
                    emb_clusters = deps.synthesis_engine.cluster_findings_from_vectors(
                        findings, vectors
                    )
                    # Prefer embedding clustering if it yields at least as many clusters
                    if emb_clusters:
                        clusters = emb_clusters
        except Exception as exc:  # pragma: no cover - safe fallback
            logfire.warning("Embedding clustering failed; using TF-IDF", error=str(exc))

        theme_clusters: list[ThemeCluster] = []
        if clusters and isinstance(clusters[0], ThemeCluster):
            theme_clusters = clusters
        else:
            for index, cluster in enumerate(clusters):
                if isinstance(cluster, dict):
                    theme_clusters.append(
                        ThemeCluster(
                            theme_name=cluster.get("name", f"Theme {index + 1}"),
                            description=cluster.get("description", "Clustered findings"),
                            findings=cluster.get("findings", []),
                            coherence_score=cluster.get("coherence", 0.5),
                            importance_score=cluster.get("importance", 0.5),
                        )
                    )

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

        if deps.cache_manager:
            deps.cache_manager.set(
                "identify_theme_clusters",
                cache_content_key,
                theme_clusters,
            )

        logfire.info("Identified theme clusters", count=len(theme_clusters))
        return theme_clusters

    except Exception as exc:  # pragma: no cover - logging fallback only
        logfire.error("Failed to identify theme clusters", error=str(exc))
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
    deps: ResearchExecutorDependencies,
    findings: list[HierarchicalFinding],
) -> list[Contradiction]:
    """Detect contradictions between findings using advanced analysis."""

    logfire.info("Detecting contradictions", findings=len(findings))

    if not findings or len(findings) < 2:
        return []

    try:
        cache_content_key = _generate_cache_key("detect_contradictions", findings)

        if deps.cache_manager:
            cached_result = deps.cache_manager.get("detect_contradictions", cache_content_key)
            if cached_result:
                logfire.debug("Using cached contradiction analysis")
                return cached_result

        # Inject embedding service into detector if available for topic matching
        try:
            if getattr(deps, "embedding_service", None) is not None and hasattr(
                deps.contradiction_detector, "detect_contradictions_with_vectors"
            ):
                # Precompute embeddings asynchronously and call the vector-aware method
                vectors = await deps.embedding_service.embed_batch([f.finding for f in findings])  # type: ignore[union-attr]
                # Align threshold with similarity threshold (slightly stricter)
                if hasattr(deps.contradiction_detector, "topic_similarity_threshold"):
                    deps.contradiction_detector.topic_similarity_threshold = max(  # type: ignore[attr-defined]
                        0.0, min(1.0, getattr(deps, "similarity_threshold", 0.6))
                    )
                contradictions = deps.contradiction_detector.detect_contradictions_with_vectors(  # type: ignore[attr-defined]
                    findings,
                    vectors if vectors else [],
                )
            else:
                contradictions = deps.contradiction_detector.detect_contradictions(findings)
        except Exception:
            contradictions = deps.contradiction_detector.detect_contradictions(findings)

        if deps.cache_manager:
            deps.cache_manager.set(
                "detect_contradictions",
                cache_content_key,
                contradictions,
            )

        logfire.info("Detected contradictions", count=len(contradictions))
        return contradictions

    except Exception as exc:  # pragma: no cover
        logfire.error("Failed to detect contradictions", error=str(exc))
        return []


async def analyze_patterns(
    deps: ResearchExecutorDependencies,
    findings: list[HierarchicalFinding],
    clusters: list[ThemeCluster],
) -> list[PatternAnalysis]:
    """Analyze patterns across findings and clusters."""

    logfire.info(
        "Analyzing patterns",
        findings=len(findings),
        clusters=len(clusters),
    )

    patterns: list[PatternAnalysis] = []

    if not findings:
        return patterns

    try:
        cache_content_key = _generate_cache_key("analyze_patterns", findings, clusters)

        if deps.cache_manager:
            cached_result = deps.cache_manager.get("analyze_patterns", cache_content_key)
            if cached_result:
                logfire.debug("Using cached pattern analysis")
                return cached_result

        detected_patterns = deps.pattern_recognizer.detect_patterns(findings)

        # detected_patterns already returns list[PatternAnalysis], so use them directly
        patterns.extend(detected_patterns)

        if len(findings) > 3:
            high_confidence_findings = [f for f in findings if f.confidence_score > 0.8]
            if len(high_confidence_findings) > len(findings) * 0.6:
                patterns.append(
                    PatternAnalysis(
                        pattern_type=PatternType.CONVERGENCE,
                        pattern_name="High Confidence Convergence",
                        description="Majority of findings show high confidence convergence",
                        strength=0.8,
                        finding_ids=[str(index) for index in range(len(high_confidence_findings))],
                        implications=["Strong consensus in research findings"],
                        confidence_factors={"source_agreement": 0.8, "data_consistency": 0.75},
                    )
                )

        temporal_findings = [f for f in findings if f.temporal_relevance]
        if temporal_findings:
            patterns.append(
                PatternAnalysis(
                    pattern_type=PatternType.TEMPORAL,
                    pattern_name="Temporal Evolution",
                    description="Findings show temporal progression",
                    strength=0.6,
                    finding_ids=[str(index) for index in range(len(temporal_findings))],
                    temporal_span="recent",
                    implications=["Research shows evolution over time"],
                )
            )

        if deps.cache_manager:
            deps.cache_manager.set(
                "analyze_patterns",
                cache_content_key,
                patterns,
            )

        logfire.info("Analyzed patterns", count=len(patterns))
        return patterns

    except Exception as exc:  # pragma: no cover
        logfire.error("Failed to analyze patterns", error=str(exc))
        return []


async def generate_executive_summary(
    deps: ResearchExecutorDependencies,
    findings: list[HierarchicalFinding],
    contradictions: list[Contradiction],
    patterns: list[PatternAnalysis],
) -> ExecutiveSummary:
    """Generate an executive summary of the research."""

    logfire.info("Generating executive summary")

    try:
        # Prefer selection_score (if provided) then fall back to importance_score
        def _score(f: HierarchicalFinding) -> float:
            try:
                if isinstance(getattr(f, "metadata", None), dict):
                    val = f.metadata.get("selection_score")
                    if isinstance(val, (int | float)):
                        return float(val)
            except Exception:
                pass
            return float(getattr(f, "importance_score", 0.0))

        sorted_findings = sorted(findings, key=_score, reverse=True)
        key_findings = [finding.finding for finding in sorted_findings[:5]]

        avg_confidence = (
            sum(finding.confidence_score for finding in findings) / len(findings) if findings else 0
        )
        confidence_assessment = f"Overall confidence: {avg_confidence:.2f} - "
        if avg_confidence > 0.8:
            confidence_assessment += "High confidence in research findings"
        elif avg_confidence > 0.6:
            confidence_assessment += "Moderate confidence with some uncertainties"
        else:
            confidence_assessment += "Low confidence, further research recommended"

        critical_gaps: list[str] = []
        if not findings:
            critical_gaps.append("No findings extracted from available sources")
        if len(contradictions) > 3:
            critical_gaps.append(
                f"Multiple contradictions ({len(contradictions)}) require resolution"
            )
        if avg_confidence < 0.6:
            critical_gaps.append("Low overall confidence in findings")

        recommended_actions: list[str] = []
        if critical_gaps:
            recommended_actions.append("Address identified gaps through additional research")
        if contradictions:
            recommended_actions.append("Investigate and resolve contradictory findings")
        if patterns:
            for pattern in patterns[:2]:
                if pattern.implications:
                    recommended_actions.append(f"Consider implications: {pattern.implications[0]}")

        risk_factors: list[str] = []
        if contradictions:
            risk_factors.append(f"{len(contradictions)} unresolved contradictions")
        low_confidence = [finding for finding in findings if finding.confidence_score < 0.5]
        if low_confidence:
            risk_factors.append(f"{len(low_confidence)} low-confidence findings")

        return ExecutiveSummary(
            key_findings=key_findings,
            confidence_assessment=confidence_assessment,
            critical_gaps=critical_gaps,
            recommended_actions=recommended_actions,
            risk_factors=risk_factors,
        )

    except Exception as exc:  # pragma: no cover
        logfire.error("Failed to generate executive summary", error=str(exc))
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
    """Assess the quality of the synthesis."""

    logfire.info("Assessing synthesis quality")

    try:
        metrics: dict[str, float] = {}

        search_count = len(deps.search_results) if deps.search_results else 0
        expected_findings = max(10, search_count * 2)
        completeness = min(1.0, len(findings) / expected_findings) if expected_findings else 0.0
        metrics["completeness"] = completeness

        if clusters:
            avg_coherence = sum(cluster.coherence_score for cluster in clusters) / len(clusters)
            metrics["coherence"] = avg_coherence
        else:
            metrics["coherence"] = 0.5

        if findings:
            avg_confidence = sum(finding.confidence_score for finding in findings) / len(findings)
            metrics["average_confidence"] = avg_confidence
        else:
            metrics["average_confidence"] = 0.0

        contradiction_penalty = min(0.3, len(contradictions) * 0.05)
        metrics["reliability"] = max(0.0, 1.0 - contradiction_penalty)

        metrics["overall_quality"] = (
            metrics["completeness"] * 0.25
            + metrics["coherence"] * 0.25
            + metrics["average_confidence"] * 0.25
            + metrics["reliability"] * 0.25
        )

        if deps.metrics_collector:
            await deps.metrics_collector.record_synthesis_metrics(metrics)

        return metrics

    except Exception as exc:  # pragma: no cover
        logfire.error("Failed to assess synthesis quality", error=str(exc))
        return {
            "completeness": 0.0,
            "coherence": 0.0,
            "average_confidence": 0.0,
            "reliability": 0.0,
            "overall_quality": 0.0,
        }


def _generate_cache_key(*args: Any) -> str:
    """Generate cache key from arguments."""

    serialized = dumps_for_cache(args)
    return hashlib.md5(serialized.encode()).hexdigest()[:16]


def _extract_findings_fallback(
    source_content: str, source_metadata: dict[str, Any] | None = None
) -> list[HierarchicalFinding]:
    """Fallback method for extracting findings when services are unavailable."""

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


__all__ = [
    "ResearchExecutorDependencies",
    "extract_hierarchical_findings",
    "identify_theme_clusters",
    "detect_contradictions",
    "analyze_patterns",
    "generate_executive_summary",
    "assess_synthesis_quality",
    "_generate_cache_key",
    "_extract_findings_fallback",
]
