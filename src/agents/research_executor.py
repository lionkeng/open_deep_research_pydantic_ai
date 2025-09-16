"""Research executor agent implementation."""

from __future__ import annotations

from typing import Any

import httpx
import logfire
from pydantic_ai import RunContext

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
    Contradiction,
    ExecutiveSummary,
    HierarchicalFinding,
    ImportanceLevel,
    PatternAnalysis,
    ResearchFinding,
    ResearchResults,
    SynthesisMetadata,
    ThemeCluster,
)

from .research_executor_tools import (
    ResearchExecutorDependencies,
    analyze_patterns,
    assess_synthesis_quality,
    detect_contradictions,
    extract_hierarchical_findings,
    generate_executive_summary,
    identify_theme_clusters,
)


class ResearchExecutorAgent(BaseResearchAgent[ResearchDependencies, ResearchResults]):
    """Base research agent wrapper around the synthesis helpers."""

    def __init__(
        self,
        config: AgentConfiguration | None = None,
        dependencies: ResearchDependencies | None = None,
    ) -> None:
        if config is None:
            config = AgentConfiguration(
                agent_name="research_executor",
                agent_type="synthesis",
            )
        super().__init__(config=config, dependencies=dependencies)

        @self.agent.instructions
        async def add_synthesis_context(ctx: RunContext[ResearchDependencies]) -> str:
            """Provide dynamic instructions derived from workflow context."""

            deps = ctx.deps
            query = deps.research_state.user_query
            stage = deps.research_state.current_stage.value
            search_queries = getattr(deps, "search_queries", None)
            query_count = len(search_queries.queries) if search_queries else 0
            search_results = getattr(deps, "search_results", []) or []
            search_result_count = len(search_results)

            return (
                "# RESEARCH EXECUTION CONTEXT\n"
                f"Current Stage: {stage}\n"
                f"Original Query: {query}\n"
                f"Search Queries Provided: {query_count}\n"
                f"Search Results Available: {search_result_count}\n"
                "Use the registered synthesis tools to extract findings, cluster themes, "
                "detect contradictions, and assess quality before generating the summary."
            )

        @self.agent.tool
        async def tool_extract_hierarchical_findings(
            ctx: RunContext[ResearchDependencies],
            source_content: str,
            source_metadata: dict[str, Any] | None = None,
        ) -> list[HierarchicalFinding]:
            """Extract structured findings from a raw source snippet.

            Args:
                ctx: Run context exposing dependencies.
                source_content: Raw text pulled from a search result.
                source_metadata: Optional metadata about the source.

            Returns:
                Hierarchical findings enriched with confidence and
                importance scores.
            """

            executor_deps = self._build_executor_dependencies(ctx.deps)
            return await extract_hierarchical_findings(
                executor_deps,
                source_content,
                source_metadata,
            )

        @self.agent.tool
        async def tool_identify_theme_clusters(
            ctx: RunContext[ResearchDependencies],
            findings: list[HierarchicalFinding],
            min_cluster_size: int = 2,
        ) -> list[ThemeCluster]:
            """Cluster findings into coherent thematic groups.

            Args:
                ctx: Run context containing dependencies.
                findings: Findings to group by similarity.
                min_cluster_size: Minimum findings required for a
                    cluster.

            Returns:
                A list of theme clusters summarizing the grouped
                findings.
            """

            executor_deps = self._build_executor_dependencies(ctx.deps)
            return await identify_theme_clusters(
                executor_deps,
                findings,
                min_cluster_size=min_cluster_size,
            )

        @self.agent.tool
        async def tool_detect_contradictions(
            ctx: RunContext[ResearchDependencies],
            findings: list[HierarchicalFinding],
        ) -> list[Contradiction]:
            """Detect contradictions across synthesized findings.

            Args:
                ctx: Run context referencing dependencies.
                findings: Findings to inspect for conflicting evidence.

            Returns:
                A list of contradictions with explanations and
                resolution suggestions.
            """

            executor_deps = self._build_executor_dependencies(ctx.deps)
            return await detect_contradictions(executor_deps, findings)

        @self.agent.tool
        async def tool_analyze_patterns(
            ctx: RunContext[ResearchDependencies],
            findings: list[HierarchicalFinding],
            clusters: list[ThemeCluster],
        ) -> list[PatternAnalysis]:
            """Identify patterns that emerge across findings and clusters.

            Args:
                ctx: Run context containing dependencies.
                findings: Findings forming the pattern search space.
                clusters: Previously generated theme clusters.

            Returns:
                Pattern analysis results describing detected trends and
                their implications.
            """

            executor_deps = self._build_executor_dependencies(ctx.deps)
            return await analyze_patterns(executor_deps, findings, clusters)

        @self.agent.tool
        async def tool_generate_executive_summary(
            ctx: RunContext[ResearchDependencies],
            findings: list[HierarchicalFinding],
            contradictions: list[Contradiction],
            patterns: list[PatternAnalysis],
        ) -> ExecutiveSummary:
            """Produce an executive summary from synthesis artifacts.

            Args:
                ctx: Run context supplying dependencies.
                findings: Findings to highlight.
                contradictions: Contradictions to call out.
                patterns: Patterns to reference in the summary.

            Returns:
                An `ExecutiveSummary` capturing key insights and
                recommended actions.
            """

            executor_deps = self._build_executor_dependencies(ctx.deps)
            return await generate_executive_summary(
                executor_deps,
                findings,
                contradictions,
                patterns,
            )

        @self.agent.tool
        async def tool_assess_synthesis_quality(
            ctx: RunContext[ResearchDependencies],
            findings: list[HierarchicalFinding],
            clusters: list[ThemeCluster],
            contradictions: list[Contradiction],
        ) -> dict[str, float]:
            """Compute quality metrics for the synthesized analysis.

            Args:
                ctx: Run context holding dependencies.
                findings: Findings evaluated in the synthesis.
                clusters: Theme clusters supporting the findings.
                contradictions: Contradictions affecting reliability.

            Returns:
                A dictionary with completeness, coherence,
                average_confidence, reliability, and overall quality
                scores.
            """

            executor_deps = self._build_executor_dependencies(ctx.deps)
            return await assess_synthesis_quality(
                executor_deps,
                findings,
                clusters,
                contradictions,
            )

    def _get_default_system_prompt(self) -> str:
        """Return a concise system prompt describing the agent role."""

        return (
            "You are a hybrid research synthesis agent that combines deterministic "
            "query execution with structured analysis. Use available findings to "
            "produce organized outputs that the downstream report generator can consume."
        )

    def _get_output_type(self) -> type[ResearchResults]:
        return ResearchResults

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
        self,
        deps: ResearchDependencies,
    ) -> ResearchExecutorDependencies:
        """Construct tool dependency bundle with optional overrides."""

        kwargs: dict[str, Any] = {
            "cache_manager": getattr(deps, "cache_manager", None),
            "parallel_executor": getattr(deps, "parallel_executor", None),
            "metrics_collector": getattr(deps, "metrics_collector", None),
            "optimization_manager": getattr(deps, "optimization_manager", None),
            "original_query": deps.research_state.user_query,
            "search_results": getattr(deps, "search_results", []) or [],
        }

        for attr in (
            "synthesis_engine",
            "contradiction_detector",
            "pattern_recognizer",
            "confidence_analyzer",
        ):
            value = getattr(deps, attr, None)
            if value is not None:
                kwargs[attr] = value

        return ResearchExecutorDependencies(**kwargs)

    async def _generate_structured_result(
        self,
        deps: ResearchDependencies,
        executor_deps: ResearchExecutorDependencies,
    ) -> ResearchResults:
        query = deps.research_state.user_query
        findings: list[HierarchicalFinding] = []

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
            executor_deps,
            findings,
            contradictions,
            patterns,
        )
        quality_metrics = await assess_synthesis_quality(
            executor_deps,
            findings,
            clusters,
            contradictions,
        )

        overall_quality = quality_metrics.get("overall_quality", 0.0) if quality_metrics else 0.0

        sources: list = [finding.source for finding in findings if finding.source is not None]

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
            "critical": [
                finding.finding
                for finding in findings
                if finding.importance == ImportanceLevel.CRITICAL
            ],
            "important": [
                finding.finding
                for finding in findings
                if finding.importance == ImportanceLevel.HIGH
            ],
            "supplementary": [
                finding.finding
                for finding in findings
                if finding.importance == ImportanceLevel.MEDIUM
            ],
            "contextual": [
                finding.finding for finding in findings if finding.importance == ImportanceLevel.LOW
            ],
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
