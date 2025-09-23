"""Research executor agent implementation."""

from __future__ import annotations

import uuid
from textwrap import dedent
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
    ResearchResults,
    ResearchSource,
    SynthesisMetadata,
    ThemeCluster,
)
from services.source_repository import InMemorySourceRepository, ensure_repository
from services.source_validation import create_default_validation_pipeline

from .research_executor_tools import (
    ResearchExecutorDependencies,
    analyze_patterns,
    assess_synthesis_quality,
    detect_contradictions,
    extract_hierarchical_findings,
    generate_executive_summary,
    identify_theme_clusters,
)

RESEARCH_EXECUTOR_SYSTEM_PROMPT = dedent(
    """
    # Role: Hybrid Research Synthesis Orchestrator

    You coordinate deterministic web research with deep synthesis. Use the
    registered tools to inspect findings, cluster themes, probe for
    contradictions, and assess quality before finalizing outputs.

    ## Tree-of-Thought Workflow
    1. Pattern Discovery
       - Map convergent, divergent, emergent, and temporal signals.
       - Use clustering to understand topical structure.
    2. Insight Formation
       - Distill the most decision-relevant findings.
       - Resolve or flag contradictions explicitly.
    3. Quality Verification
       - Evaluate completeness, coherence, confidence, and reliability.
       - Call the quality tool to numerically score the synthesis.

    ## Operational Guidelines
    - Prefer tool calls over internal speculation when data is required.
    - Always tie insights back to specific findings and sources.
    - Surface meaningful gaps, risks, and recommended next actions.
    - Produce a fully populated `ResearchResults` instance.
    """
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
            search_queries = deps.get_search_query_batch()
            search_results = getattr(deps, "search_results", []) or []

            query_summary = self._summarize_search_queries(search_queries)
            result_summary = self._summarize_search_results(search_results)

            return dedent(
                f"""
                ## Dynamic Research Context
                - Stage: {stage}
                - Query: {query}
                - Search Queries: {len(getattr(search_queries, "queries", []) or [])}
                - Search Results: {len(search_results)}

                ### Search Queries Overview
                {query_summary}

                ### Search Results Snapshot
                {result_summary}

                ### Tooling Checklist
                - Extract findings before clustering themes.
                - Examine contradictions and patterns from findings.
                - Run quality assessment prior to finalizing results.
                """
            ).strip()

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

    def _summarize_search_queries(self, search_queries: Any) -> str:
        """Create a compact summary of provided search queries."""

        if not search_queries or not getattr(search_queries, "queries", None):
            return "- No query details provided."

        snippets: list[str] = []
        for query in search_queries.queries[:3]:
            if hasattr(query, "query"):
                text = getattr(query, "query", "").strip()
                priority = getattr(query, "priority", "?")
                snippets.append(f"- {text} (priority: {priority})")
            else:
                snippets.append(f"- {str(query)}")
        if len(search_queries.queries) > 3:
            snippets.append("- …")
        return "\n".join(snippets)

    def _summarize_search_results(self, search_results: list[dict[str, Any]]) -> str:
        """Provide a snapshot of available search results."""

        if not search_results:
            return "- No search results available."

        snippets: list[str] = []
        for result in search_results[:3]:
            title = result.get("title") or "Untitled"
            url = result.get("url") or "no-url"
            snippet = result.get("content") or result.get("snippet") or ""
            snippet = " ".join(snippet.split())
            if len(snippet) > 160:
                snippet = f"{snippet[:157]}…"
            snippets.append(f"- {title} ({url}) — {snippet}")
        if len(search_results) > 3:
            snippets.append("- …")
        return "\n".join(snippets)

    def _get_default_system_prompt(self) -> str:
        """Return a concise system prompt describing the agent role."""

        return RESEARCH_EXECUTOR_SYSTEM_PROMPT

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
            "source_repository": getattr(deps, "source_repository", None),
            "embedding_service": getattr(deps, "embedding_service", None),
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
        source_repository = await ensure_repository(executor_deps.source_repository)
        if deps.source_repository is None:
            deps.source_repository = source_repository
        if executor_deps.source_repository is None:
            executor_deps.source_repository = source_repository

        validation_pipeline = None
        if getattr(deps, "http_client", None) is not None:
            validation_pipeline = create_default_validation_pipeline(
                repository=source_repository,
                http_client=deps.http_client,
            )

        for index, result in enumerate(executor_deps.search_results, start=1):
            content = result.get("content") or result.get("snippet") or ""
            if not content.strip():
                continue
            snippet = " ".join(content.split())[:500]
            metadata = {
                "title": result.get("title", "Unknown"),
                "url": result.get("url"),
                "type": result.get("type") or result.get("source_type", "unknown"),
                "publisher": result.get("metadata", {}).get("publisher"),
                "author": result.get("metadata", {}).get("author"),
                "snippet": snippet,
                "score": result.get("score"),
                "raw_index": index,
            }
            raw_source_payload = {
                "title": metadata["title"],
                "url": metadata.get("url"),
                "source_type": metadata["type"],
                "author": metadata.get("author"),
                "publisher": metadata.get("publisher"),
                "metadata": {
                    k: v
                    for k, v in metadata.items()
                    if k not in {"title", "url", "type", "author", "publisher"}
                },
            }

            if validation_pipeline is not None:
                registered_source = await validation_pipeline.validate_and_register(
                    raw_source_payload
                )
            else:
                identity = await source_repository.register(ResearchSource(**raw_source_payload))
                registered_source = await source_repository.get(identity.source_id)
            if registered_source and registered_source.source_id:
                metadata["source_id"] = registered_source.source_id
            extracted = await extract_hierarchical_findings(executor_deps, content, metadata)
            for finding in extracted:
                if registered_source:
                    finding.source = registered_source
                    if (
                        registered_source.source_id
                        and registered_source.source_id not in finding.source_ids
                    ):
                        finding.source_ids.insert(0, registered_source.source_id)
                        await source_repository.register_usage(
                            registered_source.source_id,
                            finding_id=finding.finding_id,
                        )
                findings.append(finding)

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

        ordered_sources = await source_repository.ordered_sources()

        sources: list[ResearchSource] = ordered_sources

        finding_by_id: dict[str, HierarchicalFinding] = {
            finding.finding_id: finding for finding in findings
        }
        for index, finding in enumerate(findings):
            finding_by_id.setdefault(str(index), finding)

        for cluster in clusters:
            for finding in cluster.findings:
                for source_id in finding.source_ids:
                    await source_repository.register_usage(
                        source_id,
                        finding_id=finding.finding_id,
                        cluster_id=cluster.cluster_id,
                    )

        for pattern in patterns:
            related_finding_ids = getattr(pattern, "finding_ids", [])
            for fid in related_finding_ids:
                referenced = finding_by_id.get(fid)
                if not referenced:
                    continue
                for source_id in referenced.source_ids:
                    await source_repository.register_usage(
                        source_id,
                        finding_id=referenced.finding_id,
                        pattern_id=pattern.pattern_id,
                    )

        for contradiction in contradictions:
            if not contradiction.id:
                contradiction.id = uuid.uuid4().hex
            related_findings: list[HierarchicalFinding] = []
            for attr in ("finding_1_id", "finding_2_id"):
                fid = getattr(contradiction, attr, None)
                if fid and fid in finding_by_id:
                    related_findings.append(finding_by_id[fid])
            for finding in related_findings:
                for source_id in finding.source_ids:
                    await source_repository.register_usage(
                        source_id,
                        finding_id=finding.finding_id,
                        contradiction_id=contradiction.id,
                    )

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
            "search_query_count": len(getattr(deps.get_search_query_batch(), "queries", []) or []),
            "generation_mode": "structured_fallback",
        }
        research_results = ResearchResults(
            query=query,
            findings=findings,
            theme_clusters=clusters,
            contradictions=contradictions,
            executive_summary=summary,
            patterns=patterns,
            sources=sources,
            synthesis_metadata=synthesis_metadata,
            overall_quality_score=overall_quality,
            content_hierarchy=content_hierarchy,
            key_insights=key_insights,
            data_gaps=data_gaps,
            metadata=metadata,
        )

        for source in sources:
            if not source.source_id:
                continue
            usage = await source_repository.get_usage(source.source_id)
            if usage:
                research_results.source_usage[source.source_id] = usage

        deps.research_state.metadata.sources_consulted = len(sources)

        return research_results


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
        deps.source_repository = InMemorySourceRepository()
        result = await agent_instance.run(deps)
        research_state.research_results = result
        return result
