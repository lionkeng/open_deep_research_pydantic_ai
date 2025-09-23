"""Research executor agent implementation."""

from __future__ import annotations

import uuid
from textwrap import dedent
from types import SimpleNamespace
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
from core.config import config as global_config
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
from services.dedup import DeDupService
from services.source_repository import InMemorySourceRepository, ensure_repository
from services.source_validation import create_default_validation_pipeline
from services.synthesis_tools import SearchResult as SynthesisSearchResult
from services.synthesis_tools import SynthesisTools

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

        # Metrics: record pre-dedup count
        initial_findings_count = len(findings)

        # Phase 1: optional embedding-based deduplication
        try:
            if (
                getattr(deps, "enable_embedding_similarity", False)
                and getattr(deps, "embedding_service", None) is not None
            ):
                # Simple cap to bound O(n^2) cost
                cap = max(0, int(getattr(global_config, "dedup_max_findings", 0)))
                threshold = float(getattr(global_config, "dedup_similarity_threshold", 0.7))
                if cap and len(findings) > cap:
                    top = sorted(findings, key=lambda f: f.importance_score, reverse=True)[:cap]
                    rest = [f for f in findings if f not in top]
                    deduper = DeDupService(
                        embedding_service=getattr(deps, "embedding_service", None),
                        threshold=threshold,
                    )
                    merged_top = await deduper.merge(top)
                    findings = merged_top + rest
                    logfire.info(
                        "Dedup applied with cap",
                        cap=cap,
                        in_count=len(top),
                        out_count=len(merged_top),
                        rest=len(rest),
                    )
                else:
                    deduper = DeDupService(
                        embedding_service=getattr(deps, "embedding_service", None),
                        threshold=threshold,
                    )
                    findings = await deduper.merge(findings)
        except Exception as exc:  # pragma: no cover - safe fallback
            logfire.warning("Dedup merge failed", error=str(exc))

        clusters = await identify_theme_clusters(executor_deps, findings)
        contradictions = await detect_contradictions(executor_deps, findings)
        patterns = await analyze_patterns(executor_deps, findings, clusters)

        # Optional: embedding-aware convergence analysis (uses SynthesisTools)
        convergence_points_count = 0
        try:
            if getattr(deps, "enable_embedding_similarity", False) and getattr(
                executor_deps, "search_results", None
            ):
                # Group aggregated search_results by query and build lightweight
                # items with just a `content` attribute for analysis
                grouped: dict[str, list[SimpleNamespace]] = {}
                for item in executor_deps.search_results:
                    q = str(item.get("query", "unknown"))
                    content = item.get("content") or item.get("snippet") or ""
                    if not content:
                        continue
                    grouped.setdefault(q, []).append(SimpleNamespace(content=content))

                search_payload: list[SynthesisSearchResult] = [
                    SynthesisSearchResult(query=q, results=items) for q, items in grouped.items()
                ]

                tools = SynthesisTools(
                    embedding_service=getattr(deps, "embedding_service", None),
                    enable_embedding_similarity=getattr(deps, "enable_embedding_similarity", False),
                    similarity_threshold=getattr(deps, "similarity_threshold", 0.55),
                )
                # Precompute embeddings in the caller and pass vectors in
                precomputed_vectors = None
                try:
                    if getattr(deps, "embedding_service", None) is not None:
                        # Important: extract_claims_for_convergence applies sampling/caps
                        # so vectors align with what analyze_convergence will use.
                        claims = tools.extract_claims_for_convergence(search_payload)
                        texts = [c for c, _ in claims]
                        if texts:
                            import asyncio as _asyncio
                            from time import perf_counter as _pc

                            _t0 = _pc()
                            precomputed_vectors = await _asyncio.wait_for(
                                deps.embedding_service.embed_batch(texts),  # type: ignore[union-attr]
                                timeout=20.0,
                            )
                            logfire.info(
                                "Convergence embedding precompute",
                                count=len(texts),
                                duration_ms=int((_pc() - _t0) * 1000),
                            )
                except Exception as exc:
                    logfire.warning("Convergence embedding precompute failed", error=str(exc))

                convergence_points = await tools.analyze_convergence(
                    search_payload, precomputed_vectors=precomputed_vectors or None
                )
                convergence_points_count = len(convergence_points)
                logfire.info(
                    "Convergence analysis",
                    embedding_similarity=getattr(deps, "enable_embedding_similarity", False),
                    points=convergence_points_count,
                )
        except Exception as exc:  # pragma: no cover - best-effort logging
            logfire.warning("Convergence analysis failed", error=str(exc))
        # Ranking: centrality/support/query alignment to inform summary order
        query_alignment_avg: float | None = None
        try:
            if (
                getattr(deps, "enable_embedding_similarity", False)
                and getattr(deps, "embedding_service", None) is not None
            ):
                # Embed findings and query
                finding_texts = [f.finding for f in findings]
                vectors = await deps.embedding_service.embed_batch(finding_texts)  # type: ignore[union-attr]
                # Query embedding
                query_vec = (
                    (await deps.embedding_service.embed_batch([query]))[0] if vectors else []
                )  # type: ignore[union-attr]

                # Build index per finding for quick lookup
                idx_map = {id(f): i for i, f in enumerate(findings)}

                # Compute per-cluster centroids and selection scores off-thread
                import asyncio as _asyncio

                import numpy as np  # local import for scoring only

                def _compute_ranking(
                    clusters_in, findings_in, vectors_in, query_vec_in, idx_map_in
                ) -> tuple[dict[int, float], float, int]:
                    def _cos(u: list[float], v: list[float]) -> float:
                        import math

                        if not u or not v:
                            return 0.0
                        dot = sum(a * b for a, b in zip(u, v, strict=False))
                        nu = math.sqrt(sum(a * a for a in u))
                        nv = math.sqrt(sum(b * b for b in v))
                        return (dot / (nu * nv)) if nu and nv else 0.0

                    sel_scores: dict[int, float] = {}
                    a_sum = 0.0
                    a_cnt = 0
                    for cluster in clusters_in:
                        member_idxs = [idx_map_in.get(id(f), -1) for f in cluster.findings]
                        member_idxs = [
                            i for i in member_idxs if i >= 0 and vectors_in and i < len(vectors_in)
                        ]
                        if not member_idxs:
                            continue
                        mat = np.array([vectors_in[i] for i in member_idxs])
                        centroid = mat.mean(axis=0)
                        for f in cluster.findings:
                            i = idx_map_in.get(id(f), -1)
                            if i < 0 or not vectors_in:
                                continue
                            centrality = _cos(vectors_in[i], centroid.tolist())
                            support = float(len(f.source_ids))
                            q_align = _cos(vectors_in[i], query_vec_in) if query_vec_in else 0.0
                            import math as _m

                            support_n = _m.log1p(support) / _m.log1p(5.0)
                            score = 0.5 * centrality + 0.3 * support_n + 0.2 * q_align
                            sel_scores[id(f)] = float(score)
                            a_sum += float(q_align)
                            a_cnt += 1
                    return sel_scores, a_sum, a_cnt

                from time import perf_counter as _pc

                _t_rank = _pc()
                selection_scores, align_sum, align_count = await _asyncio.to_thread(
                    _compute_ranking, clusters, findings, vectors, query_vec, idx_map
                )
                for f in findings:
                    s = selection_scores.get(id(f))
                    if s is not None:
                        f.metadata["selection_score"] = float(s)
                logfire.info(
                    "Ranking applied for summary candidates",
                    duration_ms=int((_pc() - _t_rank) * 1000),
                )
                if align_count:
                    query_alignment_avg = align_sum / align_count

                # Phase 3 (partial): representative citation ordering per finding
                # Reorder each finding's source_ids so that the most representative
                # sources (by cosine similarity to the finding text) come first.
                try:
                    from time import perf_counter as _pc

                    _t_reorder = _pc()
                    for f in findings:
                        if not f.source_ids:
                            continue
                        # Build comparable texts for each source: title + snippet
                        candidate_texts: list[tuple[str, str]] = []
                        for sid in f.source_ids[:5]:  # limit to first 5
                            # We stored snippet in metadata during extraction; fall back to title
                            text = (
                                f.metadata.get("snippet", "")
                                if isinstance(f.metadata, dict)
                                else ""
                            )
                            title = f.source.title if (f.source and f.source.title) else ""
                            candidate_texts.append((sid, (title + " " + text).strip()))

                        if not candidate_texts:
                            continue
                        # Embed finding once and candidates in a batch
                        cand_texts_only = [t for _, t in candidate_texts]
                        import asyncio as _asyncio

                        find_vec = (
                            await _asyncio.wait_for(
                                deps.embedding_service.embed_batch([f.finding]),  # type: ignore[union-attr]
                                timeout=10.0,
                            )
                        )[0]
                        cand_vecs = await _asyncio.wait_for(
                            deps.embedding_service.embed_batch(cand_texts_only),  # type: ignore[union-attr]
                            timeout=10.0,
                        )

                        def cos(u: list[float], v: list[float]) -> float:
                            import math

                            if not u or not v:
                                return 0.0
                            dot = sum(a * b for a, b in zip(u, v, strict=False))
                            nu = math.sqrt(sum(a * a for a in u))
                            nv = math.sqrt(sum(b * b for b in v))
                            return (dot / (nu * nv)) if nu and nv else 0.0

                        scored = []
                        for (sid, _txt), v in zip(candidate_texts, cand_vecs, strict=False):
                            scored.append((sid, cos(find_vec, v)))
                        scored.sort(key=lambda x: x[1], reverse=True)
                        # Reorder source_ids by similarity (keep ones not in candidates at the end)
                        ordered = [sid for sid, _ in scored] + [
                            sid for sid in f.source_ids if sid not in {s for s, _ in scored}
                        ]
                        f.source_ids = ordered[:]
                    logfire.info(
                        "Source citation reordering completed",
                        duration_ms=int((_pc() - _t_reorder) * 1000),
                    )
                except Exception as _exc:
                    # Best effort only; keep existing order if anything fails
                    pass
        except Exception as exc:  # pragma: no cover
            logfire.warning("Ranking step failed", error=str(exc))

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
        # Record convergence count as a supplementary metric
        try:
            if quality_metrics is not None:
                quality_metrics["convergence_points"] = float(convergence_points_count)
                # Dedup merge ratio
                if initial_findings_count > 0:
                    quality_metrics["dedup_merge_ratio"] = max(
                        0.0, 1.0 - (len(findings) / initial_findings_count)
                    )
                # Avg support per finding
                if findings:
                    quality_metrics["avg_support_per_finding"] = sum(
                        len(f.source_ids) for f in findings
                    ) / len(findings)
                else:
                    quality_metrics["avg_support_per_finding"] = 0.0
                # Avg cluster coherence
                if clusters:
                    quality_metrics["avg_cluster_coherence"] = sum(
                        c.coherence_score for c in clusters
                    ) / len(clusters)
                else:
                    quality_metrics["avg_cluster_coherence"] = 0.0
                # Query alignment avg if available
                if query_alignment_avg is not None:
                    quality_metrics["query_alignment_avg"] = float(query_alignment_avg)
        except Exception:
            pass

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
