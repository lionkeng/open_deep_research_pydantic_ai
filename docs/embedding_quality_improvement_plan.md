# Embedding-Driven Report Quality Improvement Plan

This plan details how to incorporate the EmbeddingService to measurably improve report quality without changing the report’s visible structure.

- Objective: Higher-quality reports (readability, cohesion, support) using embeddings for deduplication, clustering, selection, and composition.
- Constraint: No new explicit sections (e.g., “Finding Convergence”); improvements happen upstream of ReportGenerator.
- Rollout: Incremental, safe defaults; all changes are feature-flagged via global config already in place.

## Principles

- Preserve structure; improve inputs (findings, clusters, evidence, citations).
- Embeddings augment deterministic logic; always keep fallbacks.
- Observability: log before/after stats and store metrics in SynthesisMetadata/quality_metrics.

## High-Impact Enhancements

1) Embedding-based dedup + merge
- What: Group paraphrase-equivalent findings and merge them:
  - Union source_ids and supporting_evidence
  - Keep the best snippet for evidence
  - Average/weighted scores (confidence, importance)
- Effects: Fewer, stronger findings; reduced repetition; higher support per statement.
- Where:
  - New: `src/services/dedup.py` with `DeDupService(embedding_service, threshold)`
  - Flow: ResearchExecutor after `extract_hierarchical_findings()` and before `identify_theme_clusters()`
- Logs + Metrics:
  - `dedup_in`, `dedup_out`, `dedup_merged_groups`, `dedup_avg_group_size`
  - Add `dedup_merge_ratio = 1 - (out / in)` to `quality_metrics`

2) Embedding-aware theme clustering
- What: Optional clustering path using embeddings (KMeans/HDBSCAN on vectors) and coherence via cosine.
- Effects: Higher intra-cluster coherence, fewer mixed-topic clusters; better flow in sections.
- Where:
  - Update: `src/services/synthesis_engine.py`
    - Add an embedding path gated by `enable_embedding_similarity`
    - Maintain TF-IDF path as fallback
- Logs + Metrics:
  - `embedding_clustering_applied`, `clusters`, `avg_cluster_size`
  - `cluster_coherence_before/after` (if viable to compare) or simply `avg_cluster_coherence`

3) Embedding-weighted finding selection (ranking)
- What: Re-rank findings by:
  - centrality in embedding space within clusters (mean cosine to centroid)
  - support count (# of distinct sources after merge)
  - query alignment (cosine(query_embedding, finding_embedding))
- Effects: Summary and body lead with representative, well-supported insights; reduced noise.
- Where:
  - `ResearchExecutor` after clustering
  - Pass top-K ranked findings first to `generate_executive_summary`, and use ordering to compose body.
- Logs + Metrics:
  - `summary_candidates_ranked` (top-k with scores)
  - `query_alignment_avg`, `avg_support_per_finding`

4) Embedding-aware citation consolidation
- What: When multiple sources back a merged finding, choose 1–2 representative sources by cosine similarity to the finding (title/snippet), with credibility tie-breakers.
- Effects: Tighter, more relevant citations while preserving coverage.
- Where:
  - Pre-ReportGenerator step in `ResearchExecutor` (derive ordered `source_ids`)
- Logs + Metrics:
  - `citations_per_finding`, `citation_coverage` (existing), `representative_pick_rate`

5) Contradiction detection robustness (optional follow-up)
- What: Use embedding similarity + polarity detection to catch paraphrased contradictions.
- Effects: More nuanced contradiction identification supports better synthesis narrative.
- Where:
  - Update `src/services/contradiction_detector.py` (optional path gated by flag)

6) Cleaner paragraph composition (evidence fusion)
- What: For each paragraph, pick 1–2 salient evidence sentences ranked by embedding similarity and prompt the LLM to integrate them inline while preserving `[Sx]` markers.
- Effects: Smoother prose and denser support; no new structural labels.
- Where:
  - Provide `salient_evidence` inputs before ReportGenerator; reuse clean-merge guardrails for markers.
- Logs + Metrics:
  - `fusion_applied`, `avg_evidence_per_paragraph`, `clean_merge_applied`

## Wiring into the Codebase

- New: `src/services/dedup.py`
  - API: `class DeDupService: async def merge(self, findings) -> list[HierarchicalFinding]`
  - Internals: Use `EmbeddingService.embed_batch()`, reuse `pairwise_cosine_matrix` and `cluster_by_threshold`

- Update: `src/services/synthesis_engine.py`
  - Add `use_embeddings` branch (when `enable_embedding_similarity`) to cluster on embeddings and compute coherence via cosine.

- Update: `src/agents/research_executor.py`
  - After extraction: `findings = await deduper.merge(findings)`
  - After clustering: compute cluster centroids; rank findings by centrality + support + query alignment
  - Order candidates for summary/body; consolidate citations
  - Keep current pattern analysis; optional contradiction enhancement later

- Update: `src/agents/report_generator.py`
  - Accept `salient_evidence` (preselected sentences) to inform the LLM prompt subtly without changing section headings
  - Continue to run guardrailed clean-merge for the executive summary (already present)

- Config & Flags (already centralized in `core.config`)
  - `enable_embedding_similarity`, `embedding_similarity_threshold`, `enable_llm_clean_merge`
  - Expose a `dedup_similarity_threshold` if we need a separate threshold (default to same as global)

## Observability & Metrics

- Per-step logs (already partly added):
  - `Embedding batch` (backend, batch_size, computed, cache_hits, seconds)
  - `Embedding grouping applied` (threshold, clusters, avg_size)
  - `Convergence analysis` (points) — internal quality signal
- Additional quality metrics in `SynthesisMetadata.quality_metrics`:
  - `dedup_merge_ratio`
  - `avg_cluster_coherence`
  - `avg_support_per_finding`
  - `query_alignment_avg`
  - `convergence_points` (already added)

## Rollout Plan

- Phase 1 (safe, visible gains)
  - DeDupService + ranking (centrality/support/query alignment)
  - Optional convergence logs only (no section changes)

- Phase 2 (clustering quality)
  - Embedding-aware clustering + coherence metric; fallback to TF‑IDF

- Phase 3 (citations & contradictions)
  - Contextual citation consolidation; enhanced contradiction detection (optional)

- Phase 4 (composition polish)
  - Paragraph-level evidence fusion with guardrails; keep markers intact

## Risks & Mitigations

- API cost/latency: batch embeddings + cache; cap texts; only embed when enabled.
- Drift in content: we merge/weight inputs; clean-merge already enforces citation markers.
- Backwards compatibility: flags off by default; all fallbacks preserved.

## Acceptance Criteria

- With `enable_embedding_similarity=True`:
  - Lower redundancy (dedup_merge_ratio > 0), higher avg_support_per_finding, improved avg_cluster_coherence.
  - Judge eval shows improved readability and thematic clarity without harming citation quality.
- With `enable_llm_clean_merge=True`:
  - Executive summary readability improved; markers identical; logs show “Clean-merge applied”.

## Implementation Tasks (Checklist)

- [ ] Add `src/services/dedup.py` with `DeDupService`
- [ ] Wire dedup merge into `ResearchExecutor` (post-extraction)
- [ ] Add ranking step (centrality/support/query alignment) and feed to summary
- [ ] Add embedding clustering path to `SynthesisEngine` (gated by flag)
- [ ] Implement citation consolidation in executor (pre-ReportGenerator)
- [ ] (Optional) Enhance contradiction detector with embedding check
- [ ] (Optional) Add per-paragraph `salient_evidence` selection and guarded fusion
- [ ] Extend metrics + logs; update `SynthesisMetadata`
- [ ] Add unit/integration tests and update eval to track new metrics
