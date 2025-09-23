# Synthesis Engine Improvement Plan

This plan proposes two complementary enhancements to the synthesis stage:

1) Embedding‑based similarity for theme/pattern grouping (higher recall + better clustering)
2) A guardrailed LLM “clean‑merge” pass to improve narrative flow while preserving citations and structure

Both changes are optional and feature‑flagged, preserving current deterministic behavior by default.

## 0) Current State (baseline)

- Extraction, clustering, contradiction/pattern detection, executive summary, and quality scoring are orchestrated by `ResearchExecutorAgent` and implemented by deterministic utilities.
  - Orchestration: `src/agents/research_executor.py`
  - Heuristics: `src/services/synthesis_tools.py` (Jaccard/token overlap, regex‑based specificity, rule‑based contradictions, regex pattern detection)
- Report generation composes a typed `ResearchReport` and converts `[Sx]` markers to numbered footnotes.
  - `src/agents/report_generator.py`

## 1) Embedding‑Based Similarity for Grouping

### Goals
- Improve theme clustering and pattern grouping beyond token overlap by adding semantic similarity.
- Keep the pipeline deterministic: embeddings are used inside deterministic algorithms, not to replace them.
- Provide a pluggable embedding backend (OpenAI API or local `sentence-transformers`) with graceful fallback to the current heuristics.

### Design Overview
- Introduce an `EmbeddingBackend` interface and a thin `EmbeddingService` wrapper.
- Add embedding‑aware grouping utilities and use them in:
  - `_group_similar_claims` (convergence/patterns)
  - Theme clustering (optionally replacing or augmenting current clustering distance)

### Proposed Modules/Types
- New: `src/services/embeddings.py`
  - `class EmbeddingBackend(Protocol)`: `async def embed(self, texts: list[str]) -> list[list[float]]`
  - `class OpenAIEmbeddingBackend(EmbeddingBackend)`: Uses OpenAI Embeddings (e.g., text-embedding-3-small). Requires `OPENAI_API_KEY`.
  - `class LocalEmbeddingBackend(EmbeddingBackend)`: Wraps `sentence-transformers` (e.g., `all-MiniLM-L6-v2`). Optional dependency.
  - `class EmbeddingService`: Selects backend, exposes `async def embed_batch(texts) -> list[vec]` and small vector helpers (cosine similarity).

- Update: `src/agents/research_executor_tools.py`
  - Extend `ResearchExecutorDependencies` to carry an optional `embedding_service: EmbeddingService | None`.

- Update: `src/agents/research_executor.py`
  - In `_build_executor_dependencies`, attach `deps.embedding_service` when present.

- Update: `src/services/synthesis_tools.py`
  - Add embedding‑aware helpers:
    - `_embed_texts_if_available(texts: list[str], svc: EmbeddingService | None) -> list[vec] | None`
    - `_pairwise_cosine_matrix(vectors: list[list[float]]) -> list[list[float]]`
    - `_cluster_by_threshold(labels: list[str], sim: list[list[float]], threshold: float) -> list[list[int]]`
  - Enhance:
    - `_group_similar_claims(...)`: fall back to token similarity; prefer embedding similarity when available.
    - Theme extraction/clustering: if embeddings are available, compute a theme vector (e.g., centroid of member findings) and use cosine similarity threshold for grouping/merging.

### Configuration & Defaults
- New config knobs (on dependencies or a simple settings object):
  - `enable_embedding_similarity: bool = False` (default off)
  - `embedding_backend: Literal["openai", "local"] | None = None`
  - `embedding_model_name: str | None = None` (e.g., `text-embedding-3-small` or `all-MiniLM-L6-v2`)
  - `similarity_threshold: float = 0.55` (tunable; start conservative)
- Extras (pyproject.toml):
  - Optional: `embed = ["sentence-transformers>=2.7.0"]`
  - No hard dependency if only OpenAI is used.

### Pseudocode (claim grouping)
```python
claims = [(sentence, source) ...]
if embedding_service:
    vecs = await embedding_service.embed([c[0] for c in claims])
    sim = cosine_matrix(vecs)
    groups = cluster_by_threshold(range(len(claims)), sim, threshold)
else:
    groups = token_jaccard_grouping(claims)
```

### Performance Considerations
- Batch embedding requests to respect API limits.
- Cache embeddings by text hash (in‑memory first; file‑cache optional).
- For large N, limit to top‑K sentences per source when building convergence groups; expose a cap (e.g., 500 sentences).

### Tests
- Unit tests for `EmbeddingService` backends (mocked returns).
- Unit tests for `_group_similar_claims` with and without embeddings.
- Integration test: ensure ResearchResults content is stable (groups found) with a known small corpus.

## 2) Guardrailed LLM “Clean‑Merge” Pass

### Goals
- Improve narrative cohesion (readability) of the synthesized text while preserving structure and citations.
- Keep typed artifacts and provenance intact; do not allow the LLM to add or remove citations.

### Design Overview
- Add an optional LLM rewrite step after deterministic synthesis, before final rendering.
- Target text fields only (e.g., `executive_summary`, section `content`) and require preservation of `[Sx]` markers.
- Guardrails:
  - Structured output schema with the same keys (e.g., `{"executive_summary": str, "sections": [{"title": str, "content": str}]}`) for fields being rewritten.
  - Pre/post validation: the set of citation markers must remain identical; reject LLM output if markers differ.
  - Truncate long inputs to model context; iterative chunking if needed.

### Integration Options
- ReportGeneratorAgent (recommended):
  - After generating `ResearchReport`, run `llm_clean_merge(report)` if feature flag is enabled.
  - Validate markers before/after; if mismatch, log warning and keep original.

- Implementation sketch: `src/agents/report_generator.py`
  - Add `@self.agent.tool` or a helper method:
    ```python
    async def tool_guardrailed_clean_merge(ctx: RunContext[ResearchDependencies], report: ResearchReport) -> ResearchReport:
        # 1) Build structured prompt with strict instructions:
        #    - Preserve all [Sx] markers, do not add or remove
        #    - Improve clarity/flow
        #    - Return JSON with only allowed fields
        # 2) LLM call via pydantic-ai with structured output model
        # 3) Validate marker set unchanged; if not, reject
    ```
  - Add `enable_llm_clean_merge: bool = False` knob in config/deps.

- Prompt requirements
  - “Do not add, remove, or renumber [Sx] markers; keep verbatim.”
  - “Rewrite sentences for clarity; preserve technical meaning.”
  - “Output JSON with only these fields: …” (use pydantic model for schema).

### Tests
- Unit: Guardrail enforcement — generate a fake report with markers and verify that an LLM response that changes markers is rejected, while an accepted response keeps the marker set equal.
- Integration: With clean‑merge enabled, verify that `executive_summary` changes while `references` and marker mapping remain identical.

## 3) Rollout Plan

- Phase 1 (opt‑in only):
  - Add `EmbeddingService` and integrate grouping fallbacks.
  - Add guardrailed clean‑merge step (disabled by default).
  - Add config toggles and docs.

- Phase 2 (observability):
  - Log before/after stats: number of groups, average intra‑cluster similarity, marker set equality, time per step.
  - Capture counters in `SynthesisMetadata` (e.g., `embedding_used: bool`, `clean_merge_applied: bool`).

- Phase 3 (tuning):
  - Adjust thresholds by task; add per‑task knobs if needed.
  - Add local cache for embeddings; consider optional ANN (e.g., `faiss`) if scale warrants.

## 4) Risks & Mitigations

- Embedding API cost/latency → Batch + cache; keep feature optional.
- Clean‑merge drift/hallucination → Strict marker guardrails + structured output; reject on mismatch.
- Backwards compatibility → Default flags to False; deterministic path remains.

## 5) Acceptance Criteria

- With `enable_embedding_similarity=True`:
  - Convergence grouping uses embeddings; tests show increased grouping for paraphrased sentences vs token Jaccard.
- With `enable_llm_clean_merge=True`:
  - `executive_summary` (and configured fields) exhibit improved readability while the set of `[Sx]` markers is unchanged; references/footnotes remain correct.
- No regressions in direct or HTTP modes; all quality checks pass (`ruff`, `pyright`, `pytest`).

## 6) Developer Checklist

- [ ] Add `src/services/embeddings.py` and tests
- [ ] Extend `ResearchExecutorDependencies` with `embedding_service`
- [ ] Update `synthesis_tools.py` grouping/clustering functions to use embeddings when available
- [ ] Add feature flags (`enable_embedding_similarity`, thresholds)
- [ ] Implement `tool_guardrailed_clean_merge` in `ReportGeneratorAgent` (or helper) with structured output + marker validation
- [ ] Add feature flag (`enable_llm_clean_merge`) and configuration documentation
- [ ] Add unit/integration tests described above
- [ ] Update docs & README with usage and flags

## 7) Justification

- Embedding similarity provides robust paraphrase tolerance and improves recall in grouping/themes compared to token overlap alone, especially across heterogeneous sources.
- A guardrailed LLM “clean‑merge” improves readability and cohesion without sacrificing auditability by preserving citation markers; structured outputs and marker validation provide strong safety rails.
