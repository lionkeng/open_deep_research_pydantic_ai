# Source Attribution & Citation Master Plan

## Overview
This plan consolidates the previously published **Source Attribution & Footnote Integration Plan** and the follow-on **Robust Improvements to Source Tracking & Footnote Integration Plan** into a single roadmap. It preserves the near-term deliverables required to ship reliable source identifiers and Markdown footnotes while incorporating the architectural guardrails and scalability patterns identified during the robustness review. Conflicting guidance has been reconciled by defining phased milestones: phase 1 delivers the production-critical features, and phase 2+ layers in advanced context engineering capabilities once baseline behaviour is stable.

## Goals & Success Criteria
1. **Deterministic Attribution** – Every unique source receives a stable identifier (`S1`, `S2`, …) that survives through findings, synthesis stages, and final reports.
2. **Inline Citations** – Reports insert `[Sx]` markers (converted to `[^x]` footnotes for saved Markdown) so readers can trace statements back to evidence.
3. **LLM Compliance** – Prompts and post-run audits enforce citation coverage and numbering rules.
4. **Scalability** – The source subsystem scales gracefully from tens to thousands of citations using repository and caching patterns without rewriting phase-1 logic.
5. **Observability & Quality** – Metrics, logs, and automated tests detect missing citations, duplicate sources, and performance regressions.

## Architecture Decisions & Detailed Design
### 1. Source Repository Pattern (Phase 1 Deliverable)
- **Objective**: decouple source deduplication/ID assignment from the research executor.
- **Key Types**:
  ```python
  class SourceIdentity(BaseModel):
      source_id: str
      canonical_key: str
      version: int = 1

      @classmethod
      def build(cls, source_id: int, canonical_key: str) -> "SourceIdentity":
          return cls(source_id=f"S{source_id}", canonical_key=canonical_key)

  class SourceUsage(BaseModel):
      source_id: str
      finding_ids: list[str] = []
      cluster_ids: list[str] = []
      contradiction_ids: list[str] = []
      report_sections: list[str] = []
  ```
- **Repository Interface**:
  ```python
  class AbstractSourceRepository(Protocol):
      async def register(self, source: ResearchSource) -> SourceIdentity: ...
      async def get(self, source_id: str) -> ResearchSource | None: ...
      async def find_by_key(self, canonical_key: str) -> SourceIdentity | None: ...
      async def iter_all(self) -> AsyncIterator[tuple[SourceIdentity, ResearchSource]]: ...
  ```
- **In-Memory Implementation (Phase 1)**:
  - Maintain `self._sources: list[ResearchSource]` and `self._key_index: dict[str, SourceIdentity]`.
  - Canonical key = `url.lower()` when present else `sha256(title + snippet)`.
  - Registration flow:
    1. Build canonical key → check `_key_index` for existing entry.
    2. If found, reuse `SourceIdentity`; update stored metadata if the new record has richer details (publisher/year/etc.).
    3. If not found, append source to list, allocate `S{len(self._sources)}` identity, save to both collections.
    4. Return identity to caller.
  - Provide helper `def ordered_sources(self) -> list[ResearchSource]` returning sources sorted by integer portion of `Sx` so downstream rendering is deterministic.
  - Record usage: `register_usage(source_id: str, artifact_type: Literal["finding","cluster","contradiction","report_section"], artifact_id: str)` populates `SourceUsage` entries.

### 2. Attribution Metadata & Model Changes
- `models/research_executor.py`:
  - `ResearchSource` additions:
    ```python
    class ResearchSource(BaseModel):
        source_id: str | None = None
        canonical_key: str | None = None
        title: str
        url: str | None = None
        author: str | None = None
        publisher: str | None = None
        date: datetime | None = None
        source_type: str | None = None
        credibility_score: float = Field(default=0.5, ge=0.0, le=1.0)
        relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
        metadata: dict[str, Any] = Field(default_factory=dict)

        @model_validator(mode="after")
        def ensure_id(cls, values):
            if not values.source_id:
                values.source_id = f"S{values.metadata.get('legacy_index', 0)}"
            return values
    ```
  - `HierarchicalFinding` additions:
    ```python
    class HierarchicalFinding(BaseModel):
        finding_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
        source_ids: list[str] = Field(default_factory=list)
        supporting_source_ids: list[str] = Field(default_factory=list)

        @model_validator(mode="after")
        def sync_source_ids(cls, values):
            if values.source and values.source.source_id:
                if values.source.source_id not in values.source_ids:
                    values.source_ids.insert(0, values.source.source_id)
            return values
    ```
- `ResearchResults` updates:
  ```python
  class ResearchResults(BaseModel):
      sources: list[ResearchSource] = Field(default_factory=list)
      source_usage: dict[str, SourceUsage] = Field(default_factory=dict)

      def record_usage(self, source_id: str, *, finding_id: str | None = None, cluster_id: str | None = None, contradiction_id: str | None = None) -> None:
          usage = self.source_usage.setdefault(source_id, SourceUsage(source_id=source_id))
          if finding_id and finding_id not in usage.finding_ids:
              usage.finding_ids.append(finding_id)
          if cluster_id and cluster_id not in usage.cluster_ids:
              usage.cluster_ids.append(cluster_id)
          if contradiction_id and contradiction_id not in usage.contradiction_ids:
              usage.contradiction_ids.append(contradiction_id)
  ```

### 3. Context Management Roadmap (Phase 2+)
- **Append-Only Registry Adapter**: adopt the robustness plan’s `AppendOnlySourceRegistry` to guarantee stable prefixes and facilitate KV-cache reuse. Diagram:

  ```mermaid
  graph LR
      A[SourceRepository] -->|phase1| B[InMemorySourceRepository]
      A -->|phase2| C[AppendOnlySourceRepository]
      C --> D[FileBackedSourceStore]
      C --> E[RestorableSourceCompressor]
      C --> F[SemanticHasher]
  ```
- **Hierarchical Compression Levels**:
  1. `Level 0` – full content (recent sources).
  2. `Level 1` – title + key sentences.
  3. `Level 2` – title + URL + scores.
  4. `Level 3` – source ID reference only.
- **Semantic Deduplication**: integrate `SimHash` and `LSHForest` to detect near-duplicates. Reuse pseudocode from robustness doc for `compute_semantic_hash` and `find_duplicates`.
- **Attention-Aware Context Manager**: maintain `_attention_weights`, `_access_history`, and `_source_objectives` to guide agent prompts when source volume is high. Use top-k sampling with temperature 0.3 to avoid loops.

### 4. Error Handling & Recovery Enhancements
- Implement `SourceValidationPipeline` with circuit breaker semantics:
  ```python
  class SourceValidationPipeline:
      def __init__(self, repository: AbstractSourceRepository, http_client: httpx.AsyncClient, circuit_breaker: CircuitBreaker):
          self.repository = repository
          self.http_client = http_client
          self.circuit_breaker = circuit_breaker

      async def validate_and_register(self, raw_source: dict[str, Any]) -> SourceIdentity | None:
          try:
              validated_url = await self._validate_url(raw_source.get("url"))
              if self.circuit_breaker.is_closed:
                  await self._verify_content(validated_url)
              source = ResearchSource(**raw_source, url=validated_url)
              return await self.repository.register(source)
          except (SourceValidationError, httpx.HTTPError) as exc:
              return await self._register_degraded(raw_source, exc)
  ```
- Degraded registration logs warnings, marks `metadata["validation_state"] = "degraded"`, but still assigns IDs so the pipeline never stalls.

### 5. Python Engineering Best Practices
- **Async concurrency**: use `asyncio.gather(..., return_exceptions=True)` with bounded parallelism `max_concurrent=10` for URL validation.
- **Type safety**: define `SourceID: TypeAlias = str`, `FootnoteNumber: TypeAlias = int`, `ValidatedSource = TypeGuard` helper to guard CLI rendering paths.
- **Avoid mutable defaults** in Pydantic models; always use `Field(default_factory=list)`.

### 6. Testing Requirements
- **Unit Tests**:
  - Repository deduplication (property-based with Hypothesis).
  - Footnote formatter conversion ensures `[Sx]` → `[^n]` mapping is stable.
  - Audit failure cases: missing marker, orphaned source, non-contiguous numbering.
- **Integration Tests**:
  - Research executor → report generator pipeline yields deterministic IDs and footnotes.
  - CLI path writes `## Footnotes` with entries matching `ResearchResults.sources`.
- **Async Tests**: verify concurrent registration preserves unique IDs (`pytest.mark.asyncio`).

### 7. Performance Targets & Monitoring Hooks
- Targets remain as in robustness doc:
  - Registration latency < 100ms per source.
  - Deduplication check < 10ms.
  - Footnote post-processing < 1s for 100 citations.
- Instrument `SynthesisMetadata.quality_metrics` with metrics:
  - `citation_coverage` (ratio).
  - `orphaned_sources`
  - `audit_failures`
- Implement `ContextPerformanceMonitor` from robustness doc to adapt cache size, compression thresholds, and prefetching based on observed latency/hit rates.

## Execution Flow
1. **Search & Extraction** – Search results are normalised into `ResearchSource` objects and registered through the repository. The returned `source_id` is injected into extraction tool responses so `HierarchicalFinding` instances arrive with populated `source_ids`.
2. **Research Executor Assembly** – `_generate_structured_result` consolidates findings, clusters, contradictions, and summary metadata. It gathers ordered sources from the repository, calculates usage statistics, and updates `ResearchMetadata.sources_consulted` using the deduplicated count.
3. **Report Generation** – The report generator’s dynamic instructions include a “Citation Contract” summarizing available sources (`S1 – Title (URL)`) and enforcing marker usage. After the LLM responds, a `FootnoteFormatter` converts `[Sx]` markers to Markdown footnotes, builds the final `## Sources` section, and records coverage metrics.
4. **CLI Presentation** – CLI display helpers show a compact source table (ID, title, URL) beneath the rendered summary. `save_report_object` detects `[^x]` markers and writes a `## Footnotes` section; it falls back to `## References` if no markers exist.

## LLM Prompt & Compliance Strategy
- **Instruction Updates** – Embed the “Citation Contract” (modeled on LangChain’s prompts) in report generator instructions, add a checklist that warns “Citations are extremely important. Verify numbering is contiguous and no sources are dropped,” and list the source table inline for quick reference.
- **Executor Context** – Extend dynamic context to list current sources and remind downstream tools to tag findings with `source_id` values before returning.
- **Post-Generation Audit** – Implement a checker that verifies:
  - Every `[Sx]` marker maps to a repository entry.
  - No repository entry is orphaned.
  - Marker numbers are contiguous from `1..N`.
  - Citation coverage metric (`cited_sentences / total_evidence_sentences`) meets minimum thresholds.
  - Failures trigger a structured warning plus an optional retry with a “missing citations” system reminder.

## Testing & Validation
- **Unit Tests** – Cover repository registration/deduplication, ID stability, footnote formatting, and audit guardrails. Add property-based tests for deduplication correctness and async tests for concurrent registration, as suggested in the robustness review.
- **Integration Tests** – Extend the executor → report workflow tests to assert that generated reports include matching footnote markers and definitions, and that removing a marker causes the audit to fail.
- **Manual QA** – Run the CLI end-to-end, check the rendered report for annotated statements, ensure the source table matches the footnote list, and inspect logfire entries for registration counts and audit results.

## Implementation Roadmap
### Phase 1 – Production Baseline (Current Sprint)
1. Finish repository abstraction and update models with `source_id` and usage metadata.
2. Wire the repository through research executor tools and result assembly.
3. Implement prompt updates, citation contract, and post-run audit.
4. Update CLI rendering/saving and add regression/unit tests.

### Phase 2 – Scalability Enhancements
1. Introduce append-only, file-backed repository adapter with restorable compression.
2. Layer in semantic hashing for near-duplicate detection and cache-aware context preparation.
3. Add performance monitoring hooks (cache hit rate, dedup effectiveness) and adaptive strategies.

### Phase 3 – Advanced Context Engineering (Optional/Future)
1. Implement attention-aware context manager and objective prompts for agents handling hundreds of sources.
2. Integrate vector search retrieval for offloaded sources and automated freshness checks (HTTP validation, content availability).
3. Explore automated re-run workflows when citations fail audits multiple times.

## Risks & Mitigations
- **Model Non-Compliance** – Mitigated through stronger instructions, audit enforcement, and optional retries.
- **Legacy Data Without IDs** – Auto-generate IDs on load, log warnings, and backfill usage maps lazily.
- **Performance Overhead** – Measure repository latency; introduce caching/compression only when benchmarks fall short.
- **Over-Engineering Risk** – Phase gating keeps sophisticated context engineering behind a stable phase-1 API to avoid blocking near-term delivery.

## Appendix: Key Interfaces
```python
class AbstractSourceRepository(Protocol):
    async def register(self, source: ResearchSource) -> SourceID: ...
    async def get(self, source_id: SourceID) -> ResearchSource | None: ...
    async def find_by_canonical_key(self, key: str) -> SourceID | None: ...
    async def iter_sources(self) -> AsyncIterator[ResearchSource]: ...
```

```python
class FootnoteFormatter:
    def format_inline(self, source_id: SourceID) -> str:
        return f"[{source_id}]"

    def format_footnote(self, number: FootnoteNumber, source: ResearchSource) -> str:
        meta = f"{source.title}"
        if source.metadata.get("publisher"):
            meta += f" — {source.metadata['publisher']}"
        if source.metadata.get("year"):
            meta += f" ({source.metadata['year']})"
        if source.url:
            meta += f" <{source.url}>"
        return f"[^{number}]: {meta}"
```

This unified plan maintains the tangible deliverables from the original document, incorporates the rigor and scalability enhancements from the robustness review, and sequences the work so the team can deliver immediate value while building toward a highly resilient citation pipeline.
