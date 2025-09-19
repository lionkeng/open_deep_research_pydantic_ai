"""Implementation plan to strengthen hallucination safeguards in synthesis."""

# Hallucination Mitigation Implementation Plan

This plan addresses the gaps identified in the synthesis workflow across four
key risk areas. Work is staged into incremental phases so each mitigation can
be designed, validated, and reviewed independently.

## Phase 1 – Grounded Finding Extraction

### Goals
- Replace the generic fallback in `extract_hierarchical_findings` with a
  grounded alternative that quotes source spans so we never emit text that
  isn’t present in the source snippet.
- Ensure every extracted finding, regardless of extraction path, records the
  snippet boundaries used so later verification has structured anchors.

### Key Tasks
1. **Create deterministic extractor module.**
   - Add `services/text_extraction.py` with a `extract_sentences_with_spans`
     function that tokenizes a document into sentences (e.g., using
     `re.split(r"(?<=[.!?])\s+")`) and returns `(text, start, end)` tuples.
   - Select top-N sentences based on heuristics (e.g., tf-idf against the query
     terms derived from `source_metadata["title"]`/`deps.original_query`) when
     ML extraction is unavailable.
2. **Extend models with span metadata.**
   - Add optional `source_span: tuple[int, int] | None` and `evidence_span_list`
     to `HierarchicalFinding` (`src/models/research_executor.py`) with
     validation that start/end are within the snippet length and `start < end`.
   - Provide helper `apply_span(snippet: str, span: tuple[int, int]) -> str` to
     rehydrate evidence text.
3. **Integrate fallback with spans.**
   - Update `_extract_findings_fallback` in
     `src/agents/research_executor_tools.py` to call the new extractor and
     populate findings with `supporting_evidence` derived via spans, the raw
     sentence text, and the recorded offsets.
   - When ML extraction succeeds, enrich existing findings by computing spans
     via `.find` if not present; log a warning if spans cannot be resolved.
4. **Cache compatibility.**
   - Adjust cache serialization to include span metadata (ensure
     `utils.cache_serialization.dumps_for_cache` handles tuples correctly).

### Testing
- Unit tests in `tests/unit/services/test_text_extraction.py` covering
  sentence splitting, offset correctness, and selection heuristics.
- Tests in `tests/unit/agents/test_research_executor_tools.py` mocking
  `synthesis_engine.extract_themes` failures to verify fallback findings carry
  spans and match the source substring exactly.
- Regression test ensuring cached `HierarchicalFinding` objects persist spans
  across `cache_manager` get/set cycles.

### Code Review Checklist
- Confirm fallback never fabricates content outside provided text (string slice
  should always equal evidence text).
- Ensure span metadata is optional but populated whenever a deterministic span
  exists; document any edge cases where spans remain `None`.
- Verify serialization/deserialization of `HierarchicalFinding` handles spans
  and that new fields don’t break existing consumers (e.g., Pydantic models).

## Phase 2 – Pattern Generation Safeguards

### Goals
- Gate heuristic patterns (`High Confidence Convergence`, `Temporal Evolution`)
  behind quantitative checks that rely on actual evidence rather than arbitrary
  counts.
- Provide transparency on why each heuristic pattern was accepted or rejected
  so analysts can audit the decision.

### Key Tasks
1. **Quantitative gating utilities.**
   - Add a `services/pattern_metrics.py` module exposing helpers such as
     `calculate_similarity_matrix(findings, spans)` and `temporal_trend_score`.
   - Implement gating logic in `analyze_patterns` so heuristics only fire when
     metrics exceed documented thresholds (e.g., convergence requires mean
     cosine similarity > 0.7 across span-centred vectors; temporal trend needs
     Pearson r ≥ 0.5 on importance over time).
2. **Audit metadata.**
   - Extend `PatternAnalysis.confidence_factors` to include the specific metric
     values (`"mean_similarity": 0.74`, `"temporal_r": 0.58`, etc.).
   - Add a `heuristic_explanation` field to capture human-readable rationale.
3. **Runtime configuration.**
   - Introduce `analysis.enable_heuristic_patterns` in config (read from env or
     dependency settings). Default to `True`, but allow disabling via
     `ResearchExecutorDependencies` flag.
4. **Documentation.**
   - Update `docs/system_architecture.md` to explain the new gating mechanics
     and configuration knob.

### Testing
- Unit tests in `tests/unit/services/test_pattern_metrics.py` covering metric
  calculations, including edge cases (single finding, identical spans, sparse
  temporal data).
- Tests in `tests/unit/agents/test_research_executor_tools.py` verifying
  heuristic patterns are added/omitted based on metrics and config flag.
- Integration run (pytest marker `integration`) to ensure patterns carry audit
  metadata through to `ResearchResults`.

### Code Review Checklist
- Check that heuristic metrics reference documented spans or finding IDs to
  ensure traceability.
- Confirm disabling heuristics skips side-effects (no empty placeholder
  patterns, and dependent logic handles missing heuristics gracefully).
- Validate tests exercise both positive and negative gating scenarios and that
  thresholds are documented in code/comments.

## Phase 3 – Claim-Source Verification

### Goals
- Automatically verify a sample (or 100%) of generated report sentences against
  the cited source snippets to detect hallucinations before report finalization.
- Expose verification status in the outputs so human reviewers can react.

### Key Tasks
1. **Verifier service.**
   - Create `services/claim_verifier.py` with a `ClaimVerifier` class that
     accepts a `ResearchReport` and `ResearchResults` plus a target coverage
     ratio.
   - Use the spans captured in `HierarchicalFinding` to fetch the exact snippet,
     then apply fuzzy string matching (e.g., `rapidfuzz` or custom cosine
     similarity on n-grams) to score each claim→snippet pair.
   - Categorize scores: >0.85 = pass, 0.6–0.85 = warning, <0.6 = fail.
2. **Report integration.**
   - Inject the verifier step inside `ReportGeneratorAgent` just before writing
     the final `ResearchReport`; attach results to `report.metadata["claim_verification"]`
     containing per-claim statuses and overall coverage.
   - Emit warnings to logs when verification fails so monitoring can alert.
3. **Results surfacing.**
   - Update `ResearchResults.synthesis_metadata` with aggregate metrics (`pass_rate`,
     `warning_rate`, `fail_rate`) and propagate to any downstream interfaces.
4. **Manual override.**
   - Allow skipping verification via config (e.g., `REPORT_VERIFY=False`) for
     debugging, but default to enabled.

### Testing
- Unit tests in `tests/unit/services/test_claim_verifier.py` covering exact,
  paraphrased, contradictory, and missing-snippet scenarios.
- Integration test generating a short report with seeded claims (one valid, one
  fabricated) to ensure the verifier flags the fabricated claim and records the
  result in metadata.
- Negative test verifying that disabling the verifier bypasses checks but logs
  a notice.

### Code Review Checklist
- Confirm the verifier never attempts network calls and operates solely on the
  text already stored in memory / repository.
- Review fuzzy matching thresholds for false-positive/negative balance and
  ensure they are configurable.
- Ensure failure metadata does not block report creation but is clearly exposed
  to downstream users (CLI, API responses, logs).

## Phase 4 – Quality Metric Enhancements

### Goals
- Strengthen `assess_synthesis_quality` so the reliability score reflects both
  verification outcomes and source span coverage, giving stakeholders a more
  meaningful signal.

### Key Tasks
1. **Metric enrichment.**
   - Modify `assess_synthesis_quality` to accept optional `verification_metrics`
     and `span_coverage` inputs.
   - Compute `span_coverage` as `unique_span_tokens / total_report_tokens` or
     similar, rewarding analyses that rely on diverse evidence spans.
   - Blend verification failure rate into the reliability component (e.g.,
     `reliability = max(0, 1 - fail_rate - contradiction_penalty)`).
2. **Telemetry plumbing.**
   - Extend `MetricsCollector` (if enabled) to record the new metrics and update
     dashboards accordingly.
3. **Documentation & config.**
   - Update README/testing docs describing the new metrics.
   - Add fallbacks so if verification data is missing (e.g., Phase 3 disabled)
     the function reverts to previous behavior.

### Testing
- Unit tests in `tests/unit/agents/test_research_executor_tools.py` to verify
  updated score calculations with and without verification data.
- Tests ensuring metrics collector receives the enriched payload; mock collector
  to assert fields.
- Regression to confirm disabling verification maintains original scores.

### Code Review Checklist
- Validate backward compatibility: default values when verification data is
  absent should preserve existing behavior.
- Check documentation updates for the enriched metrics and sample outputs.
- Confirm telemetry paths handle the new fields without raising (e.g., optional
  keys gracefully handled).

## Phase 5 – Review & Rollout

### Goals
- Coordinate code reviews, documentation, and rollout sequencing.

### Key Tasks
1. Schedule peer review for each phase with domain owners (ML, infra, product).
2. Update runbooks and developer docs to describe new safeguards and configs.
3. Pilot the full pipeline end-to-end on staging datasets; record baseline vs.
   post-mitigation hallucination metrics.

### Testing
- Manual exploratory tests focusing on failure modes (missing sources,
  contradictory statements).
- Acceptance test suite updated to assert report metadata surfaces verification
  outcomes.

### Code Review Checklist
- Ensure documentation aligns with shipped behavior.
- Confirm staging rollout checklists are complete before production deploy.
- Verify telemetry dashboards (if any) capture new quality signals.

---

**Next steps:** Phase 1 can begin immediately with design review; subsequent
phases should start once preceding safeguards are merged and validated to avoid
overlapping risk surfaces.
