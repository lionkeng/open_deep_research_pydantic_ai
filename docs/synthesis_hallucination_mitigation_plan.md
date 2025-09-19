"""Implementation plan to strengthen hallucination safeguards in synthesis."""

# Hallucination Mitigation Implementation Plan

This plan addresses the gaps identified in the synthesis workflow across four
key risk areas. Work is staged into incremental phases so each mitigation can
be designed, validated, and reviewed independently.

## Phase 1 – Grounded Finding Extraction

### Goals
- Replace the generic fallback in `extract_hierarchical_findings` with a
  grounded alternative that quotes source spans.
- Ensure every extracted finding records the exact snippet boundary used.

### Key Tasks
1. Introduce a lightweight deterministic extractor (regex/heuristic) that
   slices top evidence sentences with offsets when ML extraction is unavailable.
2. Extend `HierarchicalFinding` to capture `source_span` metadata (start/end
   indices) for validation.
3. Update `extract_hierarchical_findings` fallback to populate findings using
   the deterministic extractor and recorded spans.

### Testing
- Unit tests mocking extractor failure to verify spans are populated and
  findings mirror underlying text.
- Regression test ensuring cached results preserve spans.

### Code Review Checklist
- Confirm fallback never fabricates content outside provided text.
- Ensure span metadata is optional but consistently populated in new paths.
- Verify serialization/deserialization of `HierarchicalFinding` handles spans.

## Phase 2 – Pattern Generation Safeguards

### Goals
- Gate heuristic patterns (`High Confidence Convergence`, `Temporal Evolution`)
  behind quantitative checks that rely on actual evidence.
- Provide transparency on why a heuristic pattern was accepted.

### Key Tasks
1. Add scoring functions that inspect supporting finding counts, variance, and
   span overlap before enabling each heuristic.
2. Annotate `PatternAnalysis.confidence_factors` with the metrics used so
   downstream consumers can audit the rationale.
3. Add a configuration flag (env/config) to disable heuristics entirely for
   safety-first deployments.

### Testing
- Unit tests covering acceptance/rejection of heuristics under varying finding
  distributions.
- Integration test simulating a synthesis run to verify patterns include the
  new audit metadata and respect the disable flag.

### Code Review Checklist
- Check that heuristic metrics reference documented spans or finding IDs.
- Confirm disabling heuristics skips side-effects (no empty placeholder
  patterns).
- Validate updated tests cover both positive and negative gating scenarios.

## Phase 3 – Claim-Source Verification

### Goals
- Automatically verify a sample of generated report sentences against the
  cited source snippets to detect hallucinations before report finalization.

### Key Tasks
1. Build a `claim_verifier` service that accepts report paragraphs and cited
   sources, performing fuzzy matching against stored snippets/spans.
2. Integrate the verifier into `report_generator` prior to final output; flag
   failures into the report metadata for visibility.
3. Surface verification scores in `ResearchResults.synthesis_metadata`.

### Testing
- Unit tests for the verifier covering exact, paraphrased, and mismatched
  claims.
- End-to-end test generating a short report with known claims to ensure the
  verifier flags incorrect citations.

### Code Review Checklist
- Confirm the verifier never attempts network calls and operates solely on
  cached text.
- Review fuzzy matching thresholds for false-positive/negative balance.
- Ensure failure metadata does not block report creation but is clearly
  exposed to downstream users.

## Phase 4 – Quality Metric Enhancements

### Goals
- Strengthen `assess_synthesis_quality` so reliability reflects verification
  outcomes and span coverage.

### Key Tasks
1. Incorporate claim verification scores and span coverage percentages into
   the reliability component.
2. Adjust the completeness heuristic to consider unique source spans rather
   than raw finding counts.
3. Update `quality_monitor` (if present) to track new metrics for telemetry.

### Testing
- Extended unit tests for `assess_synthesis_quality` verifying new inputs alter
  scores as expected.
- Ensure metrics collector receives updated payload and handles missing data.

### Code Review Checklist
- Validate backward compatibility: default values when verification data is
  absent should preserve existing behavior.
- Check documentation updates for the enriched metrics.
- Confirm telemetry paths handle the new fields without raising.

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
