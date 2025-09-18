# Source Attribution & Footnote Integration Plan

## Summary
The current pipeline surfaces source metadata inside `HierarchicalFinding.source`, but the data is not deduplicated, usage is not tracked, and the final `ResearchReport` renders a plain references list without inline citations. End users cannot see which statement came from which document when reading `research_report.md`. This plan defines the changes required to:
- assign stable identifiers to every consulted source,
- propagate those identifiers through findings, synthesis phases, and report generation,
- render Markdown footnotes in the saved report while keeping source metadata accessible for other consumers.

## Current Gaps
- `src/agents/research_executor.py:278` gathers `finding.source` objects but drops duplicates and loses ordering context.
- `ResearchResults` only exposes `sources: list[ResearchSource]` without usage metadata or identifiers.
- `ReportGeneratorAgent` instructions never mention citations, so the LLM cannot reference sources.
- `save_report_object` writes a `## References` section but lacks inline markers and does not guarantee the list aligns with the findings used in the narrative.
- CLI display helpers ignore `ResearchResults.sources` entirely; the operator has no visibility into attribution during interactive sessions.

## Goals & Success Criteria
1. **Deterministic Source Registry** – Each unique source encountered in research execution receives an ID (`S1`, `S2`, …) and persists inside `ResearchResults`.
2. **Finding-Level Attribution** – Findings, clusters, contradictions, and report sections reference source IDs so post-processing can inject citations automatically.
3. **Markdown Footnotes** – Saved reports include footnote markers (e.g., `[^1]`) at the point of use, with definitions that render the source title, URL, and optional publisher/author metadata.
4. **Backwards Compatibility** – Existing consumers of `ResearchResults.sources` continue to function; new metadata is additive.
5. **Automated Validation** – Tests cover deduplication, ID stability, citation insertion, and Markdown rendering.

## Proposed Architecture Changes
### 1. Source Identity & Deduplication
- Extend `models/research_executor.py`:
  - Add `source_id: str` to `ResearchSource` with helper constructor that generates IDs when absent.
  - Introduce a `SourceUsage` model capturing `source_id`, `first_seen_stage`, and references (finding IDs, cluster IDs, contradiction IDs).
  - Augment `ResearchResults` with `source_registry: dict[str, ResearchSource]` or `list[ResearchSource]` plus a `usage: dict[str, SourceUsage]` map while keeping the existing `sources` list for compatibility.
- In `ResearchExecutorAgent._generate_structured_result` and tool helpers, introduce a `SourceRegistry` helper that:
  - Normalizes sources by canonical key (prefer URL → title → hash of content snippet).
  - Returns a consistent ID and `ResearchSource` instance when extracting findings.
  - Updates `finding.source_id` / `finding.supporting_source_ids` before final aggregation.

### 2. Finding & Artifact Attribution
- Update `HierarchicalFinding` to store `source_id: str | None` and `supporting_source_ids: list[str]` in addition to the existing `source` object (kept for compatibility).
- Ensure downstream artifacts (`ThemeCluster`, `PatternAnalysis`, `Contradiction`, `ExecutiveSummary`) capture source IDs where applicable so the report generator understands lineage.
- Populate `ResearchMetadata.sources_consulted` using the deduplicated registry count instead of raw search results.

### 3. Report Generator Integration
- Enhance `ReportGeneratorAgent`:
  - During instruction assembly, inject a summarized table of sources with their IDs and key metadata (title, publisher, year, URL) so the model can cite them explicitly (e.g., "Use `[S1]` style markers when referencing evidence").
  - Provide a lightweight post-processing step in `run` (override base behavior) that converts `[S1]` markers to canonical footnotes. The transformation should:
    1. Walk every textual field in `ResearchReport` (executive_summary, introduction, section content, recommendations, conclusions).
    2. Replace `[S{n}]` with `[^n]`, tracking the order of first appearance to produce stable numbering.
    3. Build `report.references` (or a new `footnotes` field) from the ordered IDs, using the source registry for formatted strings like `[^1]: Title — Publisher (Year). URL`.
  - Store the mapping inside `ResearchMetadata.report` for downstream tooling (e.g., CLI UI, future exports).

### 4. Markdown Output & CLI Updates
- Adjust `save_report_object` in `src/cli.py` to detect pre-formatted `[^n]` markers:
  - Append a `## Footnotes` section instead of generic references when footnotes exist.
  - Fall back to the legacy references list when no footnotes were generated (e.g., tests without sources).
- Update `display_report_object` / `display_report_dict` to surface a concise source table (ID, title, URL) beneath the summary so CLI users can cross-check citations without reading the raw Markdown.

### 5. Telemetry & Observability
- Emit structured logs (via `logfire`) when:
  - Sources are registered/deduplicated.
  - Footnote post-processing rewrites sections (include counts of markers inserted & dangling references).
- Add metrics to `SynthesisMetadata.quality_metrics` indicating citation coverage (e.g., percentage of findings appearing in report sections).

## Implementation Steps
1. **Data Model Enhancements**
   - Modify `ResearchSource` and related models to support IDs and usage metadata.
   - Update `HierarchicalFinding` and other artifacts with source ID fields and validators ensuring IDs exist when `source` is present.
   - Add migration helpers (defaulting `source_id` when absent) to keep legacy serialized data loadable during tests.

2. **Source Registry Utility**
   - Create `source_registry.py` (likely under `src/core` or `src/utils`) encapsulating deduplication and ID assignment logic.
   - Integrate the registry within `ResearchExecutorAgent._generate_structured_result` so every extracted finding is routed through the registry before storage.

3. **Research Executor Wiring**
   - Refactor `_generate_structured_result` to:
     - Collect search result metadata, register sources, and set `finding.source_id`.
     - Populate `ResearchResults.source_registry` and `ResearchResults.sources` (ordered by ID).
     - Record usage maps tying findings/patterns/contradictions back to source IDs.

4. **Report Generator Post-Processing**
   - Override `ReportGeneratorAgent.run` (or add a dedicated method invoked post-run) to call the base implementation then process the resulting `ResearchReport` for citations.
   - Implement a `FootnoteFormatter` helper that converts `[Sx]` markers to Markdown footnotes using the registry stored in `ResearchResults` from the active `ResearchState`.
   - Update agent instructions emphasising citation markers, and surface the source table in the prompt context.

5. **CLI Presentation & Export**
   - Teach `display_report_*` helpers to render a source summary table when `research_state.research_results.source_registry` is available.
   - Update `save_report_object` to add a `## Footnotes` section with formatted definitions derived from footnote metadata.

6. **Testing**
   - Extend unit tests for `ResearchResults` to confirm deduplication, ID stability, and usage maps.
   - Add integration test covering the full workflow stubbed with deterministic findings to ensure the saved Markdown contains matching `[^n]` markers and definitions.
   - Write regression tests for CLI helpers verifying the new footnote section formatting.

## Validation Strategy
- **Unit Tests**: Target models (`ResearchSource`, `HierarchicalFinding`), registry utility, and formatter.
- **Integration Tests**: Simulate executor → report generator pipeline using canned search results to validate citations end-to-end.
- **Manual QA**: Run the CLI on a known query, inspect `research_report.md`, and confirm numbered footnotes resolve to the expected URLs.
- **Logging Review**: Verify logfire emits registration counts and footnote conversion summaries without overwhelming existing logs.

## Risks & Mitigations
- **LLM Citation Compliance**: The model might omit `[Sx]` markers. Mitigation: strengthen instructions, consider adding a prompt-time checklist, and fail fast in post-processing if findings lack citations (optionally re-run with explicit feedback in a later iteration).
- **Backward Compatibility**: Legacy cached results without source IDs may surface. Mitigation: add validators that assign IDs on load and log warnings rather than failing.
- **Markdown Rendering**: Footnote syntax must be compatible with both Markdown viewers and Rich CLI previews. Mitigation: keep to standard CommonMark footnote syntax and sanitize URLs/titles to avoid breaking formatting.
- **Performance**: Additional processing is lightweight but ensure the registry and formatter work in linear time and avoid blocking the event loop (no heavy synchronous operations in async paths).

## Open Questions
- Should we expose footnote metadata via the API JSON schema (e.g., `footnotes` array) for external clients? Decision pending stakeholder input.
- Do we need configurable citation styles beyond Markdown footnotes (APA/MLA)? For now, default to Markdown but keep formatter extensible.

## LLM Citation Compliance Plan

### Citation Behavior Goals
- Every declarative claim sourced from external material must reference at least one registered source ID.
- Citation markers should remain stable across revisions (`[S3]` should always refer to the same source).
- The saved Markdown report must expand markers into ordered footnotes (`[^3]`) while preserving bidirectional traceability back to the source registry.

### Instruction & Prompt Updates
1. **Research Executor Agent Instructions**
   - Extend the dynamic context injected via `_summarize_search_results` to include a compact source table with assigned IDs (e.g., `S1 – Title (URL)`).
   - Add an explicit checklist reminding the synthesis stack to tag findings with `source_id` values.

2. **Report Generator Agent Prompt**
   - Prepend a dedicated "Citation Contract" section to the instructions. Model the language after LangChain's Open Deep Research prompts:

     ```markdown
     ### Citation Contract
     - Assign each unique source ID exactly one marker using the form `[S{n}]`.
     - Cite **every** non-trivial statement or numeric claim; uncited assertions are not allowed.
     - Re-use the same marker each time you reference that source.
     - End the report with a `## Sources` table that maps `[S{n}]` → `Title — Publisher (Year) <URL>`.
     - If a section lacks citations, add a "Source Coverage" callout explaining why (e.g., original analysis).
     - Double-check that the number of markers in the body matches the entries in `## Sources`.
     ```

   - Include a closing checklist similar to LangChain's `compress_research_system_prompt`, emphasizing: "Citations are extremely important. Verify numbering is contiguous and no sources are dropped." This primes the LLM to self-verify before returning.

3. **Clarify Expected Marker Usage**
   - When formatting the initial prompt context, list the available source IDs next to short descriptors so the model can reference them without re-reading full metadata (e.g., `S2 – McKinsey 2024 GenAI Retrospective (https://...)`).

### Automated Compliance Hooks
- **Post-Generation Audit**: After `ResearchReport` creation and before footnote conversion, run a lightweight checker that ensures:
  - Each `[S{n}]` marker resolves to an existing registry entry.
  - No registry entry is orphaned (unused in the narrative).
  - Marker numbers are contiguous from `1..N`. If gaps or mismatches are detected, attach a warning to `SynthesisMetadata.quality_metrics` and optionally trigger a controlled retry with a "Missing citations" system message.
- **Quality Metric**: Record `citation_coverage = cited_sentences / total_evidence_sentences` in the synthesis metrics package for later observability.

### Evaluation Additions
- Add unit tests that feed canned prompts through the report generator with mocked sources to confirm the LLM instructions yield `[S{n}]` markers and that the audit rejects malformed outputs.
- Extend integration tests to assert the rendered Markdown includes matching `[^n]` definitions and that removing a citation causes the audit guardrail to flag the issue.

### Future Enhancements
- Consider incorporating automated retrieval checks (e.g., verifying URLs still respond) during post-processing, similar to LangChain's "first citation with raw content" pattern, to further bolster trust in the references.
