# Hallucination Safeguards Implementation Plan

## Objectives
- Reduce the likelihood that the report generator introduces unsupported facts or fabricated sources.
- Make hallucination regressions observable during automated QA.
- Preserve current report quality (tone, structure, latency) while adding safeguards.

## Scope
The plan covers enhancements to the report synthesis pipeline (`src/agents/report_generator.py`), supporting services (`src/services/`), prompt templates, and automated tests under `tests/`. Changes should remain compatible with existing Pydantic AI agent definitions and the citation post-processing already in place.

## Workstreams
1. Strengthen prompt-level constraints.
2. Add pre-publication citation self-checks.
3. Validate citation-to-source alignment.
4. Improve source quality controls before prompting.
5. Encourage explicit handling of uncertainty.
6. Add automated regression tests for hallucination signals.

---

## 1. Strengthen Prompt-Level Constraints

### Summary
Tighten the system prompt so the agent either cites from provided sources or clearly states that evidence is unavailable, rather than inventing facts.

### Key Tasks
- **Update citation contract** (`CITATION_CONTRACT_TEMPLATE` in `src/agents/report_generator.py`):
  - Add language that forbids speculative wording ("likely", "probably") unless explicitly supported by cited sources.
  - Require the model to state "insufficient evidence" when context does not support a claim.
  - Clarify that all quantitative values must be sourced.
- **Reinforce in Style & Voice principles** (`REPORT_GENERATOR_SYSTEM_PROMPT_TEMPLATE`):
  - Add bullet stating the report must identify evidence gaps rather than filling them.
  - Instruct that any assertion without a citation should be reframed as a question or acknowledged as uncertain.
- **Conversation metadata**: ensure `add_report_context` passes through any prior system warnings or human feedback about missing evidence so the model keeps context.

### Acceptance Criteria
- Prompt explicitly instructs against unsupported speculation and describes the fallback phrase to use when evidence is missing.
- Manual smoke-test generation produces sections that use "insufficient evidence" when data are absent.

---

## 2. Add Pre-Publication Citation Self-Checks

### Summary
Introduce a validation stage before `_apply_citation_postprocessing` finalizes the report to ensure every substantive sentence or bullet references a source marker.

### Key Tasks
- Create a helper (e.g., `_validate_citation_presence`) in `report_generator.py`:
  - Iterate through `executive_summary`, `introduction`, each `section.content`, `recommendations`, and `conclusions`.
  - Split text into sentences/bullets (use `re` or `nltk`-free heuristics to avoid new deps).
  - Flag sentences containing numbers, named entities, or factual verbs without `[S#]`.
- If violations exist:
  - Option A: raise a recoverable `CitationValidationError` for the agent to retry synthesis with the flagged guidance.
  - Option B: append a warning in the report metadata and trigger a retry with targeted feedback via agent instructions (preferred to keep user-facing output clean).
- Instrument logging with `logfire.warning` to surface missing citations during development.

### Acceptance Criteria
- Validation runs automatically on every report generation.
- Reports lacking required markers trigger a retry or at minimum a metadata warning (`report.metadata.validation_failures`).

---

## 3. Validate Citation-to-Source Alignment

### Summary
Ensure cited claims actually appear in their referenced sources by sampling cited spans and comparing them with retrieved source text.

### Key Tasks
- Extend `_apply_citation_postprocessing` to collect claim snippets (e.g., ±2 sentences around each `[S#]`).
- Add a new module `services/citation_verifier.py` containing:
  - `extract_relevant_chunk(source: ResearchSource, marker: str) -> str` using existing stored excerpts or fetching top sections from the repository.
  - `compute_similarity(claim: str, source_text: str) -> float` using cosine similarity via existing embedding infra (if available) or fallback to keyword overlap.
  - `verify_claims(claims: list[ClaimCheck]) -> list[VerificationResult]` summarizing pass/fail per marker.
- Store verification output in `report.metadata.citation_audit["alignment"]` with per-marker scores.
- Trigger `logfire.warning` for scores below a conservative threshold (e.g., 0.55 cosine similarity or <40% keyword overlap).
- (Optional stretch) Attempt auto-remediation by appending a system note prompting the agent to restate or drop unsupported claims on retry.

### Acceptance Criteria
- Metadata includes alignment metrics per citation.
- Unsupported citations produce warnings, and developers can view them in logs for investigation.

---

## 4. Improve Source Quality Controls

### Summary
Filter or tag low-credibility sources before they reach the report prompt to avoid grounding in unreliable material.

### Key Tasks
- Extend `ResearchSource` model (in `models/research_executor.py`) with fields like `credibility_score: float | None` and `source_type: Literal[...]`.
- Update ingestion pipeline to populate credibility from provenance (e.g., whitelisted domains, publication reputation, user overrides).
- Modify `summarize_sources_for_prompt` to:
  - Sort sources by descending credibility.
  - Exclude or downgrade those below a configurable threshold.
  - Annotate each source summary with trust cues ("peer-reviewed", "user-supplied" etc.).
- Ensure `source_overview` string in the prompt highlights credibility so the model prioritizes higher-quality citations.
- Add configuration knobs (e.g., via `AgentConfiguration`) so operators can tweak thresholds without code changes.

### Acceptance Criteria
- Lower-credibility sources no longer appear in the top N context unless no alternatives exist (and, if they do, they are clearly labeled).
- Prompt output demonstrates new annotations when multiple source types are present.

---

## 5. Encourage Explicit Handling of Uncertainty

### Summary
Make uncertainty and limitations first-class citizens in generated reports to deter the model from overconfident assertions.

### Key Tasks
- Extend Style & Voice rules to require reporting on limitations and data gaps with citations (placed near `# STYLE & VOICE PRINCIPLES`).
- In the Synthesis & Insights section instructions, call out the need to summarize evidence gaps and unresolved questions.
- Provide a reusable prompt snippet (e.g., `UNCERTAINTY_GUIDELINES`) that can be injected whenever the pipeline detects sparse sourcing.
- Adjust `add_report_context` to flag when available sources < `MINIMUM_CITATIONS` and include guidance for cautious phrasing.

### Acceptance Criteria
- Reports with sparse data include clearly marked limitations sections or inline caveats.
- Manual regression shows fewer assertive statements without citations when data are thin.

---

## 6. Automated Regression Tests

### Summary
Add pytest suites that fail when generated reports show hallucination risk patterns.

### Key Tasks
- Create fixtures that mock minimal `ResearchResults` and `ResearchReport` objects.
- Tests in `tests/test_report_citation_guards.py` should cover:
  - Sentences with numbers lacking `[S#]` trigger the validation logic.
  - Citation alignment verifier flags mismatched text (use synthetic sources with known content).
  - Prompt updates insert the new contract language (use snapshot tests or string contains checks).
  - Low-credibility sources are excluded from `source_overview` when alternatives exist.
- Integrate tests into CI by updating documentation (`docs/contributing.md`) to mention new commands.

### Acceptance Criteria
- `uv run pytest` fails if safeguards regress.
- Coverage includes both happy-path (valid citations) and failure-path (missing/incorrect citations) scenarios.

---

## Documentation & Rollout
- Update `AGENTS.md` with a brief description of the new safeguards and how to override thresholds.
- Add troubleshooting tips to `docs/system_architecture.md` for interpreting citation verification warnings.
- Coordinate deployment with monitoring updates so the team can observe new `logfire` signals.

## Risks & Mitigations
- **False positives in citation validation**: start with conservative heuristics; provide developer toggle via config.
- **Latency increase from alignment checks**: cache source chunks and batch similarity computations; make alignment optional with a configuration flag if latency becomes problematic.
- **Prompt token bloat**: monitor total prompt size; move verbose instructions into reusable short tags if necessary.

## Next Steps
1. Prioritize quick wins (prompt contract + validation helper) for immediate impact.
2. Schedule implementation of similarity-based alignment once supporting utilities are agreed upon.
3. Review test design with QA to ensure coverage of edge cases.
4. Plan a post-deployment audit of generated reports to confirm real-world effectiveness.
