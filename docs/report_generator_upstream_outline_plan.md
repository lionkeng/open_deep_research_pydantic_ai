# Upstream Outline + Prompt Improvements (Eliminate Labely Output)

This plan describes how to stop "Finding N…"/"Pattern Analysis…" headings and label‑like paragraph prefixes at the source by improving inputs to the Report Generator Agent. The clean‑merge step remains as a safety net, but should rarely run after these changes.

## Goals

- Provide a content‑driven section outline to the report generator.
- Nudge the agent to use natural, topic‑specific headings and flowing paragraphs.
- Gate the clean‑merge pass behind simple quality checks (default off or exec summary only).
- Keep citations `[Sx]` and current footnote logic intact.

## Data Flow Changes

- ResearchExecutor synthesizes a `section_outline`:
  - One item per major section: `title`, `bullets` (1–3 salient points), optional `salient_evidence_ids`.
  - Titles are deterministic, derived from content (no LLM). The agent refines wording, but starts from solid inputs.
- ReportGeneratorAgent receives `section_outline` in dynamic instructions and uses those titles.
- Clean‑merge only runs if headings/paragraphs fail quick quality checks.

## Model Updates (Metadata)

Add a structured outline to metadata. Example (adjust paths as needed):

`src/models/metadata.py`
```py
class ReportSectionPlan(BaseModel):
    title: str
    bullets: list[str] = Field(default_factory=list)
    salient_evidence_ids: list[str] = Field(default_factory=list)

class ReportMetadata(BaseModel):
    # ... existing fields ...
    section_outline: list[ReportSectionPlan] = Field(
        default_factory=list, description="Planned section outline provided to the agent"
    )
```

## Executor Changes (Outline Synthesis)

Compute outline right after clustering/ranking and before the agent runs.

`src/agents/research_executor.py` (conceptual placement)
```py
from models.metadata import ReportSectionPlan

# Deterministic headline from content (reusable utility)
def synthesize_headline(text: str, max_words: int = 8, max_len: int = 90) -> str:
    import re
    if not text:
        return ""
    para = re.sub(r"\[S\d+\]", "", text.split("\n\n", 1)[0]).strip()
    sent = re.split(r"(?<=[.!?])\s+", para)[0] if para else ""
    tokens = re.findall(r"[A-Za-z0-9'-]+", sent)
    stop = {"the","a","an","and","or","but","so","that","this","these","those","into","onto",
            "from","with","without","within","over","under","of","for","to","in","on","at","by",
            "is","are","was","were","be","being","been","as","it","its","their","our","your","his","her",
            "we","they","you","i","will","can","should","could","would","may","might","must","not"}
    words = [t for t in tokens if len(t) > 2 and t.lower() not in stop][:max_words]
    base = (" ".join(words) or sent).strip("-—–:;,. ")
    if len(base) > max_len:
        base = base[:max_len].rsplit(" ", 1)[0]
    return " ".join(w.capitalize() if w.islower() else w for w in base.split())

# After clustering and ranking findings
outline: list[ReportSectionPlan] = []
for cluster in results.theme_clusters:
    # Derive a content summary (first 2–3 sentences from cluster’s representative finding/evidence)
    text = some_cluster_to_text(cluster)  # implement as appropriate
    title = synthesize_headline(text) or cluster.theme_name
    bullets = pick_salient_points(cluster, k=3)    # short phrases, no trailing punctuation
    outline.append(ReportSectionPlan(title=title, bullets=bullets))

state.metadata.report.section_outline = outline
```

Notes:
- `some_cluster_to_text`: concatenate representative finding text and 1–2 best evidence sentences.
- `pick_salient_points`: reuse ranking logic to output short, specific points (no labels, minimal punctuation).

## Agent Prompt Changes

Inject the outline into the report generator’s dynamic instructions, asking the model to use those titles. Keep it structural, not a list of banned words.

`src/agents/report_generator.py` (inside `add_report_context`)
```py
# Build outline block if present
outline = getattr(metadata.report, "section_outline", []) if metadata else []
if outline:
    def fmt_item(i: int, item: Any) -> str:
        bullets = item.bullets[:3] if getattr(item, "bullets", None) else []
        btxt = "\n".join(f"    - {b}" for b in bullets)
        return f"{i+1}. {item.title}\n{btxt}"
    outline_block = "\n".join(fmt_item(i, it) for i, it in enumerate(outline))
else:
    outline_block = "(no outline provided)"

outline_instructions = (
    "Section Outline (use these as section headings; refine wording only for clarity):\n"
    + outline_block
)

# Return instructions string (append to system template inputs)
return REPORT_GENERATOR_SYSTEM_PROMPT_TEMPLATE.format(
    research_topic=research_topic,
    target_audience=target_audience,
    report_format=report_format,
    key_findings=key_findings,
    source_overview=source_overview,
    citation_contract=citation_contract,
    conversation_context=conversation_context + "\n\n" + outline_instructions,
)
```

Also adjust the template examples and structural cues:

`src/agents/report_generator.py` (within `REPORT_GENERATOR_SYSTEM_PROMPT_TEMPLATE`)
- In “Main Findings”, change “Start with a short heading …” to “Use the corresponding heading from the Section Outline; refine wording only for clarity.”
- Keep the few‑shot examples with natural, descriptive headings (no ordinals).

## Quality Gate + Clean‑Merge Policy

Make clean‑merge a safety net, gated by quick checks.

`src/agents/report_generator.py` (conceptual snippet)
```py
# After initial report generation and before citation post-processing
bad_heading_count = sum(
    1 for s in report.sections
    if looks_generic_heading(s.title, s.content)  # token-length, digit-led, poor overlap
)
ratio = bad_heading_count / max(len(report.sections), 1)

if deps.enable_llm_clean_merge and ratio > 0.25:
    report = await self._guardrailed_clean_merge(deps, report)
```

Where `looks_generic_heading` reuses the content‑overlap scoring already in the agent (no hard-coded words).

Optionally, keep the deterministic style/heading normalization behind a feature flag as a last resort during rollout.

## Testing

- Unit: outline synthesis
  - Given clustered findings, `synthesize_headline()` outputs a concise, citation-free title.
  - Bullets are short, specific, and contain no trailing punctuation.
- Unit: agent prompt wiring
  - `add_report_context` includes the outline block when `metadata.report.section_outline` exists.
  - Template string instructs to use outline headings.
- Unit: gating
  - Reports with poor heading overlap trigger clean‑merge when enabled; good ones skip it.
- Integration
  - End‑to‑end run produces headings drawn from the outline without “Finding N…” or scaffolding terms.

## Rollout Strategy

- Phase 1: Add outline generation and prompt injection; keep existing normalization and clean‑merge enabled.
- Phase 2: Gate clean‑merge on quality check; turn off deterministic normalization.
- Phase 3: Disable clean‑merge by default (exec summary only), keep flag available for incidents.

## Maintenance Notes

- Consider adding an embedding‑based similarity check behind `enable_embedding_similarity` to improve heading selection without LLMs.
- Expose minor tunables in config (max label length window, overlap thresholds) only if needed.

## References (Paths)
- Agent: src/agents/report_generator.py
- Metadata: src/models/metadata.py
- Executor: src/agents/research_executor.py
## Implementation Phases

### Phase 1 – Outline Generation & Metadata
- Implement `ReportSectionPlan` and extend `ReportMetadata.section_outline`.
- In executor, build the outline immediately after clustering/ranking: derive deterministic headings via the shared utility, collect the top evidence sentences (include source IDs), cap outline length (e.g., top 5 sections, max 2 bullets each).
- Ensure outline entries include `salient_evidence_ids` so the generator can cite without guessing.
- Add unit tests covering headline extraction/bullet trimming; after coding, run only these new tests and review the diff.

### Phase 2 – Prompt Injection & Template Updates
- Pass `section_outline` into `add_report_context` with a structured block.
- Update the system template: instruct the agent to use outline headings, weave each bullet into prose (explicit guidance: “Integrate each bullet into sentences; avoid labels such as ‘Implication:’”), and refresh few-shot examples with descriptive headings (no ordinals).
- Trim template tokens if needed to stay within budget.
- Add unit tests confirming the outline appears in the prompt and new instructions rendered; run them and review the code.

### Phase 3 – Post-Generation Validation & Iteration
- After generation, run deterministic checks (heading-to-content overlap, colon-prefix detection).
- If validation fails, log telemetry and regenerate once with an adjusted outline (e.g., expand synonyms) before any clean-merge attempt.
- Keep `apply_style_normalization` as a deterministic fallback during rollout; gate it with a feature flag for temporary use.
- Update metrics to record validation pass/fail counts.
- Add unit tests covering validation heuristics/fallback flow; run them and review the changes.

### Phase 4 – Clean-Merge Sunset
- Once validation telemetry shows high pass rates, set `enable_llm_clean_merge` default to False.
- Retain the clean-merge module only as a manual override (feature flag) and ensure validation remains in place.
- Add tests verifying the flag defaults off and validation still executes; run them and perform a final review.

## Testing & Telemetry
- At the end of each phase, add unit tests, run only those tests, and manually review the code.
- Track heading overlap scores, colon-prefix counts, and outline usage metrics to confirm clean-merge is unnecessary.
- Document regression/roll-back steps before fully disabling clean-merge.
