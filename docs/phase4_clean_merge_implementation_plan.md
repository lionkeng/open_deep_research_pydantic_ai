# Phase 4: Context-Aware Clean‑Merge for Full Report

This document specifies the design and implementation plan to generalize the guarded clean‑merge from the executive summary to the entire report, improve continuity across sections/paragraphs, and switch the sub‑agent to use dynamic instructions instead of a fixed `system_prompt`.

- Goal: Improve readability and cohesion across the whole report while strictly preserving meaning and `[Sx]` markers.
- Safety: Deterministic guardrails (marker counts, length bounds, JSON shape) with best‑effort fallbacks per field.
- Agent: Use a Pydantic AI sub‑agent with dynamic `@instructions` for constraints and continuity context.

## Scope

Apply clean‑merge when `enable_llm_clean_merge=True` to:

- `executive_summary`, `introduction`, `conclusions`
- Each `ReportSection.content` and nested `subsections[].content`
- Each string in `recommendations[]`
- Each value in `appendices{}`

Run the clean‑merge before citation post‑processing (which renumbers `[Sx]` to footnotes), preserving the current citation logic in `src/agents/report_generator.py`.

---

## Agent Design (Instructions‑First)

- Use a global instructions template constant (similar to `REPORT_GENERATOR_SYSTEM_PROMPT_TEMPLATE`).
- Provide the instructions as the first `system` message in `message_history` when running the sub‑agent.
- Keep the sub‑agent’s `system_prompt` minimal or empty; the role and constraints live in the instructions.
- User message contains only the raw field text to be rewritten.

### Global constant: CLEAN_MERGE_INSTRUCTIONS_TEMPLATE

Define a module‑level constant in `src/agents/report_generator.py`:

```
CLEAN_MERGE_INSTRUCTIONS_TEMPLATE = """
You are a senior editor. Improve clarity and flow conservatively while strictly preserving meaning and citations.

Clean‑Merge Task
Field: {field_name}

Continuity Guide
- Thesis: {thesis}
- Tone: {tone}
- Terminology: {terminology}
- Outline: {outline}
- Prev snippet: {prev_snippet}
- Next snippet: {next_snippet}
- Transition cues: {transition_cues}

Hard Constraints
1) Preserve every citation marker exactly as written, including all occurrences and their positions:
   - Do not add, remove, rename, or reorder any "[Sx]" markers.
   - Do not alter whitespace inside/around markers.
2) Do not add or remove facts. Keep named entities, quantities, dates, and metrics unchanged.
3) Keep length within ±15% of the original.
4) Maintain a polished, confident voice for senior readers; no headings, meta commentary, or process talk.
5) Output only valid JSON with a single key named "value" whose value is the rewritten string.
6) If you cannot meet these constraints, return the original text unchanged.
"""
```

Notes:

- The editor role (“You are a senior editor…”) is specified at the top of the instructions, not in `system_prompt`.
- The output schema is a single key `value`; the caller maps it to the field.

---

## Orchestration in `report_generator.py`

Add a report‑level orchestrator that calls a field‑level helper with continuity context and guardrails.

### New helpers (snippets)

```python
# src/agents/report_generator.py
from collections import Counter

class ReportGeneratorAgent(...):
    @staticmethod
    def _marker_counts(text: str | None) -> dict[str, int]:
        if not text:
            return {}
        pattern = re.compile(r"\[S(\d+)\]")
        c: Counter[str] = Counter(f"S{m.group(1)}" for m in pattern.finditer(text))
        return dict(c)

    def _length_ok(self, before: str, after: str, tol: float = 0.15) -> bool:
        if not before:
            return True
        lo = int(len(before) * (1 - tol))
        hi = int(len(before) * (1 + tol))
        return lo <= len(after) <= hi

    def _build_continuity_context(self, report: ResearchReport) -> dict[str, str]:
        # Cheap, deterministic cues; extend later if needed
        outline = ", ".join([s.title for s in report.sections])
        thesis = (report.executive_summary.split(". ")[0].strip() if report.executive_summary else "")
        tone = "polished, confident, precise"
        terminology = ", ".join(sorted(set(re.findall(r"[A-Za-z]{4,}", report.title or ""))))[:200]
        transition_cues = "Therefore; However; In contrast; As a result"
        return {
            "outline": outline,
            "thesis": thesis,
            "tone": tone,
            "terminology": terminology,
            "transition_cues": transition_cues,
        }

    def _neighbor_snippets(self, all_texts: list[str], idx: int) -> tuple[str, str]:
        # Use head/tail sentences to prime continuity
        import textwrap
        prev = all_texts[idx - 1] if idx > 0 else ""
        next_ = all_texts[idx + 1] if idx + 1 < len(all_texts) else ""
        def head(s: str) -> str:
            return " ".join(s.split(". ")[:2]).strip()
        def tail(s: str) -> str:
            parts = s.split(". ")
            return " ".join(parts[-2:]).strip() if len(parts) > 1 else s.strip()
        return (tail(prev), head(next_))
```

### Field‑level clean‑merge

```python
# src/agents/report_generator.py
from pydantic import BaseModel, Field as PydField
from pydantic_ai import Agent as PydanticAgent

class ReportGeneratorAgent(...):
    async def _clean_merge_text(
        self,
        *,
        deps: ResearchDependencies,
        field_name: str,
        text: str,
        continuity: dict[str, str],
        prev_snippet: str = "",
        next_snippet: str = "",
    ) -> str | None:
        if not text or not text.strip():
            return None

        before_counts = self._marker_counts(text)

        class _Out(BaseModel):
            value: str = PydField(description=f"Rewritten {field_name}")

        agent = PydanticAgent[ResearchDependencies, _Out](
            model=self.model,
            deps_type=type(deps),
            output_type=_Out,
            system_prompt="",  # role and constraints live in instructions below
        )

        instructions = CLEAN_MERGE_INSTRUCTIONS_TEMPLATE.format(
            field_name=field_name,
            thesis=continuity.get("thesis", ""),
            tone=continuity.get("tone", ""),
            terminology=continuity.get("terminology", ""),
            outline=continuity.get("outline", ""),
            prev_snippet=prev_snippet,
            next_snippet=next_snippet,
            transition_cues=continuity.get("transition_cues", ""),
        )

        result = await agent.run(
            deps=deps,
            message_history=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": text},
            ],
        )
        new_text = result.output.value

        # Guardrails
        if not self._length_ok(text, new_text):
            logfire.warning("clean_merge rejected (length)", field=field_name)
            return None
        if self._marker_counts(new_text) != before_counts:
            logfire.warning("clean_merge rejected (marker counts)", field=field_name)
            return None
        return new_text
```

### Report‑level orchestration

```python
# src/agents/report_generator.py
class ReportGeneratorAgent(...):
    async def _clean_merge_report(self, deps: ResearchDependencies, report: ResearchReport) -> ResearchReport:
        contx = self._build_continuity_context(report)

        # Collect (obj, attr, label) in linear reading order
        targets: list[tuple[object, str, str]] = []
        targets.append((report, "executive_summary", "executive_summary"))
        targets.append((report, "introduction", "introduction"))
        for i, s in enumerate(report.sections):
            targets.append((s, "content", f"section_{i}_content"))
            for j, ss in enumerate(s.subsections):
                targets.append((ss, "content", f"section_{i}_sub_{j}_content"))
        targets.append((report, "conclusions", "conclusions"))
        # Recommendations and appendices are after main narrative
        rec_labels = [f"recommendation_{i}" for i, _ in enumerate(report.recommendations)]
        app_labels = [f"appendix_{k}" for k in report.appendices.keys()]

        linear_texts: list[str] = [str(getattr(o, a, "")) for (o, a, _) in targets]
        for idx, (obj, attr, label) in enumerate(targets):
            text = linear_texts[idx]
            prev_snip, next_snip = self._neighbor_snippets(linear_texts, idx)
            try:
                new_text = await self._clean_merge_text(
                    deps=deps,
                    field_name=label,
                    text=text,
                    continuity=contx,
                    prev_snippet=prev_snip,
                    next_snippet=next_snip,
                )
                if new_text is not None and new_text != text:
                    setattr(obj, attr, new_text)
            except Exception as exc:
                logfire.warning("clean_merge exception; keeping original", field=label, error=str(exc))

        # Recommendations
        new_recs: list[str] = []
        for i, rec in enumerate(report.recommendations):
            prev_snip = linear_texts[-1] if linear_texts else ""
            next_snip = report.conclusions
            label = rec_labels[i]
            upd = await self._clean_merge_text(
                deps=deps, field_name=label, text=str(rec), continuity=contx,
                prev_snippet=prev_snip, next_snippet=next_snip,
            )
            new_recs.append(upd if isinstance(upd, str) else rec)
        report.recommendations = new_recs

        # Appendices
        for key, value in list(report.appendices.items()):
            label = f"appendix_{key}"
            upd = await self._clean_merge_text(
                deps=deps, field_name=label, text=str(value), continuity=contx,
                prev_snippet=report.conclusions, next_snippet="",
            )
            if isinstance(upd, str):
                report.appendices[key] = upd

        return report
```

### Integrate with `run()`

```python
# src/agents/report_generator.py (inside run)
report = await super().run(deps=deps, message_history=message_history, stream=stream)
actual_deps = deps or self.dependencies
if actual_deps and getattr(actual_deps, "enable_llm_clean_merge", False):
    try:
        report = await self._clean_merge_report(actual_deps, report)
        logfire.info("clean_merge applied across report")
    except Exception as exc:
        logfire.warning("clean_merge failed; using original", error=str(exc))
# then call self._apply_citation_postprocessing(report, actual_deps)
```

---

## Chunking for Long Fields

- Split on paragraph boundaries (`"\n\n"`) only.
- For each chunk, run `_clean_merge_text()` with prev/next chunk snippets.
- Stitch with original separators.
- Ensure each chunk’s marker counts match its original; aggregate counts must match the field’s original.

Pseudo‑code inside `_clean_merge_text` (optional extension):

```python
paras = text.split("\n\n")
if len(text) > 4000 and len(paras) > 1:
    rebuilt = []
    for i, p in enumerate(paras):
        ps, ns = (paras[i-1] if i else ""), (paras[i+1] if i+1 < len(paras) else "")
        upd = await self._clean_merge_text(
            deps=deps, field_name=f"{field_name}_p{i}", text=p, continuity=continuity,
            prev_snippet=ps, next_snippet=ns,
        )
        rebuilt.append(upd or p)
    candidate = "\n\n".join(rebuilt)
    if self._marker_counts(candidate) == self._marker_counts(text) and self._length_ok(text, candidate):
        return candidate
    return None
```

---

## Metrics & Observability

- Per‑field logs:
  - `clean_merge_attempted`, `clean_merge_applied`, `field`, `before_len`, `after_len`, `seconds`
  - Rejections by reason: `marker_mismatch`, `length`, `shape`, `exception`
- Continuity metrics (optional, cheap):
  - `continuity_cohesion_before/after`: cosine between adjacent section embeddings (title + first 2 sentences)
  - `transition_fix_rate`: share of fields where added connective cue detected
- Aggregate in `SynthesisMetadata.quality_metrics`:
  - `clean_merge_fields_attempted`, `clean_merge_fields_applied`, `clean_merge_rejects_marker_mismatch`, `clean_merge_rejects_length`, `clean_merge_rejects_shape`

---

## Configuration

- Use existing `enable_llm_clean_merge` (global flag).
- Optional future flags:
  - `clean_merge_max_len` (threshold to trigger chunking)
  - `clean_merge_fields` (allowlist)

---

## Testing Plan

- Unit tests (`tests/unit/test_embeddings_and_clean_merge.py`):
  - `test_marker_counts_guardrail()` – dropping/duplicating markers causes rejection.
  - `test_clean_merge_multi_field_apply_reject()` – mock agent returns both valid and invalid rewrites; ensure only valid fields change.
  - `test_chunking_preserves_markers()` – long multi‑paragraph input stitched correctly with same marker counts.
  - `test_instructions_include_context()` – verify instructions contain field name, prev/next snippets, and continuity keys.
- Integration (mock sub‑agent):
  - With `enable_llm_clean_merge=True`, the report‑level path runs, logs metrics, and `_apply_citation_postprocessing()` still renumbers markers correctly.

---

## Phased Rollout

- Phase A: Foundations

  - Add `_marker_counts`, length guard, continuity builder, and `_clean_merge_text` for a single field (executive summary).
  - Switch the sub‑agent to instructions‑based design; keep previous path behind flag for fallback.
  - At the completion of this phase,
    - consider adding unit tests and run the unit tests,
    - review the code, and fix issues.

- Phase B: Core Narrative

  - Extend to `introduction`, `conclusions`, and top‑level `sections[].content` (no chunking yet).
  - Add neighbor‑snippet windowing for continuity.
  - At the completion of this phase,
    - consider adding unit tests and run the unit tests,
    - review the code, and fix issues.

- Phase C: Full Coverage

  - Include `subsections`, `recommendations[]`, and `appendices{}`.
  - Add per‑field metrics and aggregate counters.
  - At the completion of this phase,
    - consider adding unit tests and run the unit tests,
    - review the code, and fix issues.

- Phase D: Scale & Robustness

  - Implement paragraph‑level chunking with stitching and per‑chunk guardrails.
  - Add optional continuity metrics (embedding‑based cohesion) gated by existing embedding flag.
  - At the completion of this phase,
    - consider adding unit tests and run the unit tests,
    - review the code, and fix issues.

- Phase E: Tuning & Evaluation
  - Adjust length tolerance, refine instructions and transition cues.
  - Expand tests; update docs and acceptance criteria.

---

## Acceptance Criteria (Phase 4)

- With `enable_llm_clean_merge=True`:
  - Clean‑merge runs across targeted fields; logs show `clean_merge_applied` events.
  - For all applied fields, `[Sx]` marker counts are identical; length within ±15%.
  - No JSON shape violations; fallbacks keep originals when constraints fail.
  - Citation post‑processing and audit still pass.

---

## Notes & Risks

- Continuity improvements are conservative by design; constraints prevent factual drift.
- Chunking’s stitching can introduce subtle rhythm changes; metrics and tests limit regressions.
- If latency becomes an issue, batch smaller fields together or cap max fields per pass.
