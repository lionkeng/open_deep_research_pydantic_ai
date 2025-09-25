"""Clean-merge utilities for the report generator agent."""

from __future__ import annotations

import re
import time
from collections import Counter
from typing import Any

import logfire
from pydantic import BaseModel
from pydantic import Field as PydField
from pydantic_ai import Agent as PydanticAgent

from core.config import config as global_config
from models.report_generator import ResearchReport
from models.research_executor import ResearchResults

from .base import ResearchDependencies

CLEAN_MERGE_INSTRUCTIONS_TEMPLATE = """
You are a senior editor.
Improve clarity and flow conservatively while strictly preserving meaning and citations.

Clean-Merge Task
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
4) Maintain a polished, confident voice for senior readers;
   no headings, meta commentary, or process talk.
5) Output only valid JSON with a single key named "value" whose value is the rewritten string.
6) If you cannot meet these constraints, return the original text unchanged.
"""


class CleanMergeOut(BaseModel):
    """Output schema for the clean-merge sub-agent."""

    value: str = PydField(description="Rewritten text output")


_clean_merge_agent: PydanticAgent[ResearchDependencies, CleanMergeOut] = PydanticAgent(
    model=global_config.get_model_config()["model"],
    deps_type=ResearchDependencies,
    output_type=CleanMergeOut,
    system_prompt="",
)


def marker_counts(text: str | None) -> dict[str, int]:
    if not text:
        return {}
    pattern = re.compile(r"\[S(\d+)\]")
    return dict(Counter(f"S{m.group(1)}" for m in pattern.finditer(text)))


def length_ok(before: str, after: str, tol: float = 0.15) -> bool:
    if not before:
        return True
    lo = int(len(before) * (1 - tol))
    hi = int(len(before) * (1 + tol))
    return lo <= len(after) <= hi


def build_continuity_context(report: ResearchReport) -> dict[str, str]:
    outline = ", ".join([getattr(s, "title", "") for s in report.sections])
    thesis = (report.executive_summary.split(". ")[0].strip()) if report.executive_summary else ""
    tone = "polished, confident, precise"
    terminology = ", ".join(sorted(set(re.findall(r"[A-Za-z]{4,}", (report.title or "")))))[:200]
    transition_cues = "Therefore; However; In contrast; As a result; Moreover; Consequently"
    return {
        "outline": outline,
        "thesis": thesis,
        "tone": tone,
        "terminology": terminology,
        "transition_cues": transition_cues,
    }


def neighbor_snippets(all_texts: list[str], idx: int) -> tuple[str, str]:
    prev = all_texts[idx - 1] if idx > 0 else ""
    next_ = all_texts[idx + 1] if idx + 1 < len(all_texts) else ""

    def head(s: str) -> str:
        parts = s.split(". ")
        return " ".join(parts[:2]).strip()

    def tail(s: str) -> str:
        parts = s.split(". ")
        return " ".join(parts[-2:]).strip() if len(parts) > 1 else s.strip()

    return tail(prev), head(next_)


async def run_clean_merge(
    *, deps: ResearchDependencies, report: ResearchReport
) -> tuple[ResearchReport, dict[str, int]]:
    """Apply the guardrailed clean-merge logic across report fields."""

    attempted = 0
    applied = 0
    reject_len = 0
    reject_marker = 0
    chunked_applied = 0

    # apply style normalization to the report
    report = apply_style_normalization(report)
    continuity = build_continuity_context(report)

    async def _clean_merge_text(
        *, field_name: str, text: str, prev_snippet: str, next_snippet: str
    ) -> str | None:
        nonlocal attempted, applied, reject_len, reject_marker, chunked_applied
        if not text or not text.strip():
            return None

        attempted += 1
        before = text
        before_counts = marker_counts(before)

        paragraphs = before.split("\n\n")
        if len(paragraphs) >= 3:
            updated_chunks: list[str] = []
            t0 = time.perf_counter()
            for i, chunk in enumerate(paragraphs):
                prev_chunk = paragraphs[i - 1] if i > 0 else prev_snippet
                next_chunk = paragraphs[i + 1] if i + 1 < len(paragraphs) else next_snippet
                instructions = CLEAN_MERGE_INSTRUCTIONS_TEMPLATE.format(
                    field_name=f"{field_name}_p{i}",
                    thesis=continuity.get("thesis", ""),
                    tone=continuity.get("tone", ""),
                    terminology=continuity.get("terminology", ""),
                    outline=continuity.get("outline", ""),
                    prev_snippet=" ".join(chunk.split(". ")[:2])
                    if prev_chunk == ""
                    else prev_chunk,
                    next_snippet=" ".join(chunk.split(". ")[-2:])
                    if next_chunk == ""
                    else next_chunk,
                    transition_cues=continuity.get("transition_cues", ""),
                )
                try:
                    result = await _clean_merge_agent.run(
                        deps=deps,
                        message_history=[
                            {"role": "system", "content": instructions},
                            {"role": "user", "content": chunk},
                        ],
                    )
                    updated = result.output.value
                except Exception as exc:  # pragma: no cover - external call
                    logfire.warning(
                        "Clean-merge chunk exception; keeping original",
                        field=field_name,
                        chunk_index=i,
                        error=str(exc),
                    )
                    updated = chunk

                if not length_ok(chunk, updated):
                    updated = chunk
                if marker_counts(updated) != marker_counts(chunk):
                    updated = chunk
                updated_chunks.append(updated)

            candidate = "\n\n".join(updated_chunks)
            if marker_counts(candidate) == before_counts and length_ok(before, candidate):
                dt = time.perf_counter() - t0
                logfire.info(
                    "Clean-merge applied (chunked)",
                    field=field_name,
                    chunks=len(paragraphs),
                    seconds=round(dt, 4),
                    len_before=len(before),
                    len_after=len(candidate),
                )
                applied += 1
                chunked_applied += 1
                return candidate

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
        t0 = time.perf_counter()
        result = await _clean_merge_agent.run(
            deps=deps,
            message_history=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": before},
            ],
        )
        after = result.output.value

        if not length_ok(before, after):
            logfire.warning("Clean-merge rejected (length)", field=field_name)
            reject_len += 1
            return None
        if marker_counts(after) != before_counts:
            logfire.warning("Clean-merge rejected (marker counts)", field=field_name)
            reject_marker += 1
            return None

        dt = time.perf_counter() - t0
        logfire.info(
            "Clean-merge applied",
            field=field_name,
            seconds=round(dt, 4),
            len_before=len(before),
            len_after=len(after),
        )
        applied += 1
        return after

    targets: list[tuple[Any, str, str]] = []
    targets.append((report, "executive_summary", "executive_summary"))
    targets.append((report, "introduction", "introduction"))
    for i, section in enumerate(report.sections):
        targets.append((section, "content", f"section_{i}_content"))
        for j, subsection in enumerate(section.subsections):
            targets.append((subsection, "content", f"section_{i}_sub_{j}_content"))
    targets.append((report, "conclusions", "conclusions"))

    linear_texts: list[str] = [str(getattr(obj, attr, "")) for (obj, attr, _) in targets]
    for idx, (obj, attr, label) in enumerate(targets):
        prev_snip, next_snip = neighbor_snippets(linear_texts, idx)
        before_text = linear_texts[idx]
        try:
            updated = await _clean_merge_text(
                field_name=label,
                text=before_text,
                prev_snippet=prev_snip,
                next_snippet=next_snip,
            )
        except Exception as exc:  # pragma: no cover - external call
            logfire.warning("Clean-merge exception; keeping original", field=label, error=str(exc))
            updated = None
        if isinstance(updated, str) and updated != before_text:
            setattr(obj, attr, updated)

    if report.recommendations:
        new_recs: list[str] = []
        for i, rec in enumerate(report.recommendations):
            try:
                updated = await _clean_merge_text(
                    field_name=f"recommendation_{i}",
                    text=str(rec),
                    prev_snippet=report.conclusions or "",
                    next_snippet="",
                )
            except Exception as exc:  # pragma: no cover - external call
                logfire.warning(
                    "Clean-merge exception; keeping original",
                    field=f"recommendation_{i}",
                    error=str(exc),
                )
                updated = None
            new_recs.append(updated if isinstance(updated, str) else rec)
        report.recommendations = new_recs

    if report.appendices:
        for key, value in list(report.appendices.items()):
            label = f"appendix_{key}"
            try:
                updated = await _clean_merge_text(
                    field_name=label,
                    text=str(value),
                    prev_snippet=report.conclusions or "",
                    next_snippet="",
                )
            except Exception as exc:  # pragma: no cover - external call
                logfire.warning(
                    "Clean-merge exception; keeping original",
                    field=label,
                    error=str(exc),
                )
                updated = None
            if isinstance(updated, str) and updated != value:
                report.appendices[key] = updated

    metrics = {
        "attempted": attempted,
        "applied": applied,
        "reject_len": reject_len,
        "reject_marker": reject_marker,
        "chunked_applied": chunked_applied,
    }
    return report, metrics


def record_clean_merge_metrics(*, metrics: dict[str, int], deps: ResearchDependencies) -> None:
    """Persist clean-merge metrics into synthesis metadata if available."""

    research_results: ResearchResults | None = getattr(
        deps.research_state, "research_results", None
    )
    if not research_results or not research_results.synthesis_metadata:
        return

    quality_metrics = research_results.synthesis_metadata.quality_metrics
    quality_metrics["clean_merge_fields_attempted"] = quality_metrics.get(
        "clean_merge_fields_attempted", 0
    ) + metrics.get("attempted", 0)
    quality_metrics["clean_merge_fields_applied"] = quality_metrics.get(
        "clean_merge_fields_applied", 0
    ) + metrics.get("applied", 0)
    quality_metrics["clean_merge_rejects_length"] = quality_metrics.get(
        "clean_merge_rejects_length", 0
    ) + metrics.get("reject_len", 0)
    quality_metrics["clean_merge_rejects_marker_mismatch"] = quality_metrics.get(
        "clean_merge_rejects_marker_mismatch", 0
    ) + metrics.get("reject_marker", 0)
    quality_metrics["clean_merge_chunked_applied"] = quality_metrics.get(
        "clean_merge_chunked_applied", 0
    ) + metrics.get("chunked_applied", 0)


def looks_generic_heading(title: str, content: str) -> bool:
    """Heuristic to detect generic headings that may need clean-merge assistance."""

    if not title:
        return True
    tokens = re.findall(r"[A-Za-z0-9'-]+", title)
    if not tokens:
        return True
    if tokens[0].isdigit():
        return True
    if len(tokens) <= 2:
        return True

    content = re.sub(r"\[S\d+\]", "", content or "").lower()
    keywords = set(re.findall(r"[a-z]{4,}", content))
    if not keywords:
        return False

    overlap = sum(1 for tok in tokens if tok.lower() in keywords)
    return (overlap / len(tokens)) < 0.25


# ---- Style normalization utilities -----------------------------------------


def _extract_keywords(text: str, *, top_k: int = 12) -> set[str]:
    if not text:
        return set()
    txt = re.sub(r"\[S\d+\]", "", str(text)).lower()
    tokens = re.findall(r"[a-z0-9'-]+", txt)
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "so",
        "that",
        "this",
        "these",
        "those",
        "into",
        "onto",
        "from",
        "with",
        "without",
        "within",
        "over",
        "under",
        "of",
        "for",
        "to",
        "in",
        "on",
        "at",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "being",
        "been",
        "as",
        "it",
        "its",
        "their",
        "our",
        "your",
        "his",
        "her",
        "we",
        "they",
        "you",
        "i",
        "will",
        "can",
        "should",
        "could",
        "would",
        "may",
        "might",
        "must",
        "not",
    }
    freq: dict[str, int] = {}
    for tok in tokens:
        if len(tok) <= 2 or tok in stop:
            continue
        freq[tok] = freq.get(tok, 0) + 1
    return {t for t, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:top_k]}


def _tokenize_heading(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"[A-Za-z0-9'-]+", text.lower())


def _overlap_score(heading: str, content_keywords: set[str]) -> float:
    h_tokens = [t for t in _tokenize_heading(heading) if len(t) > 2]
    if not h_tokens:
        return 0.0
    common = sum(1 for t in h_tokens if t in content_keywords)
    return common / len(h_tokens)


def synthesize_headline(content: str, *, max_words: int = 8, max_len: int = 90) -> str:
    if not content:
        return ""
    paragraph = content.split("\n\n", 1)[0]
    paragraph = re.sub(r"\[S\d+\]", "", paragraph).strip()
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    sentence = sentences[0].strip() if sentences else paragraph
    tokens = re.findall(r"[A-Za-z0-9'-]+", sentence)
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "so",
        "that",
        "this",
        "these",
        "those",
        "into",
        "onto",
        "from",
        "with",
        "without",
        "within",
        "over",
        "under",
        "of",
        "for",
        "to",
        "in",
        "on",
        "at",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "being",
        "been",
        "as",
        "it",
        "its",
        "their",
        "our",
        "your",
        "his",
        "her",
        "we",
        "they",
        "you",
        "i",
        "will",
        "can",
        "should",
        "could",
        "would",
        "may",
        "might",
        "must",
        "not",
    }
    words: list[str] = []
    for tok in tokens:
        low = tok.lower()
        if len(low) <= 2 or low in stop:
            continue
        words.append(tok)
        if len(words) >= max_words:
            break
    base = " ".join(words).strip() or sentence
    base = base.strip("-—–:;,. ")
    if len(base) > max_len:
        base = base[:max_len].rsplit(" ", 1)[0]
    return " ".join(w.capitalize() if w.islower() else w for w in base.split())


def strip_paragraph_qualifiers(text: str) -> str:
    if not text:
        return text

    def drop_label(paragraph: str) -> str:
        limit = 48
        candidate_idx = -1
        for delimiter in (":", "—", "–", "-"):
            idx = paragraph.find(delimiter)
            if 0 < idx <= limit:
                candidate_idx = idx if candidate_idx == -1 else min(candidate_idx, idx)
        if candidate_idx == -1:
            return paragraph
        prefix = paragraph[:candidate_idx].strip()
        suffix = paragraph[candidate_idx + 1 :].lstrip()
        tokens = re.findall(r"[A-Za-z][A-Za-z'-]+", prefix)
        if not (1 <= len(tokens) <= 5):
            return paragraph
        if re.search(r"[\d\[\]]", prefix):
            return paragraph

        def is_titleish(tok: str) -> bool:
            return tok.isupper() or (tok[0].isupper() and tok[1:].islower())

        ratio = sum(1 for t in tokens if is_titleish(t)) / len(tokens)
        suffix_first_upper = bool(suffix) and suffix[0].isupper()
        if ratio < 0.5 and not suffix_first_upper:
            return paragraph
        return suffix if suffix else paragraph

    paragraphs = str(text).split("\n\n")
    cleaned = [drop_label(p).strip() for p in paragraphs]
    return "\n\n".join(cleaned)


def normalize_title(title: str, content: str) -> str:
    if not title:
        return title
    t = str(title).strip()
    finding_prefix = bool(re.match(r"^finding\s*\d+", t, flags=re.IGNORECASE))
    t = re.sub(r"^finding\s*\d+\s*[:\-—–]*\s*", "", t, flags=re.IGNORECASE).strip()
    canonical = {
        "executive summary",
        "introduction",
        "recommendations",
        "conclusions",
        "conclusion",
    }
    if t.lower() in canonical:
        return t

    synthesized = synthesize_headline(content)
    keywords = _extract_keywords(content)
    orig_score = _overlap_score(t, keywords)
    synth_score = _overlap_score(synthesized, keywords) if synthesized else 0.0
    too_generic = len(_tokenize_heading(t)) <= 2 or bool(re.match(r"^\d+", t))
    pick_synth = (
        (synth_score - orig_score) >= 0.2
        or (orig_score < 0.2 and synth_score >= 0.3)
        or too_generic
    )
    chosen = synthesized if (synthesized and pick_synth) else t
    if finding_prefix or chosen == chosen.lower():
        chosen = " ".join(w.capitalize() if w else w for w in chosen.split())
    return chosen


def apply_style_normalization(report: ResearchReport) -> ResearchReport:
    report.executive_summary = strip_paragraph_qualifiers(report.executive_summary)
    report.introduction = strip_paragraph_qualifiers(report.introduction)
    report.conclusions = strip_paragraph_qualifiers(report.conclusions)

    for section in report.sections:
        section.content = strip_paragraph_qualifiers(section.content)
        section.title = normalize_title(section.title, section.content)
        for subsection in section.subsections:
            subsection.content = strip_paragraph_qualifiers(subsection.content)
            subsection.title = normalize_title(subsection.title, subsection.content)

    report.recommendations = [strip_paragraph_qualifiers(r) for r in report.recommendations]
    for key, value in list(report.appendices.items()):
        report.appendices[key] = strip_paragraph_qualifiers(value)
    return report


__all__ = [
    "CLEAN_MERGE_INSTRUCTIONS_TEMPLATE",
    "CleanMergeOut",
    "marker_counts",
    "length_ok",
    "build_continuity_context",
    "neighbor_snippets",
    "run_clean_merge",
    "record_clean_merge_metrics",
    "looks_generic_heading",
    "apply_style_normalization",
    "synthesize_headline",
]
