"""Helpers for producing deterministic report outlines from executor clusters."""

from __future__ import annotations

import re
from collections.abc import Sequence

from agents.headline_tokens import GENERIC_SECTION_STARTERS
from models.metadata import ReportSectionPlan
from models.research_executor import HierarchicalFinding, ThemeCluster

_CITATION_TAG = re.compile(r"\[S\d+\]")

_STOP_WORDS = {
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
    "which",
}


_LEADING_HEADLINE_SKIP_TOKENS = {
    "evidence",
    "summary",
    "key",
} | GENERIC_SECTION_STARTERS


def synthesize_headline(text: str, max_words: int = 8, max_len: int = 90) -> str:
    """Create a deterministic heading candidate from raw research content."""

    if not text:
        return ""

    cleaned = _CITATION_TAG.sub("", text)
    paragraph = cleaned.strip().split("\n\n", 1)[0]
    if not paragraph:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    sentence = sentences[0] if sentences else paragraph
    tokens = re.findall(r"[A-Za-z0-9'\-]+", sentence)
    words = [token for token in tokens if len(token) > 2 and token.lower() not in _STOP_WORDS]

    while words and words[0].lower() in _LEADING_HEADLINE_SKIP_TOKENS:
        words.pop(0)

    deduped: list[str] = []
    seen: set[str] = set()
    for token in words:
        lowered = token.lower()
        if lowered in seen:
            continue
        deduped.append(token)
        seen.add(lowered)

    words = deduped

    selected = words[:max_words]
    base = " ".join(selected) if selected else sentence
    base = base.strip("-—–:;,. ")
    if not base:
        return ""

    if len(base) > max_len:
        truncated = base[:max_len].rsplit(" ", 1)[0]
        if truncated:
            base = truncated

    capitalized = " ".join(word.capitalize() if word.islower() else word for word in base.split())
    return capitalized


def _normalize_bullet(text: str, max_length: int = 160) -> str:
    """Produce a concise bullet without trailing punctuation or citations."""

    cleaned = _CITATION_TAG.sub("", text or "")
    cleaned = " ".join(cleaned.split()).strip("-• ")
    if not cleaned:
        return ""

    while cleaned and cleaned[-1] in ".;:,":
        cleaned = cleaned[:-1].rstrip()

    if len(cleaned) > max_length:
        truncated = cleaned[:max_length].rsplit(" ", 1)[0]
        if truncated:
            cleaned = truncated

    return cleaned


def _rank_findings(findings: Sequence[HierarchicalFinding]) -> list[HierarchicalFinding]:
    return sorted(
        findings,
        key=lambda f: (f.importance_score, f.confidence_score, f.finding_id),
        reverse=True,
    )


def _cluster_summary_text(
    cluster: ThemeCluster,
    *,
    max_findings: int = 2,
    max_evidence: int = 2,
) -> str:
    parts: list[str] = []
    for finding in _rank_findings(cluster.findings)[:max_findings]:
        parts.append(finding.finding)
        if finding.supporting_evidence:
            parts.extend(finding.supporting_evidence[:max_evidence])
    if not parts:
        for fallback in (cluster.description, cluster.theme_name):
            if fallback:
                parts.append(fallback)
                break
    return " ".join(parts)


def _select_salient_bullets(
    cluster: ThemeCluster,
    *,
    max_bullets: int = 2,
) -> list[str]:
    bullets: list[str] = []
    for finding in _rank_findings(cluster.findings):
        bullet = _normalize_bullet(finding.finding)
        if not bullet or bullet in bullets:
            continue
        bullets.append(bullet)
        if len(bullets) >= max_bullets:
            break
    return bullets


def _collect_salient_evidence_ids(
    cluster: ThemeCluster,
    *,
    max_ids: int = 6,
) -> list[str]:
    collected: list[str] = []
    seen: set[str] = set()
    for finding in _rank_findings(cluster.findings):
        for source_id in list(finding.source_ids) + list(finding.supporting_source_ids):
            if not source_id or source_id in seen:
                continue
            seen.add(source_id)
            collected.append(source_id)
            if len(collected) >= max_ids:
                return collected
    return collected


def build_section_outline(
    clusters: Sequence[ThemeCluster],
    *,
    max_sections: int = 5,
    max_bullets: int = 2,
) -> list[ReportSectionPlan]:
    """Create a deterministic outline from theme clusters."""

    if not clusters:
        return []

    ordered_clusters = sorted(
        [cluster for cluster in clusters if cluster.findings],
        key=lambda c: (c.importance_score, c.coherence_score, c.cluster_id),
        reverse=True,
    )

    outline: list[ReportSectionPlan] = []
    for cluster in ordered_clusters:
        summary_text = _cluster_summary_text(cluster)
        title = synthesize_headline(summary_text)
        if not title:
            title = (cluster.theme_name or "").strip()
        title = title or "Untitled Section"

        bullets = _select_salient_bullets(cluster, max_bullets=max_bullets)
        evidence_ids = _collect_salient_evidence_ids(cluster)

        outline.append(
            ReportSectionPlan(
                title=title,
                bullets=bullets,
                salient_evidence_ids=evidence_ids,
            )
        )
        if len(outline) >= max_sections:
            break

    return outline


__all__ = [
    "build_section_outline",
    "synthesize_headline",
]
