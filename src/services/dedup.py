"""Embedding-based deduplication and merge for findings.

Safely groups semantically equivalent findings and merges them by unifying
citations and evidence while preserving structure. Gated by availability of
an EmbeddingService and intended to be a pre-clustering cleanup step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import logfire

from models.research_executor import HierarchicalFinding
from services.embeddings import (
    EmbeddingService,
    cluster_by_threshold,
    pairwise_cosine_matrix,
)


def _finding_to_text(f: HierarchicalFinding) -> str:
    parts = [f.finding]
    if f.category:
        parts.append(f"Category: {f.category}")
    if f.supporting_evidence:
        parts.extend(f.supporting_evidence[:2])
    return " ".join(parts)


@dataclass
class DeDupService:
    embedding_service: EmbeddingService | None
    threshold: float = 0.6

    async def merge(self, findings: list[HierarchicalFinding]) -> list[HierarchicalFinding]:
        """Merge paraphrase-equivalent findings using embeddings.

        Args:
            findings: Input findings

        Returns:
            De-duplicated findings with merged evidence and citations
        """
        if not findings:
            return findings
        if self.embedding_service is None:
            return findings

        texts = [_finding_to_text(f) for f in findings]
        vectors = await self.embedding_service.embed_batch(texts)
        if not vectors:
            return findings

        sim = pairwise_cosine_matrix(vectors)
        indices = list(range(len(findings)))
        clusters = cluster_by_threshold(indices, sim, self.threshold)

        if not clusters or all(len(c) == 1 for c in clusters):
            return findings

        merged: list[HierarchicalFinding] = []
        used = set()
        for group in clusters:
            if len(group) == 1:
                idx = group[0]
                if idx not in used:
                    merged.append(findings[idx])
                    used.add(idx)
                continue

            # Merge group
            members = [findings[i] for i in group]
            for i in group:
                used.add(i)

            # Choose representative text (longest finding text)
            rep = max(members, key=lambda f: len(f.finding))

            # Union citations and evidence
            source_ids: list[str] = []
            evidence: list[str] = []
            for f in members:
                for sid in f.source_ids:
                    if sid and sid not in source_ids:
                        source_ids.append(sid)
                for ev in f.supporting_evidence:
                    if ev and ev not in evidence:
                        evidence.append(ev)

            # Aggregate scores
            avg_conf = sum(f.confidence_score for f in members) / len(members)
            avg_imp = sum(f.importance_score for f in members) / len(members)

            # Shallow metadata merge: prefer representative values, fill missing keys
            merged_meta: dict[str, Any] = dict(rep.metadata or {})  # type: ignore[arg-type]
            try:
                for mf in members:
                    if mf is rep or not isinstance(getattr(mf, "metadata", None), dict):
                        continue
                    for k, v in mf.metadata.items():
                        if k not in merged_meta:
                            merged_meta[k] = v
            except Exception:
                pass

            merged_finding = HierarchicalFinding(
                finding=rep.finding,
                supporting_evidence=evidence[:5],  # cap to keep concise
                confidence=rep.confidence,
                confidence_score=avg_conf,
                importance=rep.importance,  # keep enum unchanged to avoid drift
                importance_score=avg_imp,
                source=rep.source,
                category=rep.category,
                temporal_relevance=rep.temporal_relevance,
                metadata=merged_meta,
            )
            merged_finding.source_ids = source_ids[:]
            merged.append(merged_finding)

        # Add any untouched findings
        for i, f in enumerate(findings):
            if i not in used:
                merged.append(f)

        logfire.info(
            "Dedup merge",
            dedup_in=len(findings),
            dedup_out=len(merged),
            groups=len([g for g in clusters if len(g) > 1]),
        )
        return merged


__all__ = ["DeDupService"]
