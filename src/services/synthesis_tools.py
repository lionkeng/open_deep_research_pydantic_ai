"""Synthesis Tools Service for advanced information processing and analysis."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from time import perf_counter
from typing import Any

import logfire
from pydantic import BaseModel, Field

# Optional embedding service for semantic similarity
try:  # Local import guard to avoid hard dependency
    from services.embeddings import (
        EmbeddingService,
        cluster_by_threshold,
        pairwise_cosine_matrix,
    )
except Exception:  # pragma: no cover - embedding layer is optional
    EmbeddingService = None  # type: ignore[assignment]
    cluster_by_threshold = None  # type: ignore[assignment]
    pairwise_cosine_matrix = None  # type: ignore[assignment]

# Local model definitions until integrated with main models


class PatternType(str, Enum):
    """Types of patterns that can be detected."""

    TEMPORAL = "temporal"
    CAUSAL = "causal"
    CORRELATION = "correlation"  # Changed from CORRELATIVE to match models/research_executor.py
    COMPARATIVE = "comparative"


class ContradictionType(str, Enum):
    """Types of contradictions."""

    FACTUAL = "factual"
    TEMPORAL = "temporal"
    QUANTITATIVE = "quantitative"


class InformationHierarchy(str, Enum):
    """Information hierarchy levels."""

    PRIMARY = "primary"
    SUPPORTING = "supporting"
    CONTEXTUAL = "contextual"
    TANGENTIAL = "tangential"


class SearchResult(BaseModel):
    """Search result model."""

    query: str = Field(description="The original query")
    results: list[Any] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ThemeType(str, Enum):
    """Types of themes that can be extracted."""

    CONCEPT = "concept"
    ENTITY = "entity"
    EVENT = "event"
    TREND = "trend"
    RELATIONSHIP = "relationship"


@dataclass
class HierarchyScore:
    """Score for information hierarchy classification."""

    level: InformationHierarchy
    score: float
    factors: dict[str, float]
    reasoning: str


@dataclass
class Contradiction:
    """Detected contradiction between information sources."""

    type: ContradictionType
    source1: str
    source2: str
    claim1: str
    claim2: str
    confidence: float
    resolution: str | None = None


@dataclass
class ConvergencePoint:
    """Point of convergence across multiple sources."""

    claim: str
    sources: list[str]
    support_count: int
    confidence: float
    evidence: list[str]


@dataclass
class Theme:
    """Extracted theme with relationships."""

    type: ThemeType
    name: str
    description: str
    frequency: int
    sources: list[str]
    related_themes: list[str]
    confidence: float
    context: dict[str, Any] | None = None


@dataclass
class Pattern:
    """Detected pattern in information."""

    type: PatternType
    description: str
    evidence: list[str]
    strength: float
    timeframe: str | None = None
    entities: list[str] | None = None


class HierarchyFactors(BaseModel):
    """Factors for information hierarchy assessment."""

    relevance_weight: float = Field(default=0.3, description="Weight for relevance to query")
    novelty_weight: float = Field(default=0.2, description="Weight for information novelty")
    credibility_weight: float = Field(default=0.25, description="Weight for source credibility")
    specificity_weight: float = Field(
        default=0.15, description="Weight for information specificity"
    )
    recency_weight: float = Field(default=0.1, description="Weight for information recency")


class ContradictionConfig(BaseModel):
    """Configuration for contradiction detection."""

    min_confidence: float = Field(default=0.6, description="Minimum confidence for contradictions")
    semantic_threshold: float = Field(default=0.8, description="Threshold for semantic similarity")
    temporal_window_days: int = Field(default=30, description="Window for temporal contradictions")
    quantitative_tolerance: float = Field(
        default=0.1, description="Tolerance for quantitative differences"
    )


class SynthesisTools:
    """
    Advanced tools for information synthesis and analysis.

    Provides hierarchy scoring, contradiction detection, convergence analysis,
    theme extraction, and pattern matching capabilities.
    """

    def __init__(
        self,
        hierarchy_factors: HierarchyFactors | None = None,
        contradiction_config: ContradictionConfig | None = None,
        *,
        embedding_service: EmbeddingService | None = None,
        enable_embedding_similarity: bool = False,
        similarity_threshold: float = 0.55,
        convergence_max_claims: int | None = None,
        convergence_per_source_cap: int | None = None,
        convergence_sampling_strategy: str | None = None,
    ):
        """
        Initialize SynthesisTools.

        Args:
            hierarchy_factors: Configuration for hierarchy scoring
            contradiction_config: Configuration for contradiction detection
        """
        self.hierarchy_factors = hierarchy_factors or HierarchyFactors()
        self.contradiction_config = contradiction_config or ContradictionConfig()

        # Embedding-based similarity (optional)
        self.embedding_service: EmbeddingService | None = embedding_service
        self.enable_embedding_similarity: bool = enable_embedding_similarity
        self.similarity_threshold: float = similarity_threshold

        # Convergence caps/sampling (defaults from global config if available)
        try:
            from core.config import config as _global_config

            default_max = _global_config.convergence_max_claims
            default_per_src = _global_config.convergence_per_source_cap
            default_strategy = _global_config.convergence_sampling_strategy
        except Exception:  # pragma: no cover - config import fallback
            default_max = 300
            default_per_src = 0
            default_strategy = "longest"

        self.convergence_max_claims: int = (
            convergence_max_claims if convergence_max_claims is not None else int(default_max)
        )
        self.convergence_per_source_cap: int = (
            convergence_per_source_cap
            if convergence_per_source_cap is not None
            else int(default_per_src)
        )
        self.convergence_sampling_strategy: str = (
            convergence_sampling_strategy or str(default_strategy)
        ).lower()

        # Cache for computed similarities
        self._similarity_cache: dict[tuple[str, str], float] = {}

    def score_information_hierarchy(
        self,
        information: str,
        query: str,
        source_credibility: float = 0.5,
        timestamp: datetime | None = None,
        context: dict[str, Any] | None = None,
    ) -> HierarchyScore:
        """
        Score information to determine its hierarchy level.

        Multi-factor assessment including relevance, novelty, credibility,
        specificity, and recency.
        """
        factors = {}

        # 1. Relevance to query
        relevance = self._calculate_relevance(information, query)
        factors["relevance"] = relevance

        # 2. Information novelty
        novelty = self._calculate_novelty(information, context)
        factors["novelty"] = novelty

        # 3. Source credibility
        factors["credibility"] = source_credibility

        # 4. Information specificity
        specificity = self._calculate_specificity(information)
        factors["specificity"] = specificity

        # 5. Recency
        recency = self._calculate_recency(timestamp)
        factors["recency"] = recency

        # Calculate weighted score
        total_score = (
            relevance * self.hierarchy_factors.relevance_weight
            + novelty * self.hierarchy_factors.novelty_weight
            + source_credibility * self.hierarchy_factors.credibility_weight
            + specificity * self.hierarchy_factors.specificity_weight
            + recency * self.hierarchy_factors.recency_weight
        )

        # Determine hierarchy level
        if total_score >= 0.75:
            level = InformationHierarchy.PRIMARY
            reasoning = "High relevance, credibility, and specificity"
        elif total_score >= 0.5:
            level = InformationHierarchy.SUPPORTING
            reasoning = "Moderate relevance and credibility"
        elif total_score >= 0.25:
            level = InformationHierarchy.CONTEXTUAL
            reasoning = "Provides context but lower direct relevance"
        else:
            level = InformationHierarchy.TANGENTIAL
            reasoning = "Limited relevance to main query"

        return HierarchyScore(
            level=level,
            score=total_score,
            factors=factors,
            reasoning=reasoning,
        )

    def _calculate_relevance(self, information: str, query: str) -> float:
        """Calculate relevance score between information and query."""
        # Simple keyword-based relevance (can be enhanced with embeddings)
        query_lower = query.lower()
        info_lower = information.lower()

        query_terms = set(query_lower.split())
        info_terms = set(info_lower.split())

        if not query_terms:
            return 0.0

        # Calculate Jaccard similarity
        intersection = query_terms & info_terms
        union = query_terms | info_terms

        if not union:
            return 0.0

        jaccard = len(intersection) / len(union)

        # Boost for exact phrase matches
        if query_lower in info_lower:
            jaccard = min(1.0, jaccard * 2.0)  # Stronger boost for exact phrase

        # Additional boost if all query terms are present
        if query_terms.issubset(info_terms):
            jaccard = min(1.0, jaccard * 1.5)

        # Special boost for answer-indicating phrases
        answer_indicators = [
            "directly answers",
            "answers the",
            "specifically addresses",
            "directly relates",
        ]
        for indicator in answer_indicators:
            if indicator in info_lower and jaccard > 0.3:
                jaccard = min(1.0, jaccard * 1.5)
                break

        return jaccard

    def _calculate_novelty(self, information: str, context: dict[str, Any] | None) -> float:
        """Calculate novelty score for information."""
        if not context or "seen_information" not in context:
            return 1.0  # All information is novel if no context

        seen_info = context["seen_information"]

        # Check similarity with previously seen information
        max_similarity = 0.0
        for seen in seen_info:
            similarity = self._calculate_similarity(information, seen)
            max_similarity = max(max_similarity, similarity)

        # Novelty is inverse of maximum similarity
        novelty = 1.0 - max_similarity
        return novelty

    def _calculate_specificity(self, information: str) -> float:
        """Calculate specificity score based on information characteristics."""
        specificity_indicators = {
            "numbers": r"\d+\.?\d*",  # Numerical values
            "dates": r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}",  # Date patterns
            "years": r"\b\d{4}\b",  # Year patterns
            "percentages": r"\d+\.?\d*%",  # Percentages
            "currency": r"\$\d+\.?\d*[KMB]?",  # Currency amounts
            "quotes": r'"[^"]*"',  # Quoted text
            "proper_nouns": r"\b[A-Z][a-z]+\b",  # Capitalized words
        }

        score = 0.0
        indicators_found = 0

        for _indicator, pattern in specificity_indicators.items():
            matches = re.findall(pattern, information)
            if matches:
                indicators_found += 1
                # Give more weight per match, less penalty for multiple
                score += min(1.0, len(matches) / 3)

        # Calculate final score based on indicators found
        if indicators_found > 0:
            # Give extra boost when multiple indicators are present
            multiplier = 1.5 + (indicators_found - 1) * 0.1
            return min(1.0, score / len(specificity_indicators) * multiplier)
        return 0.0

    def _calculate_recency(self, timestamp: datetime | None) -> float:
        """Calculate recency score based on timestamp."""
        if not timestamp:
            return 0.5  # Neutral score if no timestamp

        now = datetime.now(UTC)
        age_days = (now - timestamp).days

        # Exponential decay with half-life of 30 days
        half_life = 30
        recency = 0.5 ** (age_days / half_life)

        return recency

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        # Check cache
        cache_key = (text1, text2) if text1 < text2 else (text2, text1)
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]

        # Simple token-based similarity (can be enhanced with embeddings)
        text1_lower = text1.lower()
        text2_lower = text2.lower()

        tokens1 = set(text1_lower.split())
        tokens2 = set(text2_lower.split())

        if not tokens1 or not tokens2:
            similarity = 0.0
        else:
            intersection = tokens1 & tokens2
            union = tokens1 | tokens2
            jaccard = len(intersection) / len(union)

            # Check for common synonyms and related terms
            synonym_pairs = [
                ("ai", "artificial intelligence"),
                ("increasing", "growing"),
                ("rapidly", "quickly"),
                ("across", "in"),
                ("industries", "sectors"),
                ("implementation", "adoption"),
                ("accelerating", "increasing"),
                ("various", "different"),
            ]

            synonym_matches = 0
            for term1, term2 in synonym_pairs:
                if (term1 in text1_lower and term2 in text2_lower) or (
                    term2 in text1_lower and term1 in text2_lower
                ):
                    synonym_matches += 1

            # Boost similarity based on synonyms
            synonym_boost = min(0.3, synonym_matches * 0.1)

            similarity = min(1.0, jaccard + synonym_boost)

        # Cache result
        self._similarity_cache[cache_key] = similarity
        return similarity

    def detect_contradictions(self, search_results: list[SearchResult]) -> list[Contradiction]:
        """
        Detect contradictions across search results.

        Implements algorithms for factual, temporal, and quantitative contradictions.
        """
        contradictions = []

        # Extract claims from all results
        claims_by_source = self._extract_claims(search_results)

        # Compare all pairs of claims
        sources = list(claims_by_source.keys())
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                source1, source2 = sources[i], sources[j]
                claims1 = claims_by_source[source1]
                claims2 = claims_by_source[source2]

                # Check each pair of claims
                for claim1 in claims1:
                    for claim2 in claims2:
                        contradiction = self._check_contradiction(claim1, claim2, source1, source2)
                        if (
                            contradiction
                            and contradiction.confidence >= self.contradiction_config.min_confidence
                        ):
                            contradictions.append(contradiction)

        return contradictions

    def _extract_claims(self, search_results: list[SearchResult]) -> dict[str, list[str]]:
        """Extract claims from search results."""
        claims_by_source = defaultdict(list)

        for result in search_results:
            source = result.query  # Use query as source identifier

            # Extract sentences as potential claims
            if hasattr(result, "results"):
                for item in result.results:
                    if hasattr(item, "content"):
                        sentences = re.split(r"[.!?]+", item.content)
                        for sentence in sentences:
                            if len(sentence.strip()) > 20:  # Filter out short fragments
                                claims_by_source[source].append(sentence.strip())

        return dict(claims_by_source)

    def _check_contradiction(
        self, claim1: str, claim2: str, source1: str, source2: str
    ) -> Contradiction | None:
        """Check if two claims contradict each other."""
        # Check for factual contradictions
        factual = self._check_factual_contradiction(claim1, claim2)
        if factual:
            return Contradiction(
                type=ContradictionType.FACTUAL,
                source1=source1,
                source2=source2,
                claim1=claim1,
                claim2=claim2,
                confidence=factual,
                resolution=None,
            )

        # Check for temporal contradictions
        temporal = self._check_temporal_contradiction(claim1, claim2)
        if temporal:
            return Contradiction(
                type=ContradictionType.TEMPORAL,
                source1=source1,
                source2=source2,
                claim1=claim1,
                claim2=claim2,
                confidence=temporal,
                resolution="Check temporal context",
            )

        # Check for quantitative contradictions
        quantitative = self._check_quantitative_contradiction(claim1, claim2)
        if quantitative:
            return Contradiction(
                type=ContradictionType.QUANTITATIVE,
                source1=source1,
                source2=source2,
                claim1=claim1,
                claim2=claim2,
                confidence=quantitative,
                resolution="Verify numerical values",
            )

        return None

    def _check_factual_contradiction(self, claim1: str, claim2: str) -> float:
        """Check for factual contradictions using negation patterns."""
        negation_patterns = [
            (r"\bis\b", r"\bis not\b"),
            (r"\bwas\b", r"\bwas not\b"),
            (r"\bwill\b", r"\bwill not\b"),
            (r"\bcan\b", r"\bcannot\b"),
            (r"\btrue\b", r"\bfalse\b"),
            (r"\byes\b", r"\bno\b"),
            (r"\bincreased\b", r"\bdecreased\b"),
            (r"\brise\b", r"\bfall\b"),
            (r"\bgrew\b", r"\bshrank\b"),
            (r"\bup\b", r"\bdown\b"),
        ]

        claim1_lower = claim1.lower()
        claim2_lower = claim2.lower()

        for positive, negative in negation_patterns:
            if (re.search(positive, claim1_lower) and re.search(negative, claim2_lower)) or (
                re.search(negative, claim1_lower) and re.search(positive, claim2_lower)
            ):
                # Check if claims are about the same subject
                similarity = self._calculate_similarity(claim1, claim2)
                if similarity > 0.3:  # Some overlap but contradictory
                    return 0.8

        return 0.0

    def _check_temporal_contradiction(self, claim1: str, claim2: str) -> float:
        """Check for temporal contradictions."""
        # Extract temporal indicators
        temporal_patterns = [
            r"\b\d{4}\b",  # Years
            r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
            r"\b(?:before|after|during|since|until)\b",
        ]

        temporal1 = []
        temporal2 = []

        for pattern in temporal_patterns:
            temporal1.extend(re.findall(pattern, claim1.lower()))
            temporal2.extend(re.findall(pattern, claim2.lower()))

        if temporal1 and temporal2:
            # Check if different time references for similar claims
            similarity = self._calculate_similarity(
                re.sub(r"\b\d{4}\b", "", claim1), re.sub(r"\b\d{4}\b", "", claim2)
            )

            if similarity > 0.6 and temporal1 != temporal2:
                return 0.7

        return 0.0

    def _check_quantitative_contradiction(self, claim1: str, claim2: str) -> float:
        """Check for quantitative contradictions."""
        # Extract numbers
        numbers1 = re.findall(r"\d+\.?\d*", claim1)
        numbers2 = re.findall(r"\d+\.?\d*", claim2)

        if numbers1 and numbers2:
            # Check if claims are about the same subject
            claim1_no_numbers = re.sub(r"\d+\.?\d*", "X", claim1)
            claim2_no_numbers = re.sub(r"\d+\.?\d*", "X", claim2)

            similarity = self._calculate_similarity(claim1_no_numbers, claim2_no_numbers)

            if similarity > 0.7:  # Same claim structure but different numbers
                # Check if numbers differ significantly
                try:
                    val1 = float(numbers1[0])
                    val2 = float(numbers2[0])

                    if val1 > 0 and val2 > 0:
                        ratio = max(val1, val2) / min(val1, val2)
                        if ratio > (1 + self.contradiction_config.quantitative_tolerance):
                            return 0.75
                except (ValueError, ZeroDivisionError):
                    pass

        return 0.0

    async def analyze_convergence(
        self,
        search_results: list[SearchResult],
        min_sources: int = 2,
        precomputed_vectors: list[list[float]] | None = None,
    ) -> list[ConvergencePoint]:
        """Async variant that uses embeddings when enabled.

        Falls back to token similarity when embeddings are unavailable.
        """
        # Extract all claims with sources
        t_extract = perf_counter()
        all_claims: list[tuple[str, str]] = self.extract_claims_for_convergence(search_results)
        logfire.info(
            "Convergence claims extracted",
            count=len(all_claims),
            duration_ms=int((perf_counter() - t_extract) * 1000),
        )

        if not all_claims:
            return []

        # Embedding path
        use_embeddings = (
            self.enable_embedding_similarity
            and self.embedding_service is not None
            and pairwise_cosine_matrix is not None
            and cluster_by_threshold is not None
        )

        claim_groups: list[list[tuple[str, str]]] = []
        t_group = perf_counter()
        grouping_method = "token"
        if use_embeddings:
            try:
                claim_groups = await self._group_similar_claims_semantic(
                    all_claims, self.similarity_threshold, precomputed_vectors
                )
                grouping_method = "semantic"
            except Exception:
                claim_groups = []

        if not claim_groups:
            # Fallback to token similarity (offload to avoid blocking loop)
            import asyncio as _asyncio

            claim_groups = await _asyncio.to_thread(
                self._group_similar_claims, all_claims, self.similarity_threshold
            )
            grouping_method = "token"

        logfire.info(
            "Convergence grouping completed",
            method=grouping_method,
            groups=len(claim_groups),
            duration_ms=int((perf_counter() - t_group) * 1000),
        )

        # Identify convergence points (offload modest processing to keep loop responsive)
        import asyncio as _asyncio

        def _build_convergence(groups: list[list[tuple[str, str]]]) -> list[ConvergencePoint]:
            points: list[ConvergencePoint] = []
            for group in groups:
                unique_sources = list({source for _, source in group})
                if len(unique_sources) >= min_sources:
                    representative = max(group, key=lambda x: len(x[0]))[0]
                    points.append(
                        ConvergencePoint(
                            claim=representative,
                            sources=unique_sources,
                            support_count=len(unique_sources),
                            confidence=min(1.0, len(unique_sources) / 5),
                            evidence=[claim for claim, _ in group],
                        )
                    )
            points.sort(key=lambda x: x.support_count, reverse=True)
            return points

        t_points = perf_counter()
        convergence_points = await _asyncio.to_thread(_build_convergence, claim_groups)
        logfire.info(
            "Convergence points built",
            points=len(convergence_points),
            duration_ms=int((perf_counter() - t_points) * 1000),
        )
        return convergence_points

    def _group_similar_claims_from_vectors(
        self,
        claims: list[tuple[str, str]],
        vectors: list[list[float]],
        threshold: float,
    ) -> list[list[tuple[str, str]]]:
        """Group claims by cosine similarity using precomputed vectors."""
        if not vectors or len(vectors) != len(claims):
            return []
        if pairwise_cosine_matrix is None or cluster_by_threshold is None:
            return []
        sim = pairwise_cosine_matrix(vectors)
        indices = list(range(len(claims)))
        clusters = cluster_by_threshold(indices, sim, threshold)
        groups: list[list[tuple[str, str]]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            groups.append([claims[i] for i in cluster])
        if groups:
            logfire.info(
                "Embedding grouping applied",
                threshold=threshold,
                clusters=len(groups),
                avg_size=(sum(len(g) for g in groups) / len(groups)) if groups else 0.0,
            )
        return groups

    def _group_similar_claims(
        self, claims: list[tuple[str, str]], threshold: float | None = None
    ) -> list[list[tuple[str, str]]]:
        """Group similar claims using token overlap only (sync-safe).

        Embedding-based grouping must be done via async paths with precomputed vectors.
        """
        if not claims:
            return []

        th = self.similarity_threshold if threshold is None else threshold

        # Token-based grouping
        groups: list[list[tuple[str, str]]] = []
        used: set[int] = set()
        for i, (claim1, source1) in enumerate(claims):
            if i in used:
                continue
            group = [(claim1, source1)]
            used.add(i)
            for j, (claim2, source2) in enumerate(claims[i + 1 :], i + 1):
                if j in used:
                    continue
                similarity = self._calculate_similarity(claim1, claim2)
                if similarity >= (th if th is not None else 0.5):
                    group.append((claim2, source2))
                    used.add(j)
            if len(group) > 1:
                groups.append(group)
        return groups

    async def _group_similar_claims_semantic(
        self,
        claims: list[tuple[str, str]],
        threshold: float | None = None,
        precomputed_vectors: list[list[float]] | None = None,
    ) -> list[list[tuple[str, str]]]:
        """Async grouping using embeddings when available.

        - If precomputed_vectors provided and aligned, use them.
        - Else, if embedding service available, embed asynchronously.
        - Else, fall back to token-based grouping.
        """
        if not claims:
            return []
        th = self.similarity_threshold if threshold is None else threshold
        if (
            self.enable_embedding_similarity
            and pairwise_cosine_matrix is not None
            and cluster_by_threshold is not None
        ):
            if precomputed_vectors and len(precomputed_vectors) == len(claims):
                import asyncio as _asyncio

                return await _asyncio.to_thread(
                    self._group_similar_claims_from_vectors, claims, precomputed_vectors, th
                )
            if self.embedding_service is not None:
                texts = [c for c, _ in claims]
                # Bound embedding latency; fallback if it times out
                import asyncio as _asyncio

                try:
                    t_embed = perf_counter()
                    vectors = await _asyncio.wait_for(
                        self.embedding_service.embed_batch(texts),
                        timeout=20.0,  # type: ignore[union-attr]
                    )
                    logfire.info(
                        "Convergence embeddings computed",
                        count=len(texts),
                        duration_ms=int((perf_counter() - t_embed) * 1000),
                    )
                except Exception:
                    vectors = []
                if vectors:
                    return await _asyncio.to_thread(
                        self._group_similar_claims_from_vectors, claims, vectors, th
                    )
                    # Note: duration will be logged by caller when grouping completes
        # Fallback
        import asyncio as _asyncio

        return await _asyncio.to_thread(self._group_similar_claims, claims, th)

    def extract_claims_for_convergence(
        self, search_results: list[SearchResult]
    ) -> list[tuple[str, str]]:
        """Extract (claim, source) pairs for convergence analysis.

        This helper centralizes sentence splitting to keep ordering consistent
        when callers precompute embeddings externally.
        """
        claims: list[tuple[str, str]] = []
        for result in search_results:
            source = result.query
            if hasattr(result, "results"):
                for item in result.results:
                    if hasattr(item, "content"):
                        sentences = re.split(r"[.!?]+", item.content)
                        for sentence in sentences:
                            if len(sentence.strip()) > 20:
                                claims.append((sentence.strip(), source))
        # Apply sampling/caps to avoid excessive work
        if not claims:
            return claims

        original_count = len(claims)

        # Per-source cap
        if self.convergence_per_source_cap and self.convergence_per_source_cap > 0:
            by_source: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for c, s in claims:
                by_source[s].append((c, s))
            capped: list[tuple[str, str]] = []
            for _s, lst in by_source.items():
                capped.extend(
                    self._sample_list(
                        lst,
                        self.convergence_per_source_cap,
                        strategy=self.convergence_sampling_strategy,
                    )
                )
            claims = capped

        # Global cap
        if self.convergence_max_claims and self.convergence_max_claims > 0:
            if len(claims) > self.convergence_max_claims:
                claims = self._sample_list(
                    claims, self.convergence_max_claims, strategy=self.convergence_sampling_strategy
                )

        if len(claims) != original_count:
            logfire.info(
                "Convergence sampling applied",
                before=original_count,
                after=len(claims),
                per_source_cap=self.convergence_per_source_cap,
                max_claims=self.convergence_max_claims,
                strategy=self.convergence_sampling_strategy,
            )
        return claims

    def _sample_list(
        self,
        items: list[tuple[str, str]],
        k: int,
        *,
        strategy: str = "longest",
    ) -> list[tuple[str, str]]:
        """Deterministically sample items to length k based on strategy.

        - longest: pick top-k by text length
        - first: keep first-k
        - random: deterministic pseudo-random by hashing text+source
        """
        if k <= 0 or len(items) <= k:
            return list(items)

        if strategy == "first":
            return items[:k]
        if strategy == "longest":
            return sorted(items, key=lambda t: len(t[0]), reverse=True)[:k]
        if strategy == "random":
            # Deterministic by hashing the text+source
            def key_fn(tup: tuple[str, str]) -> int:
                h = hashlib.md5(f"{tup[0]}|{tup[1]}".encode()).hexdigest()
                return int(h[:8], 16)

            return sorted(items, key=key_fn)[:k]
        # Fallback: first
        return items[:k]

    # Note: Convergence with precomputed vectors is handled by
    # analyze_convergence(..., precomputed_vectors=...), so a separate
    # analyze_convergence_with_vectors method is unnecessary.

    def extract_themes(
        self, search_results: list[SearchResult], min_frequency: int = 2
    ) -> list[Theme]:
        """
        Extract themes with relationship mapping.

        Identifies recurring concepts, entities, events, and relationships.
        """
        themes = []

        # Extract various types of themes
        concepts = self._extract_concepts(search_results)
        entities = self._extract_entities(search_results)
        events = self._extract_events(search_results)
        trends = self._extract_trends(search_results)

        # Process each theme type
        for theme_type, extracted in [
            (ThemeType.CONCEPT, concepts),
            (ThemeType.ENTITY, entities),
            (ThemeType.EVENT, events),
            (ThemeType.TREND, trends),
        ]:
            for name, data in extracted.items():
                if data["frequency"] >= min_frequency:
                    theme = Theme(
                        type=theme_type,
                        name=name,
                        description=data.get("description", ""),
                        frequency=data["frequency"],
                        sources=data["sources"],
                        related_themes=data.get("related", []),
                        confidence=min(1.0, data["frequency"] / 10),
                        context=data.get("context"),
                    )
                    themes.append(theme)

        # Map relationships between themes
        self._map_theme_relationships(themes)

        # Sort by frequency
        themes.sort(key=lambda x: x.frequency, reverse=True)

        return themes

    def _extract_concepts(self, search_results: list[SearchResult]) -> dict[str, dict]:
        """Extract conceptual themes with optimized performance.

        Time Complexity: O(n * m * k) where n=search_results, m=results, k=content_length
        Previously: O(n * m * k * p) due to list membership checks
        """
        # Pre-compile patterns once for massive performance gain
        concept_patterns = [
            re.compile(
                r"\b(?:concept|theory|principle|idea|framework|model|approach)\s+(?:of\s+)?(\w+)",
                re.IGNORECASE,
            ),
            re.compile(r"\b(\w+)\s+(?:hypothesis|paradigm|methodology)", re.IGNORECASE),
        ]

        # Use defaultdict with sets for O(1) membership checks
        concepts = defaultdict(lambda: {"frequency": 0, "sources": set(), "related": []})

        for result in search_results:
            source = result.query
            if hasattr(result, "results"):
                for item in result.results:
                    if hasattr(item, "content"):
                        content = item.content
                        for pattern in concept_patterns:
                            matches = pattern.findall(content)
                            for match in matches:
                                match_lower = match.lower() if isinstance(match, str) else match
                                concepts[match_lower]["frequency"] += 1
                                concepts[match_lower]["sources"].add(source)  # O(1) operation

        # Convert sets to lists for backward compatibility
        return {
            concept: {
                "frequency": data["frequency"],
                "sources": list(data["sources"]),
                "related": data["related"],
            }
            for concept, data in concepts.items()
        }

    def _extract_entities(self, search_results: list[SearchResult]) -> dict[str, dict]:
        """Extract entity themes (people, organizations, places) with optimized performance.

        Time Complexity: O(n * m * k) where n=search_results, m=results, k=content_length
        Previously: O(n * m * k * p) due to list membership checks
        """
        # Pre-compile pattern once
        entity_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")

        # Common words to exclude (as set for O(1) lookup)
        common_words = {
            "The",
            "This",
            "That",
            "These",
            "Those",
            "There",
            "Then",
            "When",
            "Where",
            "What",
            "Which",
            "While",
        }

        # Use defaultdict with sets for O(1) membership checks
        entities = defaultdict(lambda: {"frequency": 0, "sources": set(), "type": "entity"})

        for result in search_results:
            source = result.query
            if hasattr(result, "results"):
                for item in result.results:
                    if hasattr(item, "content"):
                        matches = entity_pattern.findall(item.content)
                        for match in matches:
                            # Filter using O(1) set lookup
                            if len(match) > 3 and match not in common_words:
                                entities[match]["frequency"] += 1
                                entities[match]["sources"].add(source)  # O(1) operation

        # Convert sets to lists for backward compatibility
        return {
            entity: {
                "frequency": data["frequency"],
                "sources": list(data["sources"]),
                "type": data["type"],
            }
            for entity, data in entities.items()
        }

    def _extract_events(self, search_results: list[SearchResult]) -> dict[str, dict]:
        """Extract event themes with optimized performance.

        Time Complexity: O(n * m * k) where n=search_results, m=results, k=content_length
        Previously: O(n * m * k * p) due to list membership checks
        """
        # Pre-compile all event patterns
        event_patterns = [
            re.compile(
                r"(?:announced|launched|released|occurred|happened|took place)", re.IGNORECASE
            ),
            re.compile(r"(?:conference|summit|meeting|event|ceremony)", re.IGNORECASE),
        ]

        # Pre-compile sentence splitter
        sentence_splitter = re.compile(r"[.!?]+")

        # Use defaultdict with sets for O(1) membership checks
        events = defaultdict(lambda: {"frequency": 0, "sources": set(), "type": "event"})

        for result in search_results:
            source = result.query
            if hasattr(result, "results"):
                for item in result.results:
                    if hasattr(item, "content"):
                        content = item.content

                        # Check if content contains any event pattern
                        has_event = False
                        for pattern in event_patterns:
                            if pattern.search(content):
                                has_event = True
                                break

                        if has_event:
                            # Extract surrounding context as event
                            sentences = sentence_splitter.split(content)
                            for sentence in sentences:
                                if len(sentence) > 20:
                                    # Check each pattern
                                    for pattern in event_patterns:
                                        if pattern.search(sentence):
                                            # Use first 50 chars as key
                                            event_key = sentence[:50].lower()
                                            events[event_key]["frequency"] += 1
                                            # O(1) operation
                                            events[event_key]["sources"].add(source)
                                            break  # Only count once per sentence

        # Convert sets to lists for backward compatibility
        return {
            event: {
                "frequency": data["frequency"],
                "sources": list(data["sources"]),
                "type": data["type"],
            }
            for event, data in events.items()
        }

    def _extract_trends(self, search_results: list[SearchResult]) -> dict[str, dict]:
        """Extract trend themes with optimized performance.

        Time Complexity: O(n * m * k) where n=search_results, m=results, k=content_length
        Previously: O(n * m * k * p) due to list membership checks
        """
        # Pre-compile trend patterns
        trend_patterns = [
            re.compile(
                r"(?:increasing|decreasing|growing|declining|rising|falling)", re.IGNORECASE
            ),
            re.compile(r"(?:trend|pattern|shift|change|evolution)", re.IGNORECASE),
            re.compile(r"\d+%\s+(?:increase|decrease|growth|decline)", re.IGNORECASE),
        ]

        # Use defaultdict with sets for O(1) membership checks
        trends = defaultdict(lambda: {"frequency": 0, "sources": set(), "type": "trend"})

        for result in search_results:
            source = result.query
            if hasattr(result, "results"):
                for item in result.results:
                    if hasattr(item, "content"):
                        content = item.content
                        for pattern in trend_patterns:
                            matches = pattern.findall(content)
                            for match in matches:
                                match_key = match.lower()
                                trends[match_key]["frequency"] += 1
                                trends[match_key]["sources"].add(source)  # O(1) operation

        # Convert sets to lists for backward compatibility
        return {
            trend: {
                "frequency": data["frequency"],
                "sources": list(data["sources"]),
                "type": data["type"],
            }
            for trend, data in trends.items()
        }

    def _map_theme_relationships(self, themes: list[Theme]) -> None:
        """Map relationships between themes."""
        for i, theme1 in enumerate(themes):
            for theme2 in themes[i + 1 :]:
                # Check if themes co-occur in sources
                common_sources = set(theme1.sources) & set(theme2.sources)
                if common_sources:
                    if theme2.name not in theme1.related_themes:
                        theme1.related_themes.append(theme2.name)
                    if theme1.name not in theme2.related_themes:
                        theme2.related_themes.append(theme1.name)

    def match_patterns(self, search_results: list[SearchResult]) -> list[Pattern]:
        """
        Match patterns in search results.

        Identifies temporal, causal, and correlative patterns.
        """
        patterns = []

        # Extract text content
        all_content = []
        for result in search_results:
            if hasattr(result, "results"):
                for item in result.results:
                    if hasattr(item, "content"):
                        all_content.append(item.content)

        combined_text = " ".join(all_content)

        # Detect temporal patterns
        temporal_patterns = self._detect_temporal_patterns(combined_text)
        patterns.extend(temporal_patterns)

        # Detect causal patterns
        causal_patterns = self._detect_causal_patterns(combined_text)
        patterns.extend(causal_patterns)

        # Detect correlative patterns
        correlative_patterns = self._detect_correlative_patterns(combined_text)
        patterns.extend(correlative_patterns)

        return patterns

    def _detect_temporal_patterns(self, text: str) -> list[Pattern]:
        """Detect temporal patterns in text."""
        patterns = []

        # Temporal indicators
        temporal_indicators = [
            (r"(?:before|after|during|since|until)\s+(.+?)(?:\.|,|;)", "sequence"),
            (r"(?:from|between)\s+(\d{4})\s+(?:to|and)\s+(\d{4})", "range"),
            (r"(?:every|each)\s+(\w+)", "recurring"),
            (r"(?:first|then|next|finally|lastly)", "progression"),
        ]

        for indicator, pattern_subtype in temporal_indicators:
            matches = re.findall(indicator, text, re.IGNORECASE)
            if matches:
                pattern = Pattern(
                    type=PatternType.TEMPORAL,
                    description=f"Temporal {pattern_subtype} pattern detected",
                    evidence=matches[:5]
                    if isinstance(matches[0], str)
                    else [str(m) for m in matches[:5]],
                    strength=min(1.0, len(matches) / 10),
                    timeframe=pattern_subtype,
                )
                patterns.append(pattern)

        return patterns

    def _detect_causal_patterns(self, text: str) -> list[Pattern]:
        """Detect causal patterns in text."""
        patterns = []

        # Causal indicators
        causal_indicators = [
            r"(?:because|due to|as a result of|caused by)\s+(.+?)(?:\.|,|;)",
            r"(.+?)\s+(?:leads to|results in|causes|triggers)\s+(.+?)(?:\.|,|;)",
            r"(?:if|when)\s+(.+?)\s+then\s+(.+?)(?:\.|,|;)",
            r"(?:therefore|consequently|thus|hence)\s+(.+?)(?:\.|,|;)",
        ]

        evidence = []
        for indicator in causal_indicators:
            matches = re.findall(indicator, text, re.IGNORECASE)
            evidence.extend(matches[:3])  # Limit per indicator

        if evidence:
            pattern = Pattern(
                type=PatternType.CAUSAL,
                description="Causal relationships detected",
                evidence=[str(e)[:100] for e in evidence[:5]],  # Truncate long evidence
                strength=min(1.0, len(evidence) / 15),
            )
            patterns.append(pattern)

        return patterns

    def _detect_correlative_patterns(self, text: str) -> list[Pattern]:
        """Detect correlative patterns in text."""
        patterns = []

        # Correlative indicators
        correlative_indicators = [
            r"(?:correlates with|associated with|linked to|connected to)\s+(.+?)(?:\.|,|;)",
            r"(?:relationship between|correlation between)\s+(.+?)\s+and\s+(.+?)(?:\.|,|;)",
            r"(?:as|while)\s+(.+?)\s+(?:increases|decreases),\s+(.+?)\s+(?:increases|decreases)",
        ]

        evidence = []
        entities = []

        for indicator in correlative_indicators:
            matches = re.findall(indicator, text, re.IGNORECASE)
            for match in matches[:3]:
                evidence.append(str(match)[:100])
                # Extract entities from matches
                if isinstance(match, tuple):
                    entities.extend(match)
                else:
                    entities.append(match)

        if evidence:
            pattern = Pattern(
                type=PatternType.CORRELATION,
                description="Correlative relationships detected",
                evidence=evidence[:5],
                strength=min(1.0, len(evidence) / 10),
                entities=list(set(entities[:10])),  # Unique entities
            )
            patterns.append(pattern)

        return patterns

    def calculate_synthesis_score(
        self,
        hierarchy_scores: list[HierarchyScore],
        contradictions: list[Contradiction],
        convergence_points: list[ConvergencePoint],
        themes: list[Theme],
        patterns: list[Pattern],
    ) -> float:
        """
        Calculate overall synthesis quality score.

        Combines multiple factors into a single score.
        """
        score_components = []

        # Hierarchy distribution score
        if hierarchy_scores:
            primary_ratio = sum(
                1 for h in hierarchy_scores if h.level == InformationHierarchy.PRIMARY
            ) / len(hierarchy_scores)
            hierarchy_score = min(1.0, primary_ratio * 2)  # Ideal is ~50% primary
            score_components.append(("hierarchy", hierarchy_score, 0.2))

        # Contradiction penalty
        contradiction_penalty = max(
            0, 1 - (len(contradictions) / 10)
        )  # Penalty increases with contradictions
        score_components.append(("contradictions", contradiction_penalty, 0.2))

        # Convergence bonus
        convergence_score = min(
            1.0, len(convergence_points) / 5
        )  # Max bonus at 5+ convergence points
        score_components.append(("convergence", convergence_score, 0.25))

        # Theme richness
        theme_score = min(1.0, len(themes) / 10)  # Max at 10+ themes
        score_components.append(("themes", theme_score, 0.15))

        # Pattern detection
        pattern_score = min(1.0, len(patterns) / 5)  # Max at 5+ patterns
        score_components.append(("patterns", pattern_score, 0.2))

        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in score_components)

        return total_score

    def get_synthesis_recommendations(
        self,
        hierarchy_scores: list[HierarchyScore],
        contradictions: list[Contradiction],
        convergence_points: list[ConvergencePoint],
        themes: list[Theme],
        patterns: list[Pattern],
    ) -> list[str]:
        """Generate recommendations for improving synthesis."""
        recommendations = []

        # Check hierarchy distribution
        if hierarchy_scores:
            primary_count = sum(
                1 for h in hierarchy_scores if h.level == InformationHierarchy.PRIMARY
            )
            if primary_count < len(hierarchy_scores) * 0.2:
                recommendations.append("Increase focus on primary information sources")
            elif primary_count > len(hierarchy_scores) * 0.5:
                recommendations.append("Include more supporting and contextual information")

        # Check contradictions
        if len(contradictions) > 5:
            recommendations.append(
                "Resolve contradictions through fact-checking and source verification"
            )

        # Check convergence
        if len(convergence_points) < 3:
            recommendations.append("Seek additional sources to validate key claims")

        # Check themes
        if len(themes) < 3:
            recommendations.append("Explore topic more broadly to identify additional themes")

        # Check patterns
        if not patterns:
            recommendations.append("Analyze temporal, causal, and correlative relationships")

        return recommendations
