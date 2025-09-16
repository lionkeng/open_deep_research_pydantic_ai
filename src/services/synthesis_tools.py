"""Synthesis Tools Service for advanced information processing and analysis."""

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# Local model definitions until integrated with main models


class PatternType(str, Enum):
    """Types of patterns that can be detected."""

    TEMPORAL = "temporal"
    CAUSAL = "causal"
    CORRELATIVE = "correlative"
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
    ):
        """
        Initialize SynthesisTools.

        Args:
            hierarchy_factors: Configuration for hierarchy scoring
            contradiction_config: Configuration for contradiction detection
        """
        self.hierarchy_factors = hierarchy_factors or HierarchyFactors()
        self.contradiction_config = contradiction_config or ContradictionConfig()

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
        query_terms = set(query.lower().split())
        info_terms = set(information.lower().split())

        if not query_terms:
            return 0.0

        # Calculate Jaccard similarity
        intersection = query_terms & info_terms
        union = query_terms | info_terms

        if not union:
            return 0.0

        jaccard = len(intersection) / len(union)

        # Boost for exact phrase matches
        if query.lower() in information.lower():
            jaccard = min(1.0, jaccard * 1.5)

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
            "percentages": r"\d+\.?\d*%",  # Percentages
            "quotes": r'"[^"]*"',  # Quoted text
            "proper_nouns": r"\b[A-Z][a-z]+\b",  # Capitalized words
        }

        score = 0.0
        max_score = len(specificity_indicators)

        for _indicator, pattern in specificity_indicators.items():
            matches = re.findall(pattern, information)
            if matches:
                score += min(1.0, len(matches) / 5)  # Cap contribution per indicator

        return min(1.0, score / max_score)

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
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        if not tokens1 or not tokens2:
            similarity = 0.0
        else:
            intersection = tokens1 & tokens2
            union = tokens1 | tokens2
            similarity = len(intersection) / len(union)

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

    def analyze_convergence(
        self, search_results: list[SearchResult], min_sources: int = 2
    ) -> list[ConvergencePoint]:
        """
        Analyze convergence of information across sources.

        Identifies claims supported by multiple independent sources.
        """
        convergence_points = []

        # Extract all claims with sources
        all_claims = []
        for result in search_results:
            source = result.query
            if hasattr(result, "results"):
                for item in result.results:
                    if hasattr(item, "content"):
                        sentences = re.split(r"[.!?]+", item.content)
                        for sentence in sentences:
                            if len(sentence.strip()) > 20:
                                all_claims.append((sentence.strip(), source))

        # Group similar claims
        claim_groups = self._group_similar_claims(all_claims)

        # Identify convergence points
        for group in claim_groups:
            unique_sources = list({source for _, source in group})

            if len(unique_sources) >= min_sources:
                # Select representative claim (longest)
                representative = max(group, key=lambda x: len(x[0]))[0]

                convergence = ConvergencePoint(
                    claim=representative,
                    sources=unique_sources,
                    support_count=len(unique_sources),
                    confidence=min(1.0, len(unique_sources) / 5),  # Max confidence at 5 sources
                    evidence=[claim for claim, _ in group],
                )
                convergence_points.append(convergence)

        # Sort by support count
        convergence_points.sort(key=lambda x: x.support_count, reverse=True)

        return convergence_points

    def _group_similar_claims(
        self, claims: list[tuple[str, str]], threshold: float = 0.7
    ) -> list[list[tuple[str, str]]]:
        """Group similar claims together."""
        groups = []
        used = set()

        for i, (claim1, source1) in enumerate(claims):
            if i in used:
                continue

            group = [(claim1, source1)]
            used.add(i)

            for j, (claim2, source2) in enumerate(claims[i + 1 :], i + 1):
                if j in used:
                    continue

                similarity = self._calculate_similarity(claim1, claim2)
                if similarity >= threshold:
                    group.append((claim2, source2))
                    used.add(j)

            if len(group) > 1:
                groups.append(group)

        return groups

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
        """Extract conceptual themes."""
        concepts = defaultdict(lambda: {"frequency": 0, "sources": [], "related": []})

        # Common concept indicators
        concept_patterns = [
            r"\b(?:concept|theory|principle|idea|framework|model|approach)\s+(?:of\s+)?(\w+)",
            r"\b(\w+)\s+(?:hypothesis|paradigm|methodology)",
        ]

        for result in search_results:
            source = result.query
            if hasattr(result, "results"):
                for item in result.results:
                    if hasattr(item, "content"):
                        content = item.content.lower()
                        for pattern in concept_patterns:
                            matches = re.findall(pattern, content)
                            for match in matches:
                                concepts[match]["frequency"] += 1
                                if source not in concepts[match]["sources"]:
                                    concepts[match]["sources"].append(source)

        return dict(concepts)

    def _extract_entities(self, search_results: list[SearchResult]) -> dict[str, dict]:
        """Extract entity themes (people, organizations, places)."""
        entities = defaultdict(lambda: {"frequency": 0, "sources": [], "type": "entity"})

        # Simple entity extraction using capitalization
        entity_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"

        for result in search_results:
            source = result.query
            if hasattr(result, "results"):
                for item in result.results:
                    if hasattr(item, "content"):
                        matches = re.findall(entity_pattern, item.content)
                        for match in matches:
                            if len(match) > 3:  # Filter out short matches
                                entities[match]["frequency"] += 1
                                if source not in entities[match]["sources"]:
                                    entities[match]["sources"].append(source)

        return dict(entities)

    def _extract_events(self, search_results: list[SearchResult]) -> dict[str, dict]:
        """Extract event themes."""
        events = defaultdict(lambda: {"frequency": 0, "sources": [], "type": "event"})

        # Event indicators
        event_patterns = [
            r"(?:announced|launched|released|occurred|happened|took place)",
            r"(?:conference|summit|meeting|event|ceremony)",
        ]

        for result in search_results:
            source = result.query
            if hasattr(result, "results"):
                for item in result.results:
                    if hasattr(item, "content"):
                        content = item.content.lower()
                        for pattern in event_patterns:
                            if re.search(pattern, content):
                                # Extract surrounding context as event
                                sentences = re.split(r"[.!?]+", content)
                                for sentence in sentences:
                                    if re.search(pattern, sentence) and len(sentence) > 20:
                                        event_key = sentence[:50]  # Use first 50 chars as key
                                        events[event_key]["frequency"] += 1
                                        if source not in events[event_key]["sources"]:
                                            events[event_key]["sources"].append(source)

        return dict(events)

    def _extract_trends(self, search_results: list[SearchResult]) -> dict[str, dict]:
        """Extract trend themes."""
        trends = defaultdict(lambda: {"frequency": 0, "sources": [], "type": "trend"})

        # Trend indicators
        trend_patterns = [
            r"(?:increasing|decreasing|growing|declining|rising|falling)",
            r"(?:trend|pattern|shift|change|evolution)",
            r"\d+%\s+(?:increase|decrease|growth|decline)",
        ]

        for result in search_results:
            source = result.query
            if hasattr(result, "results"):
                for item in result.results:
                    if hasattr(item, "content"):
                        content = item.content.lower()
                        for pattern in trend_patterns:
                            matches = re.findall(pattern, content)
                            for match in matches:
                                trends[match]["frequency"] += 1
                                if source not in trends[match]["sources"]:
                                    trends[match]["sources"].append(source)

        return dict(trends)

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
                type=PatternType.CORRELATIVE,
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
