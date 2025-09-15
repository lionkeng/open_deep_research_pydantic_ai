"""Contradiction detection service for research findings."""

import re
from typing import Any

from src.models.research_executor import Contradiction, HierarchicalFinding


class ContradictionDetector:
    """Detects contradictions between research findings.

    Simplified MVP implementation focusing on two types:
    1. Direct contradictions - explicitly opposing statements
    2. Partial contradictions - conflicting implications
    """

    def __init__(self):
        """Initialize the contradiction detector."""
        # Keywords indicating opposing concepts
        self.opposition_pairs = [
            ("increase", "decrease"),
            ("improve", "worsen"),
            ("better", "worse"),
            ("success", "failure"),
            ("effective", "ineffective"),
            ("positive", "negative"),
            ("growth", "decline"),
            ("rise", "fall"),
            ("gain", "loss"),
            ("benefit", "harm"),
            ("advantage", "disadvantage"),
            ("strengthen", "weaken"),
            ("expand", "contract"),
            ("accelerate", "decelerate"),
            ("surplus", "deficit"),
        ]

        # Build reverse mapping for efficient lookup
        self.opposites: dict[str, str] = {}
        for word1, word2 in self.opposition_pairs:
            self.opposites[word1] = word2
            self.opposites[word2] = word1

    def detect_contradictions(self, findings: list[HierarchicalFinding]) -> list[Contradiction]:
        """Detect contradictions among findings.

        Args:
            findings: List of research findings to analyze

        Returns:
            List of detected contradictions
        """
        contradictions = []

        # Compare each pair of findings
        for i in range(len(findings)):
            for j in range(i + 1, len(findings)):
                contradiction = self._check_contradiction_pair(findings[i], findings[j], i, j)
                if contradiction:
                    contradictions.append(contradiction)

        return contradictions

    def _check_contradiction_pair(
        self,
        finding1: HierarchicalFinding,
        finding2: HierarchicalFinding,
        idx1: int,
        idx2: int,
    ) -> Contradiction | None:
        """Check if two findings contradict each other.

        Args:
            finding1: First finding
            finding2: Second finding
            idx1: Index of first finding
            idx2: Index of second finding

        Returns:
            Contradiction if detected, None otherwise
        """
        # Check for direct contradiction
        direct = self._is_direct_contradiction(finding1, finding2)
        if direct:
            return Contradiction(
                finding_1_id=str(idx1),
                finding_2_id=str(idx2),
                contradiction_type="direct",
                explanation=self._generate_direct_explanation(finding1, finding2),
                resolution_hint=self._generate_resolution_hint(finding1, finding2),
            )

        # Check for partial contradiction
        partial = self._is_partial_contradiction(finding1, finding2)
        if partial:
            return Contradiction(
                finding_1_id=str(idx1),
                finding_2_id=str(idx2),
                contradiction_type="partial",
                explanation=self._generate_partial_explanation(finding1, finding2),
                resolution_hint=self._generate_resolution_hint(finding1, finding2),
            )

        return None

    def _is_direct_contradiction(
        self, finding1: HierarchicalFinding, finding2: HierarchicalFinding
    ) -> bool:
        """Check for direct contradiction between findings.

        Direct contradictions occur when findings contain explicitly
        opposing statements about the same subject.

        Args:
            finding1: First finding
            finding2: Second finding

        Returns:
            True if direct contradiction detected
        """
        text1 = finding1.finding.lower()
        text2 = finding2.finding.lower()

        # Check if texts discuss similar topics (share significant words)
        words1 = set(self._extract_meaningful_words(text1))
        words2 = set(self._extract_meaningful_words(text2))

        # Need at least 2 words in common to be about the same topic
        common_words = words1.intersection(words2)
        if len(common_words) < 2:
            return False

        # Check for opposing keywords
        for word in words1:
            if word in self.opposites and self.opposites[word] in words2:
                return True

        # Check for negation patterns
        if self._has_negation_conflict(text1, text2):
            return True

        return False

    def _is_partial_contradiction(
        self, finding1: HierarchicalFinding, finding2: HierarchicalFinding
    ) -> bool:
        """Check for partial contradiction between findings.

        Partial contradictions occur when findings have conflicting
        implications or incompatible claims, even if not directly opposing.

        Args:
            finding1: First finding
            finding2: Second finding

        Returns:
            True if partial contradiction detected
        """
        # Skip if already a direct contradiction
        if self._is_direct_contradiction(finding1, finding2):
            return False

        text1 = finding1.finding.lower()
        text2 = finding2.finding.lower()

        # Check for conflicting quantitative claims
        if self._has_conflicting_quantities(text1, text2):
            return True

        # Check for mutually exclusive claims
        if self._has_mutually_exclusive_claims(text1, text2):
            return True

        # Check for conflicting temporal claims
        if self._has_temporal_conflict(text1, text2):
            return True

        return False

    def _extract_meaningful_words(self, text: str) -> list[str]:
        """Extract meaningful words from text.

        Args:
            text: Input text

        Returns:
            List of meaningful words (excluding stop words)
        """
        # Simple stop words list
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "is",
            "was",
            "are",
            "were",
            "been",
            "be",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "what",
            "which",
            "who",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "every",
            "both",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "if",
            "then",
            "else",
        }

        # Extract words (simple tokenization)
        words = []
        for word in text.split():
            # Remove punctuation
            word = word.strip(".,!?;:\"'")
            if len(word) > 2 and word not in stop_words:
                words.append(word)

        return words

    def _has_negation_conflict(self, text1: str, text2: str) -> bool:
        """Check if texts have conflicting negation patterns.

        Args:
            text1: First text
            text2: Second text

        Returns:
            True if negation conflict detected
        """
        negation_words = {"not", "no", "never", "neither", "none", "nothing"}

        # Check if one text negates what the other affirms
        for neg_word in negation_words:
            if neg_word in text1 and neg_word not in text2:
                # Extract the phrase after negation in text1
                words1 = text1.split()
                if neg_word in words1:
                    neg_idx = words1.index(neg_word)
                    if neg_idx + 1 < len(words1):
                        negated_concept = words1[neg_idx + 1]
                        # Check if text2 affirms this concept
                        if negated_concept in text2:
                            return True

        return False

    def _has_conflicting_quantities(self, text1: str, text2: str) -> bool:
        """Check for conflicting quantitative claims.

        Args:
            text1: First text
            text2: Second text

        Returns:
            True if conflicting quantities detected
        """
        # Extract numbers with their context
        pattern = r"(\d+(?:\.\d+)?)\s*(%|percent|times|x|fold)"

        matches1 = re.findall(pattern, text1)
        matches2 = re.findall(pattern, text2)

        if not matches1 or not matches2:
            return False

        # Check if discussing same metric but with different values
        for num1, unit1 in matches1:
            for num2, unit2 in matches2:
                if unit1 == unit2 or (unit1 in ["percent", "%"] and unit2 in ["percent", "%"]):
                    val1 = float(num1)
                    val2 = float(num2)
                    # Significant difference (>20% relative difference)
                    # Avoid division by zero
                    max_val = max(val1, val2)
                    if max_val > 0 and abs(val1 - val2) / max_val > 0.2:
                        return True

        return False

    def _has_mutually_exclusive_claims(self, text1: str, text2: str) -> bool:
        """Check for mutually exclusive claims.

        Args:
            text1: First text
            text2: Second text

        Returns:
            True if mutually exclusive claims detected
        """
        exclusive_patterns = [
            ("only", "also"),
            ("exclusively", "additionally"),
            ("sole", "multiple"),
            ("single", "several"),
            ("unique", "common"),
            ("always", "sometimes"),
            ("never", "occasionally"),
            ("all", "some"),
            ("none", "few"),
        ]

        for word1, word2 in exclusive_patterns:
            if word1 in text1 and word2 in text2:
                return True
            if word2 in text1 and word1 in text2:
                return True

        return False

    def _has_temporal_conflict(self, text1: str, text2: str) -> bool:
        """Check for conflicting temporal claims.

        Args:
            text1: First text
            text2: Second text

        Returns:
            True if temporal conflict detected
        """
        import re

        # Extract year patterns
        year_pattern = r"\b(19|20)\d{2}\b"
        years1 = re.findall(year_pattern, text1)
        years2 = re.findall(year_pattern, text2)

        if years1 and years2:
            # Check if making claims about different time periods
            if set(years1) != set(years2):
                # Check if both texts make absolute claims
                absolute_words = {"first", "initially", "began", "started", "ended", "stopped"}
                if any(word in text1 for word in absolute_words) and any(
                    word in text2 for word in absolute_words
                ):
                    return True

        return False

    def _generate_direct_explanation(
        self, finding1: HierarchicalFinding, finding2: HierarchicalFinding
    ) -> str:
        """Generate explanation for direct contradiction.

        Args:
            finding1: First finding
            finding2: Second finding

        Returns:
            Explanation text
        """
        # Find the opposing terms
        text1 = finding1.finding.lower()
        text2 = finding2.finding.lower()
        words1 = set(self._extract_meaningful_words(text1))
        words2 = set(self._extract_meaningful_words(text2))

        opposing_terms = []
        for word in words1:
            if word in self.opposites and self.opposites[word] in words2:
                opposing_terms.append(f"'{word}' vs '{self.opposites[word]}'")

        if opposing_terms:
            return f"Direct contradiction: {', '.join(opposing_terms[:2])}"

        return "Direct contradiction in claims about the same subject"

    def _generate_partial_explanation(
        self, finding1: HierarchicalFinding, finding2: HierarchicalFinding
    ) -> str:
        """Generate explanation for partial contradiction.

        Args:
            finding1: First finding
            finding2: Second finding

        Returns:
            Explanation text
        """
        text1 = finding1.finding.lower()
        text2 = finding2.finding.lower()

        if self._has_conflicting_quantities(text1, text2):
            return "Conflicting quantitative claims about the same metric"

        if self._has_mutually_exclusive_claims(text1, text2):
            return "Mutually exclusive claims that cannot both be true"

        if self._has_temporal_conflict(text1, text2):
            return "Conflicting temporal or chronological claims"

        return "Partial contradiction with conflicting implications"

    def _generate_resolution_hint(
        self, finding1: HierarchicalFinding, finding2: HierarchicalFinding
    ) -> str:
        """Generate hint for resolving contradiction.

        Args:
            finding1: First finding
            finding2: Second finding

        Returns:
            Resolution hint
        """
        # Compare confidence scores
        conf_diff = abs(finding1.confidence_score - finding2.confidence_score)

        if conf_diff > 0.3:
            higher_conf = (
                finding1 if finding1.confidence_score > finding2.confidence_score else finding2
            )
            return (
                f"Consider the finding with higher confidence ({higher_conf.confidence_score:.2f})"
            )

        # Compare importance levels
        if finding1.importance != finding2.importance:
            from src.models.research_executor import ImportanceLevel

            importance_order = [
                ImportanceLevel.CRITICAL,
                ImportanceLevel.HIGH,
                ImportanceLevel.MEDIUM,
                ImportanceLevel.LOW,
            ]
            idx1 = importance_order.index(finding1.importance)
            idx2 = importance_order.index(finding2.importance)

            if idx1 < idx2:
                return f"Prioritize the {finding1.importance} importance finding"
            else:
                return f"Prioritize the {finding2.importance} importance finding"

        # Check source dates if available
        if finding1.source and finding2.source:
            if finding1.source.date and finding2.source.date:
                if finding1.source.date > finding2.source.date:
                    return "Consider the more recent finding"
                elif finding2.source.date > finding1.source.date:
                    return "Consider the more recent finding"

        return "Further investigation needed - check original sources and context"

    def analyze_contradiction_patterns(self, contradictions: list[Contradiction]) -> dict[str, Any]:
        """Analyze patterns in detected contradictions.

        Args:
            contradictions: List of contradictions

        Returns:
            Analysis of contradiction patterns
        """
        if not contradictions:
            return {
                "total_contradictions": 0,
                "direct_contradictions": 0,
                "partial_contradictions": 0,
                "resolution_complexity": "low",
            }

        direct_count = sum(1 for c in contradictions if c.contradiction_type == "direct")
        partial_count = len(contradictions) - direct_count

        # Determine resolution complexity
        if direct_count > 3:
            complexity = "high"
        elif direct_count > 1 or partial_count > 3:
            complexity = "medium"
        else:
            complexity = "low"

        return {
            "total_contradictions": len(contradictions),
            "direct_contradictions": direct_count,
            "partial_contradictions": partial_count,
            "resolution_complexity": complexity,
            "requires_expert_review": direct_count > 2,
        }
