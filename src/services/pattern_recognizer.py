"""Pattern recognition service for identifying patterns in research findings."""

import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from models.research_executor import (
    HierarchicalFinding,
    PatternAnalysis,
    PatternType,
    ThemeCluster,
)


class PatternRecognizer:
    """Recognizes patterns in research findings using ML and heuristic methods."""

    def __init__(
        self,
        min_findings_for_pattern: int = 3,
        similarity_threshold: float = 0.6,
        temporal_window_days: int = 365,
    ):
        """Initialize the pattern recognizer.

        Args:
            min_findings_for_pattern: Minimum findings needed to form a pattern
            similarity_threshold: Threshold for considering findings similar
            temporal_window_days: Time window for temporal pattern detection
        """
        self.min_findings_for_pattern = min_findings_for_pattern
        self.similarity_threshold = similarity_threshold
        self.temporal_window_days = temporal_window_days

        # Initialize TF-IDF vectorizer for text similarity
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words="english",
            ngram_range=(1, 2),
        )

    def detect_patterns(
        self,
        findings: list[HierarchicalFinding],
        clusters: list[ThemeCluster] | None = None,
    ) -> list[PatternAnalysis]:
        """Detect various patterns in research findings.

        Args:
            findings: List of research findings
            clusters: Optional theme clusters for enhanced pattern detection

        Returns:
            List of detected patterns
        """
        if len(findings) < self.min_findings_for_pattern:
            return []

        patterns = []

        # Detect different types of patterns
        patterns.extend(self._detect_convergence_patterns(findings))
        patterns.extend(self._detect_divergence_patterns(findings))
        patterns.extend(self._detect_temporal_patterns(findings))
        patterns.extend(self._detect_causal_patterns(findings))
        patterns.extend(self._detect_anomaly_patterns(findings))

        # If clusters are provided, detect emergence patterns
        if clusters:
            patterns.extend(self._detect_emergence_patterns(clusters))

        # Link related patterns
        self._link_related_patterns(patterns)

        return patterns

    def _detect_convergence_patterns(
        self, findings: list[HierarchicalFinding]
    ) -> list[PatternAnalysis]:
        """Detect convergence patterns where multiple findings point to same conclusion.

        Args:
            findings: List of findings to analyze

        Returns:
            List of convergence patterns
        """
        patterns = []

        # Prepare texts for vectorization
        texts = [self._finding_to_text(f) for f in findings]

        try:
            # Vectorize findings
            X = self.vectorizer.fit_transform(texts)
            vectors = X.toarray()
        except (ValueError, AttributeError):
            return patterns

        # Find groups of similar findings
        convergence_groups = defaultdict(list)

        for i in range(len(findings)):
            for j in range(i + 1, len(findings)):
                similarity = self._cosine_similarity(vectors[i], vectors[j])

                if similarity > self.similarity_threshold:
                    # Group similar findings together
                    group_key = min(i, j)
                    convergence_groups[group_key].extend([i, j])

        # Create patterns for significant convergence groups
        for _base_idx, finding_indices in convergence_groups.items():
            unique_indices = list(set(finding_indices))

            if len(unique_indices) >= self.min_findings_for_pattern:
                # Calculate pattern strength based on similarity and importance
                involved_findings = [findings[idx] for idx in unique_indices]
                avg_importance = np.mean([f.importance_score for f in involved_findings])
                avg_confidence = np.mean([f.confidence_score for f in involved_findings])
                strength = avg_importance * 0.6 + avg_confidence * 0.4

                # Extract common theme
                common_terms = self._extract_common_terms(involved_findings)
                pattern_name = f"Convergence: {', '.join(common_terms[:3])}"

                patterns.append(
                    PatternAnalysis(
                        pattern_type=PatternType.CONVERGENCE,
                        pattern_name=pattern_name,
                        description=(
                            f"{len(unique_indices)} findings converge on similar conclusions"
                        ),
                        strength=float(strength),
                        finding_ids=[str(idx) for idx in unique_indices],
                        confidence_factors={
                            "similarity": float(similarity),
                            "importance": float(avg_importance),
                            "confidence": float(avg_confidence),
                        },
                        implications=[
                            f"Strong consensus on {common_terms[0] if common_terms else 'topic'}",
                            "High confidence in this research direction",
                        ],
                    )
                )

        return patterns

    def _detect_divergence_patterns(
        self, findings: list[HierarchicalFinding]
    ) -> list[PatternAnalysis]:
        """Detect divergence patterns where findings branch into different areas.

        Args:
            findings: List of findings to analyze

        Returns:
            List of divergence patterns
        """
        patterns = []

        # Group findings by category
        category_groups = defaultdict(list)
        for i, finding in enumerate(findings):
            if finding.category:
                category_groups[finding.category].append(i)

        # Check for divergence within categories
        for category, indices in category_groups.items():
            if len(indices) >= self.min_findings_for_pattern:
                category_findings = [findings[i] for i in indices]

                # Check variance in confidence and importance
                confidence_variance = np.var([f.confidence_score for f in category_findings])
                importance_variance = np.var([f.importance_score for f in category_findings])

                # High variance suggests divergence
                if confidence_variance > 0.1 or importance_variance > 0.1:
                    strength = min(1.0, confidence_variance + importance_variance)

                    patterns.append(
                        PatternAnalysis(
                            pattern_type=PatternType.DIVERGENCE,
                            pattern_name=f"Divergence in {category}",
                            description=f"Findings in {category} show divergent conclusions",
                            strength=float(strength),
                            finding_ids=[str(i) for i in indices],
                            confidence_factors={
                                "confidence_variance": float(confidence_variance),
                                "importance_variance": float(importance_variance),
                            },
                            implications=[
                                "Multiple valid perspectives exist",
                                "Further investigation needed to reconcile differences",
                            ],
                        )
                    )

        return patterns

    def _detect_temporal_patterns(
        self, findings: list[HierarchicalFinding]
    ) -> list[PatternAnalysis]:
        """Detect temporal patterns and trends.

        Args:
            findings: List of findings to analyze

        Returns:
            List of temporal patterns
        """
        patterns = []

        # Extract temporal information from findings
        temporal_findings = []
        for i, finding in enumerate(findings):
            # Look for years in the finding text
            years = re.findall(r"\b(19|20)\d{2}\b", finding.finding)
            if years or (finding.source and finding.source.date):
                temporal_findings.append((i, finding, years))

        if len(temporal_findings) >= self.min_findings_for_pattern:
            # Sort by extracted years or source dates
            temporal_findings.sort(
                key=lambda x: (
                    int(x[2][0])
                    if x[2]
                    else x[1].source.date.year
                    if x[1].source and x[1].source.date
                    else 2020
                )
            )

            # Check for trends
            importances = [f[1].importance_score for f in temporal_findings]
            if len(importances) > 2:
                # Simple trend detection using linear correlation
                indices = np.arange(len(importances))
                # Guard against zero-variance importances which yields NaN/RuntimeWarning
                if float(np.std(importances)) < 1e-12:
                    correlation = 0.0
                else:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        corr_mat = np.corrcoef(indices, importances)
                        correlation = float(corr_mat[0, 1]) if corr_mat.size >= 4 else 0.0
                        if np.isnan(correlation):
                            correlation = 0.0

                if abs(correlation) > 0.5:
                    trend_type = "increasing" if correlation > 0 else "decreasing"
                    strength = abs(correlation)

                    patterns.append(
                        PatternAnalysis(
                            pattern_type=PatternType.TEMPORAL,
                            pattern_name=f"Temporal {trend_type} trend",
                            description=f"Importance shows {trend_type} trend over time",
                            strength=float(strength),
                            finding_ids=[str(f[0]) for f in temporal_findings],
                            temporal_span=f"{len(temporal_findings)} time points",
                            confidence_factors={
                                "correlation": float(abs(correlation)),
                                "sample_size": float(len(temporal_findings) / 10),
                            },
                            implications=[
                                f"Topic relevance is {trend_type} over time",
                                "Consider temporal context in analysis",
                            ],
                        )
                    )

        return patterns

    def _detect_causal_patterns(self, findings: list[HierarchicalFinding]) -> list[PatternAnalysis]:
        """Detect potential causal relationships.

        Args:
            findings: List of findings to analyze

        Returns:
            List of causal patterns
        """
        patterns = []

        # Causal keywords
        causal_keywords = {
            "causes",
            "leads to",
            "results in",
            "because",
            "due to",
            "consequently",
            "therefore",
            "thus",
            "hence",
            "as a result",
            "effect of",
            "impact of",
            "influences",
            "drives",
            "triggers",
        }

        causal_findings = []
        for i, finding in enumerate(findings):
            text_lower = finding.finding.lower()
            if any(keyword in text_lower for keyword in causal_keywords):
                causal_findings.append((i, finding))

        if len(causal_findings) >= self.min_findings_for_pattern:
            # Group by similar causal relationships
            texts = [f[1].finding for f in causal_findings]

            try:
                X = self.vectorizer.fit_transform(texts)
                vectors = X.toarray()

                # Find clusters of similar causal relationships
                causal_groups = []
                used_indices = set()

                for i in range(len(causal_findings)):
                    if i in used_indices:
                        continue

                    group = [i]
                    for j in range(i + 1, len(causal_findings)):
                        if j not in used_indices:
                            similarity = self._cosine_similarity(vectors[i], vectors[j])
                            if similarity > self.similarity_threshold:
                                group.append(j)
                                used_indices.add(j)

                    if len(group) >= 2:
                        causal_groups.append(group)
                        used_indices.add(i)

                # Create patterns for causal groups
                for group in causal_groups:
                    group_findings = [causal_findings[i][1] for i in group]
                    avg_confidence = np.mean([f.confidence_score for f in group_findings])
                    strength = avg_confidence * 0.8  # Weight confidence heavily for causal claims

                    patterns.append(
                        PatternAnalysis(
                            pattern_type=PatternType.CAUSAL,
                            pattern_name="Causal relationship identified",
                            description=f"{len(group)} findings suggest causal relationships",
                            strength=float(strength),
                            finding_ids=[str(causal_findings[i][0]) for i in group],
                            confidence_factors={
                                "evidence_count": float(len(group) / 10),
                                "confidence": float(avg_confidence),
                            },
                            implications=[
                                "Potential cause-effect relationship identified",
                                "Consider validating causal claims with additional evidence",
                            ],
                        )
                    )
            except (ValueError, AttributeError):
                pass

        return patterns

    def _detect_anomaly_patterns(
        self, findings: list[HierarchicalFinding]
    ) -> list[PatternAnalysis]:
        """Detect anomalies or outliers in findings.

        Args:
            findings: List of findings to analyze

        Returns:
            List of anomaly patterns
        """
        patterns = []

        if len(findings) < 5:  # Need enough findings to detect outliers
            return patterns

        # Collect scores for analysis
        confidence_scores = np.array([f.confidence_score for f in findings])
        importance_scores = np.array([f.importance_score for f in findings])

        # Detect outliers using IQR method
        for scores, score_type in [
            (confidence_scores, "confidence"),
            (importance_scores, "importance"),
        ]:
            q1 = np.percentile(scores, 25)
            q3 = np.percentile(scores, 75)
            iqr = q3 - q1

            if iqr > 0:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                outlier_indices = np.where((scores < lower_bound) | (scores > upper_bound))[0]

                if len(outlier_indices) > 0:
                    avg_deviation = np.mean(
                        [abs(scores[i] - np.median(scores)) for i in outlier_indices]
                    )

                    patterns.append(
                        PatternAnalysis(
                            pattern_type=PatternType.ANOMALY,
                            pattern_name=f"Anomaly in {score_type}",
                            description=(
                                f"{len(outlier_indices)} findings show unusual {score_type} scores"
                            ),
                            strength=float(min(1.0, avg_deviation * 2)),
                            finding_ids=[str(i) for i in outlier_indices],
                            confidence_factors={
                                "deviation": float(avg_deviation),
                                "outlier_count": float(len(outlier_indices) / len(findings)),
                            },
                            implications=[
                                f"Unusual {score_type} levels detected",
                                "May indicate special cases or data quality issues",
                            ],
                        )
                    )

        return patterns

    def _detect_emergence_patterns(self, clusters: list[ThemeCluster]) -> list[PatternAnalysis]:
        """Detect emergence patterns from theme clusters.

        Args:
            clusters: Theme clusters to analyze

        Returns:
            List of emergence patterns
        """
        patterns = []

        # Look for clusters that combine diverse findings
        for cluster in clusters:
            if len(cluster.findings) >= self.min_findings_for_pattern:
                # Calculate diversity within cluster
                categories = [f.category for f in cluster.findings if f.category]
                category_diversity = len(set(categories)) / len(categories) if categories else 0

                confidence_range = max([f.confidence_score for f in cluster.findings]) - min(
                    [f.confidence_score for f in cluster.findings]
                )

                # High diversity suggests emergence
                if category_diversity > 0.5 or confidence_range > 0.4:
                    strength = cluster.coherence_score * cluster.importance_score

                    patterns.append(
                        PatternAnalysis(
                            pattern_type=PatternType.EMERGENCE,
                            pattern_name=f"Emergent theme: {cluster.theme_name}",
                            description=(
                                f"New pattern emerging from synthesis of "
                                f"{len(cluster.findings)} findings"
                            ),
                            strength=float(strength),
                            finding_ids=[str(i) for i in range(len(cluster.findings))],
                            confidence_factors={
                                "coherence": float(cluster.coherence_score),
                                "importance": float(cluster.importance_score),
                                "diversity": float(category_diversity),
                            },
                            implications=[
                                "Novel insight from synthesis",
                                "Cross-domain pattern identified",
                            ],
                        )
                    )

        return patterns

    def _finding_to_text(self, finding: HierarchicalFinding) -> str:
        """Convert finding to text for analysis.

        Args:
            finding: Finding to convert

        Returns:
            Text representation
        """
        parts = [finding.finding]
        if finding.supporting_evidence:
            parts.extend(finding.supporting_evidence)
        if finding.category:
            parts.append(f"Category: {finding.category}")
        return " ".join(parts)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def _extract_common_terms(self, findings: list[HierarchicalFinding]) -> list[str]:
        """Extract common terms from findings.

        Args:
            findings: List of findings

        Returns:
            List of common terms
        """
        # Combine all text
        all_text = " ".join([self._finding_to_text(f) for f in findings])

        # Simple term extraction (words that appear frequently)
        words = re.findall(r"\b[a-z]+\b", all_text.lower())

        # Filter stop words
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
        }
        words = [w for w in words if w not in stop_words and len(w) > 3]

        # Get most common terms
        counter = Counter(words)
        return [term for term, _ in counter.most_common(5)]

    def _link_related_patterns(self, patterns: list[PatternAnalysis]) -> None:
        """Link related patterns together.

        Args:
            patterns: List of patterns to link
        """
        for i, pattern1 in enumerate(patterns):
            for j, pattern2 in enumerate(patterns):
                if i != j:
                    # Check if patterns share findings
                    shared_findings = set(pattern1.finding_ids) & set(pattern2.finding_ids)
                    if len(shared_findings) > 0:
                        # Add as related pattern
                        if str(j) not in pattern1.related_patterns:
                            pattern1.related_patterns.append(str(j))

    def analyze_pattern_relationships(self, patterns: list[PatternAnalysis]) -> dict[str, Any]:
        """Analyze relationships between patterns.

        Args:
            patterns: List of patterns to analyze

        Returns:
            Analysis of pattern relationships
        """
        if not patterns:
            return {
                "total_patterns": 0,
                "pattern_types": {},
                "strongest_patterns": [],
                "pattern_network_density": 0.0,
            }

        # Count pattern types
        pattern_types = Counter([p.pattern_type for p in patterns])

        # Find strongest patterns
        strongest_patterns = sorted(patterns, key=lambda p: p.strength, reverse=True)[:3]

        # Calculate network density (how interconnected patterns are)
        total_connections = sum(len(p.related_patterns) for p in patterns)
        max_connections = len(patterns) * (len(patterns) - 1)
        network_density = total_connections / max_connections if max_connections > 0 else 0

        return {
            "total_patterns": len(patterns),
            "pattern_types": dict(pattern_types),
            "strongest_patterns": [
                {"name": p.pattern_name, "strength": p.strength} for p in strongest_patterns
            ],
            "pattern_network_density": float(network_density),
            "avg_pattern_strength": float(np.mean([p.strength for p in patterns])),
        }
