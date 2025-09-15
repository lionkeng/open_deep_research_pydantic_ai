"""Synthesis engine for ML-based clustering and analysis of research findings."""

from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from src.models.research_executor import (
    Contradiction,
    HierarchicalFinding,
    ImportanceLevel,
    ThemeCluster,
)


class SynthesisEngine:
    """ML-powered synthesis engine for research findings.

    Uses TF-IDF vectorization and KMeans clustering to identify themes
    and patterns in research data.
    """

    def __init__(
        self,
        min_cluster_size: int = 2,
        max_clusters: int = 10,
        vectorizer_max_features: int = 100,
        random_state: int = 42,
    ):
        """Initialize the synthesis engine.

        Args:
            min_cluster_size: Minimum number of findings to form a cluster
            max_clusters: Maximum number of clusters to create
            vectorizer_max_features: Maximum features for TF-IDF vectorization
            random_state: Random seed for reproducibility
        """
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.random_state = random_state

        # Initialize TF-IDF vectorizer with scikit-learn
        self.vectorizer = TfidfVectorizer(
            max_features=vectorizer_max_features,
            stop_words="english",
            ngram_range=(1, 2),  # Use unigrams and bigrams
            min_df=0.1,  # Ignore terms that appear in less than 10% of documents
            max_df=0.9,  # Ignore terms that appear in more than 90% of documents
        )

        self.last_vectorized_data: np.ndarray | None = None
        self.last_labels: np.ndarray | None = None

    def cluster_findings(self, findings: list[HierarchicalFinding]) -> list[ThemeCluster]:
        """Cluster findings into themes using ML.

        Args:
            findings: List of hierarchical findings to cluster

        Returns:
            List of theme clusters with assigned findings
        """
        if len(findings) < self.min_cluster_size:
            # Not enough findings to cluster meaningfully
            return [
                ThemeCluster(
                    theme_name="General Findings",
                    description="Ungrouped research findings",
                    findings=findings,
                    coherence_score=1.0,
                    importance_score=self._calculate_importance_score(findings),
                )
            ]

        # Prepare text data for clustering
        texts = [self._finding_to_text(f) for f in findings]

        # Vectorize the text using TF-IDF
        try:
            X = self.vectorizer.fit_transform(texts)
            self.last_vectorized_data = X.toarray()
        except ValueError:
            # If vectorization fails (e.g., all texts are too similar)
            return [
                ThemeCluster(
                    theme_name="Research Findings",
                    description="Collection of related findings",
                    findings=findings,
                    coherence_score=0.8,
                    importance_score=self._calculate_importance_score(findings),
                )
            ]

        # Determine optimal number of clusters
        optimal_k = self._find_optimal_clusters(X)

        # Perform KMeans clustering
        kmeans = KMeans(
            n_clusters=optimal_k,
            random_state=self.random_state,
            n_init=10,
        )
        labels = kmeans.fit_predict(X)
        self.last_labels = labels

        # Group findings by cluster
        clusters: dict[int, list[HierarchicalFinding]] = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(findings[idx])

        # Create ThemeCluster objects
        theme_clusters = []
        for label, cluster_findings in clusters.items():
            # Extract theme name from common terms
            theme_name = self._extract_theme_name(cluster_findings, label)

            # Calculate coherence score based on cluster tightness
            coherence = self._calculate_coherence(X, labels, label)

            # Calculate importance score
            importance = self._calculate_importance_score(cluster_findings)

            theme_clusters.append(
                ThemeCluster(
                    theme_name=theme_name,
                    description=self._generate_cluster_description(cluster_findings),
                    findings=cluster_findings,
                    coherence_score=coherence,
                    importance_score=importance,
                )
            )

        # Sort by importance score
        theme_clusters.sort(key=lambda x: x.importance_score, reverse=True)

        return theme_clusters

    def _finding_to_text(self, finding: HierarchicalFinding) -> str:
        """Convert a finding to text for vectorization.

        Args:
            finding: The finding to convert

        Returns:
            Text representation of the finding
        """
        parts = [finding.finding]

        if finding.supporting_evidence:
            parts.extend(finding.supporting_evidence)

        if finding.category:
            parts.append(f"Category: {finding.category}")

        if finding.metadata:
            for key, value in finding.metadata.items():
                if isinstance(value, str):
                    parts.append(f"{key}: {value}")

        return " ".join(parts)

    def _find_optimal_clusters(self, X: Any) -> int:
        """Find optimal number of clusters using silhouette score.

        Args:
            X: Vectorized feature matrix

        Returns:
            Optimal number of clusters
        """
        n_samples = X.shape[0]
        max_k = min(self.max_clusters, n_samples - 1)

        if max_k <= 2:
            return 2

        best_score = -1
        best_k = 2

        for k in range(2, max_k + 1):
            try:
                kmeans = KMeans(
                    n_clusters=k,
                    random_state=self.random_state,
                    n_init=5,  # Fewer initializations for speed
                )
                labels = kmeans.fit_predict(X)

                # Calculate silhouette score
                score = silhouette_score(X, labels)

                if score > best_score:
                    best_score = score
                    best_k = k
            except (ValueError, TypeError):
                # If clustering fails for this k (e.g., too few samples), skip it
                # This can happen when k is too large for the number of samples
                continue

        return best_k

    def _calculate_coherence(self, X: Any, labels: np.ndarray, cluster_label: int) -> float:
        """Calculate coherence score for a cluster.

        Args:
            X: Feature matrix
            labels: Cluster labels
            cluster_label: The specific cluster to evaluate

        Returns:
            Coherence score between 0 and 1
        """
        # Get indices of points in this cluster
        cluster_indices = np.where(labels == cluster_label)[0]

        if len(cluster_indices) < 2:
            return 1.0  # Single point cluster is perfectly coherent

        # Get cluster points - convert sparse to dense if needed
        cluster_points = X[cluster_indices]
        if hasattr(cluster_points, "toarray"):
            cluster_points = cluster_points.toarray()

        # Calculate average pairwise similarity within cluster
        similarities = []
        n_points = cluster_points.shape[0]
        for i in range(n_points):
            for j in range(i + 1, n_points):
                # Cosine similarity
                similarity = np.dot(cluster_points[i], cluster_points[j])
                norm_i = np.linalg.norm(cluster_points[i])
                norm_j = np.linalg.norm(cluster_points[j])
                if norm_i > 0 and norm_j > 0:
                    similarity = similarity / (norm_i * norm_j)
                    similarities.append(similarity)

        if similarities:
            # Convert average similarity to 0-1 scale
            avg_similarity = np.mean(similarities)
            # Cosine similarity is already between -1 and 1, normalize to 0-1
            coherence = (avg_similarity + 1) / 2
            return float(coherence)

        return 0.5  # Default moderate coherence

    def _calculate_importance_score(self, findings: list[HierarchicalFinding]) -> float:
        """Calculate importance score for a group of findings.

        Args:
            findings: List of findings

        Returns:
            Importance score between 0 and 1
        """
        if not findings:
            return 0.0

        # Calculate average importance from ImportanceLevel
        importance_scores = []
        for finding in findings:
            if finding.importance == ImportanceLevel.CRITICAL:
                importance_scores.append(1.0)
            elif finding.importance == ImportanceLevel.HIGH:
                importance_scores.append(0.8)
            elif finding.importance == ImportanceLevel.MEDIUM:
                importance_scores.append(0.5)
            else:  # LOW
                importance_scores.append(0.2)

        # Weight by confidence levels
        weighted_scores = []
        for i, finding in enumerate(findings):
            weight = finding.confidence_score
            weighted_scores.append(importance_scores[i] * weight)

        return float(np.mean(weighted_scores))

    def _extract_theme_name(self, findings: list[HierarchicalFinding], cluster_label: int) -> str:
        """Extract a theme name from cluster findings.

        Args:
            findings: Findings in the cluster
            cluster_label: Cluster label number

        Returns:
            Theme name for the cluster
        """
        if not findings:
            return f"Theme {cluster_label + 1}"

        # Collect all text from findings
        texts = [self._finding_to_text(f) for f in findings]
        combined_text = " ".join(texts)

        # Use TF-IDF to find most important terms
        try:
            tfidf = TfidfVectorizer(
                max_features=5,
                stop_words="english",
                ngram_range=(1, 2),
            )
            tfidf.fit([combined_text])
            feature_names = tfidf.get_feature_names_out()

            if len(feature_names) > 0:
                # Use top 2-3 terms as theme name
                theme_terms = feature_names[: min(3, len(feature_names))]
                return " & ".join(term.title() for term in theme_terms)
        except (ValueError, AttributeError):
            # TfidfVectorizer may fail with certain text patterns
            pass

        # Fallback to category-based naming
        categories = [f.category for f in findings if f.category]
        if categories:
            most_common = max(set(categories), key=categories.count)
            return f"{most_common.title()} Insights"

        return f"Theme {cluster_label + 1}"

    def _generate_cluster_description(self, findings: list[HierarchicalFinding]) -> str:
        """Generate a description for a cluster of findings.

        Args:
            findings: Findings in the cluster

        Returns:
            Cluster description
        """
        if not findings:
            return "Empty cluster"

        # Count key characteristics
        num_findings = len(findings)
        num_critical = sum(1 for f in findings if f.importance == ImportanceLevel.CRITICAL)
        num_high_confidence = sum(1 for f in findings if f.confidence_score >= 0.8)

        categories = list({f.category for f in findings if f.category})

        parts = [f"Cluster containing {num_findings} findings"]

        if num_critical > 0:
            parts.append(f"{num_critical} critical importance")

        if num_high_confidence > 0:
            parts.append(f"{num_high_confidence} high confidence")

        if categories:
            parts.append(f"covering {', '.join(categories[:3])}")

        return "; ".join(parts)

    def identify_contradictions(self, findings: list[HierarchicalFinding]) -> list[Contradiction]:
        """Identify contradictions in findings using vector similarity.

        Args:
            findings: List of findings to analyze

        Returns:
            List of identified contradictions
        """
        if len(findings) < 2:
            return []

        contradictions = []

        # Prepare texts for vectorization
        texts = [self._finding_to_text(f) for f in findings]

        try:
            # Vectorize findings
            X = self.vectorizer.fit_transform(texts)
            vectors = X.toarray()
        except ValueError:
            return []

        # Look for findings with high similarity but conflicting sentiment
        for i in range(len(findings)):
            for j in range(i + 1, len(findings)):
                # Calculate cosine similarity
                similarity = self._cosine_similarity(vectors[i], vectors[j])

                # High similarity suggests related topics
                if similarity > 0.5:
                    # Check for conflicting signals
                    if self._are_contradictory(findings[i], findings[j]):
                        contradictions.append(
                            Contradiction(
                                finding_1_id=str(i),
                                finding_2_id=str(j),
                                contradiction_type="direct",
                                explanation=self._explain_contradiction(findings[i], findings[j]),
                                resolution_hint=self._suggest_resolution(findings[i], findings[j]),
                            )
                        )

        return contradictions

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.

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

    def _are_contradictory(
        self, finding1: HierarchicalFinding, finding2: HierarchicalFinding
    ) -> bool:
        """Check if two findings are contradictory.

        Args:
            finding1: First finding
            finding2: Second finding

        Returns:
            True if findings appear contradictory
        """
        # Simple heuristic: Look for opposing sentiment words
        text1 = finding1.finding.lower()
        text2 = finding2.finding.lower()

        positive_words = {"increase", "improve", "better", "success", "effective", "growth"}
        negative_words = {"decrease", "worse", "failure", "ineffective", "decline", "reduce"}

        # Check if one is positive and other is negative
        text1_positive = any(word in text1 for word in positive_words)
        text1_negative = any(word in text1 for word in negative_words)
        text2_positive = any(word in text2 for word in positive_words)
        text2_negative = any(word in text2 for word in negative_words)

        return (text1_positive and text2_negative) or (text1_negative and text2_positive)

    def _explain_contradiction(
        self, finding1: HierarchicalFinding, finding2: HierarchicalFinding
    ) -> str:
        """Generate explanation for a contradiction.

        Args:
            finding1: First finding
            finding2: Second finding

        Returns:
            Explanation of the contradiction
        """
        return (
            f"Finding '{finding1.finding[:50]}...' conflicts with "
            f"'{finding2.finding[:50]}...' regarding the same topic"
        )

    def _suggest_resolution(
        self, finding1: HierarchicalFinding, finding2: HierarchicalFinding
    ) -> str:
        """Suggest resolution for a contradiction.

        Args:
            finding1: First finding
            finding2: Second finding

        Returns:
            Resolution suggestion
        """
        # Compare confidence scores
        if finding1.confidence_score > finding2.confidence_score + 0.2:
            conf = finding1.confidence_score
            return f"Consider prioritizing the first finding (confidence: {conf:.2f})"
        elif finding2.confidence_score > finding1.confidence_score + 0.2:
            conf = finding2.confidence_score
            return f"Consider prioritizing the second finding (confidence: {conf:.2f})"
        else:
            return "Further investigation needed to resolve this contradiction"

    def calculate_synthesis_metrics(self, clusters: list[ThemeCluster]) -> dict[str, float]:
        """Calculate overall synthesis quality metrics.

        Args:
            clusters: List of theme clusters

        Returns:
            Dictionary of quality metrics
        """
        if not clusters:
            return {
                "coverage": 0.0,
                "coherence": 0.0,
                "diversity": 0.0,
                "confidence": 0.0,
            }

        # Coverage: How many findings were successfully clustered
        total_findings = sum(len(c.findings) for c in clusters)
        coverage = min(1.0, total_findings / 10)  # Normalize to expected ~10 findings

        # Average coherence across clusters
        coherence = np.mean([c.coherence_score for c in clusters])

        # Diversity: Number of distinct themes
        diversity = min(1.0, len(clusters) / 5)  # Normalize to expected ~5 themes

        # Average confidence across all findings
        all_confidences = []
        for cluster in clusters:
            for finding in cluster.findings:
                all_confidences.append(finding.confidence_score)
        confidence = np.mean(all_confidences) if all_confidences else 0.0

        return {
            "coverage": float(coverage),
            "coherence": float(coherence),
            "diversity": float(diversity),
            "confidence": float(confidence),
        }
