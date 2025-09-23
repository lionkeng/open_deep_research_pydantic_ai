"""Synthesis engine for ML-based clustering and analysis of research findings."""

from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import Normalizer

from models.research_executor import (
    Contradiction,
    ContradictionType,
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
        svd_components: int = 50,
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

        self.vectorizer_max_features = vectorizer_max_features
        self.svd_components = svd_components

        self.last_vectorized_data: Any | None = None
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
            vectorizer = self._create_vectorizer(len(findings))
            X = vectorizer.fit_transform(texts)
            # Keep sparse matrix to avoid memory issues with large datasets
            self.last_vectorized_data = X
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

        # Reduce dimensionality to keep clustering stable in high-dimensional space
        reduced_matrix = self._reduce_dimensions(X)

        # Determine optimal number of clusters using reduced representations
        optimal_k = self._find_optimal_clusters(reduced_matrix)

        # Perform KMeans clustering
        kmeans = KMeans(
            n_clusters=optimal_k,
            random_state=self.random_state,
            n_init=10,
        )
        labels = kmeans.fit_predict(reduced_matrix)
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
            coherence = self._calculate_coherence(reduced_matrix, labels, label)

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

    def cluster_findings_from_vectors(
        self, findings: list[HierarchicalFinding], vectors: list[list[float]]
    ) -> list[ThemeCluster]:
        """Cluster findings using precomputed dense vectors (e.g., embeddings).

        Args:
            findings: Findings to cluster
            vectors: Dense vectors aligned with findings (len equal)

        Returns:
            List of theme clusters
        """
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        if len(findings) < self.min_cluster_size:
            return [
                ThemeCluster(
                    theme_name="General Findings",
                    description="Ungrouped research findings",
                    findings=findings,
                    coherence_score=1.0,
                    importance_score=self._calculate_importance_score(findings),
                )
            ]

        if not vectors or len(vectors) != len(findings):
            # Fallback to standard path if vectors missing
            return self.cluster_findings(findings)

        X = np.array(vectors, dtype=float)

        # Determine optimal number of clusters on dense vectors
        n_samples = X.shape[0]
        max_k = min(self.max_clusters, n_samples - 1) if n_samples > 2 else 2
        best_k = 2
        best_score = -1.0
        for k in range(2, max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=5)
                labels = kmeans.fit_predict(X)
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                continue

        kmeans = KMeans(n_clusters=best_k, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(X)

        # Group findings by cluster
        clusters: dict[int, list[HierarchicalFinding]] = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(label, []).append(findings[idx])

        theme_clusters: list[ThemeCluster] = []
        for label, cluster_findings in clusters.items():
            theme_name = self._extract_theme_name(cluster_findings, label)

            # Coherence: average pairwise cosine on cluster vectors
            cluster_indices = [i for i, lab in enumerate(labels) if lab == label]
            cluster_vecs = X[cluster_indices]
            # compute limited pairwise similarities to avoid O(n^2) explosion
            sims = []
            for i in range(len(cluster_vecs)):
                for j in range(i + 1, len(cluster_vecs)):
                    v1 = cluster_vecs[i]
                    v2 = cluster_vecs[j]
                    n1 = np.linalg.norm(v1)
                    n2 = np.linalg.norm(v2)
                    if n1 > 0 and n2 > 0:
                        sims.append(float(np.dot(v1, v2) / (n1 * n2)))
            coherence = float((np.mean(sims) + 1) / 2) if sims else 0.5

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

        theme_clusters.sort(key=lambda x: x.importance_score, reverse=True)
        return theme_clusters

    def _create_vectorizer(self, n_documents: int) -> TfidfVectorizer:
        """Create a TF-IDF vectorizer that adapts thresholds to corpus size."""

        params = self._dataset_aware_tfidf_params(n_documents)

        return TfidfVectorizer(
            max_features=min(self.vectorizer_max_features, max(50, n_documents * 10)),
            stop_words="english",
            ngram_range=(1, 2),
            **params,
        )

    def _dataset_aware_tfidf_params(self, n_documents: int) -> dict[str, Any]:
        """Derive TF-IDF bounds that scale with the number of documents."""

        if n_documents <= 5:
            # Keep all terms when data is sparse
            return {"min_df": 1, "max_df": 1.0}

        if n_documents <= 25:
            return {"min_df": 1, "max_df": 0.95}

        # For larger corpora switch to proportion-based thresholds
        min_df = max(2, int(0.02 * n_documents))
        max_df = 0.85 if n_documents < 100 else 0.8

        # Ensure min_df < total documents to avoid empty vocabularies
        min_df = min(min_df, n_documents - 1)

        return {"min_df": min_df, "max_df": max_df}

    def _reduce_dimensions(self, X: Any) -> np.ndarray:
        """Project sparse TF-IDF data into a lower-dimensional dense space."""

        n_samples, n_features = X.shape

        # Skip reduction when data is already low-dimensional
        if n_samples < 3 or n_features <= 3:
            return X.toarray() if hasattr(X, "toarray") else X

        max_components = min(self.svd_components, n_features - 1, n_samples - 1)

        if max_components < 2:
            return X.toarray() if hasattr(X, "toarray") else X

        svd = TruncatedSVD(n_components=max_components, random_state=self.random_state)
        normalizer = Normalizer(copy=False)

        reduced = svd.fit_transform(X)
        reduced = normalizer.fit_transform(reduced)

        return reduced

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

        # Get cluster points - keep as sparse matrix for memory efficiency
        cluster_points = X[cluster_indices]
        is_sparse = hasattr(cluster_points, "toarray")

        # Calculate average pairwise similarity within cluster
        similarities = []
        n_points = cluster_points.shape[0]

        # Limit pairwise comparisons for large clusters to avoid memory issues
        max_comparisons = 100
        step = max(1, n_points * (n_points - 1) // (2 * max_comparisons))

        for i in range(0, n_points, step):
            for j in range(i + 1, n_points, step):
                if j >= n_points:
                    break

                # Get vectors efficiently based on matrix type
                if is_sparse:
                    vec_i = cluster_points[i].toarray().flatten()
                    vec_j = cluster_points[j].toarray().flatten()
                else:
                    vec_i = cluster_points[i]
                    vec_j = cluster_points[j]

                # Cosine similarity calculation
                dot_product = np.dot(vec_i, vec_j)
                norm_i = np.linalg.norm(vec_i)
                norm_j = np.linalg.norm(vec_j)

                if norm_i > 0 and norm_j > 0:
                    similarity = dot_product / (norm_i * norm_j)
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
            vectorizer = self._create_vectorizer(len(findings))
            # Vectorize findings
            vectors = vectorizer.fit_transform(texts)
        except ValueError:
            return []

        # Look for findings with high similarity but conflicting sentiment
        for i in range(len(findings)):
            for j in range(i + 1, len(findings)):
                # Calculate cosine similarity with sparse matrix support
                similarity = self._cosine_similarity_sparse(vectors, i, j)

                # High similarity suggests related topics
                if similarity > 0.5:
                    # Check for conflicting signals
                    if self._are_contradictory(findings[i], findings[j]):
                        contradictions.append(
                            Contradiction(
                                id=f"synthesis-contradiction-{i}-{j}",
                                type=ContradictionType.SEMANTIC,
                                evidence_indices=[i, j],
                                description=self._explain_contradiction(findings[i], findings[j]),
                                confidence_score=0.6,
                                resolution_suggestion=self._suggest_resolution(
                                    findings[i], findings[j]
                                ),
                                contradiction_type="direct",
                                explanation=self._explain_contradiction(findings[i], findings[j]),
                                resolution_hint=self._suggest_resolution(findings[i], findings[j]),
                                finding_1_id=str(i),
                                finding_2_id=str(j),
                                severity=0.7,
                                severity_level="medium",
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

    def _cosine_similarity_sparse(self, matrix: Any, i: int, j: int) -> float:
        """Calculate cosine similarity between two vectors in a sparse matrix.

        Args:
            matrix: Sparse or dense matrix
            i: Index of first vector
            j: Index of second vector

        Returns:
            Cosine similarity score
        """
        # Get vectors efficiently
        if hasattr(matrix, "toarray"):
            # Sparse matrix - extract individual rows
            vec1 = matrix[i].toarray().flatten()
            vec2 = matrix[j].toarray().flatten()
        else:
            # Dense matrix
            vec1 = matrix[i]
            vec2 = matrix[j]

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
        if finding2.confidence_score > finding1.confidence_score + 0.2:
            conf = finding2.confidence_score
            return f"Consider prioritizing the second finding (confidence: {conf:.2f})"
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
