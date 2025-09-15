"""Confidence analysis service for research findings."""

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from src.models.research_executor import (
    ConfidenceAnalysis,
    ConfidenceLevel,
    Contradiction,
    HierarchicalFinding,
    ThemeCluster,
)


class ConfidenceAnalyzer:
    """Analyzes and calculates confidence metrics for research findings."""

    def __init__(
        self,
        min_confidence_threshold: float = 0.6,
        consistency_weight: float = 0.3,
        source_weight: float = 0.3,
        evidence_weight: float = 0.4,
    ):
        """Initialize the confidence analyzer.

        Args:
            min_confidence_threshold: Minimum acceptable confidence level
            consistency_weight: Weight for consistency in overall confidence
            source_weight: Weight for source reliability in overall confidence
            evidence_weight: Weight for evidence strength in overall confidence
        """
        self.min_confidence_threshold = min_confidence_threshold
        self.consistency_weight = consistency_weight
        self.source_weight = source_weight
        self.evidence_weight = evidence_weight

        # Ensure weights sum to 1.0
        total_weight = consistency_weight + source_weight + evidence_weight
        if abs(total_weight - 1.0) > 0.01:
            # Normalize weights
            self.consistency_weight = consistency_weight / total_weight
            self.source_weight = source_weight / total_weight
            self.evidence_weight = evidence_weight / total_weight

    def analyze_confidence(
        self,
        findings: list[HierarchicalFinding],
        contradictions: list[Contradiction] | None = None,
        clusters: list[ThemeCluster] | None = None,
    ) -> ConfidenceAnalysis:
        """Perform comprehensive confidence analysis.

        Args:
            findings: List of research findings
            contradictions: Optional list of contradictions
            clusters: Optional theme clusters

        Returns:
            Detailed confidence analysis
        """
        if not findings:
            return ConfidenceAnalysis(
                overall_confidence=0.0,
                confidence_distribution={"none": 0},
                uncertainty_areas=["No findings to analyze"],
            )

        # Calculate confidence distribution
        confidence_distribution = self._calculate_confidence_distribution(findings)

        # Calculate component scores
        source_reliability = self._calculate_source_reliability(findings)
        consistency_score = self._calculate_consistency_score(findings, contradictions)
        evidence_strength = self._calculate_evidence_strength(findings)

        # Calculate overall confidence
        overall_confidence = (
            self.consistency_weight * consistency_score
            + self.source_weight * source_reliability
            + self.evidence_weight * evidence_strength
        )

        # Analyze confidence by category
        category_confidence = self._analyze_category_confidence(findings)

        # Identify uncertainty areas
        uncertainty_areas = self._identify_uncertainty_areas(
            findings, category_confidence, contradictions
        )

        # Identify confidence gaps
        confidence_gaps = self._identify_confidence_gaps(findings, clusters, category_confidence)

        # Generate improvement recommendations
        confidence_improvements = self._generate_improvement_recommendations(
            overall_confidence,
            source_reliability,
            consistency_score,
            evidence_strength,
            uncertainty_areas,
        )

        return ConfidenceAnalysis(
            overall_confidence=float(overall_confidence),
            confidence_distribution=confidence_distribution,
            source_reliability=float(source_reliability),
            consistency_score=float(consistency_score),
            evidence_strength=float(evidence_strength),
            category_confidence=category_confidence,
            uncertainty_areas=uncertainty_areas,
            confidence_gaps=confidence_gaps,
            confidence_improvements=confidence_improvements,
        )

    def _calculate_confidence_distribution(
        self, findings: list[HierarchicalFinding]
    ) -> dict[str, int]:
        """Calculate distribution of findings by confidence level.

        Args:
            findings: List of findings

        Returns:
            Distribution of confidence levels
        """
        distribution = Counter()

        for finding in findings:
            distribution[finding.confidence.value] += 1

        return dict(distribution)

    def _calculate_source_reliability(self, findings: list[HierarchicalFinding]) -> float:
        """Calculate average source reliability.

        Args:
            findings: List of findings

        Returns:
            Average source reliability score
        """
        source_scores = []

        for finding in findings:
            if finding.source:
                # Use source's credibility score
                source_scores.append(finding.source.credibility_score)
            else:
                # No source, assign low reliability
                source_scores.append(0.3)

        if not source_scores:
            return 0.5

        return float(np.mean(source_scores))

    def _calculate_consistency_score(
        self,
        findings: list[HierarchicalFinding],
        contradictions: list[Contradiction] | None,
    ) -> float:
        """Calculate consistency score based on contradictions.

        Args:
            findings: List of findings
            contradictions: Optional list of contradictions

        Returns:
            Consistency score
        """
        if not findings:
            return 0.0

        base_consistency = 1.0

        if contradictions:
            # Reduce consistency based on contradictions
            num_findings = len(findings)
            num_contradictions = len(contradictions)

            # Count severe contradictions
            severe_contradictions = sum(
                1 for c in contradictions if c.contradiction_type == "direct" or c.severity > 0.7
            )

            # Calculate penalty
            contradiction_ratio = num_contradictions / max(num_findings, 1)
            severe_ratio = severe_contradictions / max(num_findings, 1)

            # Apply penalties
            base_consistency -= contradiction_ratio * 0.3  # Up to 30% penalty
            base_consistency -= severe_ratio * 0.2  # Additional penalty for severe

        # Also consider variance in confidence scores
        confidence_scores = [f.confidence_score for f in findings]
        if len(confidence_scores) > 1:
            confidence_variance = np.var(confidence_scores)
            # High variance reduces consistency
            base_consistency -= min(0.2, confidence_variance)

        return max(0.0, min(1.0, base_consistency))

    def _calculate_evidence_strength(self, findings: list[HierarchicalFinding]) -> float:
        """Calculate strength of supporting evidence.

        Args:
            findings: List of findings

        Returns:
            Evidence strength score
        """
        evidence_scores = []

        for finding in findings:
            # Base score from number of supporting evidence
            num_evidence = len(finding.supporting_evidence)
            evidence_score = min(1.0, num_evidence / 3)  # Cap at 3 pieces

            # Weight by finding confidence
            evidence_score *= finding.confidence_score

            # Weight by importance
            evidence_score *= 0.5 + 0.5 * finding.importance_score

            evidence_scores.append(evidence_score)

        if not evidence_scores:
            return 0.0

        return float(np.mean(evidence_scores))

    def _analyze_category_confidence(self, findings: list[HierarchicalFinding]) -> dict[str, float]:
        """Analyze confidence by category.

        Args:
            findings: List of findings

        Returns:
            Confidence scores by category
        """
        category_scores = defaultdict(list)

        for finding in findings:
            category = finding.category or "uncategorized"
            category_scores[category].append(finding.confidence_score)

        category_confidence = {}
        for category, scores in category_scores.items():
            category_confidence[category] = float(np.mean(scores))

        return category_confidence

    def _identify_uncertainty_areas(
        self,
        findings: list[HierarchicalFinding],
        category_confidence: dict[str, float],
        contradictions: list[Contradiction] | None,
    ) -> list[str]:
        """Identify areas of uncertainty.

        Args:
            findings: List of findings
            category_confidence: Confidence by category
            contradictions: Optional contradictions

        Returns:
            List of uncertainty areas
        """
        uncertainty_areas = []

        # Low confidence categories
        for category, confidence in category_confidence.items():
            if confidence < self.min_confidence_threshold:
                uncertainty_areas.append(f"{category}: low confidence ({confidence:.2f})")

        # Areas with many low-confidence findings
        low_conf_findings = [
            f
            for f in findings
            if f.confidence == ConfidenceLevel.LOW or f.confidence == ConfidenceLevel.UNCERTAIN
        ]
        if len(low_conf_findings) > len(findings) * 0.3:
            uncertainty_areas.append(
                f"{len(low_conf_findings)} findings with low/uncertain confidence"
            )

        # Areas with contradictions
        if contradictions:
            contradiction_categories = set()
            for _contradiction in contradictions:
                # Get findings involved in contradiction
                for finding in findings:
                    if finding.category:
                        contradiction_categories.add(finding.category)

            if contradiction_categories:
                uncertainty_areas.append(
                    f"Contradictions in: {', '.join(list(contradiction_categories)[:3])}"
                )

        # Findings without sources
        no_source_count = sum(1 for f in findings if f.source is None)
        if no_source_count > len(findings) * 0.2:
            uncertainty_areas.append(f"{no_source_count} findings lack source attribution")

        return uncertainty_areas

    def _identify_confidence_gaps(
        self,
        findings: list[HierarchicalFinding],
        clusters: list[ThemeCluster] | None,
        category_confidence: dict[str, float],
    ) -> list[str]:
        """Identify gaps in confidence coverage.

        Args:
            findings: List of findings
            clusters: Optional theme clusters
            category_confidence: Confidence by category

        Returns:
            List of confidence gaps
        """
        gaps = []

        # Check for categories with insufficient findings
        category_counts = Counter(f.category for f in findings if f.category)
        for category, count in category_counts.items():
            if count < 3 and category_confidence.get(category, 0) < 0.7:
                gaps.append(f"{category}: insufficient evidence (only {count} findings)")

        # Check for weak clusters
        if clusters:
            weak_clusters = [
                c for c in clusters if c.coherence_score < 0.5 or c.average_confidence() < 0.6
            ]
            if weak_clusters:
                gaps.append(f"{len(weak_clusters)} theme clusters with weak coherence/confidence")

        # Check for missing evidence
        no_evidence_findings = [f for f in findings if len(f.supporting_evidence) == 0]
        if len(no_evidence_findings) > len(findings) * 0.15:
            gaps.append(f"{len(no_evidence_findings)} findings lack supporting evidence")

        # Check temporal coverage
        findings_with_dates = [f for f in findings if f.source and f.source.date]
        if findings_with_dates and len(findings_with_dates) < len(findings) * 0.5:
            gaps.append("Insufficient temporal coverage in sources")

        return gaps

    def _generate_improvement_recommendations(
        self,
        overall_confidence: float,
        source_reliability: float,
        consistency_score: float,
        evidence_strength: float,
        uncertainty_areas: list[str],
    ) -> list[str]:
        """Generate recommendations for improving confidence.

        Args:
            overall_confidence: Overall confidence score
            source_reliability: Source reliability score
            consistency_score: Consistency score
            evidence_strength: Evidence strength score
            uncertainty_areas: List of uncertainty areas

        Returns:
            List of improvement recommendations
        """
        recommendations = []

        # Check overall confidence
        if overall_confidence < self.min_confidence_threshold:
            recommendations.append(
                "Gather additional high-quality sources to improve overall confidence"
            )

        # Check source reliability
        if source_reliability < 0.6:
            recommendations.append("Prioritize more credible and authoritative sources")
            recommendations.append("Verify findings with peer-reviewed or official sources")

        # Check consistency
        if consistency_score < 0.7:
            recommendations.append("Investigate and resolve contradictions between findings")
            recommendations.append("Seek additional evidence to clarify conflicting claims")

        # Check evidence strength
        if evidence_strength < 0.5:
            recommendations.append("Strengthen findings with more supporting evidence")
            recommendations.append("Provide specific examples and data points")

        # Address uncertainty areas
        if len(uncertainty_areas) > 3:
            recommendations.append(
                f"Focus research on {len(uncertainty_areas)} identified uncertainty areas"
            )
            recommendations.append("Consider expert consultation for low-confidence topics")

        # Specific actionable recommendations
        if source_reliability < consistency_score and source_reliability < evidence_strength:
            recommendations.insert(0, "PRIORITY: Improve source quality and credibility")
        elif consistency_score < source_reliability and consistency_score < evidence_strength:
            recommendations.insert(0, "PRIORITY: Resolve contradictions and inconsistencies")
        elif evidence_strength < source_reliability and evidence_strength < consistency_score:
            recommendations.insert(0, "PRIORITY: Strengthen evidence base")

        return recommendations[:7]  # Limit to 7 most important recommendations

    def calculate_finding_confidence(self, finding: HierarchicalFinding) -> dict[str, float]:
        """Calculate detailed confidence metrics for a single finding.

        Args:
            finding: Finding to analyze

        Returns:
            Detailed confidence metrics
        """
        metrics = {}

        # Base confidence from the finding itself
        metrics["base_confidence"] = finding.confidence_score

        # Source confidence
        if finding.source:
            metrics["source_confidence"] = finding.source.overall_quality()
        else:
            metrics["source_confidence"] = 0.3

        # Evidence confidence
        num_evidence = len(finding.supporting_evidence)
        metrics["evidence_confidence"] = min(1.0, num_evidence / 3)

        # Importance weight (higher importance requires higher confidence)
        importance_factor = 1.0 - (finding.importance_score * 0.3)
        metrics["importance_adjusted"] = metrics["base_confidence"] * importance_factor

        # Calculate composite confidence
        metrics["composite_confidence"] = float(
            np.mean(
                [
                    metrics["base_confidence"],
                    metrics["source_confidence"],
                    metrics["evidence_confidence"],
                ]
            )
        )

        return metrics

    def compare_cluster_confidence(self, clusters: list[ThemeCluster]) -> dict[str, Any]:
        """Compare confidence across theme clusters.

        Args:
            clusters: List of theme clusters

        Returns:
            Comparative analysis of cluster confidence
        """
        if not clusters:
            return {
                "highest_confidence_cluster": None,
                "lowest_confidence_cluster": None,
                "confidence_range": 0.0,
                "clusters_below_threshold": [],
            }

        cluster_confidences = []
        for cluster in clusters:
            avg_confidence = cluster.average_confidence()
            cluster_confidences.append(
                {
                    "name": cluster.theme_name,
                    "confidence": avg_confidence,
                    "finding_count": len(cluster.findings),
                }
            )

        # Sort by confidence
        cluster_confidences.sort(key=lambda x: x["confidence"], reverse=True)

        # Identify clusters below threshold
        below_threshold = [
            c for c in cluster_confidences if c["confidence"] < self.min_confidence_threshold
        ]

        confidence_values = [c["confidence"] for c in cluster_confidences]

        return {
            "highest_confidence_cluster": cluster_confidences[0] if cluster_confidences else None,
            "lowest_confidence_cluster": cluster_confidences[-1] if cluster_confidences else None,
            "confidence_range": float(max(confidence_values) - min(confidence_values)),
            "avg_cluster_confidence": float(np.mean(confidence_values)),
            "clusters_below_threshold": below_threshold,
        }
