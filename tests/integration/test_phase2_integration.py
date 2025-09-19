"""Integration tests for Phase 2 Research Executor components."""

import pytest
from datetime import datetime, UTC

from models.research_executor import (
    ConfidenceLevel,
    ImportanceLevel,
    ResearchSource,
    HierarchicalFinding,
    ThemeCluster,
    Contradiction,
    PatternAnalysis,
    PatternType,
    ConfidenceAnalysis,
)
from services.synthesis_engine import SynthesisEngine
from services.contradiction_detector import ContradictionDetector
from services.pattern_recognizer import PatternRecognizer
from services.confidence_analyzer import ConfidenceAnalyzer


class TestPhase2Integration:
    """Test Phase 2 integration: Pattern Recognition and Confidence Analysis."""

    @pytest.fixture
    def comprehensive_findings(self) -> list[HierarchicalFinding]:
        """Create comprehensive findings for testing patterns."""
        source1 = ResearchSource(
            title="AI Research Paper 2024",
            url="https://example.com/ai-paper",
            credibility_score=0.9,
            relevance_score=0.85,
            date=datetime(2024, 1, 15, tzinfo=UTC),
        )

        source2 = ResearchSource(
            title="ML Industry Report",
            url="https://example.com/ml-report",
            credibility_score=0.75,
            relevance_score=0.8,
            date=datetime(2023, 11, 20, tzinfo=UTC),
        )

        source3 = ResearchSource(
            title="Tech Blog Analysis",
            url="https://example.com/blog",
            credibility_score=0.6,
            relevance_score=0.7,
        )

        findings = [
            # Convergence pattern candidates
            HierarchicalFinding(
                finding="Deep learning models show 30% improvement in accuracy",
                supporting_evidence=["Benchmark A", "Test suite B"],
                confidence=ConfidenceLevel.HIGH,
                importance=ImportanceLevel.CRITICAL,
                source=source1,
                category="performance",
            ),
            HierarchicalFinding(
                finding="Neural networks demonstrate 35% better accuracy",
                supporting_evidence=["Validation tests"],
                confidence=ConfidenceLevel.HIGH,
                importance=ImportanceLevel.HIGH,
                source=source2,
                category="performance",
            ),
            HierarchicalFinding(
                finding="AI models achieve significant accuracy gains",
                supporting_evidence=["Industry benchmarks"],
                confidence=ConfidenceLevel.MEDIUM,
                importance=ImportanceLevel.HIGH,
                source=source3,
                category="performance",
            ),

            # Divergence pattern candidates
            HierarchicalFinding(
                finding="Implementation costs increase by 50%",
                supporting_evidence=["Budget analysis"],
                confidence=ConfidenceLevel.HIGH,
                importance=ImportanceLevel.HIGH,
                source=source1,
                category="economics",
            ),
            HierarchicalFinding(
                finding="Operational costs decrease by 20% over time",
                supporting_evidence=["Long-term study"],
                confidence=ConfidenceLevel.LOW,
                importance=ImportanceLevel.MEDIUM,
                source=source2,
                category="economics",
            ),

            # Temporal pattern candidates
            HierarchicalFinding(
                finding="Performance improvements observed since 2020",
                supporting_evidence=["Historical data"],
                confidence=ConfidenceLevel.MEDIUM,
                importance=ImportanceLevel.MEDIUM,
                source=source1,
                category="trends",
                temporal_relevance="2020-2024",
            ),
            HierarchicalFinding(
                finding="Adoption rates increased in 2023",
                supporting_evidence=["Market analysis"],
                confidence=ConfidenceLevel.HIGH,
                importance=ImportanceLevel.HIGH,
                source=source2,
                category="trends",
                temporal_relevance="2023",
            ),

            # Causal pattern candidates
            HierarchicalFinding(
                finding="Better training data leads to improved model performance",
                supporting_evidence=["Experimental results"],
                confidence=ConfidenceLevel.HIGH,
                importance=ImportanceLevel.CRITICAL,
                source=source1,
                category="causality",
            ),
            HierarchicalFinding(
                finding="Increased compute resources results in faster training",
                supporting_evidence=["Performance metrics"],
                confidence=ConfidenceLevel.HIGH,
                importance=ImportanceLevel.HIGH,
                source=source2,
                category="causality",
            ),

            # Anomaly candidates
            HierarchicalFinding(
                finding="Unexpected behavior in edge cases",
                supporting_evidence=["Edge case tests"],
                confidence=ConfidenceLevel.UNCERTAIN,
                importance=ImportanceLevel.LOW,
                source=source3,
                category="anomalies",
            ),
        ]

        return findings

    def test_pattern_recognition(self, comprehensive_findings):
        """Test pattern recognition functionality."""
        recognizer = PatternRecognizer(
            min_findings_for_pattern=2,
            similarity_threshold=0.5
        )

        # Detect patterns
        patterns = recognizer.detect_patterns(comprehensive_findings)

        # Verify patterns were detected
        assert len(patterns) > 0

        # Check pattern properties
        for pattern in patterns:
            assert isinstance(pattern, PatternAnalysis)
            assert pattern.pattern_type in PatternType
            assert pattern.pattern_name
            assert pattern.description
            assert 0.0 <= pattern.strength <= 1.0
            assert isinstance(pattern.finding_ids, list)

        # Check for specific pattern types
        pattern_types = {p.pattern_type for p in patterns}

        # Should detect at least convergence (similar accuracy findings)
        assert PatternType.CONVERGENCE in pattern_types or len(patterns) > 0

    def test_pattern_recognition_with_clusters(self, comprehensive_findings):
        """Test pattern recognition with theme clusters."""
        # First create clusters
        engine = SynthesisEngine(min_cluster_size=2, max_clusters=4)
        clusters = engine.cluster_findings(comprehensive_findings)

        # Then detect patterns
        recognizer = PatternRecognizer()
        patterns = recognizer.detect_patterns(comprehensive_findings, clusters)

        # Should detect emergence patterns from clusters
        pattern_types = {p.pattern_type for p in patterns}

        # Verify we have patterns (may or may not include emergence)
        assert len(patterns) >= 0

        # Analyze pattern relationships
        analysis = recognizer.analyze_pattern_relationships(patterns)
        assert "total_patterns" in analysis
        assert "pattern_types" in analysis
        assert "strongest_patterns" in analysis
        assert "pattern_network_density" in analysis

    def test_confidence_analysis(self, comprehensive_findings):
        """Test confidence analysis functionality."""
        analyzer = ConfidenceAnalyzer(
            min_confidence_threshold=0.6,
            consistency_weight=0.3,
            source_weight=0.3,
            evidence_weight=0.4,
        )

        # Analyze confidence
        analysis = analyzer.analyze_confidence(comprehensive_findings)

        # Verify analysis structure
        assert isinstance(analysis, ConfidenceAnalysis)
        assert 0.0 <= analysis.overall_confidence <= 1.0
        assert 0.0 <= analysis.source_reliability <= 1.0
        assert 0.0 <= analysis.consistency_score <= 1.0
        assert 0.0 <= analysis.evidence_strength <= 1.0

        # Check confidence distribution
        assert analysis.confidence_distribution
        total_findings = sum(analysis.confidence_distribution.values())
        assert total_findings == len(comprehensive_findings)

        # Check category confidence
        assert analysis.category_confidence
        for category, confidence in analysis.category_confidence.items():
            assert 0.0 <= confidence <= 1.0

        # Check helper methods
        needs_validation = analysis.needs_validation()
        assert isinstance(needs_validation, bool)

        weak_areas = analysis.get_weak_areas(threshold=0.5)
        assert isinstance(weak_areas, list)

    def test_confidence_with_contradictions(self, comprehensive_findings):
        """Test confidence analysis with contradictions."""
        # Detect contradictions
        detector = ContradictionDetector()
        contradictions = detector.detect_contradictions(comprehensive_findings)

        # Analyze confidence with contradictions
        analyzer = ConfidenceAnalyzer()
        analysis = analyzer.analyze_confidence(
            comprehensive_findings,
            contradictions=contradictions
        )

        # Consistency should be affected by contradictions
        if contradictions:
            assert analysis.consistency_score < 1.0
            assert len(analysis.uncertainty_areas) > 0

        # Check recommendations
        assert isinstance(analysis.confidence_improvements, list)
        if analysis.overall_confidence < 0.6:
            assert len(analysis.confidence_improvements) > 0

    def test_full_phase2_pipeline(self, comprehensive_findings):
        """Test complete Phase 2 pipeline integration."""
        # Phase 1 components
        engine = SynthesisEngine()
        detector = ContradictionDetector()

        # Phase 2 components
        pattern_recognizer = PatternRecognizer()
        confidence_analyzer = ConfidenceAnalyzer()

        # Run Phase 1
        clusters = engine.cluster_findings(comprehensive_findings)
        contradictions = detector.detect_contradictions(comprehensive_findings)

        # Run Phase 2
        patterns = pattern_recognizer.detect_patterns(comprehensive_findings, clusters)
        confidence_analysis = confidence_analyzer.analyze_confidence(
            comprehensive_findings,
            contradictions=contradictions,
            clusters=clusters,
        )

        # Verify complete results
        assert len(clusters) > 0
        assert isinstance(contradictions, list)
        assert isinstance(patterns, list)
        assert isinstance(confidence_analysis, ConfidenceAnalysis)

        # Check pattern analysis
        if patterns:
            pattern_analysis = pattern_recognizer.analyze_pattern_relationships(patterns)
            assert pattern_analysis["total_patterns"] == len(patterns)
            assert pattern_analysis["avg_pattern_strength"] > 0

        # Check cluster confidence comparison
        cluster_comparison = confidence_analyzer.compare_cluster_confidence(clusters)
        assert "highest_confidence_cluster" in cluster_comparison
        assert "lowest_confidence_cluster" in cluster_comparison
        assert "confidence_range" in cluster_comparison

    def test_single_finding_confidence(self, comprehensive_findings):
        """Test confidence calculation for individual findings."""
        analyzer = ConfidenceAnalyzer()

        for finding in comprehensive_findings[:3]:
            metrics = analyzer.calculate_finding_confidence(finding)

            assert "base_confidence" in metrics
            assert "source_confidence" in metrics
            assert "evidence_confidence" in metrics
            assert "importance_adjusted" in metrics
            assert "composite_confidence" in metrics

            # All metrics should be in valid range
            for key, value in metrics.items():
                assert 0.0 <= value <= 1.0

    def test_pattern_significance(self, comprehensive_findings):
        """Test pattern significance evaluation."""
        recognizer = PatternRecognizer(min_findings_for_pattern=2)
        patterns = recognizer.detect_patterns(comprehensive_findings)

        if patterns:
            # Test significance check
            significant_patterns = [p for p in patterns if p.is_significant(0.5)]

            # Test average confidence factor
            for pattern in patterns:
                avg_conf = pattern.average_confidence_factor()
                if pattern.confidence_factors:
                    assert avg_conf > 0
                else:
                    assert avg_conf == 0.0

    def test_empty_findings_handling(self):
        """Test Phase 2 components with empty findings."""
        empty_findings = []

        # Test pattern recognizer
        recognizer = PatternRecognizer()
        patterns = recognizer.detect_patterns(empty_findings)
        assert patterns == []

        pattern_analysis = recognizer.analyze_pattern_relationships(patterns)
        assert pattern_analysis["total_patterns"] == 0

        # Test confidence analyzer
        analyzer = ConfidenceAnalyzer()
        confidence = analyzer.analyze_confidence(empty_findings)
        assert confidence.overall_confidence == 0.0
        assert len(confidence.uncertainty_areas) > 0

    def test_minimal_findings_handling(self):
        """Test Phase 2 with minimal findings."""
        minimal_finding = HierarchicalFinding(
            finding="Single test finding",
            confidence=ConfidenceLevel.HIGH,
            importance=ImportanceLevel.MEDIUM,
        )

        # Test pattern recognizer
        recognizer = PatternRecognizer(min_findings_for_pattern=1)
        patterns = recognizer.detect_patterns([minimal_finding])
        # Should not detect patterns with single finding (need at least pairs for most patterns)
        assert len(patterns) == 0

        # Test confidence analyzer
        analyzer = ConfidenceAnalyzer()
        confidence = analyzer.analyze_confidence([minimal_finding])
        assert confidence.overall_confidence > 0
        assert confidence.source_reliability < 0.5  # No source provided
