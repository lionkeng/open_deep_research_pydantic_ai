"""Comprehensive tests for contradiction severity calculation system."""

import math
from datetime import datetime, timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st

from models.research_executor import (
    ConfidenceLevel,
    Contradiction,
    HierarchicalFinding,
    ImportanceLevel,
    ResearchResults,
    ResearchSource,
)
from utils.validation import ContradictionSeverityCalculator


class TestContradictionSeverityCalculator:
    """Tests for the ContradictionSeverityCalculator class."""

    def test_calculate_contradiction_severity_basic(self):
        """Test basic severity calculation with known inputs."""
        result = ContradictionSeverityCalculator.calculate_contradiction_severity(
            contradiction_type="direct",
            finding_1_confidence=0.9,
            finding_2_confidence=0.8,
            source_1_credibility=0.9,
            source_2_credibility=0.8,
            source_1_type="academic",
            source_2_type="research",
        )

        assert isinstance(result, dict)
        assert "overall_severity" in result
        assert "components" in result
        assert "metadata" in result

        # Direct contradictions should have high severity
        assert result["overall_severity"] >= 0.7
        assert 0.0 <= result["overall_severity"] <= 1.0

    def test_contradiction_type_weights(self):
        """Test that different contradiction types have appropriate weights."""
        base_params = {
            "finding_1_confidence": 0.8,
            "finding_2_confidence": 0.8,
            "source_1_credibility": 0.8,
            "source_2_credibility": 0.8,
            "source_1_type": "academic",
            "source_2_type": "academic",
        }

        # Test severity ordering
        direct = ContradictionSeverityCalculator.calculate_contradiction_severity(
            contradiction_type="direct", **base_params
        )
        semantic = ContradictionSeverityCalculator.calculate_contradiction_severity(
            contradiction_type="semantic", **base_params
        )
        minor = ContradictionSeverityCalculator.calculate_contradiction_severity(
            contradiction_type="minor", **base_params
        )

        # Direct should be most severe, minor should be least severe
        assert direct["overall_severity"] > semantic["overall_severity"]
        assert semantic["overall_severity"] > minor["overall_severity"]

    def test_confidence_impact(self):
        """Test that higher confidence increases severity."""
        base_params = {
            "contradiction_type": "direct",
            "source_1_credibility": 0.8,
            "source_2_credibility": 0.8,
            "source_1_type": "academic",
            "source_2_type": "academic",
        }

        low_confidence = ContradictionSeverityCalculator.calculate_contradiction_severity(
            finding_1_confidence=0.3,
            finding_2_confidence=0.3,
            **base_params
        )
        high_confidence = ContradictionSeverityCalculator.calculate_contradiction_severity(
            finding_1_confidence=0.9,
            finding_2_confidence=0.9,
            **base_params
        )

        assert high_confidence["overall_severity"] > low_confidence["overall_severity"]

    def test_temporal_decay(self):
        """Test that temporal distance reduces severity."""
        base_params = {
            "contradiction_type": "direct",
            "finding_1_confidence": 0.8,
            "finding_2_confidence": 0.8,
            "source_1_credibility": 0.8,
            "source_2_credibility": 0.8,
            "source_1_type": "academic",
            "source_2_type": "academic",
        }

        recent = ContradictionSeverityCalculator.calculate_contradiction_severity(
            temporal_distance_days=0.0, **base_params
        )
        old = ContradictionSeverityCalculator.calculate_contradiction_severity(
            temporal_distance_days=365.0, **base_params
        )

        assert recent["overall_severity"] > old["overall_severity"]

    def test_source_credibility_impact(self):
        """Test that source credibility affects severity."""
        base_params = {
            "contradiction_type": "direct",
            "finding_1_confidence": 0.8,
            "finding_2_confidence": 0.8,
            "source_1_type": "academic",
            "source_2_type": "academic",
        }

        low_credibility = ContradictionSeverityCalculator.calculate_contradiction_severity(
            source_1_credibility=0.2,
            source_2_credibility=0.2,
            **base_params
        )
        high_credibility = ContradictionSeverityCalculator.calculate_contradiction_severity(
            source_1_credibility=0.9,
            source_2_credibility=0.9,
            **base_params
        )

        assert high_credibility["overall_severity"] > low_credibility["overall_severity"]

    def test_importance_amplification(self):
        """Test that importance amplifies severity."""
        base_params = {
            "contradiction_type": "direct",
            "finding_1_confidence": 0.8,
            "finding_2_confidence": 0.8,
            "source_1_credibility": 0.8,
            "source_2_credibility": 0.8,
            "source_1_type": "academic",
            "source_2_type": "academic",
        }

        low_importance = ContradictionSeverityCalculator.calculate_contradiction_severity(
            importance_1=0.2, importance_2=0.2, **base_params
        )
        high_importance = ContradictionSeverityCalculator.calculate_contradiction_severity(
            importance_1=0.9, importance_2=0.9, **base_params
        )

        assert high_importance["overall_severity"] > low_importance["overall_severity"]

    def test_classify_severity_level(self):
        """Test severity level classification."""
        assert ContradictionSeverityCalculator.classify_severity_level(0.9) == "critical"
        assert ContradictionSeverityCalculator.classify_severity_level(0.7) == "high"
        assert ContradictionSeverityCalculator.classify_severity_level(0.5) == "medium"
        assert ContradictionSeverityCalculator.classify_severity_level(0.3) == "low"
        assert ContradictionSeverityCalculator.classify_severity_level(0.1) == "minimal"

    def test_get_resolution_priority(self):
        """Test resolution priority calculation."""
        # Critical severity should get priority 1
        assert ContradictionSeverityCalculator.get_resolution_priority(0.9, "direct") == 1
        assert ContradictionSeverityCalculator.get_resolution_priority(0.7, "semantic") == 1
        assert ContradictionSeverityCalculator.get_resolution_priority(0.5, "partial") == 3
        assert ContradictionSeverityCalculator.get_resolution_priority(0.1, "minor") == 5

    def test_suggest_resolution_strategy(self):
        """Test resolution strategy suggestions."""
        strategy = ContradictionSeverityCalculator.suggest_resolution_strategy(
            "direct", 0.9, {"temporal_distance_days": 30}
        )

        assert isinstance(strategy, dict)
        assert "strategy" in strategy
        assert "actions" in strategy
        assert "timeline" in strategy
        assert "severity_level" in strategy
        assert "priority" in strategy
        assert "estimated_effort" in strategy

        # Critical severity should suggest immediate investigation
        assert strategy["strategy"] == "immediate_investigation"
        assert strategy["priority"] == 1

    def test_edge_cases_handling(self):
        """Test handling of edge cases and invalid inputs."""
        # Test with NaN values
        result = ContradictionSeverityCalculator.calculate_contradiction_severity(
            contradiction_type="direct",
            finding_1_confidence=float('nan'),
            finding_2_confidence=float('inf'),
            source_1_credibility=-1.0,
            source_2_credibility=2.0,
        )

        assert 0.0 <= result["overall_severity"] <= 1.0

        # Test with None source types
        result = ContradictionSeverityCalculator.calculate_contradiction_severity(
            contradiction_type="unknown_type",
            finding_1_confidence=0.5,
            finding_2_confidence=0.5,
            source_1_credibility=0.5,
            source_2_credibility=0.5,
            source_1_type=None,
            source_2_type=None,
        )

        assert 0.0 <= result["overall_severity"] <= 1.0

    @given(
        contradiction_type=st.text(min_size=1, max_size=20),
        finding_1_confidence=st.floats(allow_nan=True, allow_infinity=True),
        finding_2_confidence=st.floats(allow_nan=True, allow_infinity=True),
        source_1_credibility=st.floats(allow_nan=True, allow_infinity=True),
        source_2_credibility=st.floats(allow_nan=True, allow_infinity=True),
        temporal_distance=st.floats(min_value=0.0, max_value=10000.0),
    )
    def test_severity_calculation_robustness(
        self,
        contradiction_type,
        finding_1_confidence,
        finding_2_confidence,
        source_1_credibility,
        source_2_credibility,
        temporal_distance,
    ):
        """Property-based test: severity calculation handles any input gracefully."""
        result = ContradictionSeverityCalculator.calculate_contradiction_severity(
            contradiction_type=contradiction_type,
            finding_1_confidence=finding_1_confidence,
            finding_2_confidence=finding_2_confidence,
            source_1_credibility=source_1_credibility,
            source_2_credibility=source_2_credibility,
            temporal_distance_days=temporal_distance,
        )

        # Should always return valid result
        assert isinstance(result, dict)
        assert "overall_severity" in result
        assert 0.0 <= result["overall_severity"] <= 1.0
        assert not math.isnan(result["overall_severity"])
        assert not math.isinf(result["overall_severity"])


class TestContradictionModel:
    """Tests for the enhanced Contradiction model."""

    def test_contradiction_creation(self):
        """Test basic contradiction creation."""
        contradiction = Contradiction(
            finding_1_id="0",
            finding_2_id="1",
            contradiction_type="direct",
            explanation="These findings directly contradict each other",
        )

        assert contradiction.finding_1_id == "0"
        assert contradiction.finding_2_id == "1"
        assert contradiction.contradiction_type == "direct"
        assert contradiction.severity == 0.5  # Default value

    def test_comprehensive_severity_calculation(self):
        """Test the comprehensive severity calculation method."""
        contradiction = Contradiction(
            finding_1_id="0",
            finding_2_id="1",
            contradiction_type="direct",
            explanation="Direct contradiction between findings",
        )

        contradiction.calculate_comprehensive_severity(
            finding_1_confidence=0.9,
            finding_2_confidence=0.8,
            source_1_credibility=0.9,
            source_2_credibility=0.8,
            source_1_type="academic",
            source_2_type="research",
            importance_1=0.9,
            importance_2=0.8,
        )

        # Verify all fields were updated
        assert contradiction.severity > 0.5  # Should be higher than default
        assert hasattr(contradiction, 'severity_components')
        assert hasattr(contradiction, 'resolution_strategy')
        assert contradiction.priority >= 1
        assert contradiction.severity_level in ["minimal", "low", "medium", "high", "critical"]

    def test_needs_immediate_resolution(self):
        """Test immediate resolution detection."""
        # High severity contradiction
        high_severity = Contradiction(
            finding_1_id="0",
            finding_2_id="1",
            contradiction_type="direct",
            explanation="Critical contradiction",
            severity=0.9,
            priority=1,
            severity_level="critical"
        )

        assert high_severity.needs_immediate_resolution()

        # Low severity contradiction
        low_severity = Contradiction(
            finding_1_id="0",
            finding_2_id="1",
            contradiction_type="minor",
            explanation="Minor inconsistency",
            severity=0.2,
            priority=5,
            severity_level="minimal"
        )

        assert not low_severity.needs_immediate_resolution()

    def test_get_severity_summary(self):
        """Test severity summary generation."""
        contradiction = Contradiction(
            finding_1_id="0",
            finding_2_id="1",
            contradiction_type="direct",
            explanation="Test contradiction",
            severity=0.8,
            priority=2,
            severity_level="high",
            resolution_strategy={"timeline": "Within 1 week", "estimated_effort": "Medium"}
        )

        summary = contradiction.get_severity_summary()

        assert isinstance(summary, dict)
        assert "overall_severity" in summary
        assert "severity_level" in summary
        assert "priority" in summary
        assert "needs_immediate_resolution" in summary
        assert summary["overall_severity"] == 0.8

    def test_get_resolution_actions(self):
        """Test resolution actions extraction."""
        contradiction = Contradiction(
            finding_1_id="0",
            finding_2_id="1",
            contradiction_type="direct",
            explanation="Test contradiction",
            resolution_strategy={
                "actions": ["Action 1", "Action 2"],
                "type_specific_actions": ["Type action 1", "Type action 2"]
            }
        )

        actions = contradiction.get_resolution_actions()

        assert isinstance(actions, list)
        assert len(actions) == 4
        assert "Action 1" in actions
        assert "Type action 1" in actions

    def test_to_markdown(self):
        """Test markdown conversion."""
        contradiction = Contradiction(
            finding_1_id="0",
            finding_2_id="1",
            contradiction_type="direct",
            explanation="Test contradiction for markdown",
            severity=0.8,
            priority=2,
            severity_level="high",
            resolution_strategy={
                "strategy": "systematic_verification",
                "timeline": "Within 1 week",
                "estimated_effort": "Medium",
                "actions": ["Cross-reference sources", "Seek expert review"]
            },
            severity_components={
                "overall_severity": 0.8,
                "raw_severity": 0.85,
                "components": {
                    "base_severity": 1.0,
                    "confidence_factor": 0.9,
                    "credibility_factor": 0.8
                },
                "metadata": {}
            }
        )

        markdown = contradiction.to_markdown()

        assert isinstance(markdown, str)
        assert "### Direct Contradiction" in markdown
        assert "**Severity:** 0.80 (High)" in markdown
        assert "**Priority:** 2/5" in markdown
        assert "Test contradiction for markdown" in markdown
        assert "systematic_verification" in markdown
        assert "Cross-reference sources" in markdown


class TestResearchResultsContradictionIntegration:
    """Tests for contradiction integration in ResearchResults."""

    @pytest.fixture
    def sample_research_results(self):
        """Create sample research results with contradictions."""
        base_time = datetime.now()  # Use consistent base time
        findings = [
            HierarchicalFinding(
                finding="AI will revolutionize healthcare",
                supporting_evidence=["Study A shows 90% improvement"],
                confidence=ConfidenceLevel.HIGH,
                confidence_score=0.9,
                importance=ImportanceLevel.CRITICAL,
                importance_score=0.95,
                source=ResearchSource(
                    title="Medical AI Study",
                    url="https://medical.edu/study1",
                    source_type="academic",
                    credibility_score=0.9,
                    date=base_time
                ),
            ),
            HierarchicalFinding(
                finding="AI implementation faces significant barriers",
                supporting_evidence=["Survey shows 70% adoption challenges"],
                confidence=ConfidenceLevel.MEDIUM,
                confidence_score=0.7,
                importance=ImportanceLevel.HIGH,
                importance_score=0.8,
                source=ResearchSource(
                    title="Industry Survey",
                    url="https://industry.com/survey",
                    source_type="research",
                    credibility_score=0.7,
                    date=base_time - timedelta(days=30)
                ),
            ),
        ]

        contradictions = [
            Contradiction(
                finding_1_id="0",
                finding_2_id="1",
                contradiction_type="perspective",
                explanation="Optimistic vs realistic views on AI adoption",
                severity=0.6,
            )
        ]

        return ResearchResults(
            query="AI in healthcare",
            findings=findings,
            sources=[f.source for f in findings],
            contradictions=contradictions,
        )

    def test_calculate_all_contradiction_severities(self, sample_research_results):
        """Test comprehensive severity calculation for all contradictions."""
        # Calculate comprehensive severities
        sample_research_results.calculate_all_contradiction_severities()

        # Severity should have been updated
        updated_contradiction = sample_research_results.contradictions[0]
        assert hasattr(updated_contradiction, 'severity_components')
        assert hasattr(updated_contradiction, 'resolution_strategy')
        assert updated_contradiction.temporal_distance_days == 30.0

    def test_weighted_contradiction_rate(self, sample_research_results):
        """Test weighted contradiction rate calculation."""
        # Calculate severities first
        sample_research_results.calculate_all_contradiction_severities()

        rate = sample_research_results.get_contradiction_rate()

        assert isinstance(rate, float)
        assert 0.0 <= rate <= 1.0

    def test_get_contradiction_summary(self, sample_research_results):
        """Test comprehensive contradiction summary."""
        # Calculate severities first
        sample_research_results.calculate_all_contradiction_severities()

        summary = sample_research_results.get_contradiction_summary()

        assert isinstance(summary, dict)
        assert "total_contradictions" in summary
        assert "weighted_contradiction_rate" in summary
        assert "severity_distribution" in summary
        assert "priority_distribution" in summary
        assert "high_priority_count" in summary
        assert "immediate_resolution_needed" in summary
        assert "average_severity" in summary
        assert "most_severe_contradiction" in summary

        assert summary["total_contradictions"] == 1

    def test_empty_contradictions_handling(self):
        """Test handling of research results with no contradictions."""
        empty_results = ResearchResults(query="Test query")

        rate = empty_results.get_contradiction_rate()
        assert rate == 0.0

        summary = empty_results.get_contradiction_summary()
        assert summary["total_contradictions"] == 0
        assert summary["weighted_contradiction_rate"] == 0.0

        # Should not raise errors
        empty_results.calculate_all_contradiction_severities()


class TestIntegrationContradictionSeverity:
    """Integration tests for the complete contradiction severity system."""

    def test_end_to_end_contradiction_processing(self):
        """Test complete end-to-end contradiction processing pipeline."""
        # Create research results with complex contradictions
        findings = [
            HierarchicalFinding(
                finding="Climate change is accelerating rapidly",
                supporting_evidence=["Temperature records", "Ice melt data"],
                confidence=ConfidenceLevel.HIGH,
                confidence_score=0.95,
                importance=ImportanceLevel.CRITICAL,
                importance_score=0.98,
                source=ResearchSource(
                    title="Climate Science Journal",
                    url="https://climate.edu/paper1",
                    source_type="academic",
                    credibility_score=0.95,
                    date=datetime.now()
                ),
            ),
            HierarchicalFinding(
                finding="Climate models show slower warming trends",
                supporting_evidence=["Revised model predictions"],
                confidence=ConfidenceLevel.MEDIUM,
                confidence_score=0.6,
                importance=ImportanceLevel.HIGH,
                importance_score=0.75,
                source=ResearchSource(
                    title="Modeling Study",
                    url="https://models.org/study",
                    source_type="research",
                    credibility_score=0.6,
                    date=datetime.now() - timedelta(days=180)
                ),
            ),
            HierarchicalFinding(
                finding="Economic impacts are manageable",
                supporting_evidence=["Cost-benefit analysis"],
                confidence=ConfidenceLevel.LOW,
                confidence_score=0.4,
                importance=ImportanceLevel.MEDIUM,
                importance_score=0.5,
                source=ResearchSource(
                    title="Economic Report",
                    url="https://econ.com/report",
                    source_type="industry",
                    credibility_score=0.5,
                    date=datetime.now() - timedelta(days=90)
                ),
            ),
        ]

        contradictions = [
            Contradiction(
                finding_1_id="0",
                finding_2_id="1",
                contradiction_type="methodological",
                explanation="Direct observations vs model predictions differ",
                domain_overlap=0.9,
            ),
            Contradiction(
                finding_1_id="0",
                finding_2_id="2",
                contradiction_type="perspective",
                explanation="Scientific urgency vs economic optimism",
                domain_overlap=0.3,
            ),
        ]

        results = ResearchResults(
            query="Climate change impacts and economics",
            findings=findings,
            sources=[f.source for f in findings],
            contradictions=contradictions,
        )

        # Process all contradictions
        results.calculate_all_contradiction_severities()

        # Verify processing worked
        for contradiction in results.contradictions:
            assert hasattr(contradiction, 'severity_components')
            assert hasattr(contradiction, 'resolution_strategy')
            assert contradiction.severity > 0.0
            assert 1 <= contradiction.priority <= 5

        # Get comprehensive summary
        summary = results.get_contradiction_summary()
        assert summary["total_contradictions"] == 2
        assert summary["average_severity"] > 0.0

        # Verify different contradictions have different severities based on their characteristics
        methodological_contradiction = results.contradictions[0]
        perspective_contradiction = results.contradictions[1]

        # Methodological contradiction should be more severe due to higher domain overlap
        # and higher confidence/importance of involved findings
        assert methodological_contradiction.severity != perspective_contradiction.severity


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
