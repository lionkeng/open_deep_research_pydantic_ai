"""Unit tests for Phase 4 model enhancements to ResearchResults."""

import json
import math
import pytest
from datetime import datetime

from src.models.research_executor import (
    ResearchResults,
    HierarchicalFinding,
    ThemeCluster,
    PatternAnalysis,
    PatternType,
    ImportanceLevel,
    ConfidenceLevel,
    ResearchSource,
    Contradiction,
    ExecutiveSummary,
)


class TestResearchResultsEnhancements:
    """Test suite for Phase 4 enhancements to ResearchResults model."""

    @pytest.fixture
    def sample_findings(self):
        """Create sample research findings for testing."""
        return [
            HierarchicalFinding(
                finding="AI is transforming industries with machine learning adoption increasing",
                supporting_evidence=["Industry reports show 40% adoption rate", "Tech sector leading transformation"],
                confidence=ConfidenceLevel.HIGH,
                confidence_score=0.9,
                importance=ImportanceLevel.CRITICAL,
                importance_score=0.95,
                source=ResearchSource(
                    title="AI Industry Report 2024",
                    url="https://source1.com/article",
                    source_type="academic"
                ),
                category="AI Transformation",
            ),
            HierarchicalFinding(
                finding="Implementation costs are high and ROI takes time to materialize",
                supporting_evidence=["Average implementation cost $2M", "ROI timeline 2-3 years"],
                confidence=ConfidenceLevel.MEDIUM,
                confidence_score=0.7,
                importance=ImportanceLevel.HIGH,
                importance_score=0.8,
                source=ResearchSource(
                    title="Cost Analysis Study",
                    url="https://source3.com/study",
                    source_type="research"
                ),
                category="Economics",
                metadata={"has_contradictions": True}
            ),
            HierarchicalFinding(
                finding="Future looks promising with widespread adoption expected by 2030",
                supporting_evidence=["Market projections show 80% adoption", "Technology maturity increasing"],
                confidence=ConfidenceLevel.HIGH,
                confidence_score=0.8,
                importance=ImportanceLevel.MEDIUM,
                importance_score=0.6,
                source=ResearchSource(
                    title="Future Trends Report",
                    url="https://source4.com/report",
                    source_type="analysis"
                ),
                category="Future Outlook",
            ),
            HierarchicalFinding(
                finding="Historical development provides context for current trends",
                supporting_evidence=["Evolution from 1950s to present"],
                confidence=ConfidenceLevel.MEDIUM,
                confidence_score=0.5,
                importance=ImportanceLevel.LOW,
                importance_score=0.3,
                source=ResearchSource(
                    title="Historical Analysis",
                    url="https://archive.edu/thesis",
                    source_type="academic"
                ),
                category="History",
            ),
        ]

    @pytest.fixture
    def sample_themes(self, sample_findings):
        """Create sample theme clusters."""
        return [
            ThemeCluster(
                theme_name="AI Transformation",
                description="AI is revolutionizing various sectors with rapid adoption",
                coherence_score=0.85,
                importance_score=0.9,
                findings=sample_findings[:2],
            ),
            ThemeCluster(
                theme_name="Economic Impact",
                description="Financial implications of AI adoption",
                coherence_score=0.7,
                importance_score=0.75,
                findings=[sample_findings[1]],
            ),
        ]

    @pytest.fixture
    def sample_patterns(self):
        """Create sample pattern analyses."""
        return [
            PatternAnalysis(
                pattern_type=PatternType.TEMPORAL,
                pattern_name="AI Adoption Trend",
                description="Increasing AI adoption across industries",
                strength=0.9,
                finding_ids=["0", "1"],
                implications=["Need for skilled workers", "Infrastructure requirements"],
            ),
            PatternAnalysis(
                pattern_type=PatternType.CORRELATION,
                pattern_name="Investment-Size Correlation",
                description="Investment correlates with company size",
                strength=0.75,
                finding_ids=["1"],
                implications=["SMEs need different strategies"],
            ),
        ]

    @pytest.fixture
    def research_results(self, sample_findings, sample_themes, sample_patterns):
        """Create a complete ResearchResults instance."""
        return ResearchResults(
            query="AI transformation research",
            findings=sample_findings,
            sources=[
                ResearchSource(title="Source 1", url="https://source1.com/article", source_type="academic"),
                ResearchSource(title="Source 2", url="https://source2.com/research", source_type="research"),
                ResearchSource(title="Source 3", url="https://source3.com/study", source_type="research"),
                ResearchSource(title="Source 4", url="https://source4.com/report", source_type="analysis"),
                ResearchSource(title="Source 5", url="https://source5.org/paper", source_type="academic"),
                ResearchSource(title="Archive", url="https://archive.edu/thesis", source_type="academic"),
            ],
            key_insights=["AI is rapidly transforming industries with increasing adoption rates."],
            theme_clusters=sample_themes,
            patterns=sample_patterns,
            data_gaps=["Small business impact unclear", "Long-term effects unknown"],
            content_hierarchy={
                "priority_levels": {
                    "Critical": ["AI transformation findings"],
                    "Important": ["Cost considerations"],
                    "Supplementary": ["Future outlook"],
                    "Contextual": ["Historical background"],
                },
                "metadata": {"organization_method": "importance-based", "version": "1.0"},
            },
        )

    def test_content_hierarchy_field_exists(self, research_results):
        """Test that content_hierarchy field exists and works correctly."""
        assert hasattr(research_results, "content_hierarchy")
        assert isinstance(research_results.content_hierarchy, dict)
        assert "priority_levels" in research_results.content_hierarchy
        assert "metadata" in research_results.content_hierarchy

    def test_patterns_field_integration(self, research_results):
        """Test that patterns field is properly integrated."""
        assert hasattr(research_results, "patterns")
        assert isinstance(research_results.patterns, list)
        assert len(research_results.patterns) == 2
        assert all(isinstance(p, PatternAnalysis) for p in research_results.patterns)

    def test_get_hierarchical_structure(self, research_results):
        """Test the get_hierarchical_structure method."""
        hierarchy = research_results.get_hierarchical_structure()

        assert isinstance(hierarchy, dict)
        assert set(hierarchy.keys()) == {"Critical", "Important", "Supplementary", "Contextual"}
        assert len(hierarchy["Critical"]) == 1
        assert len(hierarchy["Important"]) == 1
        assert len(hierarchy["Supplementary"]) == 1
        assert len(hierarchy["Contextual"]) == 1

        # Check that findings are properly categorized
        assert hierarchy["Critical"][0].importance == ImportanceLevel.CRITICAL
        assert hierarchy["Important"][0].importance == ImportanceLevel.HIGH

    def test_get_convergence_rate(self, research_results):
        """Test the convergence rate calculation."""
        convergence_rate = research_results.get_convergence_rate()

        assert isinstance(convergence_rate, float)
        assert 0.0 <= convergence_rate <= 1.0
        # Based on supporting_evidence count (2+ items = convergence)
        # All 4 findings have 1-2 pieces of evidence, 2 have 2+
        assert convergence_rate == 0.5

    def test_get_contradiction_rate(self, research_results):
        """Test the contradiction rate calculation."""
        contradiction_rate = research_results.get_contradiction_rate()

        assert isinstance(contradiction_rate, float)
        assert 0.0 <= contradiction_rate <= 1.0
        # One finding has has_contradictions metadata flag
        # But no actual contradictions in the list, so rate depends on implementation

    def test_get_source_diversity(self, research_results):
        """Test Shannon entropy calculation for source diversity."""
        diversity = research_results.get_source_diversity()

        assert isinstance(diversity, float)
        assert diversity >= 0.0
        # Should have non-zero entropy with different domains
        assert diversity > 0

        # Test with uniform distribution (maximum entropy)
        uniform_results = ResearchResults(
            query="test",
            sources=[
                ResearchSource(title="A", url="https://a.com/1", source_type="web"),
                ResearchSource(title="B", url="https://b.com/2", source_type="web"),
                ResearchSource(title="C", url="https://c.com/3", source_type="web"),
                ResearchSource(title="D", url="https://d.com/4", source_type="web"),
            ]
        )
        uniform_diversity = uniform_results.get_source_diversity()
        assert uniform_diversity == 2.0  # log2(4) = 2.0

    def test_calculate_comprehensive_quality(self, research_results):
        """Test the comprehensive quality score calculation."""
        quality_score = research_results.calculate_comprehensive_quality()

        assert isinstance(quality_score, float)
        assert 0.0 <= quality_score <= 1.0

        # Score should consider multiple factors
        assert quality_score > 0.5  # Should be reasonably good with our sample data

    def test_export_to_json(self, research_results):
        """Test JSON export functionality."""
        json_export = research_results.export_to_json()

        assert isinstance(json_export, str)

        # Parse JSON to verify structure
        data = json.loads(json_export)

        assert "timestamp" in data
        assert "synthesis" in data
        assert "overall_quality_score" in data
        assert "metrics" in data
        assert "hierarchy" in data
        assert "patterns" in data
        assert "themes" in data

        # Check metrics
        metrics = data["metrics"]
        assert "convergence_rate" in metrics
        assert "contradiction_rate" in metrics
        assert "source_diversity" in metrics
        assert metrics["total_findings"] == 4
        assert metrics["total_sources"] == 6
        assert metrics["total_patterns"] == 2
        assert metrics["total_themes"] == 2

        # Check hierarchy structure
        hierarchy = data["hierarchy"]
        assert "Critical" in hierarchy
        assert len(hierarchy["Critical"]) == 1

    def test_to_report_comprehensive(self, research_results):
        """Test the enhanced to_report method generates all sections."""
        report = research_results.to_report()

        assert isinstance(report, str)

        # Check for all required sections
        required_sections = [
            "# Research Report",
            "## Executive Summary",
            "### Key Insights",
            "### Research Quality Metrics",
            "## Hierarchical Findings",
            "### Critical Findings",
            "### Important Findings",
            "### Supplementary Findings",
            "### Contextual Findings",
            "## Theme Clusters",
            "## Pattern Analysis",
            "## Contradictions and Resolutions",
            "## Confidence Metrics",
            "## Data Gaps and Limitations",
            "## Sources and Citations",
            "## Quality Metrics Summary",
            "## Metadata",
        ]

        for section in required_sections:
            assert section in report, f"Missing section: {section}"

        # Check for quality metrics in report
        assert "Overall Quality Score" in report
        assert "Source Diversity" in report
        assert "Finding Convergence" in report
        assert "Contradiction Rate" in report

        # Check for visual indicators
        assert "✅" in report or "⚠️" in report or "❌" in report

        # Check for timestamp
        assert datetime.now().strftime("%Y-%m-%d") in report

    def test_to_report_tables(self, research_results):
        """Test that markdown tables are properly formatted in report."""
        report = research_results.to_report()

        # Check for confidence metrics table
        assert "| Topic | Confidence |" in report
        assert "|-------|------------|" in report

        # Check for quality metrics summary table
        assert "| Metric | Value | Status |" in report
        assert "|--------|-------|--------|" in report

    def test_to_report_pattern_grouping(self, research_results):
        """Test that patterns are properly grouped by type in report."""
        report = research_results.to_report()

        assert "### Trend Patterns" in report
        assert "### Correlation Patterns" in report
        assert "Increasing AI adoption" in report
        assert "Investment correlates" in report

    def test_empty_research_results(self):
        """Test that methods handle empty ResearchResults gracefully."""
        empty_results = ResearchResults(query="Empty test query")

        assert empty_results.get_convergence_rate() == 0.0
        assert empty_results.get_contradiction_rate() == 0.0
        assert empty_results.get_source_diversity() == 0.0
        assert empty_results.calculate_comprehensive_quality() == 0.0

        hierarchy = empty_results.get_hierarchical_structure()
        assert all(len(findings) == 0 for findings in hierarchy.values())

        report = empty_results.to_report()
        assert isinstance(report, str)
        assert "# Research Report" in report

        json_export = empty_results.export_to_json()
        assert isinstance(json_export, str)
        data = json.loads(json_export)
        assert data["metrics"]["total_findings"] == 0

    def test_quality_score_weighting(self):
        """Test that quality score properly weights different factors."""
        # High diversity, high convergence, no contradictions
        good_results = ResearchResults(
            query="Test query",
            findings=[
                HierarchicalFinding(
                    finding="Point 1",
                    supporting_evidence=["Evidence 1", "Evidence 2"],
                    confidence=ConfidenceLevel.HIGH,
                    confidence_score=0.9,
                    importance=ImportanceLevel.CRITICAL,
                ),
                HierarchicalFinding(
                    finding="Point 2",
                    supporting_evidence=["Evidence 3", "Evidence 4"],
                    confidence=ConfidenceLevel.HIGH,
                    confidence_score=0.85,
                    importance=ImportanceLevel.HIGH,
                ),
            ],
            sources=[
                ResearchSource(title="A", url="https://a.com/1"),
                ResearchSource(title="B", url="https://b.com/2"),
                ResearchSource(title="C", url="https://c.com/3"),
                ResearchSource(title="D", url="https://d.com/4"),
            ],
            patterns=[PatternAnalysis(pattern_type="trend", description="Strong trend", strength=0.9, confidence=0.9)],
        )

        quality = good_results.calculate_comprehensive_quality()
        assert quality > 0.7  # Should be high quality

        # Low diversity, low convergence, high contradictions
        poor_results = ResearchResults(
            query="Poor quality test",
            findings=[
                HierarchicalFinding(
                    finding="Point 1",
                    supporting_evidence=["Weak evidence"],
                    confidence=ConfidenceLevel.LOW,
                    confidence_score=0.3,
                    importance=ImportanceLevel.MEDIUM,
                    metadata={"has_contradictions": True}
                ),
                HierarchicalFinding(
                    finding="Point 2",
                    supporting_evidence=["Limited evidence"],
                    confidence=ConfidenceLevel.LOW,
                    confidence_score=0.4,
                    importance=ImportanceLevel.LOW,
                    metadata={"has_contradictions": True}
                ),
            ],
            sources=[
                ResearchSource(title="A1", url="https://a.com/1"),
                ResearchSource(title="A2", url="https://a.com/2"),
            ],
        )

        poor_quality = poor_results.calculate_comprehensive_quality()
        assert poor_quality < 0.5  # Should be low quality
        assert poor_quality < quality  # Should be worse than good results

    def test_markdown_special_characters_handling(self, research_results):
        """Test that special characters are handled properly in markdown report."""
        # Add finding with special characters
        research_results.findings.append(
            HierarchicalFinding(
                finding="Test with *asterisks* and _underscores_ and [brackets] (parentheses)",
                supporting_evidence=["Special character test"],
                confidence=ConfidenceLevel.HIGH,
                confidence_score=0.8,
                importance=ImportanceLevel.HIGH,
            )
        )

        report = research_results.to_report()
        assert isinstance(report, str)
        # Should not break markdown formatting
        assert "# Research Report" in report

    def test_source_domain_extraction(self, research_results):
        """Test that source domains are properly extracted and grouped."""
        report = research_results.to_report()

        # Check that sources are grouped by domain
        assert "### source1.com" in report or "### source2.com" in report
        assert "### archive.edu" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
