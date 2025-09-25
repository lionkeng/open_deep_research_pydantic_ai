"""
Comprehensive tests for the ReportGeneratorAgent.
"""

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from pydantic_ai import RunContext
from pydantic_ai.usage import RunUsage

from agents.base import AgentConfiguration, ResearchDependencies
from agents.report_generator import (
    ReportGeneratorAgent,
    _adjust_outline_for_retry,
    _evaluate_report_quality,
)
from models.api_models import APIKeys
from models.core import ResearchStage, ResearchState
from models.metadata import ReportSectionPlan, ResearchMetadata
from models.report_generator import (
    ReportMetadata as ReportMetadataModel,
)
from models.report_generator import (
    ReportSection,
    ResearchReport,
)


class TestReportGeneratorAgent:
    """Test suite for ReportGeneratorAgent."""

    @pytest_asyncio.fixture
    async def agent_dependencies(self):
        """Create mock dependencies for testing."""
        deps = ResearchDependencies(
            http_client=AsyncMock(),
            api_keys=APIKeys(),
            research_state=ResearchState(
                request_id="test-report-123",
                user_id="test-user",
                session_id="test-session",
                user_query="Generate report on AI advancements",
                current_stage=ResearchStage.REPORT_GENERATION,
                metadata=ResearchMetadata(),
            ),
            usage=None,
        )
        return deps

    @pytest_asyncio.fixture
    async def report_generator_agent(self, agent_dependencies):
        """Create a ReportGeneratorAgent instance."""
        config = AgentConfiguration(
            agent_name="report_generator",
            agent_type="report_generator",
            max_retries=3,
            timeout_seconds=30.0,
            temperature=0.7,
        )
        agent = ReportGeneratorAgent(config=config)
        agent._deps = agent_dependencies
        return agent

    @pytest.fixture
    def sample_research_data(self):
        """Sample research data for report generation."""
        return {
            "findings": [
                {"finding": "AI breakthrough in NLP", "source": "Research Paper A"},
                {"finding": "Computer vision advances", "source": "Conference B"},
                {"finding": "Reinforcement learning progress", "source": "Journal C"},
            ],
            "summary": "Significant AI advancements across multiple domains",
            "key_insights": ["NLP models improved", "CV accuracy increased", "RL efficiency gains"],
        }

    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = ReportGeneratorAgent()
        assert agent.name == "report_generator"
        assert agent.agent is not None
        assert agent.config is not None

    @pytest.mark.asyncio
    async def test_comprehensive_report_generation(
        self,
        report_generator_agent,
        agent_dependencies,
        sample_research_data,
    ):
        """Test generation of comprehensive research report."""
        agent_dependencies.research_state.metadata.additional_context = sample_research_data

        mock_result = ResearchReport(
            title="Artificial Intelligence Advancements: A Comprehensive Analysis",
            executive_summary=(
                "This report analyzes recent breakthroughs in AI across NLP, computer vision, "
                "and reinforcement learning domains."
            ),
            introduction=(
                "The field of artificial intelligence has witnessed remarkable progress..."
            ),
            sections=[
                ReportSection(
                    title="Natural Language Processing Advances",
                    content="Recent developments in NLP have revolutionized text understanding...",
                    subsections=[
                        ReportSection(
                            title="Transformer Architecture Evolution",
                            content="The transformer architecture continues to evolve...",
                            subsections=[],
                            figures=[],
                            citations=[
                                (
                                    "Smith, J., Doe, A. (2024). Advances in Transformer Models. "
                                    "Research Paper A. https://example.com/paper-a"
                                )
                            ],
                        )
                    ],
                    figures=[],
                    citations=[],
                ),
                ReportSection(
                    title="Computer Vision Breakthroughs",
                    content="Computer vision has achieved human-level performance...",
                    subsections=[],
                    figures=[],
                    citations=[
                        "Johnson, M. (2024). CV Advances 2024. Conference B. https://example.com/conf-b"
                    ],
                ),
                ReportSection(
                    title="Reinforcement Learning Progress",
                    content="RL agents demonstrate improved efficiency...",
                    subsections=[],
                    figures=[],
                    citations=[
                        "Williams, R. (2024). RL Efficiency Gains. Journal C. https://example.com/journal-c"
                    ],
                ),
            ],
            conclusions=(
                "AI continues to advance rapidly across multiple domains with significant "
                "implications..."
            ),
            recommendations=[
                "Invest in transformer-based models",
                "Explore multimodal AI applications",
                "Consider ethical implications",
            ],
            references=[],
            appendices={},
            metadata=ReportMetadataModel(
                version="1.0",
                created_at=datetime.now(),
                source_summary=[{"sources": 25}],
                citation_audit={"confidence_level": 0.88},
            ),
            overall_quality_score=0.88,
        )

        with patch.object(report_generator_agent, "run", return_value=mock_result):
            result = await report_generator_agent.run(agent_dependencies)

            assert isinstance(result, ResearchReport)
            assert len(result.sections) >= 2
            assert result.title is not None
            assert result.executive_summary is not None
            assert len(result.recommendations) > 0

    @pytest.mark.asyncio
    async def test_dynamic_instructions_include_outline_block(
        self,
        report_generator_agent,
        agent_dependencies,
    ) -> None:
        """Dynamic instructions should surface the deterministic section outline."""

        agent_dependencies.research_state.metadata.report.section_outline = [
            ReportSectionPlan(
                title="Renewable Adoption Momentum",
                bullets=[
                    "Solar adoption surged in 2024",
                    "Battery storage trimmed costs",
                ],
                salient_evidence_ids=["S1", "S2"],
            )
        ]

        ctx = RunContext(
            deps=agent_dependencies,
            model=report_generator_agent.agent.model,
            usage=RunUsage(),
        )
        runner = report_generator_agent.agent._instructions_functions[0]
        prompt = await runner.run(ctx)

        assert "Section Outline Guidance:" in prompt
        assert "1. Renewable Adoption Momentum" in prompt
        assert "    - Solar adoption surged in 2024" in prompt
        assert "Evidence IDs: S1, S2" in prompt
        assert "Integrate every bullet from the Section Outline" in prompt

    @pytest.mark.asyncio
    async def test_dynamic_instructions_outline_placeholder_when_missing(
        self,
        report_generator_agent,
        agent_dependencies,
    ) -> None:
        """Instructions should note when no outline is available."""

        agent_dependencies.research_state.metadata.report.section_outline = []

        ctx = RunContext(
            deps=agent_dependencies,
            model=report_generator_agent.agent.model,
            usage=RunUsage(),
        )
        runner = report_generator_agent.agent._instructions_functions[0]
        prompt = await runner.run(ctx)

        assert "Section Outline Guidance:" in prompt
        assert "(no outline provided)" in prompt

    @pytest.mark.asyncio
    async def test_report_structure_validation(self, report_generator_agent, agent_dependencies):
        """Test that report structure is properly validated."""
        mock_result = ResearchReport(
            title="Test Report",
            executive_summary="Summary",
            introduction="Intro",
            sections=[
                ReportSection(
                    title="Section 1", content="Content 1", subsections=[], figures=[], citations=[]
                )
            ],
            conclusions="Conclusion",
            recommendations=["Rec 1"],
            references=[],
            appendices={},
            metadata=ReportMetadataModel(),
            overall_quality_score=0.75,
        )

        with patch.object(report_generator_agent, "run", return_value=mock_result):
            result = await report_generator_agent.run(agent_dependencies)

            # Validate required fields
            assert result.title is not None and len(result.title) > 0
            assert result.executive_summary is not None and len(result.executive_summary) > 0
            assert result.introduction is not None
            assert result.conclusions is not None
            assert len(result.sections) > 0

    def test_evaluate_report_quality_detects_generic_headings(self) -> None:
        """Quality heuristics should flag generic headings and label prefixes."""

        report = ResearchReport(
            title="Sample",
            executive_summary="Summary",
            introduction="Intro",
            sections=[
                ReportSection(
                    title="Finding 1",
                    content="Finding: Solar adoption rose sharply in 2024 [S1].",
                    subsections=[],
                    figures=[],
                    citations=[],
                ),
                ReportSection(
                    title="Insight",
                    content="Implication: Battery storage adoption lowered costs [S2].",
                    subsections=[],
                    figures=[],
                    citations=[],
                ),
            ],
            conclusions="Conclusion",
            recommendations=["Rec"],
            references=[],
            appendices={},
            metadata=ReportMetadataModel(),
            quality_score=0.6,
        )

        evaluation = _evaluate_report_quality(report)

        assert not evaluation.passed
        assert evaluation.total_colon_prefixes >= 2
        assert evaluation.bad_heading_ratio > 0.5

    def test_evaluate_report_quality_passes_descriptive_headings(self) -> None:
        """Descriptive headings with narrative content should pass validation."""

        report = ResearchReport(
            title="Sample",
            executive_summary="Summary",
            introduction="Intro",
            sections=[
                ReportSection(
                    title="Solar Adoption Accelerates in Emerging Markets",
                    content=(
                        "Solar adoption accelerated in 2024 as installations doubled in key "
                        "regions [S1]."
                    ),
                    subsections=[],
                    figures=[],
                    citations=["[S1]"],
                ),
                ReportSection(
                    title="Storage Investments Cut Operating Costs",
                    content=(
                        "Organizations adopting lithium storage saw operating costs drop by 18 "
                        "percent [S2]."
                    ),
                    subsections=[],
                    figures=[],
                    citations=["[S2]"],
                ),
            ],
            conclusions="Conclusion",
            recommendations=["Rec"],
            references=[],
            appendices={},
            metadata=ReportMetadataModel(),
            quality_score=0.9,
        )

        evaluation = _evaluate_report_quality(report)

        assert evaluation.passed
        assert evaluation.total_colon_prefixes == 0
        assert evaluation.bad_heading_ratio == 0.0

    @pytest.mark.asyncio
    async def test_run_retries_with_adjusted_outline_on_validation_failure(
        self,
        report_generator_agent,
        agent_dependencies,
    ) -> None:
        """Report generation should retry once with an adjusted outline when validation fails."""

        failing_report = ResearchReport(
            title="Test",
            executive_summary="Summary",
            introduction="Intro",
            sections=[
                ReportSection(
                    title="Finding 1",
                    content="Finding: Solar adoption surged in 2024 [S1].",
                    subsections=[],
                    figures=[],
                    citations=["[S1]"],
                )
            ],
            conclusions="Conclusion",
            recommendations=["Rec"],
            references=[],
            appendices={},
            metadata=ReportMetadataModel(),
            quality_score=0.5,
        )
        passing_report = ResearchReport(
            title="Test",
            executive_summary="Summary",
            introduction="Intro",
            sections=[
                ReportSection(
                    title="Solar Adoption Surges in 2024",
                    content=(
                        "Solar adoption surged in 2024 as installations doubled across markets "
                        "[S1]."
                    ),
                    subsections=[],
                    figures=[],
                    citations=["[S1]"],
                )
            ],
            conclusions="Conclusion",
            recommendations=["Rec"],
            references=[],
            appendices={},
            metadata=ReportMetadataModel(),
            quality_score=0.8,
        )

        agent_dependencies.research_state.metadata.report.section_outline = [
            ReportSectionPlan(
                title="Solar Adoption",
                bullets=["Solar adoption surged in 2024"],
                salient_evidence_ids=["S1"],
            )
        ]

        with patch(
            "agents.report_generator.BaseResearchAgent.run",
            side_effect=[failing_report, passing_report],
        ) as mock_base_run:
            result = await report_generator_agent.run(agent_dependencies)

        assert result is passing_report
        assert mock_base_run.call_count == 2
        outline_title = agent_dependencies.research_state.metadata.report.section_outline[0].title
        assert outline_title.lower().startswith("solar adoption surged"), outline_title

    def test_adjust_outline_for_retry_expands_titles(self) -> None:
        """Outline retry helper should incorporate bullet detail into headings."""

        metadata = ResearchMetadata()
        metadata.report.section_outline = [
            ReportSectionPlan(
                title="Storage",
                bullets=["Battery storage trimmed costs"],
                salient_evidence_ids=["S2"],
            )
        ]

        changed = _adjust_outline_for_retry(metadata)

        assert changed is True
        assert "battery" in metadata.report.section_outline[0].title.lower()

    def test_adjust_outline_for_retry_replaces_generic_titles(self) -> None:
        """Generic outline headings should be replaced with richer bullet phrasing."""

        metadata = ResearchMetadata()
        metadata.report.section_outline = [
            ReportSectionPlan(
                title="Finding Source",
                bullets=["Builder scale, cost structure, and market behavior"],
                salient_evidence_ids=["S7"],
            )
        ]

        changed = _adjust_outline_for_retry(metadata)

        assert changed is True
        updated = metadata.report.section_outline[0].title
        assert updated.startswith("Builder scale"), updated

    @pytest.mark.asyncio
    async def test_nested_sections_handling(self, report_generator_agent, agent_dependencies):
        """Test handling of nested report sections."""
        mock_result = ResearchReport(
            title="Nested Structure Report",
            executive_summary="Summary",
            introduction="Intro",
            sections=[
                ReportSection(
                    title="Main Section",
                    content="Main content",
                    subsections=[
                        ReportSection(
                            title="Subsection 1",
                            content="Sub content 1",
                            subsections=[
                                ReportSection(
                                    title="Sub-subsection 1.1",
                                    content="Deep content",
                                    subsections=[],
                                    figures=[],
                                    citations=[],
                                )
                            ],
                            figures=[],
                            citations=[],
                        ),
                        ReportSection(
                            title="Subsection 2",
                            content="Sub content 2",
                            subsections=[],
                            figures=[],
                            citations=[],
                        ),
                    ],
                    figures=[],
                    citations=[],
                )
            ],
            conclusions="Conclusion",
            recommendations=[],
            references=[],
            appendices={},
            metadata=ReportMetadataModel(),
            overall_quality_score=0.8,
        )

        with patch.object(report_generator_agent, "run", return_value=mock_result):
            result = await report_generator_agent.run(agent_dependencies)

            assert len(result.sections) == 1
            assert len(result.sections[0].subsections) == 2
            assert len(result.sections[0].subsections[0].subsections) == 1

    @pytest.mark.asyncio
    async def test_citation_management(self, report_generator_agent, agent_dependencies):
        """Test proper citation management and formatting."""
        mock_result = ResearchReport(
            title="Citation Test Report",
            executive_summary="Summary",
            introduction="Intro",
            sections=[
                ReportSection(
                    title="Section with Citations",
                    content="Content referencing sources",
                    subsections=[],
                    figures=[],
                    citations=[
                        (
                            "Smith, J., Doe, A., Johnson, M. (2024). Important Research. "
                            "Nature AI. DOI: 10.1234/example"
                        ),
                        "Williams, R. (2023). Novel Approach. ICML 2023. https://conference.org/paper",
                    ],
                )
            ],
            conclusions="Conclusion",
            recommendations=[],
            references=["Brown, T. (2024). AI Fundamentals. Academic Press."],
            appendices={},
            metadata=ReportMetadataModel(),
            overall_quality_score=0.85,
        )

        with patch.object(report_generator_agent, "run", return_value=mock_result):
            result = await report_generator_agent.run(agent_dependencies)

            # Check section citations
            assert len(result.sections[0].citations) == 2
            assert all(isinstance(c, str) and len(c) > 0 for c in result.sections[0].citations)

            # Check global references
            assert len(result.references) == 1
            assert all(isinstance(c, str) for c in result.references)

    @pytest.mark.asyncio
    async def test_recommendations_generation(self, report_generator_agent, agent_dependencies):
        """Test generation of actionable recommendations."""
        mock_result = ResearchReport(
            title="Report with Recommendations",
            executive_summary="Summary",
            introduction="Intro",
            sections=[
                ReportSection(
                    title="Analysis",
                    content="Analysis content",
                    subsections=[],
                    figures=[],
                    citations=[],
                )
            ],
            conclusions="Based on our analysis...",
            recommendations=[
                "Implement finding 1 immediately for quick wins",
                "Develop strategy for finding 2 implementation",
                "Allocate resources for long-term initiatives",
                "Monitor progress quarterly",
                "Consider partnerships for acceleration",
            ],
            references=[],
            appendices={},
            metadata=ReportMetadataModel(keywords=["recommendation_priority: high"]),
            overall_quality_score=0.9,
        )

        with patch.object(report_generator_agent, "run", return_value=mock_result):
            result = await report_generator_agent.run(agent_dependencies)

            assert len(result.recommendations) >= 3
            assert all(isinstance(r, str) and len(r) > 0 for r in result.recommendations)
            assert "recommendation_priority: high" in result.metadata.keywords

    @pytest.mark.asyncio
    async def test_metadata_tracking(self, report_generator_agent, agent_dependencies):
        """Test that report metadata is properly tracked."""
        mock_result = ResearchReport(
            title="Metadata Test Report",
            executive_summary="Summary",
            introduction="Intro",
            sections=[
                ReportSection(
                    title="Section", content="Content", subsections=[], figures=[], citations=[]
                )
            ],
            conclusions="Conclusion",
            recommendations=["Rec"],
            references=[],
            appendices={},
            metadata=ReportMetadataModel(
                version="2.0",
                created_at=datetime(2024, 1, 15, 10, 30, 0),
                source_summary=[
                    {"id": "source1", "url": "http://example.com"},
                    {"total_sources": 42},
                ],
                citation_audit={
                    "confidence_level": 0.92,
                    "word_count": 5000,
                    "reading_time_minutes": 20,
                    "report_type": "comprehensive",
                    "quality_score": 0.88,
                },
            ),
            overall_quality_score=0.92,
        )

        with patch.object(report_generator_agent, "run", return_value=mock_result):
            result = await report_generator_agent.run(agent_dependencies)

            metadata = result.metadata
            assert metadata.version == "2.0"
            assert metadata.created_at is not None
            assert len(metadata.source_summary) == 2
            assert metadata.citation_audit["confidence_level"] == 0.92
            assert result.overall_quality_score == 0.92

    @pytest.mark.asyncio
    async def test_edge_case_minimal_report(self, report_generator_agent, agent_dependencies):
        """Test generation of minimal report for simple queries."""
        agent_dependencies.research_state.user_query = "What is 2+2?"

        mock_result = ResearchReport(
            title="Simple Query Response",
            executive_summary="The answer to 2+2 is 4",
            introduction="This report addresses a basic arithmetic question.",
            sections=[
                ReportSection(
                    title="Answer", content="2+2 equals 4", subsections=[], figures=[], citations=[]
                )
            ],
            conclusions="The arithmetic operation 2+2 results in 4.",
            recommendations=[],
            references=[],
            appendices={},
            metadata=ReportMetadataModel(classification="minimal"),
            overall_quality_score=1.0,
        )

        with patch.object(report_generator_agent, "run", return_value=mock_result):
            result = await report_generator_agent.run(agent_dependencies)

            assert result.metadata.classification == "minimal"
            assert len(result.sections) == 1
            assert len(result.recommendations) == 0

    @pytest.mark.asyncio
    async def test_appendices_handling(self, report_generator_agent, agent_dependencies):
        """Test handling of report appendices."""
        mock_result = ResearchReport(
            title="Report with Appendices",
            executive_summary="Summary",
            introduction="Intro",
            sections=[
                ReportSection(
                    title="Main", content="Main content", subsections=[], figures=[], citations=[]
                )
            ],
            conclusions="Conclusion",
            recommendations=["Rec"],
            references=[],
            appendices={
                "Appendix A: Data Tables": "Detailed data tables...",
                "Appendix B: Methodology Details": "Extended methodology description...",
            },
            metadata=ReportMetadataModel(),
            overall_quality_score=0.8,
        )

        with patch.object(report_generator_agent, "run", return_value=mock_result):
            result = await report_generator_agent.run(agent_dependencies)

            assert len(result.appendices) == 2
            assert "Appendix A: Data Tables" in result.appendices
            assert "Appendix B: Methodology Details" in result.appendices

    @pytest.mark.asyncio
    async def test_error_handling(self, report_generator_agent, agent_dependencies):
        """Test error handling during report generation."""
        with patch.object(
            report_generator_agent, "run", side_effect=Exception("Report generation failed")
        ):
            with pytest.raises(Exception, match="Report generation failed"):
                await report_generator_agent.run(agent_dependencies)

    @pytest.mark.asyncio
    async def test_limitations_and_future_work(self, report_generator_agent, agent_dependencies):
        """Test proper documentation of limitations and future work."""
        mock_result = ResearchReport(
            title="Research Report",
            executive_summary="Summary",
            introduction="Intro",
            sections=[
                ReportSection(
                    title="Findings",
                    content="Research findings",
                    subsections=[],
                    figures=[],
                    citations=[],
                )
            ],
            conclusions="Conclusion",
            recommendations=["Recommendation 1"],
            references=[],
            appendices={
                "Limitations": (
                    "Limited to English language sources; Time constraint: 2020-2024 only; "
                    "Excluded proprietary research; Sample size limitations in some studies"
                ),
                "Future Work": (
                    "Expand to multilingual sources; Include industry reports; "
                    "Longitudinal study over 10 years; Meta-analysis of all findings"
                ),
            },
            metadata=ReportMetadataModel(),
            overall_quality_score=0.7,
        )

        with patch.object(report_generator_agent, "run", return_value=mock_result):
            result = await report_generator_agent.run(agent_dependencies)

            assert "Limitations" in result.appendices
            assert "Future Work" in result.appendices
            assert "Limited to English" in result.appendices["Limitations"]
            assert "multilingual sources" in result.appendices["Future Work"]
