"""
Comprehensive tests for the ReportGeneratorAgent.
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.report_generator import ReportGeneratorAgent
from src.agents.base import ResearchDependencies, AgentConfiguration
from src.models.report_generator import ResearchReport, ReportSection, Citation
from src.models.api_models import APIKeys, ResearchMetadata
from src.models.core import ResearchState, ResearchStage


class TestReportGeneratorAgent:
    """Test suite for ReportGeneratorAgent."""

    @pytest.fixture
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
                current_stage=ResearchStage.REPORT_GENERATION
            ),
            metadata=ResearchMetadata(),
            usage=None
        )
        return deps

    @pytest.fixture
    def report_generator_agent(self, agent_dependencies):
        """Create a ReportGeneratorAgent instance."""
        config = AgentConfiguration(
            max_retries=3,
            timeout_seconds=30.0,
            temperature=0.7
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
                {"finding": "Reinforcement learning progress", "source": "Journal C"}
            ],
            "summary": "Significant AI advancements across multiple domains",
            "key_insights": ["NLP models improved", "CV accuracy increased", "RL efficiency gains"]
        }

    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = ReportGeneratorAgent()
        assert agent.name == "report_generator"
        assert agent.agent is not None
        assert agent.config is not None

    @pytest.mark.asyncio
    async def test_comprehensive_report_generation(self, report_generator_agent, agent_dependencies, sample_research_data):
        """Test generation of comprehensive research report."""
        agent_dependencies.metadata.additional_context = sample_research_data

        mock_result = MagicMock()
        mock_result.data = ResearchReport(
            title="Artificial Intelligence Advancements: A Comprehensive Analysis",
            executive_summary="This report analyzes recent breakthroughs in AI across NLP, computer vision, and reinforcement learning domains.",
            introduction="The field of artificial intelligence has witnessed remarkable progress...",
            sections=[
                ReportSection(
                    title="Natural Language Processing Advances",
                    content="Recent developments in NLP have revolutionized text understanding...",
                    subsections=[
                        ReportSection(
                            title="Transformer Architecture Evolution",
                            content="The transformer architecture continues to evolve...",
                            subsections=[],
                            key_findings=["GPT-4 capabilities", "Multimodal understanding"],
                            citations=[
                                Citation(
                                    source="Research Paper A",
                                    title="Advances in Transformer Models",
                                    authors=["Smith, J.", "Doe, A."],
                                    year=2024,
                                    url="https://example.com/paper-a"
                                )
                            ]
                        )
                    ],
                    key_findings=["Improved language understanding", "Better context retention"],
                    citations=[]
                ),
                ReportSection(
                    title="Computer Vision Breakthroughs",
                    content="Computer vision has achieved human-level performance...",
                    subsections=[],
                    key_findings=["Object detection accuracy", "Real-time processing"],
                    citations=[
                        Citation(
                            source="Conference B",
                            title="CV Advances 2024",
                            authors=["Johnson, M."],
                            year=2024,
                            url="https://example.com/conf-b"
                        )
                    ]
                ),
                ReportSection(
                    title="Reinforcement Learning Progress",
                    content="RL agents demonstrate improved efficiency...",
                    subsections=[],
                    key_findings=["Sample efficiency improved", "Generalization enhanced"],
                    citations=[
                        Citation(
                            source="Journal C",
                            title="RL Efficiency Gains",
                            authors=["Williams, R."],
                            year=2024,
                            url="https://example.com/journal-c"
                        )
                    ]
                )
            ],
            conclusion="AI continues to advance rapidly across multiple domains with significant implications...",
            recommendations=[
                "Invest in transformer-based models",
                "Explore multimodal AI applications",
                "Consider ethical implications"
            ],
            methodology="Systematic literature review and expert analysis",
            limitations=["Limited to publicly available research", "Rapid field evolution"],
            future_work=["Extended analysis of emerging architectures", "Industry application studies"],
            appendices=[],
            citations=[],
            metadata={
                "report_version": "1.0",
                "generation_date": datetime.now().isoformat(),
                "total_sources": 25,
                "confidence_level": 0.88
            }
        )

        with patch.object(report_generator_agent.agent, 'run', return_value=mock_result):
            result = await report_generator_agent.execute(agent_dependencies)

            assert isinstance(result, ResearchReport)
            assert len(result.sections) >= 2
            assert result.title is not None
            assert result.executive_summary is not None
            assert len(result.recommendations) > 0

    @pytest.mark.asyncio
    async def test_report_structure_validation(self, report_generator_agent, agent_dependencies):
        """Test that report structure is properly validated."""
        mock_result = MagicMock()
        mock_result.data = ResearchReport(
            title="Test Report",
            executive_summary="Summary",
            introduction="Intro",
            sections=[
                ReportSection(
                    title="Section 1",
                    content="Content 1",
                    subsections=[],
                    key_findings=["Finding 1"],
                    citations=[]
                )
            ],
            conclusion="Conclusion",
            recommendations=["Rec 1"],
            methodology="Method",
            limitations=["Limit 1"],
            future_work=["Future 1"],
            appendices=[],
            citations=[],
            metadata={}
        )

        with patch.object(report_generator_agent.agent, 'run', return_value=mock_result):
            result = await report_generator_agent.execute(agent_dependencies)

            # Validate required fields
            assert result.title is not None and len(result.title) > 0
            assert result.executive_summary is not None and len(result.executive_summary) > 0
            assert result.introduction is not None
            assert result.conclusion is not None
            assert len(result.sections) > 0

    @pytest.mark.asyncio
    async def test_nested_sections_handling(self, report_generator_agent, agent_dependencies):
        """Test handling of nested report sections."""
        mock_result = MagicMock()
        mock_result.data = ResearchReport(
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
                                    key_findings=["Deep finding"],
                                    citations=[]
                                )
                            ],
                            key_findings=["Sub finding 1"],
                            citations=[]
                        ),
                        ReportSection(
                            title="Subsection 2",
                            content="Sub content 2",
                            subsections=[],
                            key_findings=["Sub finding 2"],
                            citations=[]
                        )
                    ],
                    key_findings=["Main finding"],
                    citations=[]
                )
            ],
            conclusion="Conclusion",
            recommendations=[],
            methodology="Method",
            limitations=[],
            future_work=[],
            appendices=[],
            citations=[],
            metadata={}
        )

        with patch.object(report_generator_agent.agent, 'run', return_value=mock_result):
            result = await report_generator_agent.execute(agent_dependencies)

            assert len(result.sections) == 1
            assert len(result.sections[0].subsections) == 2
            assert len(result.sections[0].subsections[0].subsections) == 1

    @pytest.mark.asyncio
    async def test_citation_management(self, report_generator_agent, agent_dependencies):
        """Test proper citation management and formatting."""
        mock_result = MagicMock()
        mock_result.data = ResearchReport(
            title="Citation Test Report",
            executive_summary="Summary",
            introduction="Intro",
            sections=[
                ReportSection(
                    title="Section with Citations",
                    content="Content referencing sources",
                    subsections=[],
                    key_findings=["Finding 1"],
                    citations=[
                        Citation(
                            source="Journal Article",
                            title="Important Research",
                            authors=["Smith, J.", "Doe, A.", "Johnson, M."],
                            year=2024,
                            url="https://doi.org/10.1234/example",
                            doi="10.1234/example",
                            publication="Nature AI"
                        ),
                        Citation(
                            source="Conference Paper",
                            title="Novel Approach",
                            authors=["Williams, R."],
                            year=2023,
                            url="https://conference.org/paper",
                            conference="ICML 2023"
                        )
                    ]
                )
            ],
            conclusion="Conclusion",
            recommendations=[],
            methodology="Method",
            limitations=[],
            future_work=[],
            appendices=[],
            citations=[
                Citation(
                    source="Book",
                    title="AI Fundamentals",
                    authors=["Brown, T."],
                    year=2024,
                    publisher="Academic Press"
                )
            ],
            metadata={}
        )

        with patch.object(report_generator_agent.agent, 'run', return_value=mock_result):
            result = await report_generator_agent.execute(agent_dependencies)

            # Check section citations
            assert len(result.sections[0].citations) == 2
            assert all(c.source is not None for c in result.sections[0].citations)
            assert all(c.title is not None for c in result.sections[0].citations)
            assert all(c.year is not None for c in result.sections[0].citations)

            # Check global citations
            assert len(result.citations) == 1

    @pytest.mark.asyncio
    async def test_recommendations_generation(self, report_generator_agent, agent_dependencies):
        """Test generation of actionable recommendations."""
        mock_result = MagicMock()
        mock_result.data = ResearchReport(
            title="Report with Recommendations",
            executive_summary="Summary",
            introduction="Intro",
            sections=[
                ReportSection(
                    title="Analysis",
                    content="Analysis content",
                    subsections=[],
                    key_findings=["Finding 1", "Finding 2"],
                    citations=[]
                )
            ],
            conclusion="Based on our analysis...",
            recommendations=[
                "Implement finding 1 immediately for quick wins",
                "Develop strategy for finding 2 implementation",
                "Allocate resources for long-term initiatives",
                "Monitor progress quarterly",
                "Consider partnerships for acceleration"
            ],
            methodology="Method",
            limitations=[],
            future_work=[],
            appendices=[],
            citations=[],
            metadata={"recommendation_priority": "high"}
        )

        with patch.object(report_generator_agent.agent, 'run', return_value=mock_result):
            result = await report_generator_agent.execute(agent_dependencies)

            assert len(result.recommendations) >= 3
            assert all(isinstance(r, str) and len(r) > 0 for r in result.recommendations)
            assert result.metadata.get("recommendation_priority") == "high"

    @pytest.mark.asyncio
    async def test_metadata_tracking(self, report_generator_agent, agent_dependencies):
        """Test that report metadata is properly tracked."""
        mock_result = MagicMock()
        mock_result.data = ResearchReport(
            title="Metadata Test Report",
            executive_summary="Summary",
            introduction="Intro",
            sections=[
                ReportSection(
                    title="Section",
                    content="Content",
                    subsections=[],
                    key_findings=["Finding"],
                    citations=[]
                )
            ],
            conclusion="Conclusion",
            recommendations=["Rec"],
            methodology="Method",
            limitations=["Limit"],
            future_work=["Future"],
            appendices=[],
            citations=[],
            metadata={
                "report_version": "2.0",
                "generation_date": "2024-01-15T10:30:00",
                "total_sources": 42,
                "confidence_level": 0.92,
                "word_count": 5000,
                "reading_time_minutes": 20,
                "report_type": "comprehensive",
                "quality_score": 0.88
            }
        )

        with patch.object(report_generator_agent.agent, 'run', return_value=mock_result):
            result = await report_generator_agent.execute(agent_dependencies)

            metadata = result.metadata
            assert "report_version" in metadata
            assert "generation_date" in metadata
            assert metadata["total_sources"] == 42
            assert 0.0 <= metadata["confidence_level"] <= 1.0
            assert metadata.get("word_count", 0) > 0

    @pytest.mark.asyncio
    async def test_edge_case_minimal_report(self, report_generator_agent, agent_dependencies):
        """Test generation of minimal report for simple queries."""
        agent_dependencies.research_state.user_query = "What is 2+2?"

        mock_result = MagicMock()
        mock_result.data = ResearchReport(
            title="Simple Query Response",
            executive_summary="The answer to 2+2 is 4",
            introduction="This report addresses a basic arithmetic question.",
            sections=[
                ReportSection(
                    title="Answer",
                    content="2+2 equals 4",
                    subsections=[],
                    key_findings=["2+2=4"],
                    citations=[]
                )
            ],
            conclusion="The arithmetic operation 2+2 results in 4.",
            recommendations=[],
            methodology="Basic arithmetic",
            limitations=["N/A"],
            future_work=[],
            appendices=[],
            citations=[],
            metadata={"report_type": "minimal"}
        )

        with patch.object(report_generator_agent.agent, 'run', return_value=mock_result):
            result = await report_generator_agent.execute(agent_dependencies)

            assert result.metadata.get("report_type") == "minimal"
            assert len(result.sections) == 1
            assert len(result.recommendations) == 0

    @pytest.mark.asyncio
    async def test_appendices_handling(self, report_generator_agent, agent_dependencies):
        """Test handling of report appendices."""
        mock_result = MagicMock()
        mock_result.data = ResearchReport(
            title="Report with Appendices",
            executive_summary="Summary",
            introduction="Intro",
            sections=[
                ReportSection(
                    title="Main",
                    content="Main content",
                    subsections=[],
                    key_findings=["Finding"],
                    citations=[]
                )
            ],
            conclusion="Conclusion",
            recommendations=["Rec"],
            methodology="Method",
            limitations=["Limit"],
            future_work=["Future"],
            appendices=[
                ReportSection(
                    title="Appendix A: Data Tables",
                    content="Detailed data tables...",
                    subsections=[],
                    key_findings=[],
                    citations=[]
                ),
                ReportSection(
                    title="Appendix B: Methodology Details",
                    content="Extended methodology description...",
                    subsections=[],
                    key_findings=[],
                    citations=[]
                )
            ],
            citations=[],
            metadata={}
        )

        with patch.object(report_generator_agent.agent, 'run', return_value=mock_result):
            result = await report_generator_agent.execute(agent_dependencies)

            assert len(result.appendices) == 2
            assert all("Appendix" in a.title for a in result.appendices)

    @pytest.mark.asyncio
    async def test_error_handling(self, report_generator_agent, agent_dependencies):
        """Test error handling during report generation."""
        with patch.object(report_generator_agent.agent, 'run', side_effect=Exception("Report generation failed")):
            with pytest.raises(Exception, match="Report generation failed"):
                await report_generator_agent.execute(agent_dependencies)

    @pytest.mark.asyncio
    async def test_limitations_and_future_work(self, report_generator_agent, agent_dependencies):
        """Test proper documentation of limitations and future work."""
        mock_result = MagicMock()
        mock_result.data = ResearchReport(
            title="Research Report",
            executive_summary="Summary",
            introduction="Intro",
            sections=[
                ReportSection(
                    title="Findings",
                    content="Research findings",
                    subsections=[],
                    key_findings=["Finding 1"],
                    citations=[]
                )
            ],
            conclusion="Conclusion",
            recommendations=["Recommendation 1"],
            methodology="Systematic review",
            limitations=[
                "Limited to English language sources",
                "Time constraint: 2020-2024 only",
                "Excluded proprietary research",
                "Sample size limitations in some studies"
            ],
            future_work=[
                "Expand to multilingual sources",
                "Include industry reports",
                "Longitudinal study over 10 years",
                "Meta-analysis of all findings"
            ],
            appendices=[],
            citations=[],
            metadata={}
        )

        with patch.object(report_generator_agent.agent, 'run', return_value=mock_result):
            result = await report_generator_agent.execute(agent_dependencies)

            assert len(result.limitations) >= 2
            assert len(result.future_work) >= 2
            assert all(isinstance(l, str) for l in result.limitations)
            assert all(isinstance(f, str) for f in result.future_work)
