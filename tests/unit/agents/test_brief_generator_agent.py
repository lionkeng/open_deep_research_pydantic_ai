"""
Comprehensive tests for the BriefGeneratorAgent.
"""

import asyncio
import pytest
import pytest_asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.brief_generator import BriefGeneratorAgent
from src.agents.base import ResearchDependencies, AgentConfiguration
from src.models.api_models import APIKeys
from src.models.brief_generator import ResearchBrief, ResearchObjective, ResearchMethodology
from src.models.metadata import ResearchMetadata
from src.models.core import ResearchState, ResearchStage


class TestBriefGeneratorAgent:
    """Test suite for BriefGeneratorAgent."""

    @pytest.fixture
    def agent_dependencies(self):
        """Create mock dependencies for testing."""
        deps = ResearchDependencies(
            http_client=AsyncMock(),
            api_keys=APIKeys(),
            research_state=ResearchState(
                request_id="test-789",
                user_id="test-user",
                session_id="test-session",
                user_query="Research renewable energy technologies",
                current_stage=ResearchStage.BRIEF_GENERATION,
                metadata=ResearchMetadata()
            ),
            usage=None
        )
        return deps

    @pytest.fixture
    def brief_generator_agent(self, agent_dependencies):
        """Create a BriefGeneratorAgent instance."""
        config = AgentConfiguration(
            agent_name="brief_generator",
            agent_type="brief_generator",
            max_retries=3,
            timeout_seconds=30.0,
            temperature=0.7
        )
        agent = BriefGeneratorAgent(config=config)
        agent._deps = agent_dependencies
        return agent

    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = BriefGeneratorAgent()
        assert agent.name == "brief_generator"
        assert agent.agent is not None
        assert agent.config is not None

    @pytest.mark.asyncio
    async def test_comprehensive_brief_generation(self, brief_generator_agent, agent_dependencies):
        """Test generation of comprehensive research brief."""
        brief_data = ResearchBrief(
            title="Comprehensive Analysis of Renewable Energy Technologies",
            executive_summary="This research examines current renewable energy technologies including solar, wind, and hydroelectric power, analyzing their efficiency, costs, and future potential.",
            objectives=[
                ResearchObjective(
                    objective="Analyze current renewable energy technologies",
                    priority=5,
                    success_criteria="Comprehensive coverage of major technologies"
                ),
                ResearchObjective(
                    objective="Compare efficiency and cost metrics",
                    priority=4,
                    success_criteria="Quantitative comparison with data"
                ),
                ResearchObjective(
                    objective="Assess future potential and trends",
                    priority=4,
                    success_criteria="Evidence-based projections"
                )
            ],
            scope="Global renewable energy market with focus on solar, wind, and hydroelectric technologies from 2020-2024",
            methodology=ResearchMethodology(
                approach="Literature review, data analysis, expert interviews, and case studies",
                data_sources=["Energy databases", "Industry reports", "Academic papers"],
                analysis_methods=["Comparative analysis", "Trend analysis", "Cost-benefit analysis"],
                quality_checks=["Data validation", "Source verification", "Expert review"]
            ),
            constraints=[
                "Temporal: Focus on 2020-2024 data",
                "Geographic: Global scope with emphasis on leading markets"
            ],
            deliverables=["Comprehensive report", "Executive summary", "Data visualizations"],
            timeline_estimate="2 weeks for complete research",
            success_metrics=["Coverage of major technologies", "Data-driven insights", "Actionable recommendations"]
        )

        mock_result = MagicMock()
        mock_result.output = brief_data
        mock_result.usage = MagicMock(return_value=None)

        with patch.object(brief_generator_agent.agent, 'run', return_value=mock_result):
            result = await brief_generator_agent.run(deps=agent_dependencies)

            assert result is not None
            assert isinstance(result, ResearchBrief)
            assert len(result.objectives) >= 2
            assert result.title is not None and len(result.title) > 0
            assert result.methodology is not None

    @pytest.mark.asyncio
    async def test_simple_query_brief(self, brief_generator_agent, agent_dependencies):
        """Test brief generation for simple queries."""
        agent_dependencies.research_state.user_query = "What is Bitcoin?"

        brief_data = ResearchBrief(
            title="Understanding Bitcoin: Digital Currency Overview",
            executive_summary="Research brief on Bitcoin cryptocurrency, its technology, and market dynamics.",
            objectives=[
                ResearchObjective(
                    objective="Explain Bitcoin technology",
                    priority=5,
                    success_criteria="Clear technical explanation"
                ),
                ResearchObjective(
                    objective="Analyze market dynamics",
                    priority=4,
                    success_criteria="Current market analysis"
                )
            ],
            scope="Bitcoin technology and market overview",
            methodology=ResearchMethodology(
                approach="Literature review and market data analysis",
                data_sources=["Public sources", "Market data"],
                analysis_methods=["Technical analysis", "Market analysis"],
                quality_checks=["Data verification"]
            ),
            constraints=[],
            deliverables=["Research summary"],
            timeline_estimate="1 day",
            success_metrics=["Clear explanation", "Accurate data"]
        )

        mock_result = MagicMock()
        mock_result.output = brief_data
        mock_result.usage = MagicMock(return_value=None)

        with patch.object(brief_generator_agent.agent, 'run', return_value=mock_result):
            result = await brief_generator_agent.run(deps=agent_dependencies)

            assert result is not None
            assert "Bitcoin" in result.title
            assert len(result.objectives) >= 1
            assert result.scope is not None

    @pytest.mark.asyncio
    async def test_objectives_generation(self, brief_generator_agent, agent_dependencies):
        """Test that objectives are properly generated with importance levels."""
        brief_data = ResearchBrief(
            title="Test Research Brief",
            executive_summary="Test summary",
            objectives=[
                ResearchObjective(
                    objective="Primary objective",
                    priority=5,
                    success_criteria="Must achieve"
                ),
                ResearchObjective(
                    objective="Secondary objective",
                    priority=4,
                    success_criteria="Should achieve"
                ),
                ResearchObjective(
                    objective="Tertiary objective",
                    priority=3,
                    success_criteria="Nice to have"
                )
            ],
            scope="Test scope",
            methodology=ResearchMethodology(
                approach="Test methodology",
                data_sources=[],
                analysis_methods=[],
                quality_checks=[]
            ),
            constraints=[],
            deliverables=["Report"],
            timeline_estimate="1 week",
            success_metrics=["Metric 1"],
        )

        mock_result = MagicMock()
        mock_result.output = brief_data
        mock_result.usage = MagicMock(return_value=None)

        with patch.object(brief_generator_agent.agent, 'run', return_value=mock_result):
            result = await brief_generator_agent.run(deps=agent_dependencies)

            assert result is not None
            assert len(result.objectives) == 3
            # Check objectives are properly ordered by priority
            assert result.objectives[0].priority == 5
            assert result.objectives[1].priority == 4
            assert result.objectives[2].priority == 3

    @pytest.mark.asyncio
    async def test_edge_case_minimal_brief(self, brief_generator_agent, agent_dependencies):
        """Test generation of minimal brief for edge cases."""
        agent_dependencies.research_state.user_query = "?"

        brief_data = ResearchBrief(
            title="Unclear Research Query",
            executive_summary="Unable to generate comprehensive brief due to unclear query",
            objectives=[
                ResearchObjective(
                    objective="Clarify research intent",
                    priority=5,
                    success_criteria="Clear research question defined"
                )
            ],
            scope="To be determined",
            methodology=ResearchMethodology(
                approach="To be determined after clarification",
                data_sources=[],
                analysis_methods=[],
                quality_checks=[]
            ),
            constraints=[],
            deliverables=["TBD"],
            timeline_estimate="TBD",
            success_metrics=["Clear research direction"],
        )

        mock_result = MagicMock()
        mock_result.output = brief_data
        mock_result.usage = MagicMock(return_value=None)

        with patch.object(brief_generator_agent.agent, 'run', return_value=mock_result):
            result = await brief_generator_agent.run(deps=agent_dependencies)

            assert result is not None
            assert result.methodology is not None

    @pytest.mark.asyncio
    async def test_error_handling(self, brief_generator_agent, agent_dependencies):
        """Test error handling during brief generation."""
        with patch.object(brief_generator_agent.agent, 'run', side_effect=Exception("Generation failed")):
            with pytest.raises(Exception, match="Generation failed"):
                await brief_generator_agent.run(deps=agent_dependencies)

    @pytest.mark.asyncio
    async def test_complex_multi_domain_brief(self, brief_generator_agent, agent_dependencies):
        """Test brief generation for complex multi-domain queries."""
        agent_dependencies.research_state.user_query = "Analyze the intersection of AI, healthcare, and regulatory compliance in the EU"

        brief_data = ResearchBrief(
            title="AI in Healthcare: EU Regulatory Compliance Analysis",
            executive_summary="Comprehensive analysis of AI applications in healthcare within the EU regulatory framework",
            objectives=[
                ResearchObjective(
                    objective="Map AI healthcare applications",
                    priority=5,
                    success_criteria="Complete landscape overview"
                ),
                ResearchObjective(
                    objective="Analyze EU regulatory requirements",
                    priority=5,
                    success_criteria="Full compliance mapping"
                ),
                ResearchObjective(
                    objective="Identify compliance challenges",
                    priority=4,
                    success_criteria="Actionable insights"
                )
            ],
            scope="EU healthcare AI market with focus on GDPR, MDR, and AI Act compliance",
            methodology=ResearchMethodology(
                approach="Regulatory analysis, case studies, expert interviews",
                data_sources=["Regulatory documents", "Case studies", "Expert interviews"],
                analysis_methods=["Regulatory analysis", "Case analysis"],
                quality_checks=["Expert validation"]
            ),
            constraints=[
                "Geographic: EU member states only",
                "Regulatory: Current regulations as of 2024"
            ],
            deliverables=["Compliance framework", "Best practices guide", "Risk assessment"],
            timeline_estimate="3 weeks",
            success_metrics=["Comprehensive coverage", "Actionable framework"]
        )

        mock_result = MagicMock()
        mock_result.output = brief_data
        mock_result.usage = MagicMock(return_value=None)

        with patch.object(brief_generator_agent.agent, 'run', return_value=mock_result):
            result = await brief_generator_agent.run(deps=agent_dependencies)

            assert result is not None
            assert "AI" in result.title
            assert "healthcare" in result.title.lower()
            assert len(result.objectives) >= 3
