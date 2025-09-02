"""
Comprehensive tests for the BriefGeneratorAgent.
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.brief_generator import BriefGeneratorAgent
from src.agents.base import ResearchDependencies, AgentConfiguration
from src.models.brief_generator import ResearchBrief, ResearchObjective, ResearchConstraint
from src.models.api_models import APIKeys, ResearchMetadata
from src.models.core import ResearchState, ResearchStage


class TestBriefGeneratorAgent:
    """Test suite for BriefGeneratorAgent."""

    @pytest.fixture
    async def agent_dependencies(self):
        """Create mock dependencies for testing."""
        deps = ResearchDependencies(
            http_client=AsyncMock(),
            api_keys=APIKeys(),
            research_state=ResearchState(
                request_id="test-789",
                user_id="test-user",
                session_id="test-session",
                user_query="Research renewable energy technologies",
                current_stage=ResearchStage.BRIEF_GENERATION
            ),
            metadata=ResearchMetadata(),
            usage=None
        )
        return deps

    @pytest.fixture
    def brief_generator_agent(self, agent_dependencies):
        """Create a BriefGeneratorAgent instance."""
        config = AgentConfiguration(
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
        assert agent.agent_name == "brief_generator"
        assert agent.agent is not None
        assert agent.result_validator is not None

    @pytest.mark.asyncio
    async def test_comprehensive_brief_generation(self, brief_generator_agent, agent_dependencies):
        """Test generation of comprehensive research brief."""
        mock_result = MagicMock()
        mock_result.data = ResearchBrief(
            title="Comprehensive Analysis of Renewable Energy Technologies",
            executive_summary="This research examines current renewable energy technologies including solar, wind, and hydroelectric power, analyzing their efficiency, costs, and future potential.",
            objectives=[
                ResearchObjective(
                    objective="Analyze current renewable energy technologies",
                    importance="critical",
                    success_criteria="Comprehensive coverage of major technologies"
                ),
                ResearchObjective(
                    objective="Compare efficiency and cost metrics",
                    importance="high",
                    success_criteria="Quantitative comparison with data"
                ),
                ResearchObjective(
                    objective="Assess future potential and trends",
                    importance="high",
                    success_criteria="Evidence-based projections"
                )
            ],
            scope="Global renewable energy market with focus on solar, wind, and hydroelectric technologies from 2020-2024",
            methodology="Literature review, data analysis, expert interviews, and case studies",
            key_questions=[
                "What are the current efficiency rates of different renewable technologies?",
                "How do costs compare across technologies?",
                "What are the main barriers to adoption?",
                "What technological breakthroughs are expected?"
            ],
            constraints=[
                ResearchConstraint(
                    constraint_type="temporal",
                    description="Focus on 2020-2024 data"
                ),
                ResearchConstraint(
                    constraint_type="geographic",
                    description="Global scope with emphasis on leading markets"
                )
            ],
            deliverables=["Comprehensive report", "Executive summary", "Data visualizations"],
            timeline="2 weeks for complete research",
            success_metrics=["Coverage of major technologies", "Data-driven insights", "Actionable recommendations"],
            assumptions=["Data availability from public sources", "Current trends continue"],
            risks=["Data gaps in emerging markets", "Rapid technology changes"],
            stakeholders=["Energy policy makers", "Investors", "Technology developers"],
            resources_required=["Access to energy databases", "Industry reports", "Expert network"],
            metadata={
                "brief_version": "1.0",
                "generated_at": datetime.now().isoformat(),
                "confidence_score": 0.9
            }
        )

        with patch.object(brief_generator_agent.agent, 'run', return_value=mock_result):
            result = await brief_generator_agent.execute(agent_dependencies)

            assert isinstance(result, ResearchBrief)
            assert len(result.objectives) >= 2
            assert len(result.key_questions) >= 3
            assert result.title is not None and len(result.title) > 0
            assert result.methodology is not None

    @pytest.mark.asyncio
    async def test_simple_query_brief(self, brief_generator_agent, agent_dependencies):
        """Test brief generation for simple queries."""
        agent_dependencies.research_state.user_query = "What is Bitcoin?"

        mock_result = MagicMock()
        mock_result.data = ResearchBrief(
            title="Understanding Bitcoin: Digital Currency Overview",
            executive_summary="Research brief on Bitcoin cryptocurrency, its technology, and market dynamics.",
            objectives=[
                ResearchObjective(
                    objective="Explain Bitcoin technology",
                    importance="critical",
                    success_criteria="Clear technical explanation"
                ),
                ResearchObjective(
                    objective="Analyze market dynamics",
                    importance="high",
                    success_criteria="Current market analysis"
                )
            ],
            scope="Bitcoin technology and market overview",
            methodology="Literature review and market data analysis",
            key_questions=["What is Bitcoin?", "How does it work?", "What is its current market status?"],
            constraints=[],
            deliverables=["Research summary"],
            timeline="1 day",
            success_metrics=["Clear explanation", "Accurate data"],
            assumptions=["Public data available"],
            risks=["Market volatility"],
            stakeholders=["General audience"],
            resources_required=["Public sources"],
            metadata={"confidence_score": 0.85}
        )

        with patch.object(brief_generator_agent.agent, 'run', return_value=mock_result):
            result = await brief_generator_agent.execute(agent_dependencies)

            assert "Bitcoin" in result.title
            assert len(result.objectives) >= 1
            assert result.scope is not None

    @pytest.mark.asyncio
    async def test_objectives_generation(self, brief_generator_agent, agent_dependencies):
        """Test that objectives are properly generated with importance levels."""
        mock_result = MagicMock()
        mock_result.data = ResearchBrief(
            title="Test Research Brief",
            executive_summary="Test summary",
            objectives=[
                ResearchObjective(
                    objective="Primary objective",
                    importance="critical",
                    success_criteria="Must achieve"
                ),
                ResearchObjective(
                    objective="Secondary objective",
                    importance="high",
                    success_criteria="Should achieve"
                ),
                ResearchObjective(
                    objective="Tertiary objective",
                    importance="medium",
                    success_criteria="Nice to have"
                )
            ],
            scope="Test scope",
            methodology="Test methodology",
            key_questions=["Q1"],
            constraints=[],
            deliverables=["Report"],
            timeline="1 week",
            success_metrics=["Metric 1"],
            assumptions=[],
            risks=[],
            stakeholders=["Test stakeholder"],
            resources_required=["Test resource"],
            metadata={}
        )

        with patch.object(brief_generator_agent.agent, 'run', return_value=mock_result):
            result = await brief_generator_agent.execute(agent_dependencies)

            assert len(result.objectives) == 3
            importance_levels = [obj.importance for obj in result.objectives]
            assert "critical" in importance_levels
            assert all(obj.success_criteria is not None for obj in result.objectives)

    @pytest.mark.asyncio
    async def test_constraints_handling(self, brief_generator_agent, agent_dependencies):
        """Test proper handling of research constraints."""
        mock_result = MagicMock()
        mock_result.data = ResearchBrief(
            title="Constrained Research",
            executive_summary="Research with multiple constraints",
            objectives=[
                ResearchObjective(
                    objective="Test objective",
                    importance="high",
                    success_criteria="Test criteria"
                )
            ],
            scope="Limited scope",
            methodology="Constrained methodology",
            key_questions=["Q1"],
            constraints=[
                ResearchConstraint(
                    constraint_type="temporal",
                    description="Last 12 months only"
                ),
                ResearchConstraint(
                    constraint_type="budget",
                    description="Limited to public data sources"
                ),
                ResearchConstraint(
                    constraint_type="scope",
                    description="Focus on North America"
                )
            ],
            deliverables=["Report"],
            timeline="1 week",
            success_metrics=["Metric 1"],
            assumptions=[],
            risks=[],
            stakeholders=["Stakeholder"],
            resources_required=["Resource"],
            metadata={}
        )

        with patch.object(brief_generator_agent.agent, 'run', return_value=mock_result):
            result = await brief_generator_agent.execute(agent_dependencies)

            assert len(result.constraints) == 3
            constraint_types = [c.constraint_type for c in result.constraints]
            assert "temporal" in constraint_types
            assert "budget" in constraint_types
            assert "scope" in constraint_types

    @pytest.mark.asyncio
    async def test_edge_case_minimal_brief(self, brief_generator_agent, agent_dependencies):
        """Test generation of minimal brief for edge cases."""
        agent_dependencies.research_state.user_query = "?"

        mock_result = MagicMock()
        mock_result.data = ResearchBrief(
            title="Unclear Research Query",
            executive_summary="Unable to generate comprehensive brief due to unclear query",
            objectives=[
                ResearchObjective(
                    objective="Clarify research intent",
                    importance="critical",
                    success_criteria="Clear research question defined"
                )
            ],
            scope="To be determined",
            methodology="To be determined after clarification",
            key_questions=["What is the research question?"],
            constraints=[],
            deliverables=["TBD"],
            timeline="TBD",
            success_metrics=["Clear research direction"],
            assumptions=[],
            risks=["Unclear objectives"],
            stakeholders=["TBD"],
            resources_required=["TBD"],
            metadata={"needs_clarification": True}
        )

        with patch.object(brief_generator_agent.agent, 'run', return_value=mock_result):
            result = await brief_generator_agent.execute(agent_dependencies)

            assert result.metadata.get("needs_clarification") is True
            assert len(result.objectives) >= 1

    @pytest.mark.asyncio
    async def test_metadata_generation(self, brief_generator_agent, agent_dependencies):
        """Test that metadata is properly generated."""
        mock_result = MagicMock()
        mock_result.data = ResearchBrief(
            title="Test Brief",
            executive_summary="Test",
            objectives=[
                ResearchObjective(
                    objective="Test",
                    importance="high",
                    success_criteria="Test"
                )
            ],
            scope="Test",
            methodology="Test",
            key_questions=["Test"],
            constraints=[],
            deliverables=["Test"],
            timeline="Test",
            success_metrics=["Test"],
            assumptions=[],
            risks=[],
            stakeholders=["Test"],
            resources_required=["Test"],
            metadata={
                "brief_version": "1.0",
                "generated_at": datetime.now().isoformat(),
                "confidence_score": 0.8,
                "query_complexity": "medium",
                "estimated_effort": "moderate"
            }
        )

        with patch.object(brief_generator_agent.agent, 'run', return_value=mock_result):
            result = await brief_generator_agent.execute(agent_dependencies)

            assert result.metadata is not None
            assert "brief_version" in result.metadata
            assert "confidence_score" in result.metadata
            assert 0.0 <= result.metadata["confidence_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_error_handling(self, brief_generator_agent, agent_dependencies):
        """Test error handling during brief generation."""
        with patch.object(brief_generator_agent.agent, 'run', side_effect=Exception("Generation failed")):
            with pytest.raises(Exception, match="Generation failed"):
                await brief_generator_agent.execute(agent_dependencies)

    @pytest.mark.asyncio
    async def test_complex_multi_domain_brief(self, brief_generator_agent, agent_dependencies):
        """Test brief generation for complex multi-domain queries."""
        agent_dependencies.research_state.user_query = "Analyze the intersection of AI, healthcare, and regulatory compliance in the EU"

        mock_result = MagicMock()
        mock_result.data = ResearchBrief(
            title="AI in Healthcare: EU Regulatory Compliance Analysis",
            executive_summary="Comprehensive analysis of AI applications in healthcare within the EU regulatory framework",
            objectives=[
                ResearchObjective(
                    objective="Map AI healthcare applications",
                    importance="critical",
                    success_criteria="Complete landscape overview"
                ),
                ResearchObjective(
                    objective="Analyze EU regulatory requirements",
                    importance="critical",
                    success_criteria="Full compliance mapping"
                ),
                ResearchObjective(
                    objective="Identify compliance challenges",
                    importance="high",
                    success_criteria="Actionable insights"
                )
            ],
            scope="EU healthcare AI market with focus on GDPR, MDR, and AI Act compliance",
            methodology="Regulatory analysis, case studies, expert interviews",
            key_questions=[
                "What are key EU regulations for healthcare AI?",
                "How do companies ensure compliance?",
                "What are common compliance failures?"
            ],
            constraints=[
                ResearchConstraint(
                    constraint_type="geographic",
                    description="EU member states only"
                ),
                ResearchConstraint(
                    constraint_type="regulatory",
                    description="Current regulations as of 2024"
                )
            ],
            deliverables=["Compliance framework", "Best practices guide", "Risk assessment"],
            timeline="3 weeks",
            success_metrics=["Comprehensive coverage", "Actionable framework"],
            assumptions=["Regulations remain stable"],
            risks=["Regulatory changes", "Interpretation variations"],
            stakeholders=["Healthcare providers", "AI developers", "Regulators"],
            resources_required=["Legal databases", "Regulatory texts", "Expert consultations"],
            metadata={"domains": ["AI", "healthcare", "regulation"], "complexity": "high"}
        )

        with patch.object(brief_generator_agent.agent, 'run', return_value=mock_result):
            result = await brief_generator_agent.execute(agent_dependencies)

            assert len(result.objectives) >= 3
            assert "domains" in result.metadata
            assert len(result.metadata["domains"]) >= 2
            assert result.metadata.get("complexity") == "high"
