"""
Integration tests for agent pipeline data flow and interactions.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from agents.base import ResearchDependencies
from agents.factory import AgentFactory, AgentType
from models.api_models import APIKeys
from models.compression import CompressedContent
from models.core import ResearchStage, ResearchState
from models.metadata import ResearchMetadata
from models.report_generator import ReportSection, ResearchReport
from models.research_executor import ResearchFinding, ResearchResults, ResearchSource
from models.research_plan_models import TransformedQuery


class TestAgentPipelineIntegration:
    """Test integration between agents in the research pipeline."""

    @pytest_asyncio.fixture
    async def pipeline_dependencies(self):
        """Create dependencies for pipeline testing."""
        return ResearchDependencies(
            http_client=AsyncMock(),
            api_keys=APIKeys(),
            research_state=ResearchState(
                request_id="pipeline-test",
                user_id="test-user",
                session_id="test-session",
                user_query="Research the impact of AI on healthcare",
                current_stage=ResearchStage.CLARIFICATION,
                metadata=ResearchMetadata(),
            ),
            usage=None,
        )

    @pytest.mark.asyncio
    async def test_clarification_to_transformation_flow(self, pipeline_dependencies):
        """Test data flow from clarification to query transformation."""
        # Create agents
        clarification_agent = AgentFactory.create_agent(
            AgentType.CLARIFICATION, pipeline_dependencies
        )
        transformation_agent = AgentFactory.create_agent(
            AgentType.QUERY_TRANSFORMATION, pipeline_dependencies
        )

        # Mock clarification output
        clarification_result = MagicMock()
        clarification_result.clarification_needed = False
        clarification_result.transformed_query = (
            "AI applications in healthcare diagnostics and treatment"
        )
        clarification_result.confidence_score = 0.9

        with patch.object(clarification_agent, "run") as mock_run:
            mock_run.return_value = clarification_result
            # Execute clarification
            clarified = await clarification_agent.run(pipeline_dependencies)

            # Update state with clarification result
            pipeline_dependencies.research_state.metadata.transformed_query = {
                "query": clarified.transformed_query
            }

            # Mock transformation output
            transformation_result = TransformedQuery(
                original_query=clarified.transformed_query,
                transformed_query="Comprehensive analysis of AI in healthcare focusing on diagnostics and treatment",
                supporting_questions=[
                    "How is AI used in medical imaging?",
                    "What are AI treatment recommendations?",
                ],
                transformation_rationale="Expanded query for comprehensive research",
                specificity_score=0.85,
                missing_dimensions=[],
                clarification_responses={},
                transformation_metadata={},
            )

            with patch.object(transformation_agent, "run") as mock_transform_run:
                mock_transform_run.return_value = transformation_result
                # Execute transformation
                transformed = await transformation_agent.run(pipeline_dependencies)

                # Verify data flow
                assert transformed.original_query == clarified.transformed_query
                assert transformed.specificity_score > 0.8

    @pytest.mark.asyncio
    async def test_research_to_compression_flow(self, pipeline_dependencies):
        """Test data flow from research execution to compression."""
        # Create agents
        research_agent = AgentFactory.create_agent(
            AgentType.RESEARCH_EXECUTOR, pipeline_dependencies
        )
        compression_agent = AgentFactory.create_agent(AgentType.COMPRESSION, pipeline_dependencies)

        # Mock research output
        research_result = ResearchResults(
            query="AI in healthcare",
            findings=[
                ResearchFinding(
                    finding="AI improves diagnostic accuracy by 20%",
                    confidence_level=0.9,
                    source=ResearchSource(
                        title="Medical AI Study",
                        url="https://example.com",
                        publish_date="2024-01-01",
                        author="Researchers",
                        credibility_score=0.95,
                    ),
                    relevance_score=0.9,
                    key_insights=["Accuracy improvement"],
                    supporting_evidence=["Study data"],
                    contradictions=[],
                )
            ]
            * 10,  # Multiple findings for compression
            total_sources_consulted=20,
            search_strategies_used=["academic"],
            confidence_score=0.88,
            execution_metadata={},
        )

        with patch.object(research_agent, "run", return_value=research_result):
            research = await research_agent.run(pipeline_dependencies)

            # Pass findings to compression
            pipeline_dependencies.metadata.additional_context = {
                "findings": [
                    f.model_dump() if hasattr(f, "model_dump") else f.__dict__
                    for f in research.findings
                ]
            }

            # Mock compression output
            compression_result = CompressedContent(
                original_length=10000,
                compressed_length=2000,
                compression_ratio=5.0,
                summary="AI significantly improves healthcare diagnostics",
                key_points=["20% accuracy improvement", "Multiple applications"],
                themes={"diagnostics": ["accuracy", "efficiency"]},
                preserved_details=["Key statistics"],
                confidence_score=0.85,
                metadata={},
            )

            with patch.object(compression_agent, "run", return_value=compression_result):
                compressed = await compression_agent.run(pipeline_dependencies)

                # Verify compression
                assert compressed.compression_ratio > 1.0
                assert len(compressed.key_points) > 0

    @pytest.mark.asyncio
    async def test_compression_to_report_generation_flow(self, pipeline_dependencies):
        """Test data flow from compression to report generation."""
        # Create agents
        compression_agent = AgentFactory.create_agent(AgentType.COMPRESSION, pipeline_dependencies)
        report_agent = AgentFactory.create_agent(AgentType.REPORT_GENERATOR, pipeline_dependencies)

        # Mock compression output
        compression_result = CompressedContent(
            original_length=10000,
            compressed_length=2000,
            compression_ratio=5.0,
            summary="AI healthcare research summary",
            key_points=["Point 1", "Point 2", "Point 3"],
            themes={"theme1": ["detail1"], "theme2": ["detail2"]},
            preserved_details=["Critical finding"],
            confidence_score=0.85,
            metadata={},
        )

        with patch.object(compression_agent, "run", return_value=compression_result):
            compressed = await compression_agent.run(pipeline_dependencies)

            # Pass compressed data to report generation
            pipeline_dependencies.metadata.additional_context = {
                "summary": compressed.summary,
                "key_points": compressed.key_points,
                "themes": compressed.themes,
            }

            # Mock report generation
            report_result = ResearchReport(
                title="AI in Healthcare: Research Report",
                executive_summary=compressed.summary,
                introduction="Introduction based on research",
                sections=[
                    ReportSection(
                        title=f"Theme: {theme}",
                        content=f"Analysis of {theme}",
                        subsections=[],
                        key_findings=compressed.key_points[:2],
                        citations=[],
                    )
                    for theme in compressed.themes.keys()
                ],
                conclusion="Conclusion based on compressed findings",
                recommendations=["Recommendation 1", "Recommendation 2"],
                methodology="Research methodology",
                limitations=["Limitation 1"],
                future_work=["Future research"],
                appendices=[],
                citations=[],
                metadata={},
            )

            with patch.object(report_agent, "run", return_value=report_result):
                report = await report_agent.run(pipeline_dependencies)

                # Verify report incorporates compressed data
                assert report.executive_summary == compressed.summary
                assert len(report.sections) == len(compressed.themes)

    @pytest.mark.asyncio
    async def test_full_pipeline_execution(self, pipeline_dependencies):
        """Test complete pipeline from clarification to report."""
        agents = {
            AgentType.CLARIFICATION: AgentFactory.create_agent(
                AgentType.CLARIFICATION, pipeline_dependencies
            ),
            AgentType.QUERY_TRANSFORMATION: AgentFactory.create_agent(
                AgentType.QUERY_TRANSFORMATION, pipeline_dependencies
            ),
            AgentType.RESEARCH_EXECUTOR: AgentFactory.create_agent(
                AgentType.RESEARCH_EXECUTOR, pipeline_dependencies
            ),
            AgentType.COMPRESSION: AgentFactory.create_agent(
                AgentType.COMPRESSION, pipeline_dependencies
            ),
            AgentType.REPORT_GENERATOR: AgentFactory.create_agent(
                AgentType.REPORT_GENERATOR, pipeline_dependencies
            ),
        }

        # Track execution order
        execution_order = []

        async def mock_run(agent_type):
            execution_order.append(agent_type)
            await asyncio.sleep(0.01)  # Simulate work
            return MagicMock()

        # Mock all agent executions
        for agent_type, agent in agents.items():
            with patch.object(agent, "run", side_effect=lambda deps, at=agent_type: mock_run(at)):
                pass

        # Execute pipeline
        for agent_type in [
            AgentType.CLARIFICATION,
            AgentType.QUERY_TRANSFORMATION,
            AgentType.RESEARCH_EXECUTOR,
            AgentType.COMPRESSION,
            AgentType.REPORT_GENERATOR,
        ]:
            await agents[agent_type].run(pipeline_dependencies)

        # Verify execution order
        assert len(execution_order) == 5

    @pytest.mark.asyncio
    async def test_pipeline_error_propagation(self, pipeline_dependencies):
        """Test error propagation through the pipeline."""
        # Create agents
        research_agent = AgentFactory.create_agent(
            AgentType.RESEARCH_EXECUTOR, pipeline_dependencies
        )
        compression_agent = AgentFactory.create_agent(AgentType.COMPRESSION, pipeline_dependencies)

        # Mock research failure
        with patch.object(research_agent, "run", side_effect=Exception("Research failed")):
            with pytest.raises(Exception, match="Research failed"):
                await research_agent.run(pipeline_dependencies)

            # Compression should handle missing data gracefully
            compression_result = CompressedContent(
                original_length=0,
                compressed_length=0,
                compression_ratio=1.0,
                summary="No data available due to research failure",
                key_points=[],
                themes={},
                preserved_details=[],
                confidence_score=0.0,
                metadata={"error": "upstream_failure"},
            )

            with patch.object(compression_agent, "run", return_value=compression_result):
                result = await compression_agent.run(pipeline_dependencies)
                assert result.metadata.get("error") == "upstream_failure"

    @pytest.mark.asyncio
    async def test_context_preservation_across_agents(self, pipeline_dependencies):
        """Test that context is preserved across agent boundaries."""
        # Initial context
        initial_context = {
            "user_preference": "technical_detail",
            "domain": "healthcare",
            "urgency": "high",
        }
        pipeline_dependencies.metadata.additional_context = initial_context.copy()

        # Create agents
        agents = [
            AgentFactory.create_agent(AgentType.CLARIFICATION, pipeline_dependencies),
            AgentFactory.create_agent(AgentType.QUERY_TRANSFORMATION, pipeline_dependencies),
        ]

        # Mock executions that preserve and add to context
        for i, agent in enumerate(agents):

            async def mock_run_with_context(deps, agent_num=i):
                # Verify initial context is preserved
                assert deps.metadata.additional_context["user_preference"] == "technical_detail"
                assert deps.metadata.additional_context["domain"] == "healthcare"

                # Add agent-specific context
                deps.metadata.additional_context[f"agent_{agent_num}_processed"] = True
                return MagicMock()

            with patch.object(agent, "run", side_effect=mock_run_with_context):
                await agent.run(pipeline_dependencies)

        # Verify all context is preserved
        assert (
            pipeline_dependencies.metadata.additional_context["user_preference"]
            == "technical_detail"
        )
        assert all(
            f"agent_{i}_processed" in pipeline_dependencies.metadata.additional_context
            for i in range(2)
        )

    @pytest.mark.asyncio
    async def test_parallel_agent_execution(self, pipeline_dependencies):
        """Test parallel execution of independent agents."""
        # Create multiple independent agents
        agents = [
            AgentFactory.create_agent(AgentType.RESEARCH_EXECUTOR, pipeline_dependencies),
            AgentFactory.create_agent(AgentType.RESEARCH_EXECUTOR, pipeline_dependencies),
            AgentFactory.create_agent(AgentType.RESEARCH_EXECUTOR, pipeline_dependencies),
        ]

        execution_times = []

        async def mock_run_with_delay(deps):
            start = asyncio.get_event_loop().time()
            await asyncio.sleep(0.1)  # Simulate work
            execution_times.append(asyncio.get_event_loop().time() - start)
            return MagicMock()

        # Mock all agents
        for agent in agents:
            agent.run = mock_run_with_delay

        # Execute in parallel
        start = asyncio.get_event_loop().time()
        results = await asyncio.gather(*[agent.run(pipeline_dependencies) for agent in agents])
        total_time = asyncio.get_event_loop().time() - start

        # Verify parallel execution (should be much faster than sequential)
        assert len(results) == 3
        assert total_time < 0.3  # Should be ~0.1s for parallel, would be ~0.3s for sequential
