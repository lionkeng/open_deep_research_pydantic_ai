"""Unit tests for QueryTransformationAgent with TransformedQuery."""

import uuid
import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.query_transformation import QueryTransformationAgent
from src.agents.base import ResearchDependencies, AgentConfiguration
from src.models.api_models import APIKeys
from src.models.core import ResearchState, ResearchStage
from src.models.metadata import ResearchMetadata, ClarificationMetadata
from src.models.clarification import (
    ClarificationRequest, ClarificationResponse,
    ClarificationQuestion, ClarificationAnswer
)
from src.models.research_plan_models import (
    TransformedQuery,
    ResearchPlan,
    ResearchObjective,
    ResearchMethodology
)
from src.models.search_query_models import (
    SearchQuery,
    SearchQueryBatch,
    SearchQueryType,
    ExecutionStrategy
)


class TestQueryTransformationAgent:
    """Test suite for QueryTransformationAgent with new architecture."""

    @pytest.fixture
    def agent_dependencies(self):
        """Create mock dependencies for testing."""
        deps = ResearchDependencies(
            http_client=AsyncMock(),
            api_keys=APIKeys(),
            research_state=ResearchState(
                request_id="test-456",
                user_id="test-user",
                session_id="test-session",
                user_query="How does machine learning work?",
                current_stage=ResearchStage.CLARIFICATION,
                metadata=ResearchMetadata()
            ),
            usage=None
        )
        return deps

    @pytest.fixture
    def transformation_agent(self, agent_dependencies):
        """Create a QueryTransformationAgent instance."""
        config = AgentConfiguration(
            agent_name="query_transformation",
            agent_type="transformation",
            max_retries=3,
            timeout=30.0
        )
        agent = QueryTransformationAgent(config=config, dependencies=agent_dependencies)
        return agent

    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = QueryTransformationAgent()
        assert agent.name == "query_transformation"
        assert agent.agent is not None
        assert agent.config is not None

    @pytest.mark.asyncio
    async def test_enhanced_query_generation(self, transformation_agent, agent_dependencies):
        """Test generation of TransformedQuery with SearchQueryBatch and ResearchPlan."""
        # Create research objectives first
        objective_id = str(uuid.uuid4())
        objectives = [
            ResearchObjective(
                id=objective_id,
                objective="Explain fundamental machine learning concepts",
                priority="PRIMARY",
                success_criteria="Clear understanding of ML basics"
            )
        ]

        # Create search queries linked to objectives
        queries = [
            SearchQuery(
                id=str(uuid.uuid4()),
                query="machine learning algorithms overview",
                query_type=SearchQueryType.FACTUAL,
                priority=1,
                max_results=10,
                rationale="Understand core ML algorithms",
                objective_id=objective_id  # Link to objective
            ),
            SearchQuery(
                id=str(uuid.uuid4()),
                query="neural networks deep learning",
                query_type=SearchQueryType.ANALYTICAL,
                priority=2,
                max_results=8,
                rationale="Explore neural network concepts",
                objective_id=objective_id  # Link to objective
            )
        ]

        batch = SearchQueryBatch(
            queries=queries,
            execution_strategy=ExecutionStrategy.PARALLEL,
            max_parallel=5
        )

        # Create research plan

        methodology = ResearchMethodology(
            approach="Literature review and synthesis",
            data_sources=["Academic papers", "Technical documentation"],
            analysis_methods=["Comparative analysis", "Concept mapping"]
        )

        plan = ResearchPlan(
            objectives=objectives,
            methodology=methodology,
            expected_deliverables=["Comprehensive ML overview"]
        )

        # Create enhanced query
        enhanced_query = TransformedQuery(
            original_query="How does machine learning work?",
            search_queries=batch,
            research_plan=plan,
            transformation_rationale="Expanded query to cover ML fundamentals",
            confidence_score=0.85
        )

        mock_result = MagicMock()
        mock_result.output = enhanced_query

        with patch.object(transformation_agent.agent, 'run', return_value=mock_result):
            result = await transformation_agent.run(deps=agent_dependencies)

            assert isinstance(result, TransformedQuery)
            assert result.search_queries == batch
            assert result.research_plan == plan
            assert len(result.search_queries.queries) == 2
            assert result.confidence_score == 0.85

    @pytest.mark.asyncio
    async def test_search_query_batch_priorities(self, transformation_agent, agent_dependencies):
        """Test that SearchQueryBatch properly handles query priorities."""
        queries = [
            SearchQuery(
                id=str(uuid.uuid4()),
                query="high priority query",
                query_type=SearchQueryType.FACTUAL,
                priority=1,
                max_results=15,
                rationale="Critical information"
            ),
            SearchQuery(
                id=str(uuid.uuid4()),
                query="medium priority query",
                query_type=SearchQueryType.EXPLORATORY,
                priority=3,
                max_results=10,
                rationale="Supporting information"
            ),
            SearchQuery(
                id=str(uuid.uuid4()),
                query="low priority query",
                query_type=SearchQueryType.COMPARATIVE,
                priority=5,
                max_results=5,
                rationale="Additional context"
            )
        ]

        batch = SearchQueryBatch(
            queries=queries,
            execution_strategy=ExecutionStrategy.HIERARCHICAL
        )

        # Verify hierarchical execution sorts by priority
        assert batch.queries[0].priority <= batch.queries[1].priority
        assert batch.queries[1].priority <= batch.queries[2].priority

    @pytest.mark.asyncio
    async def test_research_plan_objectives(self, transformation_agent, agent_dependencies):
        """Test ResearchPlan with multiple objectives and dependencies."""
        obj1_id = str(uuid.uuid4())
        obj2_id = str(uuid.uuid4())

        objectives = [
            ResearchObjective(
                id=obj1_id,
                objective="Understand basic ML concepts",
                priority="PRIMARY",
                success_criteria="Can explain supervised vs unsupervised learning"
            ),
            ResearchObjective(
                id=obj2_id,
                objective="Explore advanced ML techniques",
                priority="SECONDARY",
                success_criteria="Understand deep learning architectures",
                dependencies=[obj1_id]
            )
        ]

        methodology = ResearchMethodology(
            approach="Progressive learning approach",
            data_sources=["Textbooks", "Research papers"],
            analysis_methods=["Conceptual analysis"]
        )

        plan = ResearchPlan(
            objectives=objectives,
            methodology=methodology,
            expected_deliverables=["ML learning guide"]
        )

        # Test dependency order
        dependency_order = plan.get_dependency_order()
        assert dependency_order[0].id == obj1_id
        assert dependency_order[1].id == obj2_id

    @pytest.mark.asyncio
    async def test_transformation_with_clarification(self, transformation_agent, agent_dependencies):
        """Test query transformation with clarification metadata."""
        # Setup clarification in metadata
        questions = [
            ClarificationQuestion(
                question="What aspect of ML interests you?",
                is_required=True,
                question_type="choice",
                choices=["Algorithms", "Applications", "Theory"]
            )
        ]
        request = ClarificationRequest(questions=questions)

        answers = [
            ClarificationAnswer(
                question_id=questions[0].id,
                answer="Applications",
                skipped=False
            )
        ]
        response = ClarificationResponse(
            request_id="test-456",
            answers=answers
        )

        agent_dependencies.research_state.metadata.clarification = ClarificationMetadata(
            assessment={
                "assessment_reasoning": "Query needs focus area specification",
                "missing_dimensions": ["specific domain", "technical depth"]
            },
            request=request,
            response=response
        )

        # Create focused search queries based on clarification
        queries = [
            SearchQuery(
                id=str(uuid.uuid4()),
                query="machine learning applications industry",
                query_type=SearchQueryType.EXPLORATORY,
                priority=1,
                max_results=12,
                rationale="Focus on ML applications per user preference"
            )
        ]

        batch = SearchQueryBatch(queries=queries)

        objectives = [
            ResearchObjective(
                id=str(uuid.uuid4()),
                objective="Explore ML applications in industry",
                priority="PRIMARY",
                success_criteria="Comprehensive application overview"
            )
        ]

        methodology = ResearchMethodology(
            approach="Application-focused research",
            data_sources=["Industry reports", "Case studies"],
            analysis_methods=["Use case analysis"]
        )

        plan = ResearchPlan(
            objectives=objectives,
            methodology=methodology,
            expected_deliverables=["ML applications report"]
        )

        mock_result = MagicMock()
        mock_result.output = TransformedQuery(
            original_query="How does machine learning work?",
            search_queries=batch,
            research_plan=plan,
            clarification_context={"focus": "Applications"},
            transformation_rationale="Focused on applications based on clarification",
            confidence_score=0.95
        )

        with patch.object(transformation_agent.agent, 'run', return_value=mock_result):
            result = await transformation_agent.run(deps=agent_dependencies)

            assert result.confidence_score == 0.95
            assert "Applications" in result.transformation_rationale
            assert result.clarification_context["focus"] == "Applications"

    @pytest.mark.asyncio
    async def test_execution_strategies(self, transformation_agent, agent_dependencies):
        """Test different execution strategies in SearchQueryBatch."""
        # Test PARALLEL strategy
        parallel_queries = [
            SearchQuery(
                id=str(uuid.uuid4()),
                query=f"parallel query {i}",
                query_type=SearchQueryType.FACTUAL,
                priority=2,
                max_results=10,
                rationale=f"Query {i}"
            )
            for i in range(3)
        ]

        parallel_batch = SearchQueryBatch(
            queries=parallel_queries,
            execution_strategy=ExecutionStrategy.PARALLEL
        )

        groups = parallel_batch.get_execution_groups()
        assert len(groups) == 1  # All queries in one group for parallel
        assert len(groups[0]) == 3

        # Test SEQUENTIAL strategy
        sequential_batch = SearchQueryBatch(
            queries=parallel_queries,
            execution_strategy=ExecutionStrategy.SEQUENTIAL
        )

        groups = sequential_batch.get_execution_groups()
        assert len(groups) == 3  # Each query in its own group
        assert all(len(g) == 1 for g in groups)

        # Test ADAPTIVE strategy
        adaptive_batch = SearchQueryBatch(
            queries=parallel_queries,
            execution_strategy=ExecutionStrategy.ADAPTIVE,
            max_parallel=2
        )

        groups = adaptive_batch.get_execution_groups()
        assert len(groups) == 2  # Should respect max_parallel

    @pytest.mark.asyncio
    async def test_complex_multi_domain_query(self, transformation_agent, agent_dependencies):
        """Test transformation of complex multi-domain queries."""
        agent_dependencies.research_state.user_query = (
            "Compare quantum computing and classical computing for cryptography, "
            "including performance, security, and future implications"
        )

        # Create comprehensive search queries
        queries = [
            SearchQuery(
                id=str(uuid.uuid4()),
                query="quantum computing cryptography algorithms",
                query_type=SearchQueryType.ANALYTICAL,
                priority=1,
                max_results=15,
                rationale="Quantum cryptography methods"
            ),
            SearchQuery(
                id=str(uuid.uuid4()),
                query="classical computing cryptographic performance",
                query_type=SearchQueryType.COMPARATIVE,
                priority=1,
                max_results=15,
                rationale="Classical crypto performance"
            ),
            SearchQuery(
                id=str(uuid.uuid4()),
                query="quantum vs classical security comparison",
                query_type=SearchQueryType.COMPARATIVE,
                priority=2,
                max_results=12,
                rationale="Security comparison"
            ),
            SearchQuery(
                id=str(uuid.uuid4()),
                query="post-quantum cryptography future",
                query_type=SearchQueryType.TEMPORAL,
                priority=3,
                max_results=10,
                rationale="Future implications"
            )
        ]

        batch = SearchQueryBatch(
            queries=queries,
            execution_strategy=ExecutionStrategy.ADAPTIVE
        )

        objectives = [
            ResearchObjective(
                id=str(uuid.uuid4()),
                objective="Compare quantum and classical computing for cryptography",
                priority="PRIMARY",
                success_criteria="Clear comparison across all dimensions"
            ),
            ResearchObjective(
                id=str(uuid.uuid4()),
                objective="Analyze security implications",
                priority="PRIMARY",
                success_criteria="Security assessment complete"
            ),
            ResearchObjective(
                id=str(uuid.uuid4()),
                objective="Project future developments",
                priority="SECONDARY",
                success_criteria="Future roadmap identified"
            )
        ]

        methodology = ResearchMethodology(
            approach="Comparative analysis with future projection",
            data_sources=["Academic research", "Industry reports", "Security bulletins"],
            analysis_methods=["Comparative analysis", "Trend analysis", "Security assessment"]
        )

        plan = ResearchPlan(
            objectives=objectives,
            methodology=methodology,
            expected_deliverables=[
                "Comparative analysis report",
                "Security assessment",
                "Future implications analysis"
            ]
        )

        mock_result = MagicMock()
        mock_result.output = TransformedQuery(
            original_query=agent_dependencies.research_state.user_query,
            search_queries=batch,
            research_plan=plan,
            transformation_rationale="Multi-dimensional comparison of quantum vs classical computing",
            confidence_score=0.9
        )

        with patch.object(transformation_agent.agent, 'run', return_value=mock_result):
            result = await transformation_agent.run(deps=agent_dependencies)

            assert len(result.search_queries.queries) == 4
            assert len(result.research_plan.objectives) == 3
            assert len(result.research_plan.expected_deliverables) == 3

            # Verify coverage of all aspects
            query_texts = " ".join([q.query for q in result.search_queries.queries])
            assert "quantum" in query_texts
            assert "classical" in query_texts
            assert "cryptography" in query_texts
            assert "security" in query_texts
            assert "future" in query_texts

    @pytest.mark.asyncio
    async def test_error_handling(self, transformation_agent, agent_dependencies):
        """Test error handling during transformation."""
        with patch.object(transformation_agent.agent, 'run', side_effect=Exception("Transformation failed")):
            with pytest.raises(Exception, match="Transformation failed"):
                await transformation_agent.run(deps=agent_dependencies)

    @pytest.mark.asyncio
    async def test_execution_summary(self, transformation_agent, agent_dependencies):
        """Test the execution summary generation."""
        queries = [
            SearchQuery(
                id=str(uuid.uuid4()),
                query="test query",
                query_type=SearchQueryType.FACTUAL,
                priority=1,
                max_results=10,
                rationale="Test"
            )
        ]

        batch = SearchQueryBatch(
            queries=queries,
            execution_strategy=ExecutionStrategy.PARALLEL
        )

        objectives = [
            ResearchObjective(
                id=str(uuid.uuid4()),
                objective="Test objective",
                priority="PRIMARY",
                success_criteria="Test success"
            )
        ]

        methodology = ResearchMethodology(
            approach="Test approach",
            data_sources=["Test source"],
            analysis_methods=["Test method"]
        )

        plan = ResearchPlan(
            objectives=objectives,
            methodology=methodology,
            expected_deliverables=["Test deliverable"]
        )

        enhanced_query = TransformedQuery(
            original_query="test",
            search_queries=batch,
            research_plan=plan,
            confidence_score=0.8,
            assumptions_made=["Assumption 1"],
            potential_gaps=["Gap 1"]
        )

        summary = enhanced_query.get_execution_summary()

        assert summary["total_queries"] == 1
        assert summary["execution_strategy"] == "PARALLEL"
        assert summary["primary_objectives"] == 1
        assert summary["total_objectives"] == 1
        assert summary["confidence"] == 0.8
        assert summary["has_assumptions"] is True
        assert summary["has_gaps"] is True
