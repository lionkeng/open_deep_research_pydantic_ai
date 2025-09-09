"""Unit tests for QueryTransformationAgent with TransformedQuery."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.agents.base import ResearchDependencies
from src.agents.query_transformation import QueryTransformationAgent
from src.models.api_models import APIKeys
from src.models.core import ResearchState
from src.models.metadata import ResearchMetadata

# Clarification models removed - not needed for these tests
from src.models.research_plan_models import (
    ResearchMethodology,
    ResearchObjective,
    ResearchPlan,
    TransformedQuery,
)
from src.models.search_query_models import SearchQuery, SearchQueryBatch, SearchQueryType


def create_test_transformed_query(
    original_query: str = "test query", num_objectives: int = 2, queries_per_objective: int = 2
) -> TransformedQuery:
    """Helper to create properly linked TransformedQuery for tests."""
    objectives = []
    queries = []

    for i in range(num_objectives):
        obj_id = str(uuid.uuid4())
        objectives.append(
            ResearchObjective(
                id=obj_id,
                objective=f"Test objective {i + 1}",
                priority="PRIMARY" if i == 0 else "SECONDARY",
                success_criteria=f"Test success criteria {i + 1}",
            )
        )

        for j in range(queries_per_objective):
            queries.append(
                SearchQuery(
                    id=str(uuid.uuid4()),
                    query=f"test query {i}-{j}",
                    query_type=SearchQueryType.FACTUAL,
                    priority=5 - i,
                    max_results=10,
                    rationale="Test rationale",
                    objective_id=obj_id,  # Link to objective
                )
            )

    batch = SearchQueryBatch(queries=queries)

    plan = ResearchPlan(
        objectives=objectives,
        methodology=ResearchMethodology(
            approach="Test methodology",
            data_sources=["Source 1", "Source 2"],
            analysis_methods=["Method 1", "Method 2"],
            quality_criteria=["Criteria 1"],
            limitations=["Limitation 1"],
        ),
        expected_deliverables=["Deliverable 1"],
    )

    return TransformedQuery(
        original_query=original_query,
        search_queries=batch,
        research_plan=plan,
        key_concepts=["concept1", "concept2"],
        search_strategy="Test strategy",
        assumptions_made=["Assumption 1"],
        knowledge_gaps=["Gap 1"],
    )


@pytest.fixture
def transformation_agent():
    """Create a QueryTransformationAgent instance."""
    agent = QueryTransformationAgent()
    agent._agent = None  # Reset lazy initialization
    return agent


@pytest.fixture
def agent_dependencies():
    """Create dependencies for the agent."""
    return ResearchDependencies(
        http_client=AsyncMock(spec=httpx.AsyncClient),
        api_keys=APIKeys(),
        research_state=ResearchState(
            request_id="test-request-123",
            user_id="test-user",
            session_id="test-session",
            user_query="How does machine learning work?",
            metadata=ResearchMetadata(),  # Use default factory
        ),
    )


class TestQueryTransformationAgent:
    """Test suite for QueryTransformationAgent."""

    @pytest.mark.asyncio
    async def test_enhanced_query_generation(self, transformation_agent, agent_dependencies):
        """Test generation of TransformedQuery with SearchQueryBatch and ResearchPlan."""
        # Create research objectives first
        objective_id = str(uuid.uuid4())
        objectives = [
            ResearchObjective(
                id=objective_id,
                objective="Understand ML fundamentals",
                priority="PRIMARY",
                success_criteria="Clear explanation of core concepts",
            ),
            ResearchObjective(
                id=str(uuid.uuid4()),
                objective="Explore ML applications",
                priority="SECONDARY",
                success_criteria="Identify practical use cases",
            ),
        ]

        # Create search queries linked to objectives
        queries = [
            SearchQuery(
                id=str(uuid.uuid4()),
                query="machine learning fundamentals algorithms",
                query_type=SearchQueryType.FACTUAL,
                priority=5,
                max_results=10,
                rationale="Core concepts understanding",
                objective_id=objective_id,  # Link to PRIMARY objective
            ),
            SearchQuery(
                id=str(uuid.uuid4()),
                query="supervised vs unsupervised learning",
                query_type=SearchQueryType.COMPARATIVE,
                priority=4,
                max_results=10,
                rationale="Compare learning paradigms",
                objective_id=objectives[1].id,  # Link to SECONDARY objective
            ),
        ]

        batch = SearchQueryBatch(queries=queries)

        # Create research plan
        plan = ResearchPlan(
            objectives=objectives,
            methodology=ResearchMethodology(
                approach="Systematic review",
                data_sources=["Academic papers", "Technical blogs"],
                analysis_methods=["Literature review", "Synthesis"],
                quality_criteria=["Accuracy", "Clarity"],
            ),
            expected_deliverables=["Comprehensive ML overview"],
        )

        # Create enhanced query
        enhanced_query = TransformedQuery(
            original_query="How does machine learning work?",
            search_queries=batch,
            research_plan=plan,
            key_concepts=["machine learning", "algorithms", "training"],
            search_strategy="Broad to specific approach",
            assumptions_made=["Basic technical understanding"],
            knowledge_gaps=["Deep learning specifics"],
        )

        # Mock the agent run
        mock_result = MagicMock()
        mock_result.output = enhanced_query

        with patch.object(transformation_agent.agent, "run", return_value=mock_result):
            result = await transformation_agent.run(deps=agent_dependencies)

            assert isinstance(result, TransformedQuery)
            assert result.search_queries == batch
            assert result.research_plan == plan
            assert len(result.search_queries.queries) == 2
            assert len(result.research_plan.objectives) == 2

    @pytest.mark.asyncio
    async def test_query_diversity(self, transformation_agent, agent_dependencies):
        """Test that agent generates diverse query types."""
        # Create test data using helper
        enhanced_query = create_test_transformed_query(
            original_query=agent_dependencies.research_state.user_query,
            num_objectives=3,
            queries_per_objective=2,
        )

        # Update query types for diversity
        enhanced_query.search_queries.queries[0].query_type = SearchQueryType.FACTUAL
        enhanced_query.search_queries.queries[1].query_type = SearchQueryType.ANALYTICAL
        enhanced_query.search_queries.queries[2].query_type = SearchQueryType.COMPARATIVE
        enhanced_query.search_queries.queries[3].query_type = SearchQueryType.EXPLORATORY

        mock_result = MagicMock()
        mock_result.output = enhanced_query

        with patch.object(transformation_agent.agent, "run", return_value=mock_result):
            result = await transformation_agent.run(deps=agent_dependencies)

            query_types = {q.query_type for q in result.search_queries.queries}
            assert SearchQueryType.FACTUAL in query_types
            assert SearchQueryType.ANALYTICAL in query_types
            assert len(query_types) >= 2  # At least 2 different types

    @pytest.mark.asyncio
    async def test_query_prioritization(self, transformation_agent, agent_dependencies):
        """Test that queries are properly prioritized."""
        enhanced_query = create_test_transformed_query(
            original_query=agent_dependencies.research_state.user_query,
            num_objectives=2,
            queries_per_objective=3,
        )

        # Set different priorities
        for i, query in enumerate(enhanced_query.search_queries.queries):
            query.priority = 5 - (i // 2)  # Descending priorities

        mock_result = MagicMock()
        mock_result.output = enhanced_query

        with patch.object(transformation_agent.agent, "run", return_value=mock_result):
            result = await transformation_agent.run(deps=agent_dependencies)

            priorities = [q.priority for q in result.search_queries.queries]
            assert max(priorities) == 5
            assert min(priorities) >= 1

    @pytest.mark.asyncio
    async def test_system_prompt_content(self, transformation_agent):
        """Test that system prompt contains key instructions."""
        prompt = transformation_agent._get_default_system_prompt()

        # Check for key components
        assert "query" in prompt.lower()
        assert "search" in prompt.lower()
        assert "research" in prompt.lower()

    @pytest.mark.asyncio
    async def test_agent_initialization(self, transformation_agent):
        """Test that agent is properly initialized."""
        # Agent uses lazy initialization internally
        assert transformation_agent is not None
        assert hasattr(transformation_agent, "run")

    @pytest.mark.asyncio
    async def test_with_clarification_context(self, transformation_agent, agent_dependencies):
        """Test query transformation with clarification context."""
        # Set clarified query in research state
        agent_dependencies.research_state.clarified_query = "Machine Learning fundamentals"

        enhanced_query = create_test_transformed_query(
            original_query="Machine Learning fundamentals",
            num_objectives=2,
            queries_per_objective=2,
        )

        mock_result = MagicMock()
        mock_result.output = enhanced_query

        with patch.object(transformation_agent.agent, "run", return_value=mock_result):
            result = await transformation_agent.run(deps=agent_dependencies)

            assert result.original_query == enhanced_query.original_query
            assert len(result.search_queries.queries) > 0

    @pytest.mark.asyncio
    async def test_transformation_with_clarification(
        self, transformation_agent, agent_dependencies
    ):
        """Test query transformation with clarification needs."""
        # Create test data with properly linked objectives and queries
        enhanced_query = create_test_transformed_query(
            original_query="ambiguous technical query", num_objectives=2, queries_per_objective=2
        )

        # Add a clarification objective
        clarification_obj_id = str(uuid.uuid4())
        clarification_objective = ResearchObjective(
            id=clarification_obj_id,
            objective="Clarify ambiguous terms",
            priority="SECONDARY",
            success_criteria="Clear understanding of terminology",
        )
        enhanced_query.research_plan.objectives.append(clarification_objective)

        # Add clarification queries linked to the clarification objective
        clarification_queries = [
            SearchQuery(
                id=str(uuid.uuid4()),
                query="Define technical term X",
                query_type=SearchQueryType.EXPLORATORY,  # Changed from CLARIFICATION
                priority=3,
                max_results=5,
                rationale="Clarify ambiguous term",
                objective_id=clarification_obj_id,
            ),
            SearchQuery(
                id=str(uuid.uuid4()),
                query="Explain concept Y in context",
                query_type=SearchQueryType.EXPLORATORY,  # Changed from CLARIFICATION
                priority=3,
                max_results=5,
                rationale="Understand concept",
                objective_id=clarification_obj_id,
            ),
        ]
        enhanced_query.search_queries.queries.extend(clarification_queries)

        mock_result = MagicMock()
        mock_result.output = enhanced_query

        with patch.object(transformation_agent.agent, "run", return_value=mock_result):
            result = await transformation_agent.run(deps=agent_dependencies)

            # Verify exploratory queries exist (used for clarification)
            exploratory_queries = [
                q
                for q in result.search_queries.queries
                if q.query_type == SearchQueryType.EXPLORATORY
            ]
            assert len(exploratory_queries) > 0

            # Verify all queries have valid objective_ids
            objective_ids = {obj.id for obj in result.research_plan.objectives}
            for query in result.search_queries.queries:
                assert query.objective_id in objective_ids

    @pytest.mark.asyncio
    async def test_complex_multi_domain_query(self, transformation_agent, agent_dependencies):
        """Test transformation of a complex query spanning multiple domains."""
        # Update the query in dependencies
        agent_dependencies.research_state.user_query = (
            "How to build scalable microservices with business value?"
        )

        # Create complex objectives
        objectives = []
        queries = []

        # Primary objective
        primary_id = str(uuid.uuid4())
        objectives.append(
            ResearchObjective(
                id=primary_id,
                objective="Analyze technical architecture",
                priority="PRIMARY",
                success_criteria="Complete architecture overview",
            )
        )

        # Technical queries for primary objective
        queries.extend(
            [
                SearchQuery(
                    id=str(uuid.uuid4()),
                    query="System architecture best practices",
                    query_type=SearchQueryType.FACTUAL,
                    priority=5,
                    max_results=10,
                    rationale="Architecture foundations",
                    objective_id=primary_id,
                ),
                SearchQuery(
                    id=str(uuid.uuid4()),
                    query="Microservices design patterns",
                    query_type=SearchQueryType.EXPLORATORY,
                    priority=5,
                    max_results=10,
                    rationale="Design patterns",
                    objective_id=primary_id,
                ),
            ]
        )

        # Supporting objective 1 - Performance
        perf_id = str(uuid.uuid4())
        objectives.append(
            ResearchObjective(
                id=perf_id,
                objective="Review performance metrics",
                priority="SECONDARY",
                success_criteria="Performance analysis",
            )
        )

        queries.extend(
            [
                SearchQuery(
                    id=str(uuid.uuid4()),
                    query="Performance benchmarking methods",
                    query_type=SearchQueryType.ANALYTICAL,
                    priority=4,
                    max_results=10,
                    rationale="Performance measurement",
                    objective_id=perf_id,
                ),
                SearchQuery(
                    id=str(uuid.uuid4()),
                    query="Latency optimization techniques",
                    query_type=SearchQueryType.FACTUAL,
                    priority=4,
                    max_results=10,
                    rationale="Performance optimization",
                    objective_id=perf_id,
                ),
            ]
        )

        # Supporting objective 2 - Business
        business_id = str(uuid.uuid4())
        objectives.append(
            ResearchObjective(
                id=business_id,
                objective="Evaluate business impact",
                priority="SECONDARY",
                success_criteria="Business value assessment",
            )
        )

        queries.extend(
            [
                SearchQuery(
                    id=str(uuid.uuid4()),
                    query="ROI calculation frameworks",
                    query_type=SearchQueryType.ANALYTICAL,
                    priority=3,
                    max_results=10,
                    rationale="ROI analysis",
                    objective_id=business_id,
                ),
                SearchQuery(
                    id=str(uuid.uuid4()),
                    query="Market adoption trends",
                    query_type=SearchQueryType.EXPLORATORY,
                    priority=3,
                    max_results=10,
                    rationale="Market research",
                    objective_id=business_id,
                ),
            ]
        )

        batch = SearchQueryBatch(queries=queries)

        plan = ResearchPlan(
            objectives=objectives,
            methodology=ResearchMethodology(
                approach="Multi-domain analysis",
                data_sources=["Technical docs", "Case studies", "Market reports"],
                analysis_methods=[
                    "Technical review",
                    "Performance analysis",
                    "Business evaluation",
                ],
                quality_criteria=["Technical feasibility", "Performance metrics", "ROI"],
            ),
            expected_deliverables=["Complete microservices strategy"],
        )

        enhanced_query = TransformedQuery(
            original_query=agent_dependencies.research_state.user_query,
            search_queries=batch,
            research_plan=plan,
            key_concepts=["microservices", "scalability", "business value"],
            search_strategy="Multi-domain comprehensive analysis",
            assumptions_made=["Cloud deployment", "Modern stack"],
            knowledge_gaps=["Specific technology choices"],
        )

        mock_result = MagicMock()
        mock_result.output = enhanced_query

        with patch.object(transformation_agent.agent, "run", return_value=mock_result):
            result = await transformation_agent.run(deps=agent_dependencies)

            # Verify comprehensive transformation
            assert len(result.research_plan.objectives) >= 3
            assert len(result.search_queries.queries) >= 6

            # Verify query diversity
            query_types = {q.query_type for q in result.search_queries.queries}
            assert SearchQueryType.FACTUAL in query_types
            assert SearchQueryType.ANALYTICAL in query_types
            assert SearchQueryType.EXPLORATORY in query_types

            # Verify all queries are properly linked
            objective_ids = {obj.id for obj in result.research_plan.objectives}
            for query in result.search_queries.queries:
                assert query.objective_id in objective_ids

            # Verify PRIMARY objective has queries
            primary_objectives = [
                obj for obj in result.research_plan.objectives if obj.priority == "PRIMARY"
            ]
            for primary_obj in primary_objectives:
                associated_queries = [
                    q for q in result.search_queries.queries if q.objective_id == primary_obj.id
                ]
                assert len(associated_queries) > 0

    @pytest.mark.asyncio
    async def test_execution_summary(self, transformation_agent, agent_dependencies):
        """Test that execution results are properly summarized."""
        # Create test data with properly linked objectives and queries
        enhanced_query = create_test_transformed_query(
            original_query="test query for summary", num_objectives=2, queries_per_objective=2
        )

        mock_result = MagicMock()
        mock_result.output = enhanced_query

        with patch.object(transformation_agent.agent, "run", return_value=mock_result):
            result = await transformation_agent.run(deps=agent_dependencies)

            # Verify result structure
            assert result.original_query == "test query for summary"
            assert len(result.research_plan.objectives) == 2
            assert len(result.search_queries.queries) == 4  # 2 objectives * 2 queries each

            # Verify PRIMARY objective exists and has queries
            primary_objectives = [
                obj for obj in result.research_plan.objectives if obj.priority == "PRIMARY"
            ]
            assert len(primary_objectives) >= 1

            for primary_obj in primary_objectives:
                associated_queries = [
                    q for q in result.search_queries.queries if q.objective_id == primary_obj.id
                ]
                assert len(associated_queries) > 0

            # Verify all queries are linked to valid objectives
            objective_ids = {obj.id for obj in result.research_plan.objectives}
            for query in result.search_queries.queries:
                assert query.objective_id in objective_ids
