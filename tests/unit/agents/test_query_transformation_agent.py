"""
Comprehensive tests for the QueryTransformationAgent.
"""

import asyncio
import pytest
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.query_transformation import QueryTransformationAgent
from src.agents.base import ResearchDependencies, AgentConfiguration
from src.models.query_transformation import TransformedQuery
from src.models.api_models import APIKeys
from src.models.core import ResearchState, ResearchStage
from src.models.metadata import ResearchMetadata, ClarificationMetadata
from src.models.clarification import (
    ClarificationRequest, ClarificationResponse,
    ClarificationQuestion, ClarificationAnswer
)

class TestQueryTransformationAgent:
    """Test suite for QueryTransformationAgent."""

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
            timeout_seconds=30.0,
            temperature=0.7
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
    async def test_simple_query_transformation(self, transformation_agent, agent_dependencies):
        """Test transformation of simple queries."""
        mock_result = MagicMock()
        mock_result.output = TransformedQuery(
            original_query="How does machine learning work?",
            transformed_query="Explain the fundamental principles, algorithms, and applications of machine learning",
            supporting_questions=[
                "What are the main types of machine learning?",
                "How do neural networks learn from data?",
                "What are common machine learning algorithms?"
            ],
            search_keywords=["machine learning", "neural networks", "algorithms"],
            research_scope="Machine learning fundamentals and applications",
            expected_output_type="comprehensive explanation",
            transformation_rationale="Expanded query to cover key aspects of machine learning",
            specificity_score=0.8,
            confidence_score=0.85,
            assumptions_made=[],
            ambiguities_resolved=[],
            ambiguities_remaining=[]
        )

        with patch.object(transformation_agent.agent, 'run', return_value=mock_result):
            result = await transformation_agent.run(deps=agent_dependencies)

            assert isinstance(result, TransformedQuery)
            assert len(result.transformed_query) > len(result.original_query)
            assert len(result.supporting_questions) >= 3
            assert result.specificity_score > 0.7

    @pytest.mark.asyncio
    async def test_transformation_with_clarification(self, transformation_agent, agent_dependencies):
        """Test transformation using clarification data."""
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

        mock_result = MagicMock()
        mock_result.output = TransformedQuery(
            original_query="How does machine learning work?",
            transformed_query="How are machine learning applications implemented in real-world scenarios",
            supporting_questions=[
                "What are common ML applications in industry?",
                "How to implement ML models in production?",
                "What are best practices for ML deployment?"
            ],
            search_keywords=["ML applications", "deployment", "production"],
            research_scope="Machine learning applications and implementation",
            expected_output_type="practical guide",
            transformation_rationale="Focused on applications based on clarification",
            specificity_score=0.9,
            confidence_score=0.95,
            clarification_coverage=1.0,
            assumptions_made=[],
            ambiguities_resolved=["Focus area: Applications"],
            ambiguities_remaining=[]
        )

        with patch.object(transformation_agent.agent, 'run', return_value=mock_result):
            result = await transformation_agent.run(deps=agent_dependencies)

            assert result.clarification_coverage == 1.0
            assert len(result.ambiguities_resolved) > 0
            assert result.confidence_score > 0.9

    @pytest.mark.asyncio
    async def test_vague_query_transformation(self, transformation_agent, agent_dependencies):
        """Test transformation of vague queries."""
        agent_dependencies.research_state.user_query = "Tell me about stuff"

        mock_result = MagicMock()
        mock_result.output = TransformedQuery(
            original_query="Tell me about stuff",
            transformed_query="Provide general information about common topics of interest",
            supporting_questions=[
                "What are popular educational topics?",
                "What information is commonly requested?",
                "What are trending subjects?"
            ],
            search_keywords=["general information", "topics"],
            research_scope="General knowledge overview",
            expected_output_type="summary",
            transformation_rationale="Query too vague - made assumptions about general interest",
            specificity_score=0.3,
            confidence_score=0.4,
            assumptions_made=["User wants general educational content", "Focus on popular topics"],
            ambiguities_resolved=[],
            ambiguities_remaining=["Specific topic area", "Depth of information needed"]
        )

        with patch.object(transformation_agent.agent, 'run', return_value=mock_result):
            result = await transformation_agent.run(deps=agent_dependencies)

            assert result.specificity_score < 0.5
            assert len(result.assumptions_made) > 0
            assert len(result.ambiguities_remaining) > 0

    @pytest.mark.asyncio
    async def test_supporting_questions_generation(self, transformation_agent, agent_dependencies):
        """Test that supporting questions are properly generated."""
        mock_result = MagicMock()
        mock_result.output = TransformedQuery(
            original_query="How to build a web app?",
            transformed_query="Step-by-step guide to building modern web applications",
            supporting_questions=[
                "What technology stack to choose?",
                "How to set up the development environment?",
                "What are best practices for web development?",
                "How to deploy web applications?"
            ],
            search_keywords=["web development", "deployment", "frameworks"],
            research_scope="Full-stack web development",
            expected_output_type="tutorial",
            transformation_rationale="Expanded to comprehensive web development guide",
            specificity_score=0.7,
            confidence_score=0.8,
            assumptions_made=["Modern web technologies", "Full-stack coverage"],
            ambiguities_resolved=[],
            ambiguities_remaining=[]
        )

        with patch.object(transformation_agent.agent, 'run', return_value=mock_result):
            result = await transformation_agent.run(deps=agent_dependencies)

            assert len(result.supporting_questions) >= 3
            assert len(result.supporting_questions) <= 5
            # Validator should ensure 3-5 questions
            assert all(isinstance(q, str) for q in result.supporting_questions)

    @pytest.mark.asyncio
    async def test_search_keywords_extraction(self, transformation_agent, agent_dependencies):
        """Test keyword extraction for search."""
        mock_result = MagicMock()
        mock_result.output = TransformedQuery(
            original_query="Python async programming best practices",
            transformed_query="Comprehensive guide to Python asynchronous programming patterns and best practices",
            supporting_questions=[
                "What is asyncio and how does it work?",
                "Common async patterns in Python?",
                "Performance considerations for async code?"
            ],
            search_keywords=[
                "python", "asyncio", "async", "await",
                "coroutines", "event loop", "concurrency"
            ],
            research_scope="Python async programming",
            expected_output_type="technical guide",
            transformation_rationale="Focused on Python async best practices",
            specificity_score=0.85,
            confidence_score=0.9,
            assumptions_made=[],
            ambiguities_resolved=[],
            ambiguities_remaining=[]
        )

        with patch.object(transformation_agent.agent, 'run', return_value=mock_result):
            result = await transformation_agent.run(deps=agent_dependencies)

            assert len(result.search_keywords) >= 3
            assert len(result.search_keywords) <= 10
            assert all(isinstance(k, str) for k in result.search_keywords)

    @pytest.mark.asyncio
    async def test_scope_definition(self, transformation_agent, agent_dependencies):
        """Test scope fields population."""
        agent_dependencies.research_state.user_query = "COVID-19 impact on tech industry in 2020"

        mock_result = MagicMock()
        mock_result.output = TransformedQuery(
            original_query="COVID-19 impact on tech industry in 2020",
            transformed_query="Analysis of COVID-19 pandemic effects on technology sector during 2020",
            supporting_questions=[
                "How did remote work affect tech companies?",
                "What was the financial impact on tech stocks?",
                "Which tech sectors grew during the pandemic?"
            ],
            search_keywords=["COVID-19", "technology", "2020", "pandemic", "remote work"],
            research_scope="Technology industry pandemic impact analysis",
            temporal_scope="2020",
            geographic_scope="Global",
            domain_scope="Technology industry",
            expected_output_type="analytical report",
            transformation_rationale="Structured analysis of pandemic impact",
            specificity_score=0.9,
            confidence_score=0.95,
            assumptions_made=[],
            ambiguities_resolved=[],
            ambiguities_remaining=[]
        )

        with patch.object(transformation_agent.agent, 'run', return_value=mock_result):
            result = await transformation_agent.run(deps=agent_dependencies)

            assert result.temporal_scope == "2020"
            assert result.geographic_scope == "Global"
            assert result.domain_scope == "Technology industry"

    @pytest.mark.asyncio
    async def test_error_handling(self, transformation_agent, agent_dependencies):
        """Test error handling during transformation."""
        with patch.object(transformation_agent.agent, 'run', side_effect=Exception("Transformation failed")):
            with pytest.raises(Exception, match="Transformation failed"):
                await transformation_agent.run(deps=agent_dependencies)

    @pytest.mark.asyncio
    async def test_model_validators(self):
        """Test TransformedQuery model validators."""
        # Test supporting questions validator
        query = TransformedQuery(
            original_query="test",
            transformed_query="expanded test",
            supporting_questions=["Q1", "Q2"],  # Less than 3
            search_keywords=["test"],
            research_scope="test scope",
            expected_output_type="test",
            transformation_rationale="test",
            confidence_score=0.5
        )
        # Validator should pad to 3 questions
        assert len(query.supporting_questions) == 3

        # Test coherence validator
        query2 = TransformedQuery(
            original_query="test",
            transformed_query="test",  # Same as original
            supporting_questions=["Q1", "Q2", "Q3"],
            search_keywords=["test"],
            research_scope="test scope",
            expected_output_type="test",
            transformation_rationale="test",
            confidence_score=0.9  # High confidence despite no transformation
        )
        # Validator should reduce confidence
        assert query2.confidence_score == 0.5
