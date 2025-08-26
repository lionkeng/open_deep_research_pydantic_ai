"""Unit tests for Phase 1 clarification system improvements."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime

from open_deep_research_with_pydantic_ai.agents.clarification import ClarificationAgent, ClarifyWithUser
from open_deep_research_with_pydantic_ai.agents.base import ResearchDependencies
from open_deep_research_with_pydantic_ai.models.research import ResearchState
from open_deep_research_with_pydantic_ai.models.api_models import APIKeys, ResearchMetadata
from pydantic_ai.usage import RunUsage
import httpx


class TestClarificationAgentPhase1:
    """Test suite for Phase 1 clarification improvements."""

    def setup_method(self):
        """Set up test fixtures."""
        self.agent = ClarificationAgent()

    def test_broad_indicators_patterns(self):
        """Test that broad indicator patterns are correctly defined."""
        assert len(self.agent.BROAD_INDICATORS) > 0
        assert any("what is" in pattern for pattern in self.agent.BROAD_INDICATORS)
        assert any("explain" in pattern for pattern in self.agent.BROAD_INDICATORS)
        assert any("compare" in pattern for pattern in self.agent.BROAD_INDICATORS)

    def test_context_flags_structure(self):
        """Test that context flags are properly structured."""
        required_flags = ["audience_level", "purpose", "scope", "specificity"]
        for flag in required_flags:
            assert flag in self.agent.CONTEXT_FLAGS
            assert isinstance(self.agent.CONTEXT_FLAGS[flag], list)
            assert len(self.agent.CONTEXT_FLAGS[flag]) > 0

    def test_assess_query_breadth_broad_queries(self):
        """Test breadth assessment for broad queries."""
        broad_queries = [
            ("What is machine learning?", 0.4),  # Should have high breadth score
            ("How does blockchain work?", 0.4),
            ("Explain quantum computing", 0.4),
            ("Research artificial intelligence", 0.4),
            ("Compare databases", 0.4),
        ]

        for query, min_expected_score in broad_queries:
            score, metadata = self.agent._assess_query_breadth(query, [])
            assert score >= min_expected_score, f"Query '{query}' should have breadth score >= {min_expected_score}, got {score}"
            assert len(metadata["broad_indicators_found"]) > 0, f"Query '{query}' should have broad indicators"

    def test_assess_query_breadth_specific_queries(self):
        """Test breadth assessment for specific queries."""
        specific_queries = [
            ("Compare React hooks useState vs useReducer for TypeScript form management", 0.3),
            ("Implement JWT authentication in FastAPI with Python 3.11", 0.3),
            ("PostgreSQL performance tuning for time-series data with 100M records", 0.3),
        ]

        for query, max_expected_score in specific_queries:
            score, metadata = self.agent._assess_query_breadth(query, [])
            assert score <= max_expected_score, f"Query '{query}' should have breadth score <= {max_expected_score}, got {score}"

    def test_assess_query_breadth_metadata(self):
        """Test that breadth assessment returns proper metadata."""
        query = "What is the best programming language?"
        conversation = ["I'm a beginner developer", "I want to build web applications"]

        score, metadata = self.agent._assess_query_breadth(query, conversation)

        # Check required metadata fields
        required_fields = ["broad_indicators_found", "missing_context_flags", "word_count", "has_specific_terms", "has_constraints"]
        for field in required_fields:
            assert field in metadata, f"Metadata should contain {field}"

        # Check word count
        assert metadata["word_count"] == len(query.split())
        assert metadata["word_count"] > 0

    def test_has_specific_terms(self):
        """Test detection of specific technical terms."""
        specific_queries = [
            "Python 3.11 performance improvements",
            "React hooks API documentation",
            "Docker containerization with AWS ECS",
            "PostgreSQL query optimization techniques",
        ]

        for query in specific_queries:
            assert self.agent._has_specific_terms(query), f"Query '{query}' should be detected as having specific terms"

        # Test queries without specific terms
        general_queries = [
            "How to learn programming?",
            "Best practices for development",
            "What is software engineering?",
        ]

        for query in general_queries:
            assert not self.agent._has_specific_terms(query), f"Query '{query}' should not be detected as having specific terms"

    def test_has_constraints(self):
        """Test detection of explicit constraints."""
        constrained_queries = [
            "Compare React vs Angular for large-scale applications",
            "Step-by-step tutorial for Docker deployment",
            "Database design specifically for e-commerce",
            "Performance optimization using caching techniques",
        ]

        for query in constrained_queries:
            assert self.agent._has_constraints(query), f"Query '{query}' should be detected as having constraints"

        # Test queries without constraints
        unconstrained_queries = [
            "Tell me about databases",
            "What is cloud computing?",
            "Explain machine learning",
        ]

        for query in unconstrained_queries:
            assert not self.agent._has_constraints(query), f"Query '{query}' should not be detected as having constraints"

    def test_enhanced_system_prompt_structure(self):
        """Test that the enhanced system prompt has all required sections."""
        prompt = self.agent._get_default_system_prompt()

        # Check for key sections
        required_sections = [
            "CRITICAL ASSESSMENT CRITERIA",
            "Audience Level & Purpose",
            "Scope & Focus Areas",
            "Source & Quality Requirements",
            "Deliverable Specifications",
            "EXAMPLES OF QUERIES REQUIRING CLARIFICATION",
            "INFORMATION GATHERING GUIDELINES",
            "OUTPUT REQUIREMENTS",
        ]

        for section in required_sections:
            assert section in prompt, f"System prompt should contain section: {section}"

        # Check for conditional logic instructions
        assert "If you NEED to ask a clarifying question:" in prompt
        assert "If you DO NOT need clarification:" in prompt
        assert "need_clarification: true" in prompt
        assert "need_clarification: false" in prompt

    def test_system_prompt_date_placeholder(self):
        """Test that system prompt contains date placeholder."""
        prompt = self.agent._get_default_system_prompt()
        assert "{date}" in prompt, "System prompt should contain {date} placeholder"

    @pytest.mark.asyncio
    async def test_assess_query_with_breadth_integration(self):
        """Test that assess_query integrates breadth assessment properly."""
        # Create mock dependencies
        async with httpx.AsyncClient() as client:
            state = ResearchState(
                request_id="test-123",
                user_id="test-user",
                session_id="test-session",
                user_query="What is machine learning?"
            )

            deps = ResearchDependencies(
                http_client=client,
                api_keys=APIKeys(),
                research_state=state,
                metadata=ResearchMetadata(),
                usage=RunUsage()
            )

            # Mock the agent run to avoid actual LLM call
            with patch.object(self.agent, 'model') as mock_model, \
                 patch('pydantic_ai.Agent') as MockAgent:

                # Setup mock agent
                mock_agent_instance = Mock()
                mock_result = ClarifyWithUser(
                    need_clarification=True,
                    question="What is your technical background level?",
                    verification=""
                )
                mock_agent_instance.run = AsyncMock(return_value=mock_result)
                MockAgent.return_value = mock_agent_instance

                result = await self.agent.assess_query("What is machine learning?", deps)

                # Verify result structure
                assert isinstance(result, ClarifyWithUser)
                assert result.need_clarification is True
                assert result.question != ""

                # Check that breadth assessment was stored in metadata
                assert "breadth_assessment" in deps.research_state.metadata
                breadth_data = deps.research_state.metadata["breadth_assessment"]
                assert "score" in breadth_data
                assert "metadata" in breadth_data
                assert 0.0 <= breadth_data["score"] <= 1.0

    @pytest.mark.asyncio
    async def test_assess_query_error_handling(self):
        """Test error handling in assess_query method."""
        # Create mock dependencies
        async with httpx.AsyncClient() as client:
            state = ResearchState(
                request_id="test-123",
                user_id="test-user",
                session_id="test-session",
                user_query="Test query"
            )

            deps = ResearchDependencies(
                http_client=client,
                api_keys=APIKeys(),
                research_state=state,
                metadata=ResearchMetadata(),
                usage=RunUsage()
            )

            # Mock agent to raise an error
            with patch('pydantic_ai.Agent') as MockAgent:
                mock_agent_instance = Mock()
                mock_agent_instance.run = AsyncMock(side_effect=Exception("LLM Error"))
                MockAgent.return_value = mock_agent_instance

                result = await self.agent.assess_query("Test query", deps)

                # Should return conservative fallback
                assert isinstance(result, ClarifyWithUser)
                assert result.need_clarification is True
                assert "more specific details" in result.question.lower()

    @pytest.mark.asyncio
    async def test_refine_query_success(self):
        """Test successful query refinement."""
        # Create mock dependencies
        async with httpx.AsyncClient() as client:
            state = ResearchState(
                request_id="test-123",
                user_id="test-user",
                session_id="test-session",
                user_query="Test query"
            )

            deps = ResearchDependencies(
                http_client=client,
                api_keys=APIKeys(),
                research_state=state,
                metadata=ResearchMetadata(),
                usage=RunUsage()
            )

            # Mock the refinement agent
            with patch('pydantic_ai.Agent') as MockAgent:
                mock_agent_instance = Mock()
                refined_query = "How can I optimize React application performance for initial load time under 2 seconds using code splitting and caching?"
                mock_agent_instance.run = AsyncMock(return_value=refined_query)
                MockAgent.return_value = mock_agent_instance

                original_query = "How to improve website performance?"
                user_responses = {
                    "What technology?": "React application",
                    "What metrics?": "Initial load time under 2 seconds"
                }

                result = await self.agent.refine_query(original_query, user_responses, deps)

                assert result == refined_query
                assert len(result) > len(original_query)

    @pytest.mark.asyncio
    async def test_refine_query_error_handling(self):
        """Test query refinement error handling."""
        # Create mock dependencies
        async with httpx.AsyncClient() as client:
            state = ResearchState(
                request_id="test-123",
                user_id="test-user",
                session_id="test-session",
                user_query="Test query"
            )

            deps = ResearchDependencies(
                http_client=client,
                api_keys=APIKeys(),
                research_state=state,
                metadata=ResearchMetadata(),
                usage=RunUsage()
            )

            # Mock agent to raise an error
            with patch('pydantic_ai.Agent') as MockAgent:
                mock_agent_instance = Mock()
                mock_agent_instance.run = AsyncMock(side_effect=Exception("Refinement Error"))
                MockAgent.return_value = mock_agent_instance

                original_query = "Test query"
                user_responses = {"Q1": "A1"}

                result = await self.agent.refine_query(original_query, user_responses, deps)

                # Should return original query on error
                assert result == original_query

    @pytest.mark.asyncio
    async def test_should_ask_another_question_limits(self):
        """Test question limiting logic."""
        # Create mock dependencies
        async with httpx.AsyncClient() as client:
            state = ResearchState(
                request_id="test-123",
                user_id="test-user",
                session_id="test-session",
                user_query="Test query"
            )
            state.metadata = {"clarification_count": 2}  # At limit

            deps = ResearchDependencies(
                http_client=client,
                api_keys=APIKeys(),
                research_state=state,
                metadata=ResearchMetadata(),
                usage=RunUsage()
            )

            # Should not ask another question when at limit
            result = await self.agent.should_ask_another_question(deps, max_questions=2)
            assert result is False

            # Should ask when under limit
            state.metadata["clarification_count"] = 1
            result = await self.agent.should_ask_another_question(deps, max_questions=2)
            assert result is True

    def test_breadth_scoring_edge_cases(self):
        """Test breadth scoring edge cases."""
        # Test empty query
        score, metadata = self.agent._assess_query_breadth("", [])
        assert 0.0 <= score <= 1.0
        assert metadata["word_count"] == 0

        # Test very long specific query
        long_query = "How to implement OAuth 2.0 authentication flow with PKCE extension in a React TypeScript application using Auth0 service for protecting REST API endpoints built with FastAPI framework and PostgreSQL database running on AWS ECS containers with proper token refresh handling and logout functionality"
        score, metadata = self.agent._assess_query_breadth(long_query, [])
        assert score < 0.5  # Should be recognized as specific despite length

        # Test query with conversation context
        query = "What is the best approach?"
        conversation = ["I'm building a web application", "I need to choose a database", "It's for an e-commerce site"]
        score_with_context, _ = self.agent._assess_query_breadth(query, conversation)

        score_without_context, _ = self.agent._assess_query_breadth(query, [])
        # Context should generally reduce breadth score
        assert score_with_context <= score_without_context
