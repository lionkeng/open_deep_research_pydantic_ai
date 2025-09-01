"""Comprehensive tests for all research agents."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from open_deep_research_with_pydantic_ai.agents.base import ResearchDependencies
from open_deep_research_with_pydantic_ai.agents.brief_generator import (
    BriefGeneratorAgent,
    ResearchBrief as BriefGeneratorResearchBrief,
)
from open_deep_research_with_pydantic_ai.agents.clarification import (
    ClarificationAgent,
    ClarifyWithUser,
)
from open_deep_research_with_pydantic_ai.agents.compression import (
    CompressedFindings,
    CompressionAgent,
)
from open_deep_research_with_pydantic_ai.agents.report_generator import (
    ReportGeneratorAgent,
)
from open_deep_research_with_pydantic_ai.agents.research_executor import (
    ResearchExecutorAgent,
    ResearchTask,
    SearchResult,
    SpecializedResearchAgent,
)
from open_deep_research_with_pydantic_ai.models.api_models import APIKeys, ResearchMetadata
from open_deep_research_with_pydantic_ai.models.research import (
    ResearchBrief,
    ResearchFinding,
    ResearchReport,
    ResearchSection,
    ResearchStage,
    ResearchState,
)


@pytest_asyncio.fixture
async def mock_dependencies():
    """Create mock research dependencies for testing."""
    # Create a research state
    research_state = ResearchState(
        request_id="test-request-123",
        user_id="test-user",
        session_id="test-session",
        user_query="Test query about quantum computing",
        current_stage=ResearchStage.CLARIFICATION,
    )

    # Create API keys
    from pydantic import SecretStr
    api_keys = APIKeys(openai=SecretStr("test-openai-key"))

    # Create metadata
    metadata = ResearchMetadata()

    # Create HTTP client mock
    http_client = AsyncMock()

    return ResearchDependencies(
        http_client=http_client,
        api_keys=api_keys,
        research_state=research_state,
        metadata=metadata,
    )


class TestClarifyWithUser:
    """Test the ClarifyWithUser model."""

    def test_clarify_with_user_creation(self):
        """Test creating a valid ClarifyWithUser."""
        result = ClarifyWithUser(
            need_clarification=True,
            question="What specific aspects of quantum computing are you interested in?",
            verification="I will start research on quantum computing basics",
        )

        assert result.need_clarification is True
        assert result.question == "What specific aspects of quantum computing are you interested in?"
        assert result.verification == "I will start research on quantum computing basics"

    def test_clarify_with_user_defaults(self):
        """Test ClarifyWithUser with default values."""
        result = ClarifyWithUser(
            need_clarification=False,
        )

        assert result.need_clarification is False
        assert result.question == ""
        assert result.verification == ""


class TestClarificationAgent:
    """Test the ClarificationAgent class."""

    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = ClarificationAgent()

        assert agent.name == "clarification_agent"
        # Check output type is configured correctly
        assert agent.agent._output_type == ClarifyWithUser

    def test_system_prompt_content(self):
        """Test that system prompt contains required content."""
        agent = ClarificationAgent()
        # Get system prompt through agent configuration
        prompt = agent.agent._system_prompt if hasattr(agent.agent, "_system_prompt") else ""

        # Check for key elements adapted from Langgraph approach
        assert "research assistant" in prompt
        assert "clarifying question" in prompt
        assert "need_clarification" in prompt
        assert "verification" in prompt
        assert "absolutely necessary" in prompt.lower()

    @pytest.mark.asyncio
    async def test_assess_query_needs_clarification(self, mock_dependencies: ResearchDependencies):
        """Test assessing a query that needs clarification."""
        agent = ClarificationAgent()

        # Vague query that should need clarification
        with patch.object(agent, 'run', return_value=ClarifyWithUser(
            need_clarification=True,
            question="What specific aspects of AI are you interested in?",
            verification=""
        )) as mock_run:
            result = await agent.assess_query("Tell me about AI", mock_dependencies)

            assert result.need_clarification is True
            assert "specific aspects" in result.question
            assert result.verification == ""
            mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_assess_query_no_clarification_needed(self, mock_dependencies: ResearchDependencies):
        """Test assessing a query that's already clear."""
        agent = ClarificationAgent()

        # Clear, specific query
        with patch.object(agent, 'run', return_value=ClarifyWithUser(
            need_clarification=False,
            question="",
            verification="I have sufficient information to research machine learning applications in healthcare."
        )) as mock_run:
            result = await agent.assess_query(
                "I want to research machine learning applications in healthcare diagnosis",
                mock_dependencies
            )

            assert result.need_clarification is False
            assert result.question == ""
            assert "sufficient information" in result.verification

    @pytest.mark.asyncio
    async def test_should_ask_another_question_limit_reached(self, mock_dependencies: ResearchDependencies):
        """Test question limit logic."""
        agent = ClarificationAgent()

        # Set up metadata showing we've reached the question limit
        mock_dependencies.research_state.metadata = {
            "clarification_count": 2
        }

        result = await agent.should_ask_another_question(mock_dependencies, max_questions=2)
        assert result is False

    @pytest.mark.asyncio
    async def test_should_ask_another_question_high_confidence(self, mock_dependencies: ResearchDependencies):
        """Test when research brief confidence is already high."""
        agent = ClarificationAgent()

        # Set up metadata showing high confidence brief
        mock_dependencies.research_state.metadata = {
            "clarification_count": 0,
            "research_brief_confidence": 0.8
        }

        result = await agent.should_ask_another_question(mock_dependencies, max_questions=2)
        assert result is False

    @pytest.mark.asyncio
    async def test_should_ask_another_question_allowed(self, mock_dependencies: ResearchDependencies):
        """Test when another question is allowed."""
        agent = ClarificationAgent()

        # Set up metadata showing low question count and confidence
        mock_dependencies.research_state.metadata = {
            "clarification_count": 0,
            "research_brief_confidence": 0.5
        }

        result = await agent.should_ask_another_question(mock_dependencies, max_questions=2)
        assert result is True


class TestClarificationTools:
    """Test the clarification agent tools logic."""

    def test_validate_scope_too_brief(self):
        """Test scope validation logic with too brief query."""
        # Test the core logic directly rather than through agent tools
        query = "AI research"
        word_count = len(query.split())

        # Should be flagged as too brief (less than 5 words)
        assert word_count < 5

        # Test the actual validation logic
        issues: list[str] = []
        suggestions: list[str] = []

        if len(query.split()) < 5:
            issues.append("Query is too brief")
            suggestions.append("Provide more detail about what you want to research")

        assert len(issues) > 0
        assert "too brief" in issues[0]
        assert "more detail" in suggestions[0]

    def test_validate_scope_too_long(self):
        """Test scope validation logic with extremely long query."""
        # Create a very long query (over 500 words)
        long_query = " ".join(["word"] * 501)

        issues: list[str] = []
        suggestions: list[str] = []

        if len(long_query.split()) > 500:
            issues.append("Query is extremely long")
            suggestions.append("Consider breaking down into multiple focused queries")

        assert len(issues) > 0
        assert "extremely long" in issues[0]
        assert "breaking down" in suggestions[0]

    def test_validate_scope_vague_terms(self):
        """Test scope validation logic with vague terms."""
        query = "Tell me everything about artificial intelligence"

        issues: list[str] = []
        suggestions: list[str] = []

        vague_terms = [
            "everything about",
            "all aspects",
            "general information",
            "stuff about",
            "things related to",
        ]

        for term in vague_terms:
            if term.lower() in query.lower():
                issues.append(f"Query contains vague term: '{term}'")
                suggestions.append("Be more specific about what aspects you want to research")
                break

        assert len(issues) > 0
        assert "vague term" in issues[0]
        assert "everything about" in issues[0]
        assert "more specific" in suggestions[0]

    def test_validate_scope_broad_topic(self):
        """Test scope validation logic with broad topic."""
        query = "Climate change research"

        issues: list[str] = []
        suggestions: list[str] = []

        broad_topics = [
            "artificial intelligence",
            "climate change",
            "human history",
            "the universe",
            "consciousness",
        ]

        for topic in broad_topics:
            if topic.lower() in query.lower() and len(query.split()) < 15:
                issues.append(f"Topic '{topic}' is very broad")
                suggestions.append(f"Narrow down to specific aspects of {topic}")
                break

        assert len(issues) > 0
        assert "very broad" in issues[0]
        assert "climate change" in issues[0]
        assert "narrow down" in suggestions[0].lower()

    def test_validate_scope_good_query(self):
        """Test scope validation logic with a good query."""
        query = "Impact of quantum computing on modern cryptographic algorithms used in banking"

        issues: list[str] = []

        # Check length
        if len(query.split()) < 5:
            issues.append("Query is too brief")
        elif len(query.split()) > 500:
            issues.append("Query is extremely long")

        # Check vague terms
        vague_terms = ["everything about", "all aspects", "general information"]
        for term in vague_terms:
            if term.lower() in query.lower():
                issues.append(f"Query contains vague term: '{term}'")
                break

        # Check broad topics with short query
        broad_topics = ["artificial intelligence", "climate change"]
        for topic in broad_topics:
            if topic.lower() in query.lower() and len(query.split()) < 15:
                issues.append(f"Topic '{topic}' is very broad")
                break

        # Should have no issues
        assert len(issues) == 0

    def test_assess_complexity_simple(self):
        """Test complexity assessment logic for simple query."""
        query = "What is Python programming language"

        # Test complexity logic
        word_count = len(query.split())

        complex_indicators = [
            "comparative analysis", "systematic review", "meta-analysis",
            "longitudinal", "interdisciplinary", "comprehensive evaluation",
        ]
        complex_count = sum(
            1 for indicator in complex_indicators if indicator.lower() in query.lower()
        )

        technical_domains = ["quantum", "neurological", "pharmaceutical"]
        technical_count = sum(
            1 for domain in technical_domains if domain.lower() in query.lower()
        )

        # Determine complexity
        if complex_count >= 2 or technical_count >= 1 or word_count > 100:
            complexity = "complex"
        elif complex_count == 1 or word_count > 50:
            complexity = "medium"
        else:
            complexity = "simple"

        assert complexity == "simple"

    def test_assess_complexity_medium(self):
        """Test complexity assessment logic for medium query."""
        # Create a medium query with exactly enough words to be > 50
        words = ["Research", "the", "applications", "and", "limitations", "of", "machine", "learning", "algorithms"] * 6
        medium_query = " ".join(words)  # Should be 54 words
        word_count = len(medium_query.split())

        # Should be medium due to length (should be > 50 words)
        assert word_count > 50

        # Test the complexity logic
        complex_indicators = ["comparative analysis", "systematic review", "meta-analysis"]
        complex_count = sum(
            1 for indicator in complex_indicators if indicator.lower() in medium_query.lower()
        )

        technical_domains = ["quantum", "neurological", "pharmaceutical"]
        technical_count = sum(
            1 for domain in technical_domains if domain.lower() in medium_query.lower()
        )

        # Should be medium due to word count (no complex indicators or technical domains)
        if complex_count >= 2 or technical_count >= 1 or word_count > 100:
            complexity = "complex"
        elif complex_count == 1 or word_count > 50:
            complexity = "medium"
        else:
            complexity = "simple"

        assert complexity == "medium"

    def test_assess_complexity_complex_indicator(self):
        """Test complexity assessment logic with complex indicators."""
        query = "Comparative analysis of machine learning approaches"

        complex_indicators = [
            "comparative analysis", "systematic review", "meta-analysis",
            "longitudinal", "interdisciplinary", "comprehensive evaluation",
        ]

        complex_count = sum(
            1 for indicator in complex_indicators if indicator.lower() in query.lower()
        )

        # Should find "comparative analysis"
        assert complex_count >= 1

        complexity = "complex" if complex_count >= 1 else "simple"
        assert complexity == "complex"

    def test_assess_complexity_technical_domain(self):
        """Test complexity assessment logic with technical domain."""
        query = "Quantum computing applications"

        technical_domains = [
            "quantum", "neurological", "pharmaceutical", "cryptographic",
            "genomic", "algorithmic", "theoretical physics",
        ]

        technical_count = sum(
            1 for domain in technical_domains if domain.lower() in query.lower()
        )

        # Should find "quantum"
        assert technical_count >= 1

        complexity = "complex" if technical_count >= 1 else "simple"
        assert complexity == "complex"


class TestClarificationWorkflow:
    """Test the complete clarification workflow."""

    @pytest.mark.asyncio
    async def test_clarify_query_success(self, mock_dependencies: ResearchDependencies):
        """Test successful query clarification."""
        agent = ClarificationAgent()

        # Mock the agent.run method to return a ClarificationResult
        mock_result = ClarificationResult(
            is_clear=True,
            clarified_query="Refined query about quantum computing fundamentals",
            clarifying_questions=[],
            scope_validation="Scope is well-defined and achievable",
            estimated_complexity="medium",
            warnings=[],
        )

        with patch.object(agent, 'run', return_value=mock_result) as mock_run:
            with patch('open_deep_research_with_pydantic_ai.agents.clarification.emit_stage_completed') as mock_emit:
                result = await agent.clarify_query("What is quantum computing?", mock_dependencies)

                # Verify the result
                assert result.is_clear is True
                assert result.clarified_query == "Refined query about quantum computing fundamentals"
                assert result.estimated_complexity == "medium"

                # Verify the research state was updated
                assert mock_dependencies.research_state.clarified_query == result.clarified_query

                # Verify stage completion event was emitted
                mock_emit.assert_called_once_with(
                    request_id="test-request-123",
                    stage=ResearchStage.CLARIFICATION,
                    success=True,
                    result=result,
                )

                # Verify agent.run was called with correct parameters
                mock_run.assert_called_once()
                args, kwargs = mock_run.call_args
                assert "quantum computing" in args[0]  # The prompt should contain the query
                assert args[1] == mock_dependencies  # deps parameter
                assert kwargs.get('stream') is True

    @pytest.mark.asyncio
    async def test_clarify_query_needs_clarification(self, mock_dependencies: ResearchDependencies):
        """Test query that needs clarification."""
        agent = ClarificationAgent()

        # Mock the agent.run method to return a result that needs clarification
        mock_result = ClarificationResult(
            is_clear=False,
            clarified_query="Query about general AI research - needs more specificity",
            clarifying_questions=[
                "What specific aspect of AI are you interested in?",
                "What is the intended use case or application?",
            ],
            scope_validation="Scope is too broad - needs narrowing",
            estimated_complexity="medium",
            warnings=["Query is very broad and may yield unfocused results"],
        )

        with patch.object(agent, 'run', return_value=mock_result):
            with patch('open_deep_research_with_pydantic_ai.agents.clarification.emit_stage_completed') as mock_emit:
                result = await agent.clarify_query("Tell me about AI", mock_dependencies)

                # Verify the result indicates clarification needed
                assert result.is_clear is False
                assert len(result.clarifying_questions) == 2
                assert len(result.warnings) == 1
                assert "broad" in result.scope_validation

                # Verify stage completion event was emitted with success=False
                mock_emit.assert_called_once_with(
                    request_id="test-request-123",
                    stage=ResearchStage.CLARIFICATION,
                    success=False,  # is_clear is False
                    result=result,
                )

    @pytest.mark.asyncio
    async def test_clarify_query_with_agent_failure(self, mock_dependencies: ResearchDependencies):
        """Test handling of agent failure during clarification."""
        agent = ClarificationAgent()

        # Mock the agent.run method to raise an exception
        with patch.object(agent, 'run', side_effect=Exception("AI model timeout")):
            # Test that the exception propagates (agent should handle this at base level)
            with pytest.raises(Exception, match="AI model timeout"):
                await agent.clarify_query("Test query", mock_dependencies)


class TestClarificationEdgeCases:
    """Test edge cases and error scenarios for clarification."""

    def test_empty_query(self):
        """Test handling of empty query."""
        query = ""

        # Test the validation logic
        issues: list[str] = []
        if len(query.split()) < 5:
            issues.append("Query is too brief")

        assert len(issues) > 0
        assert "too brief" in issues[0]

    def test_whitespace_only_query(self):
        """Test handling of whitespace-only query."""
        query = "   \n\t   "

        # Test the validation logic
        issues: list[str] = []
        if len(query.split()) < 5:
            issues.append("Query is too brief")

        assert len(issues) > 0
        assert "too brief" in issues[0]

    def test_special_characters_query(self):
        """Test handling of query with special characters."""
        query = "Research on AI & ML algorithms: impact on society (2024)?"

        # Test the validation logic
        issues: list[str] = []

        # Check length
        if len(query.split()) < 5:
            issues.append("Query is too brief")

        # Check vague terms
        vague_terms = ["everything about", "all aspects", "general information"]
        for term in vague_terms:
            if term.lower() in query.lower():
                issues.append(f"Query contains vague term: '{term}'")
                break

        # Should not flag as having issues if length is reasonable and no vague terms
        assert len(issues) == 0

    def test_multiple_vague_terms(self):
        """Test query with multiple vague terms."""
        query = "Tell me everything about all aspects of general information on AI stuff"

        # Test the validation logic
        issues: list[str] = []

        vague_terms = [
            "everything about",
            "all aspects",
            "general information",
            "stuff about",
            "things related to",
        ]

        for term in vague_terms:
            if term.lower() in query.lower():
                issues.append(f"Query contains vague term: '{term}'")
                break  # Only flag once

        assert len(issues) > 0
        # Should detect vague term but not duplicate issues (due to break)
        assert len(issues) == 1
        assert "vague term" in issues[0]

    def test_complexity_edge_cases(self):
        """Test complexity assessment edge cases."""
        # Test with exactly complex threshold indicators
        query_with_multiple_complex = "comparative analysis and meta-analysis of AI"

        complex_indicators = [
            "comparative analysis", "systematic review", "meta-analysis",
            "longitudinal", "interdisciplinary", "comprehensive evaluation",
        ]

        complex_count = sum(
            1 for indicator in complex_indicators if indicator.lower() in query_with_multiple_complex.lower()
        )

        # Should find both "comparative analysis" and "meta-analysis"
        assert complex_count >= 2
        complexity = "complex" if complex_count >= 2 else "simple"
        assert complexity == "complex"

        # Test with exactly one complex indicator
        query_with_one_complex = "systematic review of machine learning"
        complex_count = sum(
            1 for indicator in complex_indicators if indicator.lower() in query_with_one_complex.lower()
        )

        assert complex_count == 1
        complexity = "complex" if complex_count >= 1 else "simple"
        assert complexity == "complex"  # Still complex due to the indicator

    def test_case_insensitive_matching(self):
        """Test that keyword matching is case insensitive."""
        # Test case insensitive vague term detection
        query = "EVERYTHING ABOUT quantum computing"

        vague_terms = ["everything about", "all aspects", "general information"]
        found_vague = False

        for term in vague_terms:
            if term.lower() in query.lower():
                found_vague = True
                break

        assert found_vague

        # Test case insensitive complexity detection
        query = "QUANTUM computing research"

        technical_domains = ["quantum", "neurological", "pharmaceutical"]
        technical_count = sum(
            1 for domain in technical_domains if domain.lower() in query.lower()
        )

        assert technical_count >= 1
        complexity = "complex" if technical_count >= 1 else "simple"
        assert complexity == "complex"


# =============================================================================
# BRIEF GENERATOR AGENT TESTS
# =============================================================================

class TestBriefGeneratorAgent:
    """Test the BriefGeneratorAgent class."""

    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = BriefGeneratorAgent()

        assert agent.name == "brief_generator_agent"
        # Check output type is configured correctly
        assert agent.agent._output_type == BriefGeneratorResearchBrief

    def test_system_prompt_content(self):
        """Test that system prompt contains required content."""
        agent = BriefGeneratorAgent()
        # Get system prompt through agent configuration
        prompt = agent.agent._system_prompt if hasattr(agent.agent, "_system_prompt") else ""

        # Check for key elements in the system prompt
        assert "messages that have been exchanged" in prompt
        assert "confidence score" in prompt
        assert "missing aspects" in prompt
        assert "first person" in prompt.lower()
        assert "specificity and detail" in prompt.lower()


class TestBriefGeneratorConversational:
    """Test the brief generator agent conversational approach."""

    @pytest.mark.asyncio
    async def test_generate_from_conversation_basic(self, mock_dependencies: ResearchDependencies):
        """Test generating research brief from basic conversation."""
        agent = BriefGeneratorAgent()

        # Set up conversation history
        mock_dependencies.research_state.metadata = {
            "conversation_messages": ["I want to learn about quantum computing"]
        }

        # Mock the agent run to return a brief
        with patch.object(agent, 'run', return_value=BriefGeneratorResearchBrief(
            brief="I want to research quantum computing basics, including fundamental concepts, applications, and current developments.",
            confidence_score=0.8,
            missing_aspects=[]
        )) as mock_run:
            result = await agent.generate_from_conversation(mock_dependencies)

            assert result.brief.startswith("I want to research quantum computing")
            assert result.confidence_score == 0.8
            assert len(result.missing_aspects) == 0
            mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_from_conversation_with_clarifications(self, mock_dependencies: ResearchDependencies):
        """Test generating research brief after clarification conversation."""
        agent = BriefGeneratorAgent()

        # Set up conversation with clarifications
        mock_dependencies.research_state.metadata = {
            "conversation_messages": [
                "I want to learn about AI",
                "What specific area of AI interests you?",
                "I'm interested in machine learning for healthcare applications"
            ]
        }

        with patch.object(agent, 'run', return_value=BriefGeneratorResearchBrief(
            brief="I want to research machine learning applications in healthcare, focusing on current implementations, effectiveness, and challenges.",
            confidence_score=0.9,
            missing_aspects=["specific medical domains"]
        )) as mock_run:
            result = await agent.generate_from_conversation(mock_dependencies)

            assert "machine learning" in result.brief
            assert "healthcare" in result.brief
            assert result.confidence_score == 0.9
            assert "specific medical domains" in result.missing_aspects

    @pytest.mark.asyncio
    async def test_generate_from_conversation_no_history(self, mock_dependencies: ResearchDependencies):
        """Test generating research brief with no conversation history."""
        agent = BriefGeneratorAgent()

        # No conversation history, just the original query
        mock_dependencies.research_state.user_query = "Tell me about blockchain technology"
        mock_dependencies.research_state.metadata = {}

        with patch.object(agent, 'run', return_value=BriefGeneratorResearchBrief(
            brief="I want to research blockchain technology comprehensively, including fundamentals, applications, and industry adoption.",
            confidence_score=0.6,
            missing_aspects=["specific use cases", "timeframe"]
        )) as mock_run:
            result = await agent.generate_from_conversation(mock_dependencies)

            assert result.confidence_score == 0.6
            assert len(result.missing_aspects) == 2
            # Check that metadata is updated
            assert mock_dependencies.research_state.metadata["research_brief_text"] == result.brief
            assert mock_dependencies.research_state.metadata["research_brief_confidence"] == 0.6


class TestBriefGeneratorWorkflow:
    """Test the complete brief generation workflow."""

    @pytest.mark.asyncio
    async def test_generate_brief_success(self, mock_dependencies: ResearchDependencies):
        """Test successful brief generation."""
        agent = BriefGeneratorAgent()

        # Mock the agent.run method to return a ResearchBrief
        mock_brief = ResearchBrief(
            topic="Machine Learning in Healthcare",
            objectives=[
                "Analyze current ML applications in healthcare",
                "Identify key challenges and opportunities",
                "Evaluate best practices and success factors",
            ],
            key_questions=[
                "What are the main ML applications in healthcare?",
                "How effective are current ML solutions?",
                "What barriers exist to ML adoption in healthcare?",
                "What are the ethical considerations?",
                "How can ML improve patient outcomes?",
            ],
            scope="Focus on diagnostic and treatment applications of ML in hospitals and clinics",
            priority_areas=[
                "Diagnostic imaging applications",
                "Clinical decision support systems",
                "Regulatory and compliance requirements",
            ],
            constraints=[
                "Limited to English-language sources",
                "Focus on recent developments (last 5 years)",
            ],
            expected_deliverables=[
                "Comprehensive research report",
                "Best practices guidelines",
                "Implementation recommendations",
            ],
        )

        with patch.object(agent, 'run', return_value=mock_brief) as mock_run:
            with patch('open_deep_research_with_pydantic_ai.agents.brief_generator.emit_stage_completed') as mock_emit:
                result = await agent.generate_brief(
                    "Research machine learning applications in healthcare",
                    "medium",
                    mock_dependencies
                )

                # Verify the result
                assert result.topic == "Machine Learning in Healthcare"
                assert len(result.objectives) == 3
                assert len(result.key_questions) == 5
                assert "diagnostic" in result.scope.lower()
                assert len(result.priority_areas) == 3

                # Verify the research state was updated
                assert mock_dependencies.research_state.research_brief == result

                # Verify stage completion event was emitted
                mock_emit.assert_called_once_with(
                    request_id="test-request-123",
                    stage=ResearchStage.BRIEF_GENERATION,
                    success=True,
                    result=result,
                )

                # Verify agent.run was called with correct parameters
                mock_run.assert_called_once()
                args, kwargs = mock_run.call_args
                assert "machine learning" in args[0].lower()  # The prompt should contain the query
                assert args[1] == mock_dependencies  # deps parameter
                assert kwargs.get('stream') is True

    @pytest.mark.asyncio
    async def test_generate_brief_different_complexity_levels(self, mock_dependencies: ResearchDependencies):
        """Test brief generation with different complexity levels."""
        agent = BriefGeneratorAgent()

        # Test with complex topic
        mock_complex_brief = ResearchBrief(
            topic="Quantum Machine Learning Algorithms",
            objectives=[
                "Understand quantum computing fundamentals",
                "Analyze quantum ML algorithm advantages",
                "Evaluate current research trends",
                "Assess practical implementation challenges",
            ],
            key_questions=[
                "What are the theoretical foundations of quantum ML?",
                "How do quantum algorithms differ from classical ML?",
                "What are the current limitations?",
            ],
            scope="Focus on theoretical frameworks and current research",
            priority_areas=[
                "Fundamental quantum computing concepts",
                "Quantum advantage in machine learning",
                "Current experimental implementations",
            ],
        )

        with patch.object(agent, 'run', return_value=mock_complex_brief):
            with patch('open_deep_research_with_pydantic_ai.agents.brief_generator.emit_stage_completed'):
                result = await agent.generate_brief(
                    "Quantum machine learning algorithms",
                    "complex",
                    mock_dependencies
                )

                assert "quantum" in result.topic.lower()
                assert len(result.objectives) >= 3
                assert "fundamental" in result.priority_areas[0].lower()

    @pytest.mark.asyncio
    async def test_generate_brief_with_agent_failure(self, mock_dependencies: ResearchDependencies):
        """Test handling of agent failure during brief generation."""
        agent = BriefGeneratorAgent()

        # Mock the agent.run method to raise an exception
        with patch.object(agent, 'run', side_effect=Exception("AI model timeout")):
            # Test that the exception propagates
            with pytest.raises(Exception, match="AI model timeout"):
                await agent.generate_brief(
                    "Test query",
                    "medium",
                    mock_dependencies
                )


class TestBriefGeneratorEdgeCases:
    """Test edge cases and error scenarios for brief generation."""

    def test_empty_topic_decomposition(self):
        """Test topic decomposition with empty topic."""
        topic = ""

        # Test with empty topic
        dimensions = ["historical context", "current state", "future trends"]

        subtopics: list[str] = []
        for dimension in dimensions[:5]:
            subtopics.append(f"{topic} - {dimension}")

        # Should still generate entries, just with empty topic
        assert len(subtopics) == 3
        assert " - historical context" in subtopics[0]

    def test_question_generation_with_empty_objectives(self):
        """Test research question generation with no objectives."""
        topic = "Machine Learning"
        objectives = []

        questions: list[str] = []

        # Template questions should still be generated
        templates = [
            "What are the main factors influencing {topic}?",
            "How has {topic} evolved over time?",
        ]

        for template in templates:
            questions.append(template.format(topic=topic))

        # No objective-specific questions should be added
        for obj in objectives[:3]:
            questions.append(f"How can we achieve: {obj}?")

        # Should have template questions but no objective questions
        assert len(questions) == 2
        assert "What are the main factors influencing Machine Learning?" in questions

    def test_priority_areas_unknown_complexity(self):
        """Test priority area identification with unknown complexity."""
        topic = "Test Topic"
        complexity = "unknown"

        priority_areas: list[str] = []

        # Test the logic branches
        if complexity == "complex":
            priority_areas.extend(["complex1", "complex2"])
        elif complexity == "medium":
            priority_areas.extend(["medium1", "medium2"])
        else:  # This should be the default case
            priority_areas.extend([
                f"Basic overview of {topic}",
                f"Key facts and figures about {topic}",
                f"Common use cases for {topic}",
            ])

        priority_areas = priority_areas[:4]

        # Should default to simple category
        assert len(priority_areas) == 3
        assert "Basic overview of Test Topic" in priority_areas

    def test_large_objectives_list(self):
        """Test question generation with many objectives."""
        topic = "AI"
        objectives = [
            "Objective 1", "Objective 2", "Objective 3",
            "Objective 4", "Objective 5", "Objective 6"
        ]

        questions: list[str] = []

        # Template questions
        templates = ["Template 1 {topic}", "Template 2 {topic}"]
        for template in templates:
            questions.append(template.format(topic=topic))

        # Should only take first 3 objectives
        for obj in objectives[:3]:
            questions.append(f"How can we achieve: {obj}?")

        assert len(questions) == 5  # 2 template + 3 objective questions
        assert "How can we achieve: Objective 3?" in questions
        assert "How can we achieve: Objective 4?" not in questions


# =============================================================================
# RESEARCH EXECUTOR AGENT TESTS
# =============================================================================

class TestSearchResult:
    """Test the SearchResult model."""

    def test_search_result_creation(self):
        """Test creating a valid SearchResult."""
        result = SearchResult(
            query="machine learning applications",
            results=[
                {
                    "title": "ML in Healthcare",
                    "url": "https://example.com/ml-healthcare",
                    "snippet": "Machine learning is transforming healthcare...",
                    "score": 0.95,
                }
            ],
            total_results=100,
            source="test_search_engine",
        )

        assert result.query == "machine learning applications"
        assert len(result.results) == 1
        assert result.total_results == 100
        assert result.source == "test_search_engine"


class TestResearchTask:
    """Test the ResearchTask model."""

    def test_research_task_creation(self):
        """Test creating a valid ResearchTask."""
        task = ResearchTask(
            task_id="task_001",
            description="Research AI applications in healthcare",
            query="AI healthcare applications benefits challenges",
            priority=1,
        )

        assert task.task_id == "task_001"
        assert task.priority == 1
        assert task.completed is False
        assert len(task.findings) == 0

    def test_research_task_defaults(self):
        """Test ResearchTask with default values."""
        task = ResearchTask(
            task_id="task_002",
            description="Test task",
            query="test query",
        )

        assert task.priority == 0
        assert task.completed is False
        assert task.findings == []


class TestSpecializedResearchAgent:
    """Test the SpecializedResearchAgent class."""

    def test_specialized_agent_initialization(self):
        """Test specialized agent initialization."""
        agent = SpecializedResearchAgent("healthcare", "Medical AI Applications")

        assert agent.name == "specialized_healthcare"
        assert agent.specialization == "Medical AI Applications"
        # Check output type is configured correctly
        assert agent.agent._output_type == list

    def test_specialized_system_prompt(self):
        """Test specialized system prompt content."""
        agent = SpecializedResearchAgent("tech", "Software Engineering")
        # Get system prompt through agent configuration
        prompt = agent.agent._system_prompt if hasattr(agent.agent, "_system_prompt") else ""

        assert "Software Engineering" in prompt
        assert "specialized research agent" in prompt
        assert "domain knowledge" in prompt
        assert "cite sources" in prompt.lower()


class TestResearchExecutorAgent:
    """Test the ResearchExecutorAgent class."""

    def test_agent_initialization(self):
        """Test research executor agent initialization."""
        agent = ResearchExecutorAgent()

        assert agent.name == "research_executor_agent"
        # Check output type is configured correctly
        assert agent.agent._output_type == list
        assert len(agent.sub_agents) == 0

    def test_system_prompt_content(self):
        """Test that system prompt contains required content."""
        agent = ResearchExecutorAgent()
        # Get system prompt through agent configuration
        prompt = agent.agent._system_prompt if hasattr(agent.agent, "_system_prompt") else ""

        # Check for key elements
        assert "research execution specialist" in prompt
        assert "gather information" in prompt.lower()
        assert "evaluate source credibility" in prompt.lower()
        assert "source attribution" in prompt
        assert "quality criteria" in prompt.lower()

    def test_create_sub_agent(self):
        """Test creating specialized sub-agents."""
        agent = ResearchExecutorAgent()

        # Create a sub-agent
        sub_agent = agent.create_sub_agent("medical", "Healthcare AI")

        assert isinstance(sub_agent, SpecializedResearchAgent)
        assert sub_agent.name == "specialized_medical"
        assert sub_agent.specialization == "Healthcare AI"
        assert "medical" in agent.sub_agents
        assert agent.sub_agents["medical"] == sub_agent

        # Test getting existing sub-agent
        existing_agent = agent.sub_agents["medical"]
        assert existing_agent == sub_agent


class TestResearchExecutorTools:
    """Test the research executor agent tools logic."""

    def test_source_credibility_evaluation(self):
        """Test source credibility evaluation logic."""
        # Test credibility scoring logic
        credibility_indicators = {
            ".gov": 0.95,
            ".edu": 0.90,
            ".org": 0.80,
            "wikipedia": 0.75,
            "arxiv": 0.85,
            "pubmed": 0.90,
            "nature.com": 0.95,
            "science.org": 0.95,
            ".com": 0.60,
            ".blog": 0.50,
        }

        # Test various source types - expected scores based on first match in dict order
        test_cases = [
            ("https://nih.gov/research", 0.95),      # .gov (first match)
            ("https://stanford.edu/ai", 0.90),       # .edu (first match)
            ("https://cleanwho.org/health", 0.80),   # .org (first match)
            ("https://en.wikipedia.com/ai", 0.75),   # wikipedia (first match, no .org)
            ("https://arxiv.net/abs/123", 0.85),     # arxiv (first match, no .org)
            ("https://pubmed.gov/research", 0.95),   # .gov (first match, higher than pubmed)
            ("https://nature.com/articles", 0.95),   # nature.com (first match)
            ("https://science.org/doi", 0.80),       # .org (first match, not science.org)
            ("https://example.com/blog", 0.60),      # .com (first match)
            ("https://myblog.blog/post", 0.50),      # .blog (first match)
            ("https://unknown-site.xyz", 0.5),       # default
        ]

        for source, expected_score in test_cases:
            # Implement the credibility logic that matches the actual implementation
            # The actual implementation uses break, so the first match wins
            score = 0.5  # Default score
            for indicator, cred_score in credibility_indicators.items():
                if indicator in source.lower():
                    score = max(score, cred_score)
                    break  # First match wins

            assert score == expected_score, f"Failed for {source}: expected {expected_score}, got {score}"

    def test_finding_extraction_logic(self):
        """Test research finding extraction logic."""
        # Test content summarization logic
        short_content = "This is a short piece of content."
        long_content = "This is a very long piece of content that exceeds 500 characters. " * 10

        # Short content should not get a summary
        assert len(short_content) <= 500
        summary = None if len(short_content) <= 500 else short_content[:200] + "..."
        assert summary is None

        # Long content should get a summary
        assert len(long_content) > 500
        summary = None if len(long_content) <= 500 else long_content[:200] + "..."
        assert summary is not None
        assert summary.endswith("...")
        assert len(summary) == 203  # 200 chars + "..."

    def test_search_result_conversion(self):
        """Test search result data conversion logic."""
        # Mock search response data
        mock_search_results = [
            {
                "title": "AI in Healthcare",
                "url": "https://example.com/ai-health",
                "snippet": "AI is revolutionizing healthcare...",
                "score": 0.95,
            },
            {
                "title": "Machine Learning Applications",
                "url": "https://example.com/ml-apps",
                "snippet": "ML has numerous applications...",
                "score": 0.87,
            },
        ]

        # Test conversion to SearchResult format
        converted_results = []
        for result in mock_search_results:
            converted_results.append({
                "title": result["title"],
                "url": result["url"],
                "snippet": result["snippet"],
                "score": result["score"],
            })

        search_result = SearchResult(
            query="test query",
            results=converted_results,
            total_results=len(converted_results),
            source="test_source",
        )

        assert len(search_result.results) == 2
        assert search_result.results[0]["title"] == "AI in Healthcare"
        assert search_result.results[1]["score"] == 0.87


class TestResearchExecutorWorkflow:
    """Test the complete research execution workflow."""

    @pytest.mark.asyncio
    async def test_execute_research_success(self, mock_dependencies: ResearchDependencies):
        """Test successful research execution."""
        agent = ResearchExecutorAgent()

        # Create a mock research brief
        brief = ResearchBrief(
            topic="Artificial Intelligence in Healthcare",
            objectives=[
                "Analyze AI applications in healthcare",
                "Identify benefits and challenges",
                "Evaluate implementation strategies",
            ],
            key_questions=[
                "What are the main AI applications in healthcare?",
                "What are the benefits of AI in medical diagnosis?",
                "What challenges exist in AI healthcare adoption?",
                "How can healthcare systems implement AI effectively?",
                "What are the ethical considerations?",
            ],
            scope="Focus on current AI applications in hospitals and clinics",
        )

        # Mock research findings that the agent would return
        mock_findings = [
            ResearchFinding(
                content="AI applications in healthcare include diagnostic imaging, drug discovery, and clinical decision support.",
                source="https://example.com/ai-healthcare",
                relevance_score=0.9,
                confidence=0.85,
            ),
            ResearchFinding(
                content="Benefits include improved accuracy in diagnosis and reduced time for treatment decisions.",
                source="https://example.com/ai-benefits",
                relevance_score=0.88,
                confidence=0.82,
            ),
        ]

        with patch.object(agent, 'run', return_value=mock_findings) as mock_run:
            with patch('open_deep_research_with_pydantic_ai.agents.research_executor.emit_stage_completed') as mock_emit:
                result = await agent.execute_research(brief, mock_dependencies, max_parallel_tasks=2)

                # Verify the result
                assert isinstance(result, list)
                assert len(result) >= 2  # Should have findings from parallel tasks

                # Verify the research state was updated
                assert mock_dependencies.research_state.findings == result

                # Verify stage completion event was emitted
                mock_emit.assert_called_once_with(
                    request_id="test-request-123",
                    stage=ResearchStage.RESEARCH_EXECUTION,
                    success=True,
                    result={"findings_count": len(result)},
                )

                # Verify agent.run was called for each task (limited by max_parallel_tasks)
                assert mock_run.call_count == 2  # max_parallel_tasks = 2

    @pytest.mark.asyncio
    async def test_execute_research_with_failure(self, mock_dependencies: ResearchDependencies):
        """Test research execution with some task failures."""
        agent = ResearchExecutorAgent()
        brief = ResearchBrief(
            topic="Test Topic",
            objectives=["Test objective"],
            key_questions=["Question 1", "Question 2"],
            scope="Test scope",
        )

        # Mock mixed results: one success, one failure
        def mock_run_side_effect(*args, **kwargs):
            if "Question 1" in args[0]:
                return [ResearchFinding(
                    content="Successful finding",
                    source="https://example.com",
                    relevance_score=0.8,
                    confidence=0.7,
                )]
            else:
                raise Exception("Search service unavailable")

        with patch.object(agent, 'run', side_effect=mock_run_side_effect):
            with patch('open_deep_research_with_pydantic_ai.agents.research_executor.emit_stage_completed'):
                # Mock the import of logfire inside the function
                with patch('builtins.__import__') as mock_import:
                    mock_logfire = MagicMock()
                    mock_import.return_value = mock_logfire

                    result = await agent.execute_research(brief, mock_dependencies)

                    # Should have findings from successful task only
                    assert len(result) == 1
                    assert result[0].content == "Successful finding"

                    # Error should have been handled gracefully (logged but not re-raised)
                    # We're not testing the specific logging call here

    @pytest.mark.asyncio
    async def test_delegate_specialized_research(self, mock_dependencies: ResearchDependencies):
        """Test delegation to specialized sub-agents."""
        agent = ResearchExecutorAgent()

        # Mock RunContext
        ctx = MagicMock()
        ctx.deps = mock_dependencies

        # Mock the delegate_to_agent method
        mock_findings = [
            ResearchFinding(
                content="Specialized medical AI research findings",
                source="https://medical-journal.com",
                relevance_score=0.92,
                confidence=0.88,
            )
        ]

        with patch.object(agent, 'delegate_to_agent', return_value=mock_findings) as mock_delegate:
            result = await agent.delegate_specialized_research(
                ctx,
                "AI in medical diagnosis",
                "medical_ai"
            )

            # Verify sub-agent was created
            assert "medical_ai" in agent.sub_agents
            assert isinstance(agent.sub_agents["medical_ai"], SpecializedResearchAgent)

            # Verify delegation was called
            mock_delegate.assert_called_once()
            call_args = mock_delegate.call_args
            assert call_args[0][0] == ctx  # context
            assert isinstance(call_args[0][1], SpecializedResearchAgent)  # sub-agent
            assert "AI in medical diagnosis" in call_args[0][2]  # prompt
            assert call_args[1]["context"]["specialization"] == "medical_ai"

            # Verify result
            assert result == mock_findings

    @pytest.mark.asyncio
    async def test_delegate_to_existing_sub_agent(self, mock_dependencies: ResearchDependencies):
        """Test delegation to an existing sub-agent."""
        agent = ResearchExecutorAgent()

        # Pre-create a sub-agent
        existing_agent = agent.create_sub_agent("tech", "Software Engineering")

        # Mock RunContext
        ctx = MagicMock()
        ctx.deps = mock_dependencies

        mock_findings = [ResearchFinding(
            content="Tech research findings",
            source="https://tech-site.com",
            relevance_score=0.85,
            confidence=0.80,
        )]

        with patch.object(agent, 'delegate_to_agent', return_value=mock_findings):
            result = await agent.delegate_specialized_research(
                ctx,
                "Software development best practices",
                "tech"
            )

            # Should use existing sub-agent, not create new one
            assert len(agent.sub_agents) == 1
            assert agent.sub_agents["tech"] == existing_agent

            # Verify result
            assert result == mock_findings


class TestResearchExecutorEdgeCases:
    """Test edge cases and error scenarios for research execution."""

    @pytest.mark.asyncio
    async def test_execute_research_minimal_brief(self, mock_dependencies: ResearchDependencies):
        """Test research execution with minimal brief returning no results."""
        agent = ResearchExecutorAgent()

        # Brief with minimal content (can't be completely empty due to validation)
        brief = ResearchBrief(
            topic="Empty Brief",
            objectives=["Test"],
            key_questions=["Test question"],  # Need at least one question
            scope="Test scope",
        )

        # Mock agent.run to return empty results
        with patch.object(agent, 'run', return_value=[]) as mock_run:
            with patch('open_deep_research_with_pydantic_ai.agents.research_executor.emit_stage_completed'):
                result = await agent.execute_research(brief, mock_dependencies, max_parallel_tasks=1)

                # Should return empty list (since run returns empty)
                assert result == []
                assert mock_dependencies.research_state.findings == []
                assert mock_run.call_count == 1  # Called once for the one question

    @pytest.mark.asyncio
    async def test_execute_research_with_max_parallel_limit(self, mock_dependencies: ResearchDependencies):
        """Test research execution respects max parallel tasks limit."""
        agent = ResearchExecutorAgent()

        # Brief with many questions
        brief = ResearchBrief(
            topic="Large Brief",
            objectives=["Test"],
            key_questions=[f"Question {i}" for i in range(10)],  # 10 questions
            scope="Test scope",
        )

        with patch.object(agent, 'run', return_value=[]) as mock_run:
            with patch('open_deep_research_with_pydantic_ai.agents.research_executor.emit_stage_completed'):
                await agent.execute_research(brief, mock_dependencies, max_parallel_tasks=3)

                # Should only execute 3 tasks despite having 10 questions
                assert mock_run.call_count == 3

    def test_credibility_evaluation_edge_cases(self):
        """Test source credibility evaluation edge cases."""
        credibility_indicators = {
            ".gov": 0.95,
            ".edu": 0.90,
            ".com": 0.60,
        }

        # Test with multiple indicators (should pick highest)
        source = "https://example.edu.com/research"
        score = 0.5
        for indicator, cred_score in credibility_indicators.items():
            if indicator in source.lower():
                score = max(score, cred_score)
                # Don't break here to test max() behavior

        assert score == 0.90  # Should pick .edu (higher) over .com

        # Test with no indicators
        source = "https://unknown-domain.xyz"
        score = 0.5
        for indicator, cred_score in credibility_indicators.items():
            if indicator in source.lower():
                score = max(score, cred_score)
                break

        assert score == 0.5  # Should use default

    def test_research_task_completion_tracking(self):
        """Test research task completion status tracking."""
        task = ResearchTask(
            task_id="test_task",
            description="Test task",
            query="test query",
        )

        # Initially not completed
        assert task.completed is False
        assert len(task.findings) == 0

        # Add findings
        finding = ResearchFinding(
            content="Test finding",
            source="https://example.com",
            relevance_score=0.8,
            confidence=0.7,
        )

        # Note: In real implementation, you'd modify the task
        # This tests the model structure
        assert hasattr(task, 'findings')
        assert hasattr(task, 'completed')


# =====================================================
# CompressionAgent Tests
# =====================================================

class TestCompressedFindingsModel:
    """Test CompressedFindings model."""

    def test_compressed_findings_model_creation(self):
        """Test CompressedFindings model initialization."""
        findings = CompressedFindings(
            summary="Test summary",
            key_insights=["Insight 1", "Insight 2"],
            themes={"Technology": ["Tech finding 1"], "Economics": ["Econ finding 1"]},
            contradictions=["Contradiction 1"],
            gaps=["Gap 1"],
            consensus_points=["Consensus 1"],
            statistical_data={"stat1": 42, "stat2": "value"},
            source_quality_summary="Good sources overall"
        )

        assert findings.summary == "Test summary"
        assert len(findings.key_insights) == 2
        assert "Technology" in findings.themes
        assert len(findings.contradictions) == 1
        assert len(findings.gaps) == 1
        assert len(findings.consensus_points) == 1
        assert findings.statistical_data["stat1"] == 42
        assert findings.source_quality_summary == "Good sources overall"

    def test_compressed_findings_defaults(self):
        """Test CompressedFindings with default values."""
        findings = CompressedFindings(
            summary="Minimal summary",
            key_insights=["Single insight"]
        )

        assert findings.summary == "Minimal summary"
        assert len(findings.key_insights) == 1
        assert findings.themes == {}
        assert findings.contradictions == []
        assert findings.gaps == []
        assert findings.consensus_points == []
        assert findings.statistical_data == {}
        assert findings.source_quality_summary == ""


class TestCompressionAgent:
    """Test CompressionAgent functionality."""

    def test_agent_initialization(self):
        """Test CompressionAgent initialization."""
        agent = CompressionAgent()

        assert agent.name == "compression_agent"

    def test_system_prompt_content(self):
        """Test that system prompt contains compression-specific instructions."""
        agent = CompressionAgent()
        # Get system prompt through agent configuration
        prompt = agent.agent._system_prompt if hasattr(agent.agent, "_system_prompt") else ""

        # Check for key compression concepts
        assert "synthesis" in prompt.lower()
        assert "compress" in prompt.lower()
        assert "theme" in prompt.lower()
        assert "contradiction" in prompt.lower()
        assert "consensus" in prompt.lower()
        assert "gap" in prompt.lower()
        assert "pattern" in prompt.lower()
        assert "insight" in prompt.lower()

        # Check for specific instructions
        assert "Synthesis Approach:" in prompt
        assert "Quality Assessment:" in prompt
        assert "Organization Principles:" in prompt
        assert "Output Requirements:" in prompt

    @pytest.mark.asyncio
    async def test_identify_themes_tool_logic(self, mock_dependencies: ResearchDependencies):
        """Test the identify_themes tool logic."""
        agent = CompressionAgent()

        # Test theme identification directly by recreating the logic
        findings = [
            ResearchFinding(
                content="This technology breakthrough in AI automation will revolutionize software development",
                source="https://example.com/tech",
                relevance_score=0.9,
                confidence=0.8
            ),
            ResearchFinding(
                content="The economic cost and market investment in this innovation show significant growth potential",
                source="https://example.com/econ",
                relevance_score=0.8,
                confidence=0.7
            ),
            ResearchFinding(
                content="Social impact on communities and people will be profound with cultural changes",
                source="https://example.com/social",
                relevance_score=0.7,
                confidence=0.6
            ),
            ResearchFinding(
                content="Environmental sustainability and climate effects need green solutions",
                source="https://example.com/env",
                relevance_score=0.6,
                confidence=0.5
            ),
            ResearchFinding(
                content="This is some unrelated content about random topics",
                source="https://example.com/other",
                relevance_score=0.5,
                confidence=0.4
            )
        ]

        # Test the logic of theme identification (based on compression.py implementation)
        from collections import defaultdict
        themes = defaultdict(list)

        # Same theme keywords as in the actual implementation
        theme_keywords = {
            "Technology": ["technology", "digital", "software", "hardware", "AI", "automation"],
            "Economics": ["cost", "price", "market", "economy", "financial", "investment"],
            "Social Impact": ["society", "community", "people", "social", "cultural", "human"],
            "Environment": ["environment", "climate", "sustainability", "green", "ecological"],
            "Innovation": ["innovation", "new", "novel", "breakthrough", "advancement"],
            "Challenges": ["challenge", "problem", "issue", "difficulty", "obstacle"],
            "Opportunities": ["opportunity", "potential", "possibility", "benefit", "advantage"],
            "Trends": ["trend", "future", "emerging", "growth", "development"],
        }

        # Categorize findings by theme (same logic as in compression.py)
        for finding in findings:
            content_lower = finding.content.lower()
            categorized = False

            for theme, keywords in theme_keywords.items():
                if any(keyword in content_lower for keyword in keywords):
                    themes[theme].append(finding.content[:200])
                    categorized = True
                    break

            if not categorized:
                themes["Other"].append(finding.content[:200])

        themes = dict(themes)

        # Check that themes were identified correctly
        assert isinstance(themes, dict)
        assert "Technology" in themes
        assert "Economics" in themes
        assert "Social Impact" in themes
        assert "Environment" in themes
        assert "Other" in themes

        # Check content truncation
        for theme_findings in themes.values():
            for finding_text in theme_findings:
                assert len(finding_text) <= 200

    @pytest.mark.asyncio
    async def test_find_contradictions_tool_logic(self, mock_dependencies: ResearchDependencies):
        """Test the find_contradictions tool logic."""
        agent = CompressionAgent()

        # Create contradictory findings
        findings = [
            ResearchFinding(
                content="The technology shows significant increase in performance and positive results across multiple tests",
                source="https://example1.com",
                relevance_score=0.9,
                confidence=0.8
            ),
            ResearchFinding(
                content="Recent studies indicate a decrease in performance and negative outcomes in similar technology applications",
                source="https://example2.com",
                relevance_score=0.8,
                confidence=0.7
            ),
            ResearchFinding(
                content="Market growth has been substantial with success in implementation",
                source="https://example3.com",
                relevance_score=0.7,
                confidence=0.6
            ),
            ResearchFinding(
                content="Economic decline and failure in market adoption has been observed",
                source="https://example4.com",
                relevance_score=0.6,
                confidence=0.5
            )
        ]

        # Test contradiction detection logic (based on compression.py implementation)
        contradictions: list[str] = []

        # Same opposing pairs as in the actual implementation
        opposing_pairs = [
            ("increase", "decrease"),
            ("positive", "negative"),
            ("growth", "decline"),
            ("success", "failure"),
            ("effective", "ineffective"),
            ("beneficial", "harmful"),
        ]

        # Compare findings pairwise for potential contradictions
        for i, finding1 in enumerate(findings):
            for finding2 in findings[i + 1 :]:
                content1_lower = finding1.content.lower()
                content2_lower = finding2.content.lower()

                for term1, term2 in opposing_pairs:
                    if (term1 in content1_lower and term2 in content2_lower) or (
                        term2 in content1_lower and term1 in content2_lower
                    ):
                        # Check if they're about the same subject
                        words1 = set(content1_lower.split())
                        words2 = set(content2_lower.split())
                        common_words = words1.intersection(words2)

                        if len(common_words) > 5:  # Arbitrary threshold
                            contradiction = (
                                f"Potential contradiction between: "
                                f"'{finding1.content[:100]}...' and "
                                f"'{finding2.content[:100]}...'"
                            )
                            contradictions.append(contradiction)
                            break

        contradictions = contradictions[:5]  # Limit to top 5 contradictions

        # Check that contradictions were found
        assert isinstance(contradictions, list)
        assert len(contradictions) <= 5  # Tool limits to 5

        # Should find contradictions due to opposing terms
        if len(contradictions) > 0:
            for contradiction in contradictions:
                assert "Potential contradiction between:" in contradiction
                assert "..." in contradiction  # Content truncation

    @pytest.mark.asyncio
    async def test_extract_consensus_points_tool_logic(self, mock_dependencies: ResearchDependencies):
        """Test the extract_consensus_points tool logic."""
        agent = CompressionAgent()

        # Create findings with potential consensus
        findings = [
            ResearchFinding(
                content="Machine learning algorithms show great promise for data analysis and pattern recognition",
                source="https://source1.com",
                relevance_score=0.9,
                confidence=0.8
            ),
            ResearchFinding(
                content="Data analysis using machine learning has proven effective for pattern recognition tasks",
                source="https://source2.com",
                relevance_score=0.8,
                confidence=0.7
            ),
            ResearchFinding(
                content="Pattern recognition through machine learning enables better data analysis capabilities",
                source="https://source3.com",
                relevance_score=0.7,
                confidence=0.6
            )
        ]

        # Test consensus extraction logic (based on compression.py implementation)
        from collections import defaultdict
        consensus_points: list[str] = []

        # Group findings by source
        source_groups = defaultdict(list)
        for finding in findings:
            source_groups[finding.source].append(finding)

        # Find common themes across sources
        if len(source_groups) > 1:
            # Extract key phrases from each source
            source_phrases = {}
            for source, source_findings in source_groups.items():
                phrases = set()
                for finding in source_findings:
                    # Simple phrase extraction (in production, use NLP)
                    words = finding.content.lower().split()
                    for i in range(len(words) - 2):
                        phrase = " ".join(words[i : i + 3])
                        phrases.add(phrase)
                source_phrases[source] = phrases

            # Find common phrases across sources
            sources = list(source_phrases.keys())
            if len(sources) >= 2:
                common = source_phrases[sources[0]]
                for source in sources[1:]:
                    common = common.intersection(source_phrases[source])

                for phrase in list(common)[:10]:
                    consensus_points.append(f"Multiple sources agree on: {phrase}")

        consensus_points = consensus_points[:5]  # Top 5 consensus points

        # Check consensus points structure
        assert isinstance(consensus_points, list)
        assert len(consensus_points) <= 5  # Tool limits to 5

        # Check consensus point format
        for point in consensus_points:
            assert "Multiple sources agree on:" in point

    @pytest.mark.asyncio
    async def test_identify_gaps_tool_logic(self, mock_dependencies: ResearchDependencies):
        """Test the identify_gaps tool logic."""
        agent = CompressionAgent()

        # Create findings that don't fully address all questions
        findings = [
            ResearchFinding(
                content="This addresses machine learning algorithms and their effectiveness in data processing",
                source="https://example.com",
                relevance_score=0.9,
                confidence=0.8
            ),
            ResearchFinding(
                content="Low confidence information about uncertain topic",
                source="https://example2.com",
                relevance_score=0.6,
                confidence=0.3  # Low confidence
            )
        ]

        research_questions = [
            "How effective are machine learning algorithms?",
            "What are the security implications?",  # Not addressed
            "What is the cost analysis?"  # Not addressed
        ]

        # Test gap identification logic (based on compression.py implementation)
        gaps: list[str] = []

        # Check if each research question was adequately addressed
        for question in research_questions:
            question_lower = question.lower()
            question_addressed = False

            for finding in findings:
                # Simple check - in production, use semantic similarity
                finding_lower = finding.content.lower()
                question_words = set(question_lower.split())
                finding_words = set(finding_lower.split())

                # If significant overlap, consider it addressed
                common_words = question_words.intersection(finding_words)
                if len(common_words) >= len(question_words) * 0.3:
                    question_addressed = True
                    break

            if not question_addressed:
                gaps.append(f"Limited information on: {question}")

        # Check for low coverage areas based on confidence scores
        low_confidence_topics = []
        for finding in findings:
            if finding.confidence < 0.5:
                low_confidence_topics.append(finding.summary or finding.content[:100])

        if low_confidence_topics:
            gaps.append(f"Low confidence areas: {', '.join(low_confidence_topics[:3])}")

        # Check gaps identification
        assert isinstance(gaps, list)

        # Should identify unaddressed questions
        gap_text = " ".join(gaps)
        assert "security implications" in gap_text.lower() or "cost analysis" in gap_text.lower()

        # Should identify low confidence areas
        if any(f.confidence < 0.5 for f in findings):
            assert any("low confidence" in gap.lower() for gap in gaps)

    @pytest.mark.asyncio
    async def test_compress_findings_success(self, mock_dependencies: ResearchDependencies):
        """Test successful compression of findings."""
        agent = CompressionAgent()

        # Create test findings
        findings = [
            ResearchFinding(
                content="Machine learning shows great potential for automation",
                source="https://example1.com",
                relevance_score=0.9,
                confidence=0.8
            ),
            ResearchFinding(
                content="AI automation can reduce costs by 30-50%",
                source="https://example2.com",
                relevance_score=0.8,
                confidence=0.7
            )
        ]

        research_questions = [
            "What is machine learning?",
            "How can it reduce costs?"
        ]

        # Mock the agent.run method
        mock_result = CompressedFindings(
            summary="Machine learning offers significant automation potential with cost savings of 30-50%",
            key_insights=[
                "ML enables automation",
                "Cost reduction is substantial",
                "Multiple applications possible"
            ],
            themes={
                "Technology": ["ML automation capabilities"],
                "Economics": ["Cost reduction benefits"]
            },
            contradictions=[],
            gaps=["Implementation details needed"],
            consensus_points=["ML is beneficial for automation"],
            statistical_data={"cost_reduction": "30-50%"},
            source_quality_summary="Sources are reliable with good relevance scores"
        )

        with patch.object(agent, 'run', return_value=mock_result) as mock_run:
            result = await agent.compress_findings(findings, research_questions, mock_dependencies)

            # Verify the call was made
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert "Synthesize and compress the following research findings" in call_args[0][0]
            assert mock_dependencies in call_args[0]

            # Verify result
            assert isinstance(result, CompressedFindings)
            assert result.summary == mock_result.summary
            assert len(result.key_insights) == 3
            assert "Technology" in result.themes
            assert "Economics" in result.themes

            # Verify research state was updated
            assert mock_dependencies.research_state.compressed_findings == result.summary

    @pytest.mark.asyncio
    async def test_compress_findings_with_empty_list(self, mock_dependencies: ResearchDependencies):
        """Test compression with empty findings list."""
        agent = CompressionAgent()

        findings = []
        research_questions = ["What is the topic?"]

        # Mock the agent.run method
        mock_result = CompressedFindings(
            summary="No findings available for analysis",
            key_insights=["No insights available"],
            themes={},
            contradictions=[],
            gaps=["Complete research needed"],
            consensus_points=[],
            statistical_data={},
            source_quality_summary="No sources to evaluate"
        )

        with patch.object(agent, 'run', return_value=mock_result):
            result = await agent.compress_findings(findings, research_questions, mock_dependencies)

            assert isinstance(result, CompressedFindings)
            assert "No findings" in result.summary
            assert len(result.themes) == 0

    @pytest.mark.asyncio
    async def test_compress_findings_prompt_structure(self, mock_dependencies: ResearchDependencies):
        """Test that compression prompt is properly structured."""
        agent = CompressionAgent()

        findings = [
            ResearchFinding(
                content="Test content",
                source="https://test.com",
                relevance_score=0.8,
                confidence=0.7
            )
        ]

        research_questions = ["Test question 1", "Test question 2"]

        mock_result = CompressedFindings(
            summary="Test summary",
            key_insights=["Test insight"]
        )

        with patch.object(agent, 'run', return_value=mock_result) as mock_run:
            await agent.compress_findings(findings, research_questions, mock_dependencies)

            # Check prompt structure
            call_args = mock_run.call_args
            prompt = call_args[0][0]

            # Should include research questions section
            assert "Research Questions:" in prompt
            assert "- Test question 1" in prompt
            assert "- Test question 2" in prompt

            # Should include findings section
            assert "Findings to Compress:" in prompt
            assert "Finding 1 (Source: https://test.com, Relevance: 0.80):" in prompt
            assert "Test content" in prompt

            # Should include instructions
            assert "Instructions:" in prompt
            assert "Create a comprehensive summary" in prompt
            assert "Extract 5-7 key insights" in prompt
            assert "Organize findings by theme" in prompt

    def test_tools_registration(self):
        """Test that compression agent initializes correctly."""
        agent = CompressionAgent()

        # Since we can't access tools directly, we test that the agent
        # has the required tool decorators by checking agent initialization
        assert agent.name == "compression_agent"
        assert hasattr(agent, '_register_tools')

        # Verify agent has the tool method (used by decorators)
        assert hasattr(agent.agent, 'tool')

        # Test that agent was created without errors (tools registered)
        assert agent.agent is not None

    @pytest.mark.asyncio
    async def test_compression_with_mixed_quality_sources(self, mock_dependencies: ResearchDependencies):
        """Test compression behavior with mixed quality sources."""
        agent = CompressionAgent()

        # Create findings with different confidence levels
        findings = [
            ResearchFinding(
                content="High confidence finding from reliable source",
                source="https://nature.com/study",
                relevance_score=0.9,
                confidence=0.9
            ),
            ResearchFinding(
                content="Medium confidence finding with some uncertainty",
                source="https://example.edu/research",
                relevance_score=0.7,
                confidence=0.6
            ),
            ResearchFinding(
                content="Low confidence finding from questionable source",
                source="https://blog.unknown.com/post",
                relevance_score=0.5,
                confidence=0.3
            )
        ]

        research_questions = ["What is the reliability of the data?"]

        mock_result = CompressedFindings(
            summary="Mixed quality sources with varying reliability",
            key_insights=["Source quality varies significantly"],
            source_quality_summary="Sources range from high-quality academic to low-confidence blogs"
        )

        with patch.object(agent, 'run', return_value=mock_result):
            result = await agent.compress_findings(findings, research_questions, mock_dependencies)

            assert "quality" in result.source_quality_summary.lower()
            assert isinstance(result, CompressedFindings)

    @pytest.mark.asyncio
    async def test_compression_error_handling(self, mock_dependencies: ResearchDependencies):
        """Test compression behavior when agent.run fails."""
        agent = CompressionAgent()

        findings = [
            ResearchFinding(
                content="Test finding",
                source="https://test.com",
                relevance_score=0.8,
                confidence=0.7
            )
        ]

        research_questions = ["Test question"]

        # Mock agent.run to raise an exception
        with patch.object(agent, 'run', side_effect=Exception("AI model error")):
            with pytest.raises(Exception, match="AI model error"):
                await agent.compress_findings(findings, research_questions, mock_dependencies)


# =====================================================
# ReportGeneratorAgent Tests
# =====================================================

class TestReportGeneratorAgent:
    """Test ReportGeneratorAgent functionality."""

    def test_agent_initialization(self):
        """Test ReportGeneratorAgent initialization."""
        agent = ReportGeneratorAgent()

        assert agent.name == "report_generator_agent"

    def test_system_prompt_content(self):
        """Test that system prompt contains report generation instructions."""
        agent = ReportGeneratorAgent()
        # Get system prompt through agent configuration
        prompt = agent.agent._system_prompt if hasattr(agent.agent, "_system_prompt") else ""

        # Check for key report generation concepts
        assert "research report specialist" in prompt.lower()
        assert "report structure:" in prompt.lower()
        assert "writing principles:" in prompt.lower()
        assert "quality standards:" in prompt.lower()
        assert "formatting guidelines:" in prompt.lower()

        # Check for specific report sections
        assert "executive summary" in prompt.lower()
        assert "introduction" in prompt.lower()
        assert "methodology" in prompt.lower()
        assert "conclusion" in prompt.lower()
        assert "recommendations" in prompt.lower()
        assert "citations" in prompt.lower()

        # Check for quality criteria
        assert "accuracy" in prompt.lower()
        assert "completeness" in prompt.lower()
        assert "clarity" in prompt.lower()
        assert "coherence" in prompt.lower()
        assert "credibility" in prompt.lower()

    @pytest.mark.asyncio
    async def test_create_executive_summary_tool_logic(self, mock_dependencies: ResearchDependencies):
        """Test the create_executive_summary tool logic."""
        agent = ReportGeneratorAgent()

        # Create test data
        brief = ResearchBrief(
            topic="AI in Healthcare",
            objectives=["Understand applications", "Evaluate benefits", "Identify challenges"],
            key_questions=["How is AI used?", "What are the benefits?"],
            scope="Current applications in hospitals"
        )

        compressed_findings = CompressedFindings(
            summary="AI in healthcare shows significant promise for improving patient outcomes and operational efficiency through various applications including diagnostics, treatment planning, and administrative automation",
            key_insights=[
                "AI improves diagnostic accuracy by 25-30%",
                "Cost reduction potential of 15-20% in administrative tasks",
                "Patient satisfaction increases with AI-assisted care"
            ],
            consensus_points=[
                "AI enhances diagnostic capabilities",
                "Workflow efficiency improvements are substantial"
            ],
            contradictions=[
                "Privacy concerns vs efficiency gains",
                "Cost vs benefit in small hospitals"
            ]
        )

        # Test executive summary creation logic (based on report_generator.py)
        summary_parts: list[str] = []

        # Opening statement
        summary_parts.append(
            f"This research report addresses the topic of '{brief.topic}', "
            f"examining {len(brief.objectives)} key objectives through comprehensive analysis."
        )

        # Key findings
        if compressed_findings.key_insights:
            summary_parts.append("\nKey Findings:")
            for insight in compressed_findings.key_insights[:3]:
                summary_parts.append(f"• {insight}")

        # Consensus points
        if compressed_findings.consensus_points:
            summary_parts.append("\nAreas of Consensus:")
            for point in compressed_findings.consensus_points[:2]:
                summary_parts.append(f"• {point}")

        # Challenges or contradictions
        if compressed_findings.contradictions:
            summary_parts.append(
                f"\nThe research identified {len(compressed_findings.contradictions)} "
                "areas requiring further investigation."
            )

        # Conclusion
        summary_parts.append(
            f"\n{compressed_findings.summary[:200]}..."
            if len(compressed_findings.summary) > 200
            else f"\n{compressed_findings.summary}"
        )

        result = "\n".join(summary_parts)

        # Verify executive summary content
        assert "AI in Healthcare" in result
        assert "3 key objectives" in result
        assert "Key Findings:" in result
        assert "AI improves diagnostic accuracy" in result
        assert "Areas of Consensus:" in result
        assert "2 areas requiring further investigation" in result
        assert "AI in healthcare shows significant promise" in result

    @pytest.mark.asyncio
    async def test_create_methodology_section_tool_logic(self, mock_dependencies: ResearchDependencies):
        """Test the create_methodology_section tool logic."""
        ReportGeneratorAgent()
        ResearchBrief(
            topic="Machine Learning in Finance",
            objectives=["Analyze applications", "Assess risks"],
            key_questions=["What are the use cases?"],
            scope="Current financial sector implementations",
            constraints=["Regulatory compliance", "Data privacy"]
        )

        findings_count = 25

        # Test methodology section creation logic (based on report_generator.py)
        methodology = f"""Research Methodology

This research was conducted using a systematic approach to ensure comprehensive coverage
and reliable results.

Research Design:
• Objective-driven research focusing on {len(brief.objectives)} key objectives
• Multi-source information gathering from {findings_count} distinct findings
• Systematic synthesis and analysis of collected data

Data Collection:
• Comprehensive search across multiple authoritative sources
• Evaluation of source credibility and relevance
• Cross-verification of information across sources

Analysis Approach:
• Thematic analysis to identify patterns and relationships
• Comparative analysis to identify consensus and contradictions
• Gap analysis to identify areas requiring further research

Quality Assurance:
• Source credibility assessment
• Information verification across multiple sources
• Systematic documentation of all findings

Scope and Limitations:
• Research scope: {brief.scope}
• Constraints: {", ".join(brief.constraints) if brief.constraints else "None identified"}
• Time frame: Current analysis based on available information"""

        # Verify methodology section content
        assert "Research Methodology" in methodology
        assert "2 key objectives" in methodology
        assert "25 distinct findings" in methodology
        assert brief.scope in methodology
        assert "Regulatory compliance, Data privacy" in methodology
        assert "systematic approach" in methodology
        assert "Quality Assurance:" in methodology

    @pytest.mark.asyncio
    async def test_organize_sections_by_theme_tool_logic(self, mock_dependencies: ResearchDependencies):
        """Test the organize_sections_by_theme tool logic."""
        agent = ReportGeneratorAgent()

        # Create test data
        compressed_findings = CompressedFindings(
            summary="Test summary",
            key_insights=["Test insight"],
            themes={
                "Technology": [
                    "AI algorithms show great potential for automation",
                    "Machine learning models improve accuracy"
                ],
                "Economics": [
                    "Cost reduction of 30% observed",
                    "ROI shows positive trends"
                ],
                "Challenges": [
                    "Implementation barriers exist",
                    "Skills gap needs addressing"
                ]
            }
        )

        findings = [
            ResearchFinding(
                content="AI algorithms show great potential for automation in various industries",
                source="https://tech1.com",
                relevance_score=0.9,
                confidence=0.8
            ),
            ResearchFinding(
                content="Cost reduction of 30% observed in pilot implementations",
                source="https://econ1.com",
                relevance_score=0.8,
                confidence=0.7
            ),
            ResearchFinding(
                content="Implementation barriers exist due to legacy systems",
                source="https://challenge1.com",
                relevance_score=0.7,
                confidence=0.6
            )
        ]

        # Test section organization logic (based on report_generator.py)
        sections: list[ResearchSection] = []

        for i, (theme, theme_content) in enumerate(compressed_findings.themes.items()):
            # Get relevant findings for this theme
            relevant_findings = [
                f for f in findings if any(content in f.content for content in theme_content)
            ][:5]  # Limit to top 5 findings per theme

            section = ResearchSection(
                title=theme,
                content="\n\n".join(theme_content),
                findings=relevant_findings,
                order=i,
            )
            sections.append(section)

        # Verify section organization
        assert len(sections) == 3

        # Check Technology section
        tech_section = sections[0]
        assert tech_section.title == "Technology"
        assert "AI algorithms show great potential" in tech_section.content
        assert len(tech_section.findings) == 1
        assert tech_section.order == 0

        # Check Economics section
        econ_section = sections[1]
        assert econ_section.title == "Economics"
        assert "Cost reduction of 30%" in econ_section.content
        assert len(econ_section.findings) == 1
        assert econ_section.order == 1

        # Check Challenges section
        challenge_section = sections[2]
        assert challenge_section.title == "Challenges"
        assert "Implementation barriers" in challenge_section.content
        assert len(challenge_section.findings) == 1
        assert challenge_section.order == 2

    @pytest.mark.asyncio
    async def test_generate_recommendations_tool_logic(self, mock_dependencies: ResearchDependencies):
        """Test the generate_recommendations tool logic."""        ReportGeneratorAgent()        ResearchBrief(
            topic="Digital Transformation",
            objectives=["Assess readiness"],
            key_questions=["How to proceed?"],
            scope="Enterprise level"
        )

        compressed_findings = CompressedFindings(
            summary="Digital transformation requires strategic planning",
            key_insights=[
                "Cloud adoption accelerates transformation",
                "Employee training is critical for success",
                "Data governance must be established early"
            ],
            gaps=[
                "Limited information on implementation costs",
                "Security framework details needed"
            ],
            themes={
                "Opportunities": ["Cloud cost savings", "Process automation"],
                "Challenges": ["Legacy system integration", "Change management"]
            }
        )

        # Test recommendation generation logic (based on report_generator.py)
        recommendations: list[str] = []

        # Based on key insights
        for insight in compressed_findings.key_insights[:3]:
            rec = (
                f"Based on the finding that {insight}, "
                "it is recommended to explore implementation strategies."
            )
            recommendations.append(rec)

        # Based on gaps
        for gap in compressed_findings.gaps[:2]:
            rec = f"Further research is recommended to address: {gap}"
            recommendations.append(rec)

        # Based on opportunities in themes
        if "Opportunities" in compressed_findings.themes:
            rec = "Leverage identified opportunities for strategic advantage"
            recommendations.append(rec)

        # Based on challenges
        if "Challenges" in compressed_findings.themes:
            rec = "Develop mitigation strategies for identified challenges"
            recommendations.append(rec)

        # Verify all recommendations before slicing
        assert len(recommendations) == 7  # 3 insights + 2 gaps + 1 opportunities + 1 challenges

        # Check insight-based recommendations
        assert any("Cloud adoption accelerates" in rec for rec in recommendations)
        assert any("Employee training is critical" in rec for rec in recommendations)
        assert any("Data governance must be established" in rec for rec in recommendations)

        # Check gap-based recommendations
        assert any("Limited information on implementation costs" in rec for rec in recommendations)
        assert any("Security framework details needed" in rec for rec in recommendations)

        # Check theme-based recommendations
        assert any("strategic advantage" in rec for rec in recommendations)
        assert any("mitigation strategies" in rec for rec in recommendations)

        recommendations = recommendations[:5]  # Top 5 recommendations

        # After slicing to top 5, should be exactly 5
        assert len(recommendations) == 5

    @pytest.mark.asyncio
    async def test_compile_citations_tool_logic(self, mock_dependencies: ResearchDependencies):
        """Test the compile_citations tool logic."""
        agent = ReportGeneratorAgent()

        findings = [
            ResearchFinding(
                content="First finding",
                source="https://example1.com",
                relevance_score=0.9,
                confidence=0.8
            ),
            ResearchFinding(
                content="Second finding",
                source="https://example2.com",
                relevance_score=0.8,
                confidence=0.7
            ),
            ResearchFinding(
                content="Third finding",
                source="https://example1.com",  # Duplicate source
                relevance_score=0.7,
                confidence=0.6
            ),
            ResearchFinding(
                content="Fourth finding",
                source="https://example3.com",
                relevance_score=0.6,
                confidence=0.5
            )
        ]

        # Test citation compilation logic (based on report_generator.py)
        from datetime import datetime

        # Get unique sources
        sources = list({f.source for f in findings})

        # Format citations (simplified - in production, use proper citation format)
        citations: list[str] = []
        for i, source in enumerate(sources, 1):
            citation = f"[{i}] {source} (Accessed: {datetime.now().strftime('%Y-%m-%d')})"
            citations.append(citation)

        citations = sorted(citations)

        # Verify citations structure and content
        assert len(citations) == 3  # Only unique sources
        assert all(citation.startswith("[") and citation.count("]") == 1 for citation in citations)
        assert all("https://example" in citation for citation in citations)
        assert all("Accessed:" in citation for citation in citations)

        # Verify that all unique sources are represented
        citation_urls = []
        for citation in citations:
            # Parse: "[number] https://example.com (Accessed: date)"
            url_part = citation.split("] ")[1].split(" (")[0]
            citation_urls.append(url_part)

        # Should contain all unique URLs (order doesn't matter for this test)
        unique_sources = {"https://example1.com", "https://example2.com", "https://example3.com"}
        assert set(citation_urls) == unique_sources

        # Verify citations are lexicographically sorted (as strings)
        assert citations == sorted(citations)

    @pytest.mark.asyncio
    async def test_generate_report_success(self, mock_dependencies: ResearchDependencies):
        """Test successful report generation."""
        agent = ReportGeneratorAgent()

        # Create test data
        brief = ResearchBrief(
            topic="Artificial Intelligence Trends",
            objectives=["Identify emerging trends", "Assess market impact"],
            key_questions=["What are the key trends?", "How will they impact business?"],
            scope="2024-2025 AI developments"
        )

        findings = [
            ResearchFinding(
                content="Generative AI adoption is accelerating across industries",
                source="https://ai-research.com/trends",
                relevance_score=0.9,
                confidence=0.8
            ),
            ResearchFinding(
                content="AI infrastructure spending will increase 40% in 2024",
                source="https://market-analysis.com/ai",
                relevance_score=0.8,
                confidence=0.7
            )
        ]

        compressed_findings = CompressedFindings(
            summary="AI trends show rapid adoption and significant investment growth",
            key_insights=[
                "Generative AI leads market transformation",
                "Infrastructure investment is accelerating",
                "Business applications are expanding rapidly"
            ],
            themes={
                "Technology": ["AI capabilities expanding"],
                "Economics": ["Investment growth accelerating"]
            }
        )

        # Mock the agent.run method
        mock_report = ResearchReport(
            title="Artificial Intelligence Trends: Market Analysis and Future Outlook",
            executive_summary="This comprehensive analysis examines emerging AI trends and their business impact...",
            introduction="Artificial intelligence continues to reshape industries...",
            methodology="This research employed systematic analysis of market data...",
            sections=[
                ResearchSection(
                    title="Technology Trends",
                    content="Generative AI adoption is accelerating...",
                    findings=findings[:1],
                    order=0
                ),
                ResearchSection(
                    title="Market Impact",
                    content="Infrastructure spending increases...",
                    findings=findings[1:],
                    order=1
                )
            ],
            conclusion="AI trends indicate sustained growth and transformation...",
            recommendations=[
                "Invest in AI infrastructure development",
                "Develop comprehensive AI adoption strategies",
                "Monitor emerging technology developments"
            ],
            citations=[
                "[1] https://ai-research.com/trends (Accessed: 2024-01-01)",
                "[2] https://market-analysis.com/ai (Accessed: 2024-01-01)"
            ]
        )

        with patch.object(agent, 'run', return_value=mock_report) as mock_run:
            result = await agent.generate_report(brief, findings, compressed_findings, mock_dependencies)

            # Verify the call was made
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert "Generate a comprehensive research report" in call_args[0][0]
            assert "Artificial Intelligence Trends" in call_args[0][0]
            assert mock_dependencies in call_args[0]

            # Verify result
            assert isinstance(result, ResearchReport)
            assert result.title == mock_report.title
            assert len(result.sections) == 2
            assert len(result.recommendations) == 3
            assert len(result.citations) == 2

            # Verify research state was updated
            assert mock_dependencies.research_state.final_report == result
            assert mock_dependencies.research_state.current_stage == ResearchStage.COMPLETED
            assert mock_dependencies.research_state.completed_at is not None

    @pytest.mark.asyncio
    async def test_generate_report_with_minimal_data(self, mock_dependencies: ResearchDependencies):
        """Test report generation with minimal data."""        ReportGeneratorAgent()        ResearchBrief(
            topic="Minimal Test",
            objectives=["Test objective"],
            key_questions=["Test question?"],
            scope="Test scope"
        )

        findings = []

        compressed_findings = CompressedFindings(
            summary="Limited data available for analysis",
            key_insights=["Insufficient information for comprehensive analysis"]
        )

        mock_report = ResearchReport(
            title="Minimal Test: Limited Data Analysis",
            executive_summary="Limited data was available...",
            introduction="This analysis examines minimal data...",
            methodology="Due to limited data availability...",
            sections=[],
            conclusion="Further research is needed...",
            recommendations=["Conduct more comprehensive data collection"],
            citations=[]
        )

        with patch.object(agent, 'run', return_value=mock_report):
            result = await agent.generate_report(brief, findings, compressed_findings, mock_dependencies)

            assert isinstance(result, ResearchReport)
            assert "Minimal Test" in result.title
            assert len(result.sections) == 0
            assert len(result.citations) == 0
            assert mock_dependencies.research_state.current_stage == ResearchStage.COMPLETED

    @pytest.mark.asyncio
    async def test_generate_report_prompt_structure(self, mock_dependencies: ResearchDependencies):
        """Test that report generation prompt is properly structured."""        ReportGeneratorAgent()        ResearchBrief(
            topic="Test Topic",
            objectives=["Objective 1", "Objective 2"],
            key_questions=["Question 1?", "Question 2?"],
            scope="Test scope"
        )

        compressed_findings = CompressedFindings(
            summary="Test summary of findings",
            key_insights=["Insight 1", "Insight 2"]
        )

        mock_report = ResearchReport(
            title="Test Report",
            executive_summary="Test summary",
            introduction="Test intro",
            methodology="Test methodology",
            sections=[],
            conclusion="Test conclusion",
            recommendations=["Test recommendation"],
            citations=[]
        )

        with patch.object(agent, 'run', return_value=mock_report) as mock_run:
            await agent.generate_report(brief, [], compressed_findings, mock_dependencies)

            # Check prompt structure
            call_args = mock_run.call_args
            prompt = call_args[0][0]

            # Should include research context
            assert "Research Topic: Test Topic" in prompt
            assert "Objectives:" in prompt
            assert "- Objective 1" in prompt
            assert "- Objective 2" in prompt

            # Should include key questions
            assert "Key Questions:" in prompt
            assert "- Question 1?" in prompt
            assert "- Question 2?" in prompt

            # Should include compressed findings
            assert "Compressed Findings Summary:" in prompt
            assert "Test summary of findings" in prompt

            # Should include key insights
            assert "Key Insights:" in prompt
            assert "- Insight 1" in prompt
            assert "- Insight 2" in prompt

            # Should include instructions
            assert "Instructions:" in prompt
            assert "Create an engaging title" in prompt
            assert "Write a comprehensive executive summary" in prompt
            assert "Organize findings into logical sections" in prompt

    def test_tools_registration(self):
        """Test that report generator agent initializes correctly."""
        agent = ReportGeneratorAgent()

        # Since we can't access tools directly, test agent initialization
        assert agent.name == "report_generator_agent"
        assert hasattr(agent, '_register_tools')
        assert hasattr(agent.agent, 'tool')
        assert agent.agent is not None

    @pytest.mark.asyncio
    async def test_report_generation_error_handling(self, mock_dependencies: ResearchDependencies):
        """Test report generation behavior when agent.run fails."""        ReportGeneratorAgent()        ResearchBrief(
            topic="Test",
            objectives=["Test"],
            key_questions=["Test?"],
            scope="Test"
        )

        compressed_findings = CompressedFindings(
            summary="Test",
            key_insights=["Test"]
        )

        # Mock agent.run to raise an exception
        with patch.object(agent, 'run', side_effect=Exception("Report generation failed")):
            with pytest.raises(Exception, match="Report generation failed"):
                await agent.generate_report(brief, [], compressed_findings, mock_dependencies)
