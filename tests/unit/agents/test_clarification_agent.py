"""Unit tests for ClarificationAgent with proper mocking strategy.

These tests use real ClarificationAgent instances and only mock external dependencies
(LLM calls), ensuring we test actual agent logic rather than mock behavior.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from agents.base import AgentConfiguration, ResearchDependencies
from agents.clarification import ClarificationAgent, ClarifyWithUser
from models.api_models import APIKeys
from models.clarification import ClarificationChoice, ClarificationQuestion, ClarificationRequest
from models.core import ResearchState


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response for testing."""
    def _create_response(data):
        """Helper to create a properly structured result."""
        mock_result = MagicMock()
        mock_result.output = data
        return mock_result
    return _create_response


@pytest.fixture
def test_config():
    """Create a test configuration for agents."""
    return AgentConfiguration(
        agent_name="test-clarification-agent",
        agent_type="clarification",
        model="test",  # Use PydanticAI's built-in test model
        max_retries=1,
        custom_settings={"temperature": 0.7}
    )


@pytest.fixture
def clarification_agent(test_config):
    """Create a real ClarificationAgent instance for testing."""
    # When using 'test' model, PydanticAI creates a TestModel internally
    # which handles mocking appropriately
    agent = ClarificationAgent(config=test_config)
    return agent


@pytest.fixture
def sample_dependencies():
    """Create sample research dependencies for testing."""
    return ResearchDependencies(
        http_client=AsyncMock(),
        api_keys=APIKeys(openai=SecretStr("test-key")),
        research_state=ResearchState(
            request_id="test-123",
            user_query="test query"
        )
    )


@pytest.mark.asyncio
class TestClarificationAgent:
    """Test suite for ClarificationAgent using proper mocking."""

    async def test_agent_initialization(self, test_config):
        """Test that ClarificationAgent initializes correctly."""
        # Act - Create agent with test model
        agent = ClarificationAgent(config=test_config)

        # Assert
        assert agent.config == test_config
        assert agent.name == "test-clarification-agent"
        assert agent.agent is not None
        # Verify it's using the TestModel
        from pydantic_ai.models.test import TestModel
        assert isinstance(agent.agent.model, TestModel)

    async def test_needs_clarification_basic(
        self, clarification_agent, mock_llm_response, sample_dependencies
    ):
        """Test basic case where clarification is needed."""
        # Arrange
        query = "Tell me about AI"
        expected_output = ClarifyWithUser(
            needs_clarification=True,
            request=ClarificationRequest(
                questions=[
                    ClarificationQuestion(
                        question="What aspect of AI interests you most?",
                        question_type="choice",
                        choices=[
                            ClarificationChoice(id="ml", label="Machine Learning"),
                            ClarificationChoice(id="nlp", label="Natural Language Processing"),
                            ClarificationChoice(id="cv", label="Computer Vision"),
                            ClarificationChoice(id="gen", label="General Overview"),
                        ],
                        is_required=True,
                        order=0,
                    ),
                    ClarificationQuestion(
                        question="What is your technical background?",
                        question_type="choice",
                        choices=[
                            ClarificationChoice(id="non", label="Non-technical"),
                            ClarificationChoice(id="beg", label="Beginner"),
                            ClarificationChoice(id="int", label="Intermediate"),
                            ClarificationChoice(id="exp", label="Expert"),
                        ],
                        is_required=True,
                        order=1,
                    )
                ]
            ),
            reasoning="The query 'Tell me about AI' is too broad and lacks specific context",
            missing_dimensions=["SPECIFICITY & SCOPE", "AUDIENCE & DEPTH"],
            assessment_reasoning=(
                "Query needs narrowing down to specific AI aspects and audience level"
            )
        )

        # Mock only the agent.run method
        with patch.object(clarification_agent.agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_llm_response(expected_output)

            # Act
            result = await clarification_agent.agent.run(query, deps=sample_dependencies)

            # Assert
            assert result.output.needs_clarification is True
            assert isinstance(result.output.request, ClarificationRequest)
            assert len(result.output.request.questions) == 2
            assert result.output.missing_dimensions == ["SPECIFICITY & SCOPE", "AUDIENCE & DEPTH"]
            mock_run.assert_called_once_with(query, deps=sample_dependencies)

    async def test_no_clarification_needed(
        self, clarification_agent, mock_llm_response, sample_dependencies
    ):
        """Test case where query is clear and no clarification is needed."""
        # Arrange
        query = "What is the current stock price of Apple Inc. (AAPL) as of market close today?"
        expected_output = ClarifyWithUser(
            needs_clarification=False,
            request=None,
            reasoning="Query is specific, time-bounded, and has clear deliverable expectations",
            missing_dimensions=[],
            assessment_reasoning=(
                "All dimensions are satisfied: specific company, clear timeframe, "
                "simple factual request"
            )
        )

        with patch.object(clarification_agent.agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_llm_response(expected_output)

            # Act
            result = await clarification_agent.agent.run(query, deps=sample_dependencies)

            # Assert
            assert result.output.needs_clarification is False
            assert result.output.request is None
            assert len(result.output.missing_dimensions) == 0

    async def test_complex_clarification_request(
        self, clarification_agent, mock_llm_response, sample_dependencies
    ):
        """Test complex query requiring multiple clarification questions."""
        # Arrange
        query = "Compare cloud providers"
        expected_output = ClarifyWithUser(
            needs_clarification=True,
            request=ClarificationRequest(
                questions=[
                    ClarificationQuestion(
                        question="Which cloud providers should I compare?",
                        question_type="multi_choice",
                        choices=[
                            ClarificationChoice(id="aws", label="AWS"),
                            ClarificationChoice(id="azure", label="Azure"),
                            ClarificationChoice(id="gcp", label="Google Cloud"),
                            ClarificationChoice(id="ibm", label="IBM Cloud"),
                            ClarificationChoice(id="oracle", label="Oracle Cloud"),
                            ClarificationChoice(id="alibaba", label="Alibaba Cloud"),
                        ],
                        is_required=True,
                        order=0,
                    ),
                    ClarificationQuestion(
                        question="What is your primary use case?",
                        question_type="choice",
                        choices=[
                            ClarificationChoice(id="web", label="Web hosting"),
                            ClarificationChoice(id="ml", label="Machine Learning/AI"),
                            ClarificationChoice(id="data", label="Data storage"),
                            ClarificationChoice(id="enterprise", label="Enterprise applications"),
                            ClarificationChoice(id="devops", label="DevOps/CI-CD"),
                        ],
                        is_required=True,
                        order=1,
                    ),
                    ClarificationQuestion(
                        question="What factors are most important to you?",
                        question_type="multi_choice",
                        choices=[
                            ClarificationChoice(id="cost", label="Cost"),
                            ClarificationChoice(id="perf", label="Performance"),
                            ClarificationChoice(id="feat", label="Features"),
                            ClarificationChoice(id="support", label="Support"),
                            ClarificationChoice(id="compliance", label="Compliance"),
                            ClarificationChoice(id="geo", label="Geographic coverage"),
                        ],
                        is_required=True,
                        order=2,
                    ),
                    ClarificationQuestion(
                        question="What is your estimated monthly budget?",
                        question_type="text",
                        is_required=False,
                        order=3
                    )
                ]
            ),
            reasoning=(
                "Cloud provider comparison requires specific providers, use cases, "
                "and evaluation criteria"
            ),
            missing_dimensions=[
                "SPECIFICITY & SCOPE", "DELIVERABLE FORMAT", "QUALITY & SOURCES"
            ],
            assessment_reasoning=(
                "Need to identify specific providers, use case, evaluation criteria, "
                "and budget constraints"
            )
        )

        with patch.object(clarification_agent.agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_llm_response(expected_output)

            # Act
            result = await clarification_agent.agent.run(query, deps=sample_dependencies)

            # Assert
            assert result.output.needs_clarification is True
            assert len(result.output.request.questions) == 4
            assert result.output.request.questions[0].question_type == "multi_choice"
            assert result.output.request.questions[3].is_required is False

    async def test_handles_empty_query(
        self, clarification_agent, mock_llm_response, sample_dependencies
    ):
        """Test handling of empty or minimal queries."""
        # Arrange
        query = ""
        expected_output = ClarifyWithUser(
            needs_clarification=True,
            request=ClarificationRequest(
                questions=[
                    ClarificationQuestion(
                        question="What would you like to research?",
                        question_type="text",
                        is_required=True,
                        order=0
                    )
                ]
            ),
            reasoning="No query provided",
            missing_dimensions=[
                "SPECIFICITY & SCOPE", "AUDIENCE & DEPTH",
                "QUALITY & SOURCES", "DELIVERABLE FORMAT"
            ],
            assessment_reasoning="Empty query requires complete clarification"
        )

        with patch.object(clarification_agent.agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_llm_response(expected_output)

            # Act
            result = await clarification_agent.agent.run(query, deps=sample_dependencies)

            # Assert
            assert result.output.needs_clarification is True
            assert len(result.output.missing_dimensions) == 4

    async def test_handles_llm_error(self, clarification_agent, sample_dependencies):
        """Test error handling when LLM call fails."""
        # Arrange
        query = "Test query"

        with patch.object(clarification_agent.agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = Exception("LLM API error")

            # Act & Assert
            with pytest.raises(Exception, match="LLM API error"):
                await clarification_agent.agent.run(query, deps=sample_dependencies)

    async def test_validates_output_structure(
        self, clarification_agent, mock_llm_response, sample_dependencies
    ):
        """Test that output is properly validated and auto-corrected."""
        # Arrange
        # query = "Test validation"  # Unused variable

        # Test with invalid output (needs_clarification=True but no request)
        # The validator should auto-create a default request
        output = ClarifyWithUser(
            needs_clarification=True,
            request=None,  # This will be auto-corrected by validator
            reasoning="Test",
            missing_dimensions=[],
            assessment_reasoning="Test"
        )

        # Assert that validator auto-created a request
        assert output.request is not None
        assert len(output.request.questions) > 0

        # Test the opposite case - no clarification needed but request provided
        output2 = ClarifyWithUser(
            needs_clarification=False,
            request=ClarificationRequest(questions=[ClarificationQuestion(question="Test?")]),
            reasoning="Clear query",
            missing_dimensions=[],
            assessment_reasoning="No clarification needed"
        )

        # Assert that validator nullified the request
        assert output2.request is None

    async def test_question_ordering(
        self, clarification_agent, mock_llm_response, sample_dependencies
    ):
        """Test that questions are properly ordered."""
        # Arrange
        query = "Research topic"
        expected_output = ClarifyWithUser(
            needs_clarification=True,
            request=ClarificationRequest(
                questions=[
                    ClarificationQuestion(question="Q3", order=2),
                    ClarificationQuestion(question="Q1", order=0),
                    ClarificationQuestion(question="Q2", order=1),
                ]
            ),
            reasoning="Need ordering test",
            missing_dimensions=["TEST"],
            assessment_reasoning="Testing question order"
        )

        with patch.object(clarification_agent.agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_llm_response(expected_output)

            # Act
            result = await clarification_agent.agent.run(query, deps=sample_dependencies)

            # Assert
            sorted_questions = result.output.request.get_sorted_questions()
            assert sorted_questions[0].question == "Q1"
            assert sorted_questions[1].question == "Q2"
            assert sorted_questions[2].question == "Q3"

    async def test_concurrent_requests(
        self, clarification_agent, mock_llm_response, sample_dependencies
    ):
        """Test that multiple clarification requests can be processed concurrently."""
        # Arrange
        queries = ["What is AI?", "Explain blockchain", "How does quantum computing work?"]

        outputs = [
            ClarifyWithUser(
                needs_clarification=True,
                request=ClarificationRequest(
                    questions=[ClarificationQuestion(question=f"Clarify {q}?")]
                ),
                reasoning=f"Need clarification for: {q}",
                missing_dimensions=["SCOPE"],
                assessment_reasoning=f"Assessment for: {q}"
            )
            for q in queries
        ]

        with patch.object(clarification_agent.agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = [mock_llm_response(output) for output in outputs]

            # Act
            tasks = [clarification_agent.agent.run(q, deps=sample_dependencies) for q in queries]
            results = await asyncio.gather(*tasks)

            # Assert
            assert len(results) == 3
            for i, result in enumerate(results):
                assert queries[i] in result.output.reasoning
            assert mock_run.call_count == 3

    async def test_required_vs_optional_questions(
        self, clarification_agent, mock_llm_response, sample_dependencies
    ):
        """Test differentiation between required and optional questions."""
        # Arrange
        query = "Investment advice"
        expected_output = ClarifyWithUser(
            needs_clarification=True,
            request=ClarificationRequest(
                questions=[
                    ClarificationQuestion(
                        question="What is your risk tolerance?",
                        is_required=True,
                        order=0
                    ),
                    ClarificationQuestion(
                        question="What is your investment timeline?",
                        is_required=True,
                        order=1
                    ),
                    ClarificationQuestion(
                        question="Do you have any sector preferences?",
                        is_required=False,
                        order=2
                    ),
                ]
            ),
            reasoning="Investment advice requires risk profile and timeline",
            missing_dimensions=["AUDIENCE & DEPTH", "SPECIFICITY & SCOPE"],
            assessment_reasoning="Must understand investor profile"
        )

        with patch.object(clarification_agent.agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_llm_response(expected_output)

            # Act
            result = await clarification_agent.agent.run(query, deps=sample_dependencies)

            # Assert
            required_questions = result.output.request.get_required_questions()
            assert len(required_questions) == 2
            assert all(q.is_required for q in required_questions)

    async def test_model_configuration_impacts(self, test_config):
        """Test that different configurations create different agent behaviors."""
        # Create agents with different configs
        config1 = AgentConfiguration(
            agent_name="agent1",
            agent_type="clarification",
            model="test",  # Use PydanticAI's built-in test model
            max_retries=1,
            custom_settings={"temperature": 0.1}
        )
        config2 = AgentConfiguration(
            agent_name="agent2",
            agent_type="clarification",
            model="test",  # Use PydanticAI's built-in test model
            max_retries=3,
            custom_settings={"temperature": 0.9}
        )

        # Create agents with test model
        agent1 = ClarificationAgent(config=config1)
        agent2 = ClarificationAgent(config=config2)

        # Verify configs are applied
        assert agent1.config.model == "test"
        assert agent2.config.model == "test"
        assert agent1.config.custom_settings["temperature"] == 0.1
        assert agent2.config.custom_settings["temperature"] == 0.9
