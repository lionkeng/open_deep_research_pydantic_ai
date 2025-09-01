"""Pytest configuration and fixtures for the three-phase clarification system tests."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any

from agents.base import ResearchDependencies
from models.api_models import APIKeys, ResearchMetadata
from models.core import ResearchState, ResearchStage


@pytest.fixture
def mock_api_keys():
    """Mock API keys for testing."""
    return APIKeys()


@pytest.fixture
def sample_research_state():
    """Sample research state for testing."""
    return ResearchState(
        request_id="test-request-123",
        user_id="test-user",
        session_id="test-session",
        user_query="Sample test query about quantum computing",
        current_stage=ResearchStage.CLARIFICATION,
        metadata={}
    )


@pytest.fixture
def mock_http_client():
    """Mock HTTP client for testing."""
    client = AsyncMock()

    # Mock common HTTP responses
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "ok"}
    mock_response.text = "Mock response content"

    client.get.return_value = mock_response
    client.post.return_value = mock_response

    return client


@pytest.fixture
async def sample_research_dependencies(mock_api_keys, sample_research_state, mock_http_client):
    """Sample research dependencies for testing."""
    return ResearchDependencies(
        http_client=mock_http_client,
        api_keys=mock_api_keys,
        research_state=sample_research_state,
        metadata=ResearchMetadata(),
        usage=None,
    )


@pytest.fixture
def mock_interaction_callback():
    """Mock interaction callback for CLI testing."""
    def callback(question: str) -> str:
        # Return different responses based on question content
        if "specific" in question.lower():
            return "I'm interested in machine learning applications"
        elif "aspect" in question.lower():
            return "Focus on current research and practical applications"
        elif "domain" in question.lower():
            return "Healthcare and medical diagnostics"
        else:
            return "Please provide more technical details"

    return callback


@pytest.fixture
def performance_test_queries():
    """Standard set of queries for performance testing."""
    return [
        # Specific queries (should be fast)
        "What is the current price of Bitcoin?",
        "How many days in February 2024?",
        "What is the capital of Japan?",

        # Moderate queries
        "What are the benefits of solar energy?",
        "How does machine learning work in healthcare?",
        "What are current trends in electric vehicles?",

        # Broad queries (may need clarification)
        "What is artificial intelligence?",
        "Tell me about technology",
        "How does science work?",
    ]


@pytest.fixture
def algorithm_accuracy_dataset():
    """Curated dataset for testing algorithm accuracy."""
    return [
        # Format: (query, expected_needs_clarification, confidence_level, notes)
        ("What is 2+2?", False, "high", "Simple math question"),
        ("Current Apple stock price", False, "high", "Specific data request"),
        ("Compare React vs Vue.js performance", False, "medium", "Technical comparison"),

        ("What is AI?", True, "medium", "Broad technical concept"),
        ("Tell me about space", True, "high", "Very broad topic"),
        ("How does life work?", True, "high", "Extremely broad philosophical"),

        # Edge cases
        ("", False, "low", "Empty query - should not crash"),
        ("?", False, "low", "Single character query"),
        ("AI ML DL NLP", True, "medium", "Acronym-heavy query"),
    ]


@pytest.fixture(scope="session")
def event_loop():
    """Event loop fixture for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Configure test settings
pytest_plugins = ["pytest_asyncio"]


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (may take >30s)")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "accuracy: marks tests as algorithm accuracy tests")


@pytest.fixture(autouse=True)
def configure_test_logging():
    """Configure logging for tests."""
    import logging

    # Reduce log verbosity during tests
    logging.getLogger("logfire").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    yield

    # Cleanup if needed
    pass


class MockWorkflowComponents:
    """Helper class for creating mock workflow components."""

    @staticmethod
    def create_mock_clarification_response(needs_clarification: bool = False, question: str = "") -> Dict[str, Any]:
        """Create mock clarification response."""
        return {
            "need_clarification": needs_clarification,
            "question": question,
            "verification": "Mock verification message",
        }

    @staticmethod
    def create_mock_transformation_data(specificity_score: float = 0.8) -> Dict[str, Any]:
        """Create mock transformation data."""
        return {
            "original_query": "Original test query",
            "transformed_query": "Specific transformed research question",
            "supporting_questions": ["Supporting question 1", "Supporting question 2"],
            "transformation_rationale": "Mock transformation rationale",
            "specificity_score": specificity_score,
            "missing_dimensions": [],
            "clarification_responses": {},
            "transformation_metadata": {"method": "mock"},
        }

    @staticmethod
    def create_mock_brief_result(confidence: float = 0.8) -> Dict[str, Any]:
        """Create mock brief generation result."""
        return {
            "brief": "Mock research brief with comprehensive details about the topic.",
            "confidence_score": confidence,
            "missing_aspects": [] if confidence > 0.7 else ["Additional context needed"],
        }


# Make MockWorkflowComponents available to all tests
@pytest.fixture
def mock_components():
    """Provide mock workflow components."""
    return MockWorkflowComponents
