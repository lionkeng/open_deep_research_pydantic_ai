"""Helper utilities for testing agents with proper mocking."""

from contextlib import contextmanager
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch


class MockLLMAgent:
    """Helper class for mocking PydanticAI Agent LLM calls."""

    def __init__(self, agent):
        """Initialize with the agent to mock.

        Args:
            agent: The PydanticAI Agent instance to mock
        """
        self.agent = agent
        self.call_history: List[Dict[str, Any]] = []

    @contextmanager
    def mock_response(self, response_data: Any, error: Optional[Exception] = None):
        """Context manager to mock agent.run() with specified response.

        Args:
            response_data: Data to return from the mocked LLM call
            error: Optional exception to raise instead of returning data

        Yields:
            The mock object for additional configuration if needed
        """
        mock_result = self._create_mock_result(response_data)

        with patch.object(self.agent, "run", new_callable=AsyncMock) as mock_run:
            if error:
                mock_run.side_effect = error
            else:
                mock_run.return_value = mock_result

            # Track calls for inspection
            original_call = mock_run.__call__

            async def tracked_call(*args, **kwargs):
                self.call_history.append({"args": args, "kwargs": kwargs})
                return await original_call(*args, **kwargs)

            mock_run.__call__ = tracked_call
            yield mock_run

    @contextmanager
    def mock_responses(self, response_list: List[Any]):
        """Context manager to mock multiple sequential responses.

        Args:
            response_list: List of response data for sequential calls

        Yields:
            The mock object for additional configuration
        """
        mock_results = [self._create_mock_result(data) for data in response_list]

        with patch.object(self.agent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = mock_results
            yield mock_run

    def _create_mock_result(self, data: Any):
        """Create a properly structured mock result.

        Args:
            data: The data to include in the result

        Returns:
            Mock result with proper structure
        """
        mock_result = MagicMock()
        mock_result.output = data  # Use output instead of data for consistency
        return mock_result

    def clear_history(self):
        """Clear the call history."""
        self.call_history.clear()

    def get_last_call(self) -> Optional[Dict[str, Any]]:
        """Get the last call made to the mocked agent.

        Returns:
            Dictionary with call args and kwargs, or None if no calls
        """
        return self.call_history[-1] if self.call_history else None


def create_mock_llm_response():
    """Create a factory for mock LLM responses.

    Returns:
        A function that creates mock responses with the given data
    """
    def _create_response(data):
        """Helper to create a properly structured result."""
        mock_result = MagicMock()
        mock_result.output = data
        return mock_result
    return _create_response


def assert_valid_clarification_output(output):
    """Assert that output is valid ClarifyWithUser.

    Args:
        output: The output to validate

    Raises:
        AssertionError: If validation fails
    """
    from src.agents.clarification import ClarifyWithUser

    assert isinstance(output, ClarifyWithUser)
    assert isinstance(output.needs_clarification, bool)
    assert isinstance(output.reasoning, str)
    assert len(output.reasoning) > 0
    assert isinstance(output.missing_dimensions, list)
    assert isinstance(output.assessment_reasoning, str)

    if output.needs_clarification:
        assert output.request is not None
        from src.models.clarification import ClarificationRequest
        assert isinstance(output.request, ClarificationRequest)
        assert len(output.request.questions) > 0
    else:
        assert output.request is None
