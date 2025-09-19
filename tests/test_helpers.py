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


def create_dynamic_query_response(
    query: str,
    clarification_state: Optional[Any] = None,
    conversation_history: Optional[List[Any]] = None
) -> Any:
    """Create a dynamic TransformedQuery response based on input.

    This function simulates realistic LLM behavior by generating
    responses that vary based on the input parameters.

    Args:
        query: The original query string
        clarification_state: Optional clarification state
        conversation_history: Optional conversation history

    Returns:
        A TransformedQuery with contextually appropriate content
    """
    from models.research_plan_models import (
        TransformedQuery,
        ResearchPlan,
        ResearchObjective,
        ResearchMethodology,
    )
    from models.search_query_models import (
        SearchQuery,
        SearchQueryBatch,
        SearchQueryType,
    )
    import uuid

    # Start with the base query
    research_query = query
    objectives = []
    queries = []
    key_concepts = []
    assumptions = []

    # If we have clarification, use the final query
    if clarification_state:
        # Check if it's a dict (from test) or an object
        if isinstance(clarification_state, dict):
            if clarification_state.get('is_clarified') and clarification_state.get('final_query'):
                research_query = clarification_state['final_query']
                assumptions.append("Query clarified by user")
            # Extract keywords from user responses
            if clarification_state.get('user_responses'):
                for response in clarification_state['user_responses']:
                    words = response.lower().split()
                    key_concepts.extend([w for w in words if len(w) > 3][:2])
        elif hasattr(clarification_state, 'is_clarified'):
            if clarification_state.is_clarified and clarification_state.final_query:
                research_query = clarification_state.final_query
                assumptions.append("Query clarified by user")
            # Extract keywords from user responses
            if hasattr(clarification_state, 'user_responses'):
                for response in clarification_state.user_responses:
                    words = response.lower().split()
                    key_concepts.extend([w for w in words if len(w) > 3][:2])

    # Add conversation context if available
    if conversation_history:
        context_snippets = []
        for turn in conversation_history[-3:]:  # Last 3 turns
            if hasattr(turn, 'role') and turn.role == "user":
                context_snippets.append(turn.content[:50])
        if context_snippets:
            assumptions.append(f"Building on previous discussion about: {', '.join(context_snippets[:2])}")

    # Generate objectives based on query type
    if "compare" in query.lower() or "vs" in query.lower():
        # Comparison query
        obj_id = str(uuid.uuid4())
        objectives.append(ResearchObjective(
            id=obj_id,
            objective="Compare and contrast the specified items",
            priority="PRIMARY",
            success_criteria="Clear comparison with pros and cons"
        ))
        # Generate comparison-specific queries
        queries.extend([
            SearchQuery(
                id=str(uuid.uuid4()),
                query=f"comparison {query[:50]}",
                query_type=SearchQueryType.COMPARATIVE,
                priority=5,
                max_results=10,
                rationale="Direct comparison",
                objective_id=obj_id
            ),
            SearchQuery(
                id=str(uuid.uuid4()),
                query=f"differences between {query[:40]}",
                query_type=SearchQueryType.ANALYTICAL,
                priority=4,
                max_results=10,
                rationale="Analyze differences",
                objective_id=obj_id
            )
        ])

    elif "how" in query.lower() or "tutorial" in query.lower():
        # Tutorial/How-to query
        obj_id = str(uuid.uuid4())
        objectives.append(ResearchObjective(
            id=obj_id,
            objective="Provide step-by-step guidance",
            priority="PRIMARY",
            success_criteria="Clear instructions with examples"
        ))
        queries.extend([
            SearchQuery(
                id=str(uuid.uuid4()),
                query=f"tutorial {query[:50]}",
                query_type=SearchQueryType.EXPLORATORY,
                priority=5,
                max_results=10,
                rationale="Find tutorials",
                objective_id=obj_id
            ),
            SearchQuery(
                id=str(uuid.uuid4()),
                query=f"guide how to {query[:40]}",
                query_type=SearchQueryType.FACTUAL,
                priority=4,
                max_results=10,
                rationale="Find guides",
                objective_id=obj_id
            )
        ])

    else:
        # General query
        obj_id = str(uuid.uuid4())
        objectives.append(ResearchObjective(
            id=obj_id,
            objective=f"Research and explain: {query[:100]}",
            priority="PRIMARY",
            success_criteria="Comprehensive understanding"
        ))
        queries.extend([
            SearchQuery(
                id=str(uuid.uuid4()),
                query=f"information about {query[:50]}",
                query_type=SearchQueryType.FACTUAL,
                priority=5,
                max_results=10,
                rationale="General information",
                objective_id=obj_id
            ),
            SearchQuery(
                id=str(uuid.uuid4()),
                query=f"explain {query[:50]}",
                query_type=SearchQueryType.ANALYTICAL,
                priority=4,
                max_results=10,
                rationale="Detailed explanation",
                objective_id=obj_id
            )
        ])

    # Extract key concepts if not already done
    if not key_concepts and query:
        words = query.lower().split()
        key_concepts = [w for w in words if len(w) > 3 and w not in
                       {"what", "when", "where", "which", "that", "this", "with", "from"}][:5]

    # Create the research plan
    plan = ResearchPlan(
        objectives=objectives,
        methodology=ResearchMethodology(
            approach="Systematic research and analysis",
            data_sources=["Academic sources", "Technical documentation"],
            analysis_methods=["Content analysis", "Synthesis"],
            quality_criteria=["Accuracy", "Relevance"]
        ),
        expected_deliverables=["Comprehensive research report"]
    )

    # Create SearchQueryBatch
    batch = SearchQueryBatch(queries=queries)

    return TransformedQuery(
        original_query=query,
        search_queries=batch,
        research_plan=plan,
        clarification_context={
            "clarified": clarification_state is not None,
            "has_conversation": conversation_history is not None
        },
        transformation_rationale=f"Transformed query for {'comparison' if 'compare' in query.lower() else 'tutorial' if 'how' in query.lower() else 'general'} research",
        confidence_score=0.9 if clarification_state else 0.7,
        ambiguities_resolved=["Query clarified by user"] if clarification_state else [],
        assumptions_made=assumptions if assumptions else ["No specific context provided"],
        potential_gaps=["Specific requirements may need clarification"]
    )


def assert_valid_clarification_output(output):
    """Assert that output is valid ClarifyWithUser.

    Args:
        output: The output to validate

    Raises:
        AssertionError: If validation fails
    """
    from agents.clarification import ClarifyWithUser

    assert isinstance(output, ClarifyWithUser)
    assert isinstance(output.needs_clarification, bool)
    assert isinstance(output.reasoning, str)
    assert len(output.reasoning) > 0
    assert isinstance(output.missing_dimensions, list)
    assert isinstance(output.assessment_reasoning, str)

    if output.needs_clarification:
        assert output.request is not None
        from models.clarification import ClarificationRequest
        assert isinstance(output.request, ClarificationRequest)
        assert len(output.request.questions) > 0
    else:
        assert output.request is None
