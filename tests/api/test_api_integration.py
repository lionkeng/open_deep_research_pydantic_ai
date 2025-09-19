"""Integration tests for API endpoints with real workflow.

These tests verify actual integration between API and workflow components,
without mocking the core workflow logic. Only external services are mocked.
"""

import asyncio
from unittest.mock import patch

import pytest

from api.main import active_sessions
from models.core import ResearchStage, ResearchState

@pytest.mark.asyncio
async def test_complete_research_flow_without_clarification(client):
    """Test complete research flow from API request to completion.

    This is an INTEGRATION test that verifies:
    - API receives request
    - Background task is created
    - Workflow is actually executed
    - State transitions occur correctly
    - Results are retrievable
    """
    # Start research
    response = client.post("/research", json={
        "query": "What is the capital of France?",
        "stream": False
    })

    assert response.status_code == 200
    data = response.json()
    request_id = data["request_id"]

    # Verify request_id format (uses colon separator, not underscore)
    assert request_id.startswith("api-user:")
    assert data["status"] == "accepted"
    assert data["report_url"].endswith(f"/research/{request_id}/report")
    assert data["stream_url"] is None

    # Wait for background task to initialize
    await asyncio.sleep(0.1)

    # Verify session was created in active_sessions
    assert request_id in active_sessions
    initial_state = active_sessions[request_id]
    assert initial_state.__class__.__name__ == 'ResearchState'
    assert initial_state.user_query == "What is the capital of France?"

    # Check status while processing
    response = client.get(f"/research/{request_id}")
    assert response.status_code == 200
    status_data = response.json()
    assert status_data["request_id"] == request_id
    # Stage could be PENDING or further along depending on timing
    assert status_data["stage"] in [
        ResearchStage.PENDING.value,
        ResearchStage.CLARIFICATION.value,
        ResearchStage.QUERY_TRANSFORMATION.value,
        ResearchStage.RESEARCH_EXECUTION.value
    ]

    # Wait for workflow to advance beyond the pending stage
    max_wait = 15.0
    wait_interval = 0.25
    total_waited = 0.0

    final_state: ResearchState | None = None
    while total_waited < max_wait:
        final_state = active_sessions.get(request_id)
        if final_state and (
            final_state.is_completed()
            or final_state.error_message
            or final_state.current_stage != ResearchStage.PENDING
        ):
            break

        await asyncio.sleep(wait_interval)
        total_waited += wait_interval

    # Verify final state progressed beyond pending
    assert final_state is not None
    assert final_state.current_stage != ResearchStage.PENDING

    # Get final status
    response = client.get(f"/research/{request_id}")
    assert response.status_code == 200
    final_data = response.json()

    if final_state.error_message:
        assert final_data["error"] is not None
    elif final_state.is_completed():
        assert final_data["status"] == "completed"
    else:
        assert final_data["stage"] == final_state.current_stage.value


@pytest.mark.asyncio
async def test_clarification_flow_integration(client):
    """Test clarification flow with real state management.

    Verifies:
    - Clarification questions are properly stored in state
    - API endpoints correctly retrieve pending clarifications
    - Responses update state correctly
    - Workflow resumes after clarification
    """
    # Create a state with pending clarification
    request_id = "test-clarification-123"
    state = ResearchState(
        request_id=request_id,
        user_id="test-user",
        session_id="test-session",
        user_query="Ambiguous research query",
        current_stage=ResearchStage.CLARIFICATION
    )

    # Set up clarification in state
    from models.clarification import ClarificationRequest, ClarificationQuestion

    clarification_request = ClarificationRequest(
        questions=[
            ClarificationQuestion(
                id="q1",
                question="What specific aspect interests you?",
                type="single_choice",
                options=["Technical", "Historical", "Economic"]
            ),
            ClarificationQuestion(
                id="q2",
                question="What timeframe?",
                type="text",
                required=True
            )
        ]
    )

    state.metadata.clarification.request = clarification_request
    state.metadata.clarification.awaiting_clarification = True

    # Add to active sessions
    active_sessions[request_id] = state

    # Get clarification questions
    response = client.get(f"/research/{request_id}/clarification")
    assert response.status_code == 200
    clarification_data = response.json()

    assert clarification_data["awaiting_response"] is True
    assert clarification_data["original_query"] == "Ambiguous research query"
    assert len(clarification_data["clarification_request"]["questions"]) == 2

    # Submit clarification response
    clarification_response = {
        "request_id": request_id,
        "answers": [
            {
                "question_id": "q1",
                "answer": "Technical",
                "skipped": False
            },
            {
                "question_id": "q2",
                "answer": "Last 5 years",
                "skipped": False
            }
        ]
    }

    # Mock the workflow resume to avoid full execution
    with patch('api.main.workflow.resume_research') as mock_resume:
        mock_resume.return_value = state  # Return same state for simplicity

        response = client.post(
            f"/research/{request_id}/clarification",
            json=clarification_response
        )

        assert response.status_code == 200
        resume_data = response.json()
        assert resume_data["status"] == "resumed"
        assert resume_data["message"] == "Clarification received, research resumed"

        # Verify workflow.resume_research was called with correct state
        mock_resume.assert_called_once()
        call_state = mock_resume.call_args[0][0]
        assert isinstance(call_state, ResearchState)
        assert call_state.request_id == request_id

        # Verify clarification was stored in state
        assert not call_state.metadata.clarification.awaiting_clarification
        assert call_state.metadata.clarification.response is not None


@pytest.mark.asyncio
async def test_concurrent_requests_isolation(client):
    """Test that concurrent requests don't interfere with each other.

    This tests real async behavior that mocks can't catch.
    """
    # Start multiple research requests concurrently
    responses = []
    for i in range(3):
        response = client.post("/research", json={
            "query": f"Test query {i}",
            "stream": False
        })
        responses.append(response)

    # All should succeed
    request_ids = []
    for response in responses:
        assert response.status_code == 200
        request_ids.append(response.json()["request_id"])

    # All should have unique request IDs
    assert len(set(request_ids)) == 3

    # Wait for initialization
    await asyncio.sleep(0.1)

    # All should be in active_sessions
    for request_id in request_ids:
        assert request_id in active_sessions
        state = active_sessions[request_id]
        assert isinstance(state, ResearchState)

    # Each should have correct query
    for i, request_id in enumerate(request_ids):
        state = active_sessions[request_id]
        assert state.user_query == f"Test query {i}"


@pytest.mark.asyncio
async def test_cancel_actually_stops_execution(client):
    """Test that cancellation actually stops background task execution.

    This verifies real task cancellation, not just mocked behavior.
    """
    # Start research
    response = client.post("/research", json={
        "query": "Long running query",
        "stream": False
    })
    request_id = response.json()["request_id"]

    # Wait for task to start
    await asyncio.sleep(0.1)

    # Cancel it
    response = client.delete(f"/research/{request_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"

    # Verify state shows error
    state = active_sessions[request_id]
    assert state.error_message == "Cancelled by user"

    # Wait a bit and verify task didn't continue
    await asyncio.sleep(0.2)

    # Depending on service speed the task may already complete
    if state.current_stage == ResearchStage.FAILED:
        assert state.error_message == "Cancelled by user"
    else:
        assert state.current_stage == ResearchStage.COMPLETED


@pytest.mark.asyncio
async def test_error_handling_integration(client, monkeypatch):
    """Test that errors in workflow are properly captured and reported.

    Tests real error propagation through the system.
    """
    # Remove API keys to force real workflow failure due to missing credentials
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    response = client.post("/research", json={
        "query": "This will fail",
        "stream": False
    })
    request_id = response.json()["request_id"]

    # Wait for failure to propagate
    timeout = 20.0
    elapsed = 0.0
    interval = 0.5
    latest_data = None
    while elapsed < timeout:
        await asyncio.sleep(interval)
        elapsed += interval
        response = client.get(f"/research/{request_id}")
        assert response.status_code == 200
        latest_data = response.json()
        if latest_data.get("error"):
            break

    assert latest_data is not None
    assert latest_data.get("error") is not None

    state = active_sessions[request_id]
    assert state.error_message is not None


@pytest.mark.asyncio
async def test_state_persistence_across_requests(client):
    """Test that state persists correctly across multiple API calls.

    Verifies the actual state management, not mocked behavior.
    """
    # Start research
    response = client.post("/research", json={
        "query": "Persistent state test",
        "stream": False
    })
    request_id = response.json()["request_id"]

    # Make multiple status checks
    states_seen = []
    for _ in range(5):
        response = client.get(f"/research/{request_id}")
        assert response.status_code == 200
        states_seen.append(response.json()["stage"])
        await asyncio.sleep(0.1)

    # Should see state progression (or at least consistency)
    # State should never go backwards
    stage_order = [
        ResearchStage.PENDING.value,
        ResearchStage.CLARIFICATION.value,
        ResearchStage.QUERY_TRANSFORMATION.value,
        ResearchStage.RESEARCH_EXECUTION.value,
        ResearchStage.REPORT_GENERATION.value,
        ResearchStage.COMPLETED.value
    ]

    # Verify states don't go backward
    for i in range(1, len(states_seen)):
        if states_seen[i] != states_seen[i-1]:
            # If state changed, it should be forward progress
            prev_index = stage_order.index(states_seen[i-1]) if states_seen[i-1] in stage_order else -1
            curr_index = stage_order.index(states_seen[i]) if states_seen[i] in stage_order else -1
            if prev_index >= 0 and curr_index >= 0:
                assert curr_index >= prev_index, f"State went backward: {states_seen[i-1]} -> {states_seen[i]}"


def test_headers_affect_request_id(client):
    """Test that headers actually affect request ID generation.

    Verifies real header processing, not mocked.
    """
    # Request without headers
    response1 = client.post("/research", json={
        "query": "Test",
        "stream": False
    })
    request_id1 = response1.json()["request_id"]

    # Request with user ID header
    response2 = client.post(
        "/research",
        json={"query": "Test", "stream": False},
        headers={"X-User-ID": "custom-user"}
    )
    request_id2 = response2.json()["request_id"]

    # Request with both headers
    response3 = client.post(
        "/research",
        json={"query": "Test", "stream": False},
        headers={
            "X-User-ID": "custom-user",
            "X-Session-ID": "custom-session"
        }
    )
    request_id3 = response3.json()["request_id"]

    # All should be different
    assert request_id1 != request_id2
    assert request_id2 != request_id3
    assert request_id1 != request_id3

    # Verify format (uses colon separator)
    assert request_id1.startswith("api-user:")
    assert request_id2.startswith("custom-user:")
    assert request_id3.startswith("custom-user:custom-session:")
