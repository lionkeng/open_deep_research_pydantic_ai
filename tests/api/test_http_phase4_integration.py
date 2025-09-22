"""Phase 4.2 integration tests covering HTTP research flows."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock

from api.main import active_sessions, clarification_handler
from models.core import ResearchStage
from models.clarification import ClarificationQuestion, ClarificationRequest
from models.core import ResearchState
from models.report_generator import ResearchReport


def _build_completed_state(request_id: str) -> ResearchState:
    state = ResearchState(
        request_id=request_id,
        user_id="api-user",
        session_id=None,
        user_query="How do solar panels work?",
        current_stage=ResearchStage.COMPLETED,
    )
    state.metadata.clarification.awaiting_clarification = False
    state.started_at = datetime.now()
    state.completed_at = state.started_at
    state.final_report = ResearchReport(
        title="Solar Panel Research",
        executive_summary="Summary",
        introduction="Intro",
        sections=[],
        conclusions="Conclusions",
        recommendations=[],
        references=[],
        appendices={},
        quality_score=0.9,
    )
    return state


def _create_clarification_state(request_id: str) -> ResearchState:
    state = ResearchState(
        request_id=request_id,
        user_id="user",
        session_id="session",
        user_query="Ambiguous topic",
        current_stage=ResearchStage.CLARIFICATION,
    )
    state.metadata.clarification.request = ClarificationRequest(
        questions=[
            ClarificationQuestion(
                id="q1",
                question="Which domain should we focus on?",
                type="text",
                required=True,
            )
        ]
    )
    state.metadata.clarification.awaiting_clarification = True
    return state


def test_research_flow_end_to_end_success(client, monkeypatch):
    """Research request should store completion data when workflow succeeds."""

    request_payload = {
        "query": "Explain the latest advances in solar energy storage",
        "stream": False,
    }

    async def fake_run(*args, **kwargs):
        return _build_completed_state(kwargs.get("request_id", "req-end-to-end"))

    async def immediate_submit(request_id, coro, metadata=None):
        await coro
        return request_id

    monkeypatch.setattr("api.main.workflow.run", fake_run)
    monkeypatch.setattr("api.main.task_manager.submit_research", immediate_submit)

    response = client.post("/research", json=request_payload)
    assert response.status_code == 200
    body = response.json()
    request_id = body["request_id"]
    assert body["status"] == "accepted"
    assert body["report_url"] == f"/research/{request_id}/report"

    status = client.get(f"/research/{request_id}").json()
    assert status["status"] == "completed"
    assert status["error"] is None
    assert status["report"]["title"] == "Solar Panel Research"


def test_duplicate_clarification_response_returns_conflict(client, monkeypatch):
    """Submitting a clarification twice should return a conflict error."""

    request_id = "clarify-123"
    state = _create_clarification_state(request_id)

    active_sessions[request_id] = state

    monkeypatch.setattr(
        clarification_handler,
        "submit_response",
        AsyncMock(return_value=False),
    )

    payload = {
        "request_id": request_id,
        "answers": [
            {
                "question_id": "q1",
                "answer": "Healthcare",
                "skipped": False,
            }
        ],
    }

    response = client.post(f"/research/{request_id}/clarification", json=payload)
    assert response.status_code == 409
    assert response.json()["detail"] == "Clarification already processed"
