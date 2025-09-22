"""Tests for the HTTP clarification callback helper."""

import pytest
from collections.abc import Callable

from fastapi import HTTPException

from api.main import clarification_handler, http_clarification_callback
from core.exceptions import (
    ClarificationLimitError,
    ClarificationStateError,
    SessionNotFoundError,
    SessionStateError,
)
from models.clarification import ClarificationQuestion, ClarificationRequest
from models.core import ResearchStage, ResearchState


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("factory", "expected_status", "expected_code"),
    (
        (lambda sid: SessionNotFoundError(sid), 404, "SESSION_NOT_FOUND"),
        (
            lambda sid: SessionStateError(
                session_id=sid,
                current_state="completed",
                allowed_states=["researching"],
            ),
            409,
            "SESSION_INVALID_STATE",
        ),
        (
            lambda sid: ClarificationLimitError(sid, limit=1),
            429,
            "CLARIFICATION_LIMIT_EXCEEDED",
        ),
        (
            lambda sid: ClarificationStateError(sid, reason="No pending clarification"),
            409,
            "CLARIFICATION_STATE_ERROR",
        ),
    ),
)
async def test_http_clarification_callback_raises_http_error(
    monkeypatch: pytest.MonkeyPatch,
    factory: Callable[[str], Exception],
    expected_status: int,
    expected_code: str,
) -> None:
    error = factory("req-123")
    async def fake_request_clarification(session_id: str, request: ClarificationRequest) -> None:
        raise error

    monkeypatch.setattr(
        clarification_handler, "request_clarification", fake_request_clarification
    )

    state = ResearchState(
        request_id="req-123",
        user_id="user-1",
        session_id=None,
        user_query="Initial query",
        current_stage=ResearchStage.CLARIFICATION,
    )
    request = ClarificationRequest(
        questions=[ClarificationQuestion(question="What topic should we focus on?")]
    )

    with pytest.raises(HTTPException) as exc_info:
        await http_clarification_callback(request, state)

    http_error = exc_info.value
    assert http_error.status_code == expected_status
    assert http_error.detail["request_id"] == state.request_id
    assert http_error.detail["error"] == expected_code
