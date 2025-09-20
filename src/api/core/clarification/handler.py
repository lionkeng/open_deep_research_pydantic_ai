"""Clarification flow handler for HTTP mode."""

from __future__ import annotations

import asyncio

import logfire

from core.exceptions import (
    ClarificationLimitError,
    ClarificationStateError,
    SessionNotFoundError,
    SessionStateError,
)
from models.clarification import ClarificationRequest, ClarificationResponse

from ..session import SessionManager, SessionState
from ..session.models import ClarificationExchange


class ClarificationHandler:
    """Coordinate clarification requests/responses for HTTP sessions."""

    def __init__(self, session_manager: SessionManager) -> None:
        self._session_manager = session_manager
        self._pending: dict[str, asyncio.Future[ClarificationResponse]] = {}

    async def request_clarification(
        self,
        session_id: str,
        request: ClarificationRequest,
    ) -> ClarificationExchange:
        session = await self._session_manager.get_session(session_id, for_update=True)
        if not session:
            raise SessionNotFoundError(session_id)
        if session.state not in {SessionState.RESEARCHING, SessionState.CLARIFICATION_TIMEOUT}:
            raise SessionStateError(
                session_id=session_id,
                current_state=session.state.value,
                allowed_states=[SessionState.RESEARCHING.value],
            )
        if len(session.clarification_exchanges) >= session.config.max_clarifications:
            raise ClarificationLimitError(
                session_id=session_id,
                limit=session.config.max_clarifications,
            )

        exchange = ClarificationExchange(request=request)
        session.clarification_exchanges.append(exchange)
        session.transition_to(SessionState.AWAITING_CLARIFICATION)
        await self._session_manager.update_session(session)

        key = self._make_key(session_id, len(session.clarification_exchanges) - 1)
        future: asyncio.Future[ClarificationResponse] = asyncio.get_running_loop().create_future()
        self._pending[key] = future

        asyncio.create_task(
            self._handle_timeout(
                session_id,
                len(session.clarification_exchanges) - 1,
                session.config.clarification_timeout_seconds,
            )
        )

        logfire.info("Clarification requested", session_id=session_id)
        return exchange

    async def submit_response(self, session_id: str, response: ClarificationResponse) -> bool:
        session = await self._session_manager.get_session(session_id, for_update=True)
        if not session:
            raise SessionNotFoundError(session_id)
        if session.state != SessionState.AWAITING_CLARIFICATION:
            raise SessionStateError(
                session_id=session_id,
                current_state=session.state.value,
                allowed_states=[SessionState.AWAITING_CLARIFICATION.value],
            )

        pending_index = self._find_pending_exchange(session)
        if pending_index is None:
            return False

        exchange = session.clarification_exchanges[pending_index]
        exchange.record_response(response)

        session.transition_to(SessionState.RESEARCHING)
        await self._session_manager.update_session(session)

        key = self._make_key(session_id, pending_index)
        future = self._pending.pop(key, None)
        if future and not future.done():
            future.set_result(response)

        logfire.info("Clarification response received", session_id=session_id)
        return True

    async def wait_for_response(
        self, session_id: str, exchange_index: int
    ) -> ClarificationResponse:
        key = self._make_key(session_id, exchange_index)
        future = self._pending.get(key)
        if not future:
            raise ClarificationStateError(session_id=session_id, reason="No pending clarification")
        return await future

    async def _handle_timeout(self, session_id: str, exchange_index: int, timeout: int) -> None:
        await asyncio.sleep(timeout)
        key = self._make_key(session_id, exchange_index)
        future = self._pending.get(key)
        if not future or future.done():
            return

        session = await self._session_manager.get_session(session_id, for_update=True)
        if not session:
            return
        if exchange_index >= len(session.clarification_exchanges):
            return
        exchange = session.clarification_exchanges[exchange_index]
        if exchange.response is not None:
            return

        exchange.timed_out = True
        session.transition_to(SessionState.CLARIFICATION_TIMEOUT)
        await self._session_manager.update_session(session)

        future.set_exception(TimeoutError("Clarification timed out"))
        self._pending.pop(key, None)
        logfire.warning("Clarification timed out", session_id=session_id)

    @staticmethod
    def _make_key(session_id: str, index: int) -> str:
        return f"{session_id}:{index}"

    @staticmethod
    def _find_pending_exchange(session) -> int | None:
        for idx, exchange in enumerate(session.clarification_exchanges):
            if exchange.response is None and not exchange.timed_out:
                return idx
        return None
