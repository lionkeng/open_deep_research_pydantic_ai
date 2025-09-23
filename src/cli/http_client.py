"""HTTP client for the research API with SSE support."""

from __future__ import annotations

from types import TracebackType
from typing import Any

import httpx
import logfire
from httpx_sse import aconnect_sse

from core.events import ResearchCompletedEvent, StageCompletedEvent, StreamingUpdateEvent
from core.sse_models import (
    CompletedMessage,
    ConnectionMessage,
    ErrorMessage,
    HeartbeatMessage,
    PingMessage,
    SSEEventType,
    StageCompletedMessage,
    UpdateMessage,
    parse_sse_message,
)
from models.core import ResearchStage
from models.report_generator import ResearchReport

from .stream import CLIStreamHandler


class HTTPResearchClient:
    """HTTP client wrapper for API operations + SSE streaming."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        self.base_url = base_url
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    async def __aenter__(self):
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:  # noqa: D401
        await self.close()

    async def start_research(self, query: str) -> str:
        resp = await self.client.post(
            f"{self.base_url}/research", json={"query": query, "stream": True}
        )
        resp.raise_for_status()
        return resp.json()["request_id"]

    async def stream_events(self, request_id: str, handler: CLIStreamHandler) -> None:
        async with aconnect_sse(
            self.client, "GET", f"{self.base_url}/research/{request_id}/stream"
        ) as event_source:
            async for sse in event_source.aiter_sse():
                await self._process_sse_event(sse, handler)

    async def _process_sse_event(self, sse: Any, handler: CLIStreamHandler) -> None:
        try:
            msg = parse_sse_message(sse.data)
            if sse.event == SSEEventType.UPDATE and isinstance(msg, UpdateMessage):
                event = StreamingUpdateEvent(
                    _request_id=msg.request_id, content=msg.content, stage=ResearchStage[msg.stage]
                )
                await handler.handle_streaming_update(event)
            elif sse.event == SSEEventType.STAGE_COMPLETED and isinstance(
                msg, StageCompletedMessage
            ):
                event = StageCompletedEvent(
                    _request_id=msg.request_id,
                    stage=ResearchStage[msg.stage],
                    success=msg.success,
                    result=msg.result,
                )
                await handler.handle_stage_completed(event)
            elif sse.event == SSEEventType.ERROR and isinstance(msg, ErrorMessage):
                # Convert to ErrorEvent for handler
                from core.events import ErrorEvent  # local import to avoid circulars

                await handler.handle_error(
                    ErrorEvent(
                        _request_id=msg.request_id,
                        stage=ResearchStage[msg.stage],
                        error_type=msg.error_type or "error",
                        error_message=msg.message,
                        recoverable=bool(msg.recoverable),
                    )
                )
            elif sse.event == SSEEventType.COMPLETE and isinstance(msg, CompletedMessage):
                report = None
                if msg.report:
                    try:
                        report = ResearchReport(**msg.report)
                    except Exception as e:  # pragma: no cover - defensive
                        logfire.warning(f"Failed to parse report in CompletedMessage: {e}")
                await handler.handle_research_completed(
                    ResearchCompletedEvent(
                        _request_id=msg.request_id,
                        success=msg.success,
                        duration_seconds=msg.duration or 0.0,
                        error_message=msg.error,
                        report=report,
                    )
                )
            elif sse.event == SSEEventType.CONNECTION and isinstance(msg, ConnectionMessage):
                logfire.info(f"SSE connection established: {msg.message}")
            elif sse.event == SSEEventType.PING and isinstance(msg, HeartbeatMessage | PingMessage):
                pass
        except Exception as e:
            logfire.error(f"Error processing SSE event: {e}")

    async def stream_events_with_retry(
        self, request_id: str, handler: CLIStreamHandler, max_retries: int = 5
    ) -> None:
        retry_count = 0
        while retry_count < max_retries:
            try:
                await self.stream_events(request_id, handler)
                break
            except Exception as e:
                # Connection-level errors: retry with a friendly message
                if isinstance(e, httpx.ConnectError):
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise
                    from rich.console import Console

                    Console().print(
                        "[yellow]Connection failed, retrying "
                        + f"{retry_count}/{max_retries}...[/yellow]"
                    )
                # Read timeout on SSE: the server sent no events for a while.
                # Treat as transient; reconnect and continue streaming.
                elif isinstance(e, httpx.ReadTimeout):
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise
                    from rich.console import Console

                    Console().print(
                        "[yellow]SSE timed out waiting for events; reconnecting "
                        + f"{retry_count}/{max_retries}...[/yellow]"
                    )
                else:
                    raise

    async def get_report(self, request_id: str) -> dict[str, Any]:
        resp = await self.client.get(f"{self.base_url}/research/{request_id}/report")
        resp.raise_for_status()
        return resp.json()

    async def get_clarification(self, request_id: str) -> dict[str, Any]:
        try:
            resp = await self.client.get(f"{self.base_url}/research/{request_id}/clarification")
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return {"awaiting_response": False}
            raise

    async def submit_clarification(self, request_id: str, response: Any) -> dict[str, Any]:
        payload: dict[str, Any]
        if hasattr(response, "model_dump"):
            payload = response.model_dump(mode="json")  # type: ignore[assignment]
        else:
            payload = dict(response)
        resp = await self.client.post(
            f"{self.base_url}/research/{request_id}/clarification", json=payload
        )
        resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        await self.client.aclose()
