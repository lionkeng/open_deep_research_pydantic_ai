"""FastAPI application for the research API with SSE support."""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Annotated

import logfire
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, SecretStr

from api.core.clarification import ClarificationHandler
from api.core.session import InMemorySessionStore, SessionManager, SessionState
from api.core.session.models import ClarificationExchange, SessionMetadata
from api.error_handlers import install_error_handlers
from api.sse_handler import create_sse_response
from api.task_manager import task_manager
from core.config import config as global_config
from core.context import ResearchContextManager
from core.events import research_event_bus
from core.exceptions import OpenDeepResearchError
from core.logging import configure_logging
from core.workflow import workflow
from models.api_models import APIKeys, ConversationMessage
from models.api_models import ResearchRequest as APIResearchRequest
from models.api_models import ResearchResponse as APIResearchResponse
from models.clarification import ClarificationRequest, ClarificationResponse
from models.core import ResearchStage, ResearchState


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Manage application lifespan - startup and shutdown."""
    # Startup logic
    configure_logging(enable_console=True)  # Enable console logging for FastAPI server
    logfire.info(
        "Deep Research API started",
        embedding_similarity=global_config.enable_embedding_similarity,
        similarity_threshold=global_config.embedding_similarity_threshold,
        llm_clean_merge=global_config.enable_llm_clean_merge,
    )
    await session_manager.start()

    try:
        yield  # Application runs here
    finally:
        # Shutdown logic
        await session_manager.stop()
        await research_event_bus.cleanup()
        await task_manager.shutdown()
        logfire.info("Deep Research API shutdown")


app = FastAPI(
    title="Deep Research API",
    description="AI-powered deep research system using Pydantic-AI",
    version="1.0.0",
    lifespan=lifespan,
)

install_error_handlers(app)

# Add CORS middleware with proper configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],  # Configure for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# In-memory storage for active research sessions (use Redis in production)
active_sessions: dict[str, ResearchState] = {}
_sessions_lock = asyncio.Lock()  # Lock for thread-safe access to active_sessions

session_store = InMemorySessionStore()
session_manager = SessionManager(store=session_store)
clarification_handler = ClarificationHandler(session_manager)


def load_api_keys_from_env() -> APIKeys:
    """Build APIKeys object from environment variables."""

    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    return APIKeys(
        openai=SecretStr(openai_key) if openai_key else None,
        anthropic=SecretStr(anthropic_key) if anthropic_key else None,
        tavily=SecretStr(tavily_key) if tavily_key else None,
    )


class ClarificationStatusResponse(BaseModel):
    """Status payload when clarification is pending."""

    request_id: str
    state: str
    awaiting_response: bool
    clarification_request: ClarificationRequest | None = None
    original_query: str | None = None


class ClarificationResumeResponse(BaseModel):
    """Response payload when clarification has been accepted."""

    request_id: str
    status: str
    message: str
    current_stage: str


async def http_clarification_callback(
    request: ClarificationRequest, state: ResearchState
) -> ClarificationResponse | None:
    """Trigger clarification handling for HTTP-mode sessions."""

    try:
        _ = await clarification_handler.request_clarification(state.request_id, request)
    except OpenDeepResearchError as exc:
        logfire.warning(
            "Clarification request failed",
            request_id=state.request_id,
            status_code=exc.status_code,
            error=exc.message,
        )
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.to_payload() | {"request_id": state.request_id},
        ) from exc
    return None


async def _update_session_after_run(state: ResearchState) -> None:
    if state.metadata.clarification.awaiting_clarification:
        return

    session = await session_manager.get_session(state.request_id, for_update=True)
    if not session:
        return

    session.query = state.user_query
    if state.research_results is not None:
        try:
            session.research_results = state.research_results.model_dump()
        except AttributeError:
            session.research_results = state.research_results  # type: ignore[assignment]
    if state.final_report is not None:
        session.synthesis_result = state.final_report.model_dump()

    if state.error_message:
        session.record_error(state.error_message)
        session.transition_to(SessionState.ERROR)
    elif state.is_completed():
        session.transition_to(SessionState.COMPLETED)
    else:
        session.transition_to(SessionState.SYNTHESIZING)

    await session_manager.update_session(session)


async def _mark_session_failed(session_id: str, error: str) -> None:
    session = await session_manager.get_session(session_id, for_update=True)
    if not session:
        return
    session.record_error(error)
    session.transition_to(SessionState.ERROR)
    await session_manager.update_session(session)


def _find_pending_exchange(session) -> ClarificationExchange | None:  # type: ignore[name-defined]
    for exchange in reversed(session.clarification_exchanges):
        if exchange.response is None and not exchange.timed_out:
            return exchange
    return None


async def _resume_research(state: ResearchState) -> None:
    try:
        async with ResearchContextManager(
            user_id=state.user_id,
            session_id=state.session_id,
            request_id=state.request_id,
        ):
            updated_state = await workflow.resume_research(
                state,
                api_keys=load_api_keys_from_env(),
                stream_callback=True,
                clarification_callback=http_clarification_callback,
            )

        async with _sessions_lock:
            active_sessions[state.request_id] = updated_state

        await _update_session_after_run(updated_state)
    except Exception as exc:  # pragma: no cover - resume failures logged
        logfire.error("Failed to resume research", request_id=state.request_id, error=str(exc))
        await _mark_session_failed(state.request_id, str(exc))


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Deep Research API",
        "version": "1.0.0",
        "status": "running",
    }


@app.post("/research", response_model=APIResearchResponse)
async def start_research(
    request: APIResearchRequest,
    fastapi_request: Request,
    x_user_id: Annotated[str | None, Header(alias="X-User-ID")] = None,
    x_session_id: Annotated[str | None, Header(alias="X-Session-ID")] = None,
) -> APIResearchResponse:
    """Start a new research task.

    Args:
        request: Research request parameters
        x_user_id: User ID from header (optional)
        x_session_id: Session ID from header (optional)

    Returns:
        Research response with request ID
    """
    try:
        # Use headers or defaults
        user_id = x_user_id or "api-user"
        session_id = x_session_id

        # Generate scoped request ID
        request_id = ResearchState.generate_request_id(user_id, session_id)

        # Initialize state immediately to avoid race condition
        initial_state = ResearchState(
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            user_query=request.query,
            current_stage=ResearchStage.PENDING,
        )
        initial_state.start_research()

        client_ip = fastapi_request.client.host if fastapi_request.client else None
        await session_manager.create_session(
            query=request.query,
            session_id=request_id,
            metadata=SessionMetadata(client_ip=client_ip),
        )
        async with _sessions_lock:
            active_sessions[request_id] = initial_state

        # Start research in background with user context
        _ = await task_manager.submit_research(
            request_id,
            execute_research_background(
                request_id,
                request.query,
                request.stream,
                user_id,
                session_id,
            ),
            metadata={
                "user_id": user_id,
                "session_id": session_id,
                "stream": request.stream,
            },
        )

        return APIResearchResponse(
            request_id=request_id,
            status="accepted",
            message=(
                "Research accepted. "
                + (
                    "Streaming updates enabled."
                    if request.stream
                    else "Polling required for progress."
                )
            ),
            stream_url=f"/research/{request_id}/stream" if request.stream else None,
            report_url=f"/research/{request_id}/report",
        )

    except Exception as e:
        logfire.error(f"Failed to start research: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


async def execute_research_background(
    request_id: str,
    query: str,
    stream: bool,
    user_id: str = "api-user",
    session_id: str | None = None,
) -> None:
    """Execute research in the background.

    Args:
        request_id: Research request ID
        query: Research query
        stream: Whether streaming is enabled
        user_id: User identifier
        session_id: Optional session identifier
    """
    try:
        # Set up user context for this research
        async with ResearchContextManager(
            user_id=user_id, session_id=session_id, request_id=request_id
        ):
            api_keys = load_api_keys_from_env()
            # Execute research workflow with the proper request_id
            state = await workflow.run(
                user_query=query,
                api_keys=api_keys,
                stream_callback=True if stream else None,
                request_id=request_id,
                clarification_callback=http_clarification_callback,
            )

            # Update state with user context
            state.user_id = user_id
            state.session_id = session_id

            # Store completed state
            async with _sessions_lock:
                active_sessions[request_id] = state

            await _update_session_after_run(state)

    except Exception as e:
        logfire.error(f"Background research failed: {str(e)}")
        # Store error state
        error_state = ResearchState(
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            user_query=query,
        )
        error_state.set_error(str(e))
        async with _sessions_lock:
            active_sessions[request_id] = error_state
        await _mark_session_failed(request_id, str(e))


@app.get("/research/{request_id}")
async def get_research_status(request_id: str):
    """Get the status of a research request.

    Args:
        request_id: Research request ID

    Returns:
        Current research state
    """
    if request_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Research request not found")

    state = active_sessions[request_id]

    return {
        "request_id": request_id,
        "status": "completed" if state.is_completed() else "processing",
        "stage": state.current_stage.value,
        "error": state.error_message,
        "report": state.final_report.model_dump() if state.final_report else None,
    }


@app.get("/research/{request_id}/stream")
async def stream_research_updates(request_id: str, request: Request):
    """Stream research updates via Server-Sent Events.

    Args:
        request_id: Research request ID
        request: FastAPI request object

    Returns:
        SSE stream response
    """
    # Note: If request_id not in active_sessions, the SSE handler will wait for it

    return create_sse_response(request_id, request, active_sessions)


@app.get("/research/{request_id}/report")
async def get_research_report(request_id: str):
    """Get the final research report.

    Args:
        request_id: Research request ID

    Returns:
        Final research report if available
    """
    async with _sessions_lock:
        if request_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Research request not found")
        state = active_sessions[request_id]

    if not state.final_report:
        raise HTTPException(
            status_code=400,
            detail=f"Report not yet available. Current stage: {state.current_stage.value}",
        )

    return state.final_report.model_dump()


@app.delete("/research/{request_id}")
async def cancel_research(request_id: str):
    """Cancel an ongoing research request.

    Args:
        request_id: Research request ID

    Returns:
        Cancellation confirmation
    """
    async with _sessions_lock:
        if request_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Research request not found")
        # Mark as cancelled
        state = active_sessions[request_id]
        state.set_error("Cancelled by user")

    return {
        "request_id": request_id,
        "status": "cancelled",
        "message": "Research request cancelled",
    }


@app.get("/research/{request_id}/clarification", response_model=ClarificationStatusResponse)
async def get_clarification_question(request_id: str):
    """Get pending clarification questions for a research request.

    Args:
        request_id: Research request ID

    Returns:
        Clarification request with questions if pending, otherwise 404
    """
    session = await session_manager.get_session(request_id)
    if session and session.state == SessionState.AWAITING_CLARIFICATION:
        pending = _find_pending_exchange(session)
        if pending:
            return ClarificationStatusResponse(
                request_id=request_id,
                state=session.state.value,
                awaiting_response=True,
                clarification_request=pending.request,
                original_query=session.query,
            )

    async with _sessions_lock:
        state = active_sessions.get(request_id)
    if (
        state
        and state.metadata.clarification.awaiting_clarification
        and state.metadata.clarification.request
    ):
        return ClarificationStatusResponse(
            request_id=request_id,
            state=state.current_stage.value,
            awaiting_response=True,
            clarification_request=state.metadata.clarification.request,
            original_query=state.user_query,
        )

    raise HTTPException(status_code=404, detail="No pending clarification questions")


@app.post("/research/{request_id}/clarification", response_model=ClarificationResumeResponse)
async def respond_to_clarification(request_id: str, clarification_response: ClarificationResponse):
    """Respond to clarification questions and resume research.

    Args:
        request_id: Research request ID
        clarification_response: Multi-question clarification response

    Returns:
        Updated research state
    """
    async with _sessions_lock:
        if request_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Research request not found")
        state = active_sessions[request_id]

    # Check if there's a pending clarification
    if not (state.metadata and state.metadata.clarification.awaiting_clarification):
        raise HTTPException(status_code=400, detail="No pending clarification for this request")

    # Validate against the original request
    if state.metadata.clarification.request:
        errors = clarification_response.validate_against_request(
            state.metadata.clarification.request
        )
        if errors:
            raise HTTPException(
                status_code=400,
                detail={"message": "Invalid clarification response", "errors": errors},
            )

    # Store the response
    state.metadata.clarification.response = clarification_response

    # Update conversation with Q&A pairs
    conversation = list(state.metadata.conversation_messages)
    if state.metadata.clarification.request:
        for answer in clarification_response.answers:
            if not answer.skipped and answer.answer:
                question = state.metadata.clarification.request.get_question_by_id(
                    answer.question_id
                )
                if question:
                    new_messages: list[ConversationMessage] = [
                        ConversationMessage(role="assistant", content=question.question),
                        ConversationMessage(role="user", content=answer.answer),
                    ]
                    conversation.extend(new_messages)

    state.metadata.conversation_messages = conversation
    state.metadata.clarification.awaiting_clarification = False

    processed = await clarification_handler.submit_response(request_id, clarification_response)
    if not processed:
        raise HTTPException(status_code=409, detail="Clarification already processed")

    async with _sessions_lock:
        active_sessions[request_id] = state

    asyncio.create_task(_resume_research(state))

    return ClarificationResumeResponse(
        request_id=request_id,
        status="resumed",
        message="Clarification received, research resumed",
        current_stage=state.current_stage.value,
    )


def main() -> None:
    """Run the FastAPI server."""
    import sys

    import uvicorn

    try:
        uvicorn.run(
            "api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,  # Disable reload for proper signal handling
            use_colors=True,
            log_level="info",
            access_log=True,
        )
    except KeyboardInterrupt:
        print("\n✓ Server stopped gracefully")
        sys.exit(0)


if __name__ == "__main__":
    main()
