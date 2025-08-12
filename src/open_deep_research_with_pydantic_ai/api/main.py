"""FastAPI application for the research API with SSE support."""

import asyncio
from contextlib import asynccontextmanager
from typing import Annotated

import logfire
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from open_deep_research_with_pydantic_ai.api.sse_handler import create_sse_response
from open_deep_research_with_pydantic_ai.core.context import ResearchContextManager
from open_deep_research_with_pydantic_ai.core.events import research_event_bus
from open_deep_research_with_pydantic_ai.core.workflow import workflow
from open_deep_research_with_pydantic_ai.models.api_models import APIKeys
from open_deep_research_with_pydantic_ai.models.research import ResearchStage, ResearchState


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Manage application lifespan - startup and shutdown."""
    # Startup logic
    logfire.configure()
    logfire.info("Deep Research API started")

    yield  # Application runs here

    # Shutdown logic
    await research_event_bus.cleanup()
    logfire.info("Deep Research API shutdown")


app = FastAPI(
    title="Deep Research API",
    description="AI-powered deep research system using Pydantic-AI",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware with proper configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],  # Configure for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
    expose_headers=["*"],
)


class ResearchRequest(BaseModel):
    """Request model for research endpoint."""

    query: str = Field(description="Research query")
    api_keys: APIKeys | None = Field(
        default=None, description="Optional API keys for search services"
    )
    stream: bool = Field(default=True, description="Whether to stream updates via SSE")


class ResearchResponse(BaseModel):
    """Response model for research endpoint."""

    request_id: str = Field(description="Unique request identifier")
    status: str = Field(description="Request status")
    message: str = Field(description="Status message")
    state: ResearchState | None = Field(default=None, description="Research state if completed")


# In-memory storage for active research sessions (use Redis in production)
active_sessions: dict[str, ResearchState] = {}
_sessions_lock = asyncio.Lock()  # Lock for thread-safe access to active_sessions


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Deep Research API",
        "version": "1.0.0",
        "status": "running",
    }


@app.post("/research", response_model=ResearchResponse)
async def start_research(
    request: ResearchRequest,
    x_user_id: Annotated[str | None, Header(alias="X-User-ID")] = None,
    x_session_id: Annotated[str | None, Header(alias="X-Session-ID")] = None,
):
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
        active_sessions[request_id] = initial_state

        # Start research in background with user context
        asyncio.create_task(
            execute_research_background(
                request_id,
                request.query,
                request.api_keys,
                request.stream,
                user_id,
                session_id,
            )
        )

        return ResearchResponse(
            request_id=request_id,
            status="started",
            message=(
                "Research started. "
                + (
                    f"Stream available at /research/{request_id}/stream"
                    if request.stream
                    else f"Poll /research/{request_id} for status"
                )
            ),
        )

    except Exception as e:
        logfire.error(f"Failed to start research: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


async def execute_research_background(
    request_id: str,
    query: str,
    api_keys: APIKeys | None,
    stream: bool,
    user_id: str = "api-user",
    session_id: str | None = None,
) -> None:
    """Execute research in the background.

    Args:
        request_id: Research request ID
        query: Research query
        api_keys: Optional API keys
        stream: Whether streaming is enabled
        user_id: User identifier
        session_id: Optional session identifier
    """
    try:
        # Set up user context for this research
        async with ResearchContextManager(
            user_id=user_id, session_id=session_id, request_id=request_id
        ):
            # Execute research workflow with the proper request_id and user context
            state = await workflow.execute_research(
                user_query=query,
                api_keys=api_keys,
                stream_callback=True if stream else None,
                request_id=request_id,
                user_id=user_id,
                session_id=session_id,
            )

            # Store completed state
            active_sessions[request_id] = state

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
        active_sessions[request_id] = error_state


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "open_deep_research_with_pydantic_ai.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
