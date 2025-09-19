# Phase 1: Critical Infrastructure Fixes - Detailed Implementation Guide

## Overview

Phase 1 focuses on fixing the critical issues that prevent the HTTP API from functioning correctly. These are breaking issues that must be resolved before any other improvements can be made.

## Current State vs Required State

### Problem 1: Workflow Integration Failure

**Current State** (BROKEN):
```python
# src/api/main.py line 16
from core.workflow import workflow  # workflow is imported but doesn't exist

# src/api/main.py line 176
state = await workflow.execute_research(...)  # Method doesn't exist
```

**Root Cause**:
- `workflow` is expected to be a module-level singleton instance
- `ResearchWorkflow` class exists but no instance is created
- Method name mismatch: `execute_research()` doesn't exist, should be `run()`

**Required State**:
```python
# src/core/workflow.py (at end of file)
workflow = ResearchWorkflow()  # Create singleton instance

# src/api/main.py
state = await workflow.run(...)  # Use correct method name
```

### Problem 2: Duplicate API Models

**Current State** (CAUSES DRIFT):
```python
# src/api/main.py lines 54-71
class ResearchRequest(BaseModel):  # Duplicate definition
    query: str = Field(description="Research query")
    api_keys: APIKeys | None = Field(...)
    stream: bool = Field(...)

class ResearchResponse(BaseModel):  # Duplicate definition
    request_id: str = Field(...)
    status: str = Field(...)
    message: str = Field(...)
    state: ResearchState | None = Field(...)
```

**Root Cause**:
- Models are defined both in `main.py` and `models/api_models.py`
- Missing fields in responses (`stream_url`, `report_url`)
- Validation differences between duplicates

**Required State**:
```python
# src/api/main.py
from models.api_models import ResearchRequest, ResearchResponse  # Use canonical models
# Remove lines 54-71 (duplicate definitions)
```

### Problem 3: Improper Background Task Management

**Current State** (MEMORY LEAKS):
```python
# src/api/main.py line 123
asyncio.create_task(
    execute_research_background(...)
)  # No tracking, no cleanup, no cancellation
```

**Root Cause**:
- Tasks are created but never tracked
- No cleanup on server shutdown
- No cancellation mechanism
- Lost context across async boundaries

**Required State**:
```python
# New file: src/api/task_manager.py
# Implement BackgroundTaskManager with proper lifecycle

# src/api/main.py
task_manager = BackgroundTaskManager()
await task_manager.submit_research(...)
```

## Step-by-Step Implementation Instructions

### Step 1: Fix Workflow Singleton (5 minutes)

**File**: `src/core/workflow.py`

**Action**: Add at the very end of the file (after line 771):

```python
# Create a module-level singleton instance for API usage
workflow = ResearchWorkflow()
```

**Verification**:
```bash
# Check the singleton exists
uv run python -c "from src.core.workflow import workflow; print(type(workflow))"
# Should output: <class 'src.core.workflow.ResearchWorkflow'>
```

### Step 2: Update API Method Calls (15 minutes)

**File**: `src/api/main.py`

**Changes Required**:

1. **Line 176** - Change method name:
```python
# OLD (BROKEN):
state = await workflow.execute_research(
    user_query=query,
    api_keys=api_keys,
    stream_callback=True if stream else None,
    request_id=request_id,
    user_id=user_id,
    session_id=session_id,
)

# NEW (CORRECT):
state = await workflow.run(
    user_query=query,
    api_keys=api_keys,
    conversation_history=None,  # Add this parameter
    request_id=request_id,
    stream_callback=True if stream else None,
)
```

2. **Line 119** - Add proper state initialization:
```python
# After creating initial_state
initial_state = ResearchState(...)
initial_state.start_research()  # ADD THIS LINE
active_sessions[request_id] = initial_state
```

3. **Line 381** - Fix resume_research call:
```python
# OLD:
updated_state = await workflow.resume_research(state)

# NEW:
updated_state = await workflow.resume_research(
    state,
    api_keys=request.api_keys,  # Add api_keys
    stream_callback=True  # Add streaming
)
```

### Step 3: Remove Duplicate Models (10 minutes)

**File**: `src/api/main.py`

**Action**: Delete lines 54-71 (the duplicate class definitions)

**Ensure imports are correct** (should already be at top):
```python
from models.api_models import ResearchRequest, ResearchResponse, ErrorResponse
```

**Update Response Creation** (lines 134-145):
```python
return ResearchResponse(
    request_id=request_id,
    status="accepted",  # Change from "started" to "accepted"
    message="Research started",
    stream_url=f"/research/{request_id}/stream" if request.stream else None,
    report_url=f"/research/{request_id}/report",
)
```

### Step 4: Create Background Task Manager (30 minutes)

**Create New File**: `src/api/task_manager.py`

```python
"""Background task management for FastAPI with proper lifecycle handling."""

import asyncio
import weakref
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Set
from datetime import datetime

import logfire


class BackgroundTaskManager:
    """Manages background tasks with proper lifecycle and cleanup."""

    def __init__(self):
        """Initialize the task manager."""
        self._tasks: Dict[str, asyncio.Task] = {}
        self._task_metadata: Dict[str, Dict[str, Any]] = {}
        self._completed_tasks: Set[str] = set()
        self._lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()

    async def submit_research(
        self,
        request_id: str,
        coro,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Submit a research task for background execution.

        Args:
            request_id: Unique request identifier
            coro: Coroutine to execute
            metadata: Optional metadata about the task

        Returns:
            The request_id for tracking
        """
        async with self._lock:
            # Cancel existing task if present
            if request_id in self._tasks:
                existing_task = self._tasks[request_id]
                if not existing_task.done():
                    existing_task.cancel()
                    logfire.info(f"Cancelled existing task for {request_id}")

            # Create new task
            task = asyncio.create_task(
                self._execute_with_cleanup(request_id, coro),
                name=f"research-{request_id}"
            )

            self._tasks[request_id] = task
            self._task_metadata[request_id] = {
                "created_at": datetime.now(),
                "metadata": metadata or {},
                "status": "running"
            }

            logfire.info(f"Started background task for {request_id}")
            return request_id

    async def _execute_with_cleanup(self, request_id: str, coro):
        """Execute coroutine with automatic cleanup."""
        try:
            result = await coro
            async with self._lock:
                if request_id in self._task_metadata:
                    self._task_metadata[request_id]["status"] = "completed"
                    self._task_metadata[request_id]["completed_at"] = datetime.now()
                self._completed_tasks.add(request_id)
            logfire.info(f"Task {request_id} completed successfully")
            return result

        except asyncio.CancelledError:
            async with self._lock:
                if request_id in self._task_metadata:
                    self._task_metadata[request_id]["status"] = "cancelled"
                    self._task_metadata[request_id]["cancelled_at"] = datetime.now()
            logfire.info(f"Task {request_id} was cancelled")
            raise

        except Exception as e:
            async with self._lock:
                if request_id in self._task_metadata:
                    self._task_metadata[request_id]["status"] = "failed"
                    self._task_metadata[request_id]["error"] = str(e)
                    self._task_metadata[request_id]["failed_at"] = datetime.now()
            logfire.error(f"Task {request_id} failed: {e}")
            raise

        finally:
            # Clean up after delay to allow status queries
            await asyncio.sleep(60)  # Keep metadata for 1 minute
            async with self._lock:
                self._tasks.pop(request_id, None)
                self._task_metadata.pop(request_id, None)
                self._completed_tasks.discard(request_id)

    async def cancel_task(self, request_id: str) -> bool:
        """Cancel a specific task.

        Args:
            request_id: Task to cancel

        Returns:
            True if task was cancelled, False if not found or already done
        """
        async with self._lock:
            task = self._tasks.get(request_id)
            if task and not task.done():
                task.cancel()
                logfire.info(f"Cancelled task {request_id}")
                return True
            return False

    async def get_task_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a task.

        Args:
            request_id: Task to query

        Returns:
            Task metadata if found, None otherwise
        """
        async with self._lock:
            return self._task_metadata.get(request_id)

    async def list_active_tasks(self) -> Dict[str, Dict[str, Any]]:
        """List all active tasks.

        Returns:
            Dictionary of active tasks and their metadata
        """
        async with self._lock:
            active = {}
            for request_id, task in self._tasks.items():
                if not task.done():
                    active[request_id] = self._task_metadata.get(request_id, {})
            return active

    async def shutdown(self, timeout: float = 30.0):
        """Gracefully shutdown all tasks.

        Args:
            timeout: Maximum time to wait for tasks to complete
        """
        logfire.info("Starting task manager shutdown")
        self._shutdown_event.set()

        async with self._lock:
            tasks = list(self._tasks.values())

        if not tasks:
            logfire.info("No tasks to shutdown")
            return

        # Cancel all tasks
        for task in tasks:
            if not task.done():
                task.cancel()

        # Wait for cancellation with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
            logfire.info(f"All {len(tasks)} tasks shut down gracefully")
        except asyncio.TimeoutError:
            # Force kill remaining tasks
            still_running = sum(1 for t in tasks if not t.done())
            logfire.warning(f"Forced shutdown of {still_running} tasks after timeout")
            for task in tasks:
                if not task.done():
                    task.cancel()

    async def cleanup_completed(self, older_than_seconds: int = 300):
        """Clean up metadata for completed tasks older than specified time.

        Args:
            older_than_seconds: Age threshold for cleanup
        """
        now = datetime.now()
        to_remove = []

        async with self._lock:
            for request_id, metadata in self._task_metadata.items():
                if metadata["status"] in ["completed", "failed", "cancelled"]:
                    completed_at = metadata.get(
                        "completed_at",
                        metadata.get("failed_at", metadata.get("cancelled_at"))
                    )
                    if completed_at and (now - completed_at).total_seconds() > older_than_seconds:
                        to_remove.append(request_id)

            for request_id in to_remove:
                self._tasks.pop(request_id, None)
                self._task_metadata.pop(request_id, None)
                self._completed_tasks.discard(request_id)

        if to_remove:
            logfire.info(f"Cleaned up {len(to_remove)} old task records")


# Global instance for the application
task_manager = BackgroundTaskManager()


@asynccontextmanager
async def lifespan_with_task_manager(app):
    """FastAPI lifespan context with task manager cleanup."""
    # Startup
    logfire.info("Task manager initialized")

    yield

    # Shutdown
    await task_manager.shutdown()
    logfire.info("Task manager shut down")
```

### Step 5: Integrate Task Manager into API (20 minutes)

**File**: `src/api/main.py`

**Add import**:
```python
from api.task_manager import task_manager
```

**Update lifespan context** (lines 22-33):
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    # Startup logic
    configure_logging(enable_console=True)
    logfire.info("Deep Research API started")

    yield  # Application runs here

    # Shutdown logic
    await task_manager.shutdown()  # Add this line
    await research_event_bus.cleanup()
    logfire.info("Deep Research API shutdown")
```

**Update research submission** (lines 123-132):
```python
# Replace asyncio.create_task with task manager
await task_manager.submit_research(
    request_id,
    execute_research_background(
        request_id,
        request.query,
        request.api_keys,
        request.stream,
        user_id,
        session_id,
    ),
    metadata={
        "user_id": user_id,
        "session_id": session_id,
        "query": request.query[:100]  # First 100 chars for logging
    }
)
```

**Update cancellation endpoint** (lines 265-286):
```python
@app.delete("/research/{request_id}")
async def cancel_research(request_id: str):
    """Cancel an ongoing research request."""
    # First try to cancel the task
    cancelled = await task_manager.cancel_task(request_id)

    # Then update the session state
    async with _sessions_lock:
        if request_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Research request not found")
        state = active_sessions[request_id]
        state.set_error("Cancelled by user")

    return {
        "request_id": request_id,
        "status": "cancelled",
        "message": "Research request cancelled",
        "task_cancelled": cancelled
    }
```

### Step 6: Add State Initialization (5 minutes)

**File**: `src/api/main.py`

**Update state initialization** (line 119):
```python
# Initialize state immediately to avoid race condition
initial_state = ResearchState(
    request_id=request_id,
    user_id=user_id,
    session_id=session_id,
    user_query=request.query,
    current_stage=ResearchStage.PENDING,
)
initial_state.start_research()  # ADD THIS LINE - properly initialize state
active_sessions[request_id] = initial_state
```

## Testing Phase 1 Changes

### Step 7: Create Tests (30 minutes)

**Create File**: `tests/api/test_workflow_integration.py`

```python
"""Tests for workflow integration fixes."""

import pytest
from src.core.workflow import workflow, ResearchWorkflow


def test_workflow_singleton_exists():
    """Test that workflow singleton is created."""
    assert workflow is not None
    assert isinstance(workflow, ResearchWorkflow)


def test_workflow_singleton_is_singleton():
    """Test that workflow is truly a singleton."""
    from src.core.workflow import workflow as workflow2
    assert workflow is workflow2


def test_workflow_has_correct_methods():
    """Test that workflow has the expected methods."""
    assert hasattr(workflow, 'run')
    assert hasattr(workflow, 'resume_research')
    assert callable(workflow.run)
    assert callable(workflow.resume_research)

    # Should NOT have execute_research
    assert not hasattr(workflow, 'execute_research')
```

**Create File**: `tests/api/test_task_manager.py`

```python
"""Tests for background task manager."""

import asyncio
import pytest
from src.api.task_manager import BackgroundTaskManager


@pytest.mark.asyncio
async def test_task_submission():
    """Test submitting a task."""
    manager = BackgroundTaskManager()

    async def sample_task():
        await asyncio.sleep(0.1)
        return "completed"

    request_id = await manager.submit_research(
        "test-123",
        sample_task(),
        metadata={"test": True}
    )

    assert request_id == "test-123"

    # Check status
    status = await manager.get_task_status("test-123")
    assert status is not None
    assert status["status"] == "running"

    # Wait for completion
    await asyncio.sleep(0.2)

    # Cleanup
    await manager.shutdown(timeout=1.0)


@pytest.mark.asyncio
async def test_task_cancellation():
    """Test cancelling a task."""
    manager = BackgroundTaskManager()

    async def long_task():
        await asyncio.sleep(10)

    request_id = await manager.submit_research(
        "test-456",
        long_task()
    )

    # Cancel the task
    cancelled = await manager.cancel_task("test-456")
    assert cancelled is True

    # Try to cancel again
    cancelled_again = await manager.cancel_task("test-456")
    assert cancelled_again is False  # Already cancelled

    # Cleanup
    await manager.shutdown(timeout=1.0)


@pytest.mark.asyncio
async def test_task_manager_shutdown():
    """Test graceful shutdown."""
    manager = BackgroundTaskManager()

    # Submit multiple tasks
    tasks_ids = []
    for i in range(5):
        async def task(n=i):
            await asyncio.sleep(n * 0.1)

        request_id = f"test-{i}"
        await manager.submit_research(request_id, task())
        tasks_ids.append(request_id)

    # Get active tasks
    active = await manager.list_active_tasks()
    assert len(active) > 0

    # Shutdown
    await manager.shutdown(timeout=2.0)

    # All tasks should be done
    active = await manager.list_active_tasks()
    assert len(active) == 0


@pytest.mark.asyncio
async def test_duplicate_request_id_handling():
    """Test that duplicate request IDs cancel the previous task."""
    manager = BackgroundTaskManager()

    counter = {"value": 0}

    async def incrementing_task():
        counter["value"] += 1
        await asyncio.sleep(1)
        counter["value"] += 1

    # Submit first task
    await manager.submit_research("duplicate", incrementing_task())
    await asyncio.sleep(0.1)  # Let it start

    # Submit second task with same ID
    await manager.submit_research("duplicate", incrementing_task())

    # Wait a bit
    await asyncio.sleep(0.2)

    # Counter should be 2 (first task cancelled, second started)
    assert counter["value"] == 2

    # Cleanup
    await manager.shutdown(timeout=1.0)
```

**Create File**: `tests/api/test_api_models.py`

```python
"""Tests for API model alignment."""

import pytest
from pydantic import ValidationError
from src.models.api_models import ResearchRequest, ResearchResponse


def test_research_request_validation():
    """Test ResearchRequest model validation."""
    # Valid request
    request = ResearchRequest(
        query="What is quantum computing?",
        stream=True,
        max_search_results=10
    )
    assert request.query == "What is quantum computing?"
    assert request.stream is True

    # Invalid request - empty query
    with pytest.raises(ValidationError):
        ResearchRequest(query="", stream=True)

    # Invalid request - query too long
    with pytest.raises(ValidationError):
        ResearchRequest(query="x" * 6000, stream=True)


def test_research_response_fields():
    """Test ResearchResponse has all required fields."""
    response = ResearchResponse(
        request_id="test-123",
        status="accepted",
        message="Research started",
        stream_url="/research/test-123/stream",
        report_url="/research/test-123/report"
    )

    assert response.request_id == "test-123"
    assert response.status == "accepted"
    assert response.stream_url == "/research/test-123/stream"
    assert response.report_url == "/research/test-123/report"

    # Test JSON serialization
    json_data = response.model_dump()
    assert "stream_url" in json_data
    assert "report_url" in json_data
```

### Step 8: Run Tests

```bash
# Run the new tests
uv run pytest tests/api/test_workflow_integration.py -v
uv run pytest tests/api/test_task_manager.py -v
uv run pytest tests/api/test_api_models.py -v

# Run existing API tests to ensure nothing broke
uv run pytest tests/api/ -v

# Check code quality
uv run ruff check src/api/
uv run pyright src/api/
```

## Verification Checklist

### Functionality Verification

- [ ] Workflow singleton exists and is accessible
- [ ] API can call `workflow.run()` without errors
- [ ] API can call `workflow.resume_research()` without errors
- [ ] Background tasks are properly tracked
- [ ] Tasks can be cancelled
- [ ] Server shutdown gracefully stops all tasks

### Code Quality Verification

- [ ] No duplicate model definitions
- [ ] All imports are correct
- [ ] Type hints are consistent
- [ ] No memory leaks in task manager
- [ ] Proper error handling

### Integration Verification

- [ ] Start research via POST /research
- [ ] Get status via GET /research/{id}
- [ ] Stream events via GET /research/{id}/stream
- [ ] Cancel research via DELETE /research/{id}
- [ ] Server shutdown cleanly with Ctrl+C

## Common Issues and Solutions

### Issue 1: ImportError for workflow

**Error**: `ImportError: cannot import name 'workflow' from 'core.workflow'`

**Solution**: Ensure you added `workflow = ResearchWorkflow()` at the end of `src/core/workflow.py`

### Issue 2: AttributeError for execute_research

**Error**: `AttributeError: 'ResearchWorkflow' object has no attribute 'execute_research'`

**Solution**: Change all `execute_research` calls to `run` in `src/api/main.py`

### Issue 3: ValidationError for API models

**Error**: `pydantic.errors.ValidationError` when creating responses

**Solution**: Ensure you're using models from `models.api_models` and including all required fields

### Issue 4: Tasks not cleaning up

**Error**: Memory usage increases over time

**Solution**: Verify `BackgroundTaskManager` is integrated and `shutdown()` is called in lifespan

## Success Criteria for Phase 1

1. **API Starts Successfully**
   ```bash
   uv run uvicorn src.api.main:app --reload
   # Should see: "Application startup complete"
   ```

2. **Research Can Be Initiated**
   ```bash
   curl -X POST http://localhost:8000/research \
     -H "Content-Type: application/json" \
     -d '{"query": "test query"}'
   # Should return: {"request_id": "...", "status": "accepted", ...}
   ```

3. **All Tests Pass**
   ```bash
   uv run pytest tests/api/ -v
   # All tests should pass
   ```

4. **Clean Shutdown**
   ```bash
   # Start server, then Ctrl+C
   # Should see: "Task manager shut down"
   ```

## Next Steps

After completing Phase 1:

1. **Code Review**: Run the code-reviewer agent on all changes
2. **Fix Issues**: Address any issues found by the reviewer
3. **Document Changes**: Update API documentation
4. **Performance Test**: Run basic load test with 10 concurrent requests
5. **Proceed to Phase 2**: Enhanced session management and robustness

## Summary

Phase 1 addresses the critical infrastructure issues that prevent the HTTP API from functioning. The main fixes are:

1. Creating a workflow singleton for the API to use
2. Correcting method names to match the actual workflow interface
3. Removing duplicate model definitions
4. Implementing proper background task management
5. Adding proper state initialization

These changes establish a solid foundation for the subsequent phases which will add robustness, scalability, and production features.
