"""Background task management for FastAPI with proper lifecycle handling."""

import asyncio
from collections.abc import Coroutine
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import logfire
from fastapi import FastAPI


class BackgroundTaskManager:
    """Manages background tasks with proper lifecycle and cleanup."""

    def __init__(self):
        """Initialize the task manager."""
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._task_metadata: dict[str, dict[str, Any]] = {}
        self._completed_tasks: set[str] = set()
        self._lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()

    async def submit_research(
        self,
        request_id: str,
        coro: Coroutine[Any, Any, Any],
        metadata: dict[str, Any] | None = None,
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
                self._execute_with_cleanup(request_id, coro), name=f"research-{request_id}"
            )

            self._tasks[request_id] = task
            self._task_metadata[request_id] = {
                "created_at": datetime.now(),
                "metadata": metadata or {},
                "status": "running",
            }

            logfire.info(f"Started background task for {request_id}")
            return request_id

    async def _execute_with_cleanup(self, request_id: str, coro: Coroutine[Any, Any, Any]) -> Any:
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
            cleanup_delay = 60.0
            if self._shutdown_event.is_set():
                cleanup_delay = 0.0
            if cleanup_delay > 0:
                await asyncio.sleep(cleanup_delay)
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

    async def get_task_status(self, request_id: str) -> dict[str, Any] | None:
        """Get status of a task.

        Args:
            request_id: Task to query

        Returns:
            Task metadata if found, None otherwise
        """
        async with self._lock:
            return self._task_metadata.get(request_id)

    async def list_active_tasks(self) -> dict[str, dict[str, Any]]:
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
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout)
            logfire.info(f"All {len(tasks)} tasks shut down gracefully")
        except TimeoutError:
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
                        "completed_at", metadata.get("failed_at", metadata.get("cancelled_at"))
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
async def lifespan_with_task_manager(app: FastAPI):
    """FastAPI lifespan context with task manager cleanup."""
    # Startup
    logfire.info("Task manager initialized")

    yield

    # Shutdown
    await task_manager.shutdown()
    logfire.info("Task manager shut down")


# Create singleton instance
task_manager = BackgroundTaskManager()
