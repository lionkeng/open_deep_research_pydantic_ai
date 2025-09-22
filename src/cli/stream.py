"""Streaming UI handlers for CLI (progress + event adapters)."""

from __future__ import annotations

import logfire
from rich.console import Console
from rich.progress import TaskID

from core.events import (
    ErrorEvent,
    ResearchCompletedEvent,
    StageCompletedEvent,
    StageStartedEvent,
    StreamingUpdateEvent,
)
from models.core import ResearchStage

console = Console(force_terminal=True)


class CLIStreamHandler:
    """Handler for streaming updates to the CLI with progress display."""

    def __init__(self, query: str = ""):
        from interfaces.progress_context import ProgressManager

        self.query = query
        self._research_started = False
        self.progress_manager = ProgressManager()
        self._clarification_active = False
        self._post_clarification_active = False
        self.current_task: TaskID | None = None
        self.stage_tasks: dict[ResearchStage, TaskID] = {}

    def __enter__(self) -> CLIStreamHandler:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        # Ensure any active progress is stopped to restore terminal state
        if self._clarification_active:
            try:
                self.progress_manager.stop()
            except Exception:
                pass
            self._clarification_active = False
        if self._post_clarification_active:
            try:
                self.progress_manager.stop()
            except Exception:
                pass
            self._post_clarification_active = False

    def start_research_tracking(self, query: str) -> None:
        if not self._research_started:
            self.query = query
            self._research_started = True

    async def handle_streaming_update(self, event: StreamingUpdateEvent) -> None:
        if event.stage == ResearchStage.CLARIFICATION:
            if not self._clarification_active:
                self._clarification_active = True
                self.progress_manager.start("Analyzing your query for clarity and scope...")
            elif event.content and (
                "examining" in event.content.lower() or "analyzing" in event.content.lower()
            ):
                self.progress_manager.update(event.content)
            return

        if self._clarification_active:
            return

        if not self._research_started:
            self.start_research_tracking(self.query or "Research Query")

        if not self._post_clarification_active:
            self.progress_manager.start(f"Processing {event.stage.value}...")
            self._post_clarification_active = True
        elif event.content:
            self.progress_manager.update(event.content[:80])

        if event.content:
            self.progress_manager.update(event.content[:80])

    async def handle_stage_started(self, event: StageStartedEvent) -> None:
        console.print(f"[cyan]{event.stage.value} started[/cyan]")
        logfire.info(f"Starting {event.stage.value} stage", request_id=event.request_id)

    async def handle_stage_completed(self, event: StageCompletedEvent) -> None:
        if event.stage == ResearchStage.CLARIFICATION:
            if self._clarification_active:
                self.progress_manager.stop_and_complete()
                self._clarification_active = False
                console.print()
            return

        if event.stage == ResearchStage.QUERY_TRANSFORMATION and self._clarification_active:
            self._clarification_active = False
            return

        if event.stage == ResearchStage.FAILED:
            if self._post_clarification_active:
                self.progress_manager.stop()
                self._post_clarification_active = False
            if not event.success and event.error_message:
                console.print(f"\n[red]✗ Research failed: {event.error_message}[/red]")
            else:
                console.print("\n[red]✗ Research failed[/red]")
            console.print("[dim]Exiting...[/dim]")
            raise SystemExit(1)

        if not self._research_started:
            self.start_research_tracking(self.query or "Research Query")

    async def handle_error(self, event: ErrorEvent) -> None:
        if not self._research_started:
            self.start_research_tracking(self.query or "Research Query")
        if event.error_message:
            self.progress_manager.update(f"Error: {event.error_message}")
            if event.stage == ResearchStage.FAILED:
                if self._post_clarification_active:
                    self.progress_manager.stop()
                    self._post_clarification_active = False
                console.print(f"\n[red]✗ Research failed during {event.stage.value}[/red]")
                console.print(f"[red]  Error: {event.error_message}[/red]")
                console.print("[dim]Exiting...[/dim]")
                raise SystemExit(1)
            elif event.recoverable:
                console.print(
                    "[yellow]⚠ Recoverable error in "
                    f"{event.stage.value}: {event.error_message}[/yellow]"
                )
            else:
                console.print(f"[red]✗ Error in {event.stage.value}: {event.error_message}[/red]")

    async def handle_research_completed(self, event: ResearchCompletedEvent) -> None:
        if not self._research_started:
            self.start_research_tracking(self.query or "Research Query")
        if self._clarification_active:
            self.progress_manager.stop_and_complete()
            self._clarification_active = False
        if self._post_clarification_active:
            self.progress_manager.stop_and_complete()
            self._post_clarification_active = False
