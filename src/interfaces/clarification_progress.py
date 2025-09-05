"""Clarification progress indicator for CLI mode."""

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text


@dataclass
class PhaseInfo:
    """Information about a clarification phase."""
    name: str
    description: str
    weight: float  # Relative weight for progress calculation
    started: bool = False
    completed: bool = False
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class ClarificationProgressIndicator:
    """Progress indicator for the clarification agent process."""

    def __init__(self, console: Optional[Console] = None):
        """Initialize the clarification progress indicator.
        
        Args:
            console: Optional Rich console instance to use.
        """
        self.console = console or Console()
        self._live: Optional[Live] = None
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._current_phase_idx = 0
        self._overall_progress = 0.0
        self._start_time = time.time()
        
        # Define phases with relative weights
        self.phases = [
            PhaseInfo("Query Understanding", "Analyzing the user's query structure", 0.15),
            PhaseInfo("Scope Analysis", "Determining research boundaries", 0.20),
            PhaseInfo("Context Assessment", "Evaluating available context", 0.20),
            PhaseInfo("Ambiguity Detection", "Identifying unclear elements", 0.25),
            PhaseInfo("Question Formulation", "Creating clarifying questions", 0.20),
        ]
        
        # Progress bars for visual tracking
        self.phase_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
            refresh_per_second=10,
        )
        
        self.overall_progress = Progress(
            TextColumn("[bold green]Overall Progress"),
            BarColumn(complete_style="green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console,
            refresh_per_second=10,
        )
        
        # Add tasks to progress bars
        self.phase_task_id = self.phase_progress.add_task(
            self.phases[0].description, total=100
        )
        self.overall_task_id = self.overall_progress.add_task(
            "Processing", total=100
        )

    def _create_display(self) -> Layout:
        """Create the display layout for the progress indicator."""
        layout = Layout()
        
        # Create phase status table
        phase_table = Table(title="Clarification Analysis Phases", box=None, expand=True)
        phase_table.add_column("Phase", style="cyan", no_wrap=True)
        phase_table.add_column("Status", style="yellow")
        phase_table.add_column("Duration", justify="right")
        
        for i, phase in enumerate(self.phases):
            status = "✓" if phase.completed else ("⟳" if phase.started else "•")
            style = "green" if phase.completed else ("yellow" if phase.started else "dim")
            
            duration = ""
            if phase.start_time:
                if phase.end_time:
                    duration = f"{phase.end_time - phase.start_time:.1f}s"
                else:
                    duration = f"{time.time() - phase.start_time:.1f}s"
            
            phase_table.add_row(
                phase.name,
                Text(status, style=style),
                duration
            )
        
        # Create main panel
        content = Layout()
        content.split_column(
            Layout(phase_table, size=7),
            Layout(Panel(self.phase_progress, border_style="blue", padding=(1, 2)), size=4),
            Layout(Panel(self.overall_progress, border_style="green", padding=(1, 2)), size=3),
        )
        
        main_panel = Panel(
            content,
            title="[bold cyan]🔍 Clarification Agent Analysis",
            border_style="cyan",
            padding=(1, 2),
        )
        
        layout.update(main_panel)
        return layout

    async def _update_loop(self) -> None:
        """Main update loop for the progress indicator."""
        live = None
        try:
            # Create Live display that doesn't block keyboard interrupts
            live = Live(
                self._create_display(),
                console=self.console,
                refresh_per_second=10,
                transient=True,
            )
            live.start()
            self._live = live
            
            # Simulate phase progression
            while not self._stop_event.is_set():
                # Update current phase
                if self._current_phase_idx < len(self.phases):
                    current_phase = self.phases[self._current_phase_idx]
                    
                    if not current_phase.started:
                        current_phase.started = True
                        current_phase.start_time = time.time()
                    
                    # Simulate phase progress (in real use, this would be updated by the agent)
                    phase_progress = min(100, self.phase_progress.tasks[0].completed + 2)
                    self.phase_progress.update(
                        self.phase_task_id,
                        completed=phase_progress,
                        description=current_phase.description,
                    )
                    
                    # Phase completion
                    if phase_progress >= 100:
                        current_phase.completed = True
                        current_phase.end_time = time.time()
                        self._current_phase_idx += 1
                        
                        if self._current_phase_idx < len(self.phases):
                            # Move to next phase
                            next_phase = self.phases[self._current_phase_idx]
                            self.phase_progress.update(
                                self.phase_task_id,
                                completed=0,
                                description=next_phase.description,
                            )
                
                # Update overall progress
                completed_weight = sum(
                    phase.weight for phase in self.phases if phase.completed
                )
                self._overall_progress = completed_weight * 100
                self.overall_progress.update(
                    self.overall_task_id,
                    completed=self._overall_progress,
                )
                
                # Update display
                if live:
                    live.update(self._create_display())
                
                # Check if all phases complete
                if all(phase.completed for phase in self.phases):
                    break
                
                await asyncio.sleep(0.1)  # 10 FPS update rate
                
        except (asyncio.CancelledError, KeyboardInterrupt):
            # Clean shutdown on cancellation or Ctrl+C
            if live:
                live.stop()
            raise  # Re-raise to propagate the interrupt
        except Exception as e:
            # Handle other exceptions
            if live:
                live.stop()
            raise e
        finally:
            if live:
                live.stop()
            self._live = None

    async def start(self) -> None:
        """Start the progress indicator."""
        if self._task is None or self._task.done():
            self._stop_event.clear()
            self._task = asyncio.create_task(self._update_loop())

    async def stop(self) -> None:
        """Stop the progress indicator."""
        self._stop_event.set()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        # Show completion message
        self.console.print("[bold green]✓[/bold green] Clarification analysis complete!")

    def update_phase(self, phase_name: str, progress: float = 100.0) -> None:
        """Update a specific phase's progress.
        
        Args:
            phase_name: Name of the phase to update.
            progress: Progress percentage (0-100).
        """
        for i, phase in enumerate(self.phases):
            if phase.name == phase_name:
                if not phase.started:
                    phase.started = True
                    phase.start_time = time.time()
                
                if progress >= 100:
                    phase.completed = True
                    phase.end_time = time.time()
                    
                # Update current phase if it matches
                if i == self._current_phase_idx:
                    self.phase_progress.update(
                        self.phase_task_id,
                        completed=progress,
                    )
                break

    @asynccontextmanager
    async def progress_context(self):
        """Context manager for the progress indicator."""
        try:
            await self.start()
            yield self
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            await self.stop()
            self.console.print("\n[yellow]Progress cancelled by user[/yellow]")
            raise  # Re-raise to terminate the workflow
        finally:
            await self.stop()