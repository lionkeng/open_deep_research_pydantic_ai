"""Enhanced progress tracking system with rich visual feedback.

This module provides a comprehensive progress tracking system that displays
meaningful information about AI agent activities, stage progress, and timing.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from ..models.core import ResearchStage
from .stage_descriptions import get_stage_metadata, get_stage_index
from .time_estimator import TimeEstimator


@dataclass
class StageProgress:
    """Progress information for a research stage."""
    
    stage: ResearchStage
    status: str  # "pending", "active", "completed", "error"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    current_activity: str = ""
    sub_progress: float = 0.0  # 0.0 to 1.0 for within-stage progress
    error_message: Optional[str] = None
    
    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time for this stage."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    @property
    def is_active(self) -> bool:
        """Check if this stage is currently active."""
        return self.status == "active"
    
    @property
    def is_completed(self) -> bool:
        """Check if this stage is completed."""
        return self.status in ("completed", "error")


@dataclass
class ActivityInfo:
    """Information about current agent activity."""
    
    description: str
    activity_type: str  # "analyzing", "reasoning", "searching", etc.
    progress: Optional[float] = None  # 0.0 to 1.0 if measurable
    details: Dict[str, str] = field(default_factory=dict)
    
    def get_spinner_pattern(self) -> tuple[str, str]:
        """Get spinner pattern and color for this activity type."""
        patterns = {
            "analyzing": ("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏", "cyan"),
            "reasoning": ("⠋⠙⠚⠞⠖⠦⠴⠲⠳⠓", "yellow"),
            "synthesizing": ("⢿⣻⣽⣾⣷⣯⣟⡿", "green"),
            "searching": ("◐◓◑◒", "blue"),
            "validating": ("▁▃▄▅▆▇█▇▆▅▄▃", "magenta"),
            "writing": ("⠁⠂⠄⡀⢀⠠⠐⠈", "white"),
            "thinking": ("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏", "bright_cyan"),
        }
        return patterns.get(self.activity_type, ("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏", "cyan"))


class EnhancedProgressTracker:
    """Enhanced progress tracking with rich visual feedback."""
    
    def __init__(self, console: Optional[Console] = None):
        """Initialize the enhanced progress tracker.
        
        Args:
            console: Rich console instance (creates new if None)
        """
        self.console = console or Console()
        self.time_estimator = TimeEstimator()
        
        # Progress state
        self.stage_progress: Dict[ResearchStage, StageProgress] = {}
        self.current_stage: Optional[ResearchStage] = None
        self.current_activity: Optional[ActivityInfo] = None
        self.query: str = ""
        self.start_time: float = 0.0
        
        # Rich components
        self.layout = Layout()
        self.main_progress: Optional[Progress] = None
        self.stage_progress_bar: Optional[Progress] = None
        self.activity_progress: Optional[Progress] = None
        self.live_display: Optional[Live] = None
        
        # Task tracking
        self.main_task_id: Optional[TaskID] = None
        self.stage_task_id: Optional[TaskID] = None
        self.activity_task_id: Optional[TaskID] = None
        
        # Initialize all stages as pending
        self._initialize_stages()
    
    def _initialize_stages(self):
        """Initialize all research stages as pending."""
        active_stages = [
            ResearchStage.CLARIFICATION,
            ResearchStage.BRIEF_GENERATION,
            ResearchStage.RESEARCH_EXECUTION,
            ResearchStage.COMPRESSION,
            ResearchStage.REPORT_GENERATION,
        ]
        
        for stage in active_stages:
            self.stage_progress[stage] = StageProgress(
                stage=stage,
                status="pending"
            )
    
    def start_research(self, query: str) -> None:
        """Start research tracking for a query.
        
        Args:
            query: The research query being processed
        """
        self.query = query
        self.start_time = time.time()
        
        # Get initial time estimates
        total_seconds, confidence = self.time_estimator.estimate_total_time(query)
        
        # Setup progress bars
        self._setup_progress_bars()
        self._setup_layout()
        
        # Start live display
        self.live_display = Live(
            self.layout,
            console=self.console,
            refresh_per_second=4,
            vertical_overflow="visible"
        )
        self.live_display.start()
    
    def _setup_progress_bars(self):
        """Setup the progress bar components."""
        # Main overall progress (stages)
        self.main_progress = Progress(
            TextColumn("[bold blue]Research Progress"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        )
        
        # Current stage progress
        self.stage_progress_bar = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=30),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console
        )
        
        # Current activity progress
        self.activity_progress = Progress(
            SpinnerColumn(),
            TextColumn("[dim]{task.description}"),
            console=self.console
        )
        
        # Add main task
        active_stages = [
            ResearchStage.CLARIFICATION,
            ResearchStage.BRIEF_GENERATION,
            ResearchStage.RESEARCH_EXECUTION,
            ResearchStage.COMPRESSION,
            ResearchStage.REPORT_GENERATION,
        ]
        
        self.main_task_id = self.main_progress.add_task(
            f"Processing: {self.query[:50]}...",
            total=len(active_stages)
        )
    
    def _setup_layout(self):
        """Setup the Rich layout structure."""
        self.layout.split_column(
            Layout(name="header", size=5),
            Layout(name="main", size=10),
            Layout(name="activity", size=8),
            Layout(name="status", size=3),
        )
    
    def _update_display(self):
        """Update the live display with current information."""
        if not self.live_display:
            return
        
        # Update header
        self._update_header()
        
        # Update main progress
        self._update_main_progress()
        
        # Update activity display
        self._update_activity_display()
        
        # Update status
        self._update_status()
    
    def _update_header(self):
        """Update the header panel."""
        if not self.current_stage:
            title = "Initializing Research..."
            stage_info = ""
        else:
            metadata = get_stage_metadata(self.current_stage)
            stage_index = get_stage_index(self.current_stage) + 1
            total_stages = 5  # Active stages
            
            title = f"Stage {stage_index} of {total_stages}: {metadata.title}"
            
            elapsed = time.time() - self.start_time
            elapsed_mins = int(elapsed // 60)
            elapsed_secs = int(elapsed % 60)
            
            # Get remaining time estimate
            if self.current_stage and elapsed > 5:  # Only show after 5 seconds
                remaining_secs, confidence = self.time_estimator.get_remaining_time_estimate(
                    self.current_stage, elapsed, self.query
                )
                remaining_mins = remaining_secs // 60
                remaining_display = f" • ~{remaining_mins}m remaining"
            else:
                remaining_display = ""
            
            stage_info = f"Query: {self.query[:60]}...\nElapsed: {elapsed_mins}m {elapsed_secs}s{remaining_display}"
        
        header = Panel(
            stage_info,
            title=title,
            border_style="cyan",
            padding=(0, 1)
        )
        
        self.layout["header"].update(header)
    
    def _update_main_progress(self):
        """Update the main progress display."""
        if not self.main_progress or not self.current_stage:
            return
        
        # Update main progress based on completed stages
        completed_stages = sum(
            1 for progress in self.stage_progress.values()
            if progress.is_completed
        )
        
        current_stage_progress = 0.0
        if self.current_stage in self.stage_progress:
            current_stage_progress = self.stage_progress[self.current_stage].sub_progress
        
        total_progress = completed_stages + current_stage_progress
        
        self.main_progress.update(
            self.main_task_id,
            completed=total_progress,
            description=f"Processing: {self.query[:50]}..."
        )
        
        # Create stage overview table
        stage_table = self._create_stage_table()
        
        # Combine progress bar and table
        main_panel = Panel(
            self.main_progress,
            title="Overall Progress",
            border_style="blue"
        )
        
        self.layout["main"].update(main_panel)
    
    def _create_stage_table(self) -> Table:
        """Create a table showing all stage statuses."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(width=3)  # Status icon
        table.add_column(width=25)  # Stage name
        table.add_column(width=15)  # Status/time
        
        active_stages = [
            ResearchStage.CLARIFICATION,
            ResearchStage.BRIEF_GENERATION,
            ResearchStage.RESEARCH_EXECUTION,
            ResearchStage.COMPRESSION,
            ResearchStage.REPORT_GENERATION,
        ]
        
        for stage in active_stages:
            progress = self.stage_progress.get(stage)
            metadata = get_stage_metadata(stage)
            
            if not progress:
                icon = "⏸"
                name_color = "dim"
                status_text = "Waiting"
                status_color = "dim"
            elif progress.status == "completed":
                icon = "✓"
                name_color = "green"
                elapsed = progress.elapsed_seconds
                status_text = f"{elapsed:.0f}s"
                status_color = "green"
            elif progress.status == "active":
                icon = "⚡"
                name_color = "cyan"
                elapsed = progress.elapsed_seconds
                status_text = f"{elapsed:.0f}s"
                status_color = "cyan"
            elif progress.status == "error":
                icon = "✗"
                name_color = "red"
                status_text = "Error"
                status_color = "red"
            else:
                icon = "○"
                name_color = "dim"
                status_text = "Pending"
                status_color = "dim"
            
            table.add_row(
                f"[{name_color}]{icon}[/{name_color}]",
                f"[{name_color}]{metadata.title}[/{name_color}]",
                f"[{status_color}]{status_text}[/{status_color}]"
            )
        
        return table
    
    def _update_activity_display(self):
        """Update the current activity display."""
        if not self.current_stage or not self.current_activity:
            activity_panel = Panel(
                "[dim]Preparing...[/dim]",
                title="Current Activity",
                border_style="dim"
            )
            self.layout["activity"].update(activity_panel)
            return
        
        metadata = get_stage_metadata(self.current_stage)
        agent_info = metadata.agent_info
        
        # Create activity content
        activity_content = []
        
        # Agent information
        activity_content.append(
            f"{agent_info.icon} [bold]{agent_info.name}[/bold] - {agent_info.purpose}"
        )
        activity_content.append("")
        
        # Current activity with spinner
        spinner_pattern, spinner_color = self.current_activity.get_spinner_pattern()
        activity_content.append(
            f"[{spinner_color}]⚡ {self.current_activity.description}[/{spinner_color}]"
        )
        
        # Progress bar if available
        if self.current_activity.progress is not None:
            progress_percent = int(self.current_activity.progress * 100)
            progress_bar = "█" * (progress_percent // 5) + "░" * (20 - progress_percent // 5)
            activity_content.append(f"[cyan]{progress_bar}[/cyan] {progress_percent}%")
        
        # Additional details
        if self.current_activity.details:
            activity_content.append("")
            for key, value in self.current_activity.details.items():
                activity_content.append(f"[dim]{key}: {value}[/dim]")
        
        activity_panel = Panel(
            "\n".join(activity_content),
            title="Current Activity",
            border_style="cyan",
            padding=(1, 2)
        )
        
        self.layout["activity"].update(activity_panel)
    
    def _update_status(self):
        """Update the status bar."""
        if not self.current_stage:
            status_text = "[dim]Starting research...[/dim]"
        else:
            metadata = get_stage_metadata(self.current_stage)
            status_text = f"[dim]{metadata.description}[/dim]"
        
        self.layout["status"].update(Panel(status_text, border_style="dim"))
    
    def start_stage(self, stage: ResearchStage, activity_description: str = "") -> None:
        """Start a new research stage.
        
        Args:
            stage: The stage being started
            activity_description: Initial activity description
        """
        # Complete previous stage if any
        if self.current_stage and self.current_stage in self.stage_progress:
            prev_progress = self.stage_progress[self.current_stage]
            if prev_progress.status == "active":
                prev_progress.status = "completed"
                prev_progress.end_time = time.time()
                # Record timing for learning
                self.time_estimator.record_actual_time(
                    self.current_stage, 
                    prev_progress.elapsed_seconds
                )
        
        # Start new stage
        self.current_stage = stage
        if stage not in self.stage_progress:
            self.stage_progress[stage] = StageProgress(stage=stage, status="pending")
        
        stage_progress = self.stage_progress[stage]
        stage_progress.status = "active"
        stage_progress.start_time = time.time()
        stage_progress.current_activity = activity_description
        
        # Update activity
        if activity_description:
            self.update_activity(activity_description, "analyzing")
        
        self._update_display()
    
    def update_activity(
        self, 
        description: str, 
        activity_type: str = "thinking",
        progress: Optional[float] = None,
        details: Optional[Dict[str, str]] = None
    ) -> None:
        """Update the current activity information.
        
        Args:
            description: Description of current activity
            activity_type: Type of activity (affects spinner pattern)
            progress: Optional progress value (0.0 to 1.0)
            details: Optional additional details
        """
        self.current_activity = ActivityInfo(
            description=description,
            activity_type=activity_type,
            progress=progress,
            details=details or {}
        )
        
        # Update stage progress
        if self.current_stage and self.current_stage in self.stage_progress:
            self.stage_progress[self.current_stage].current_activity = description
            if progress is not None:
                self.stage_progress[self.current_stage].sub_progress = progress
        
        self._update_display()
    
    def complete_stage(self, stage: ResearchStage, success: bool = True, error_message: Optional[str] = None) -> None:
        """Complete a research stage.
        
        Args:
            stage: The stage being completed
            success: Whether the stage completed successfully
            error_message: Error message if stage failed
        """
        if stage in self.stage_progress:
            progress = self.stage_progress[stage]
            progress.status = "completed" if success else "error"
            progress.end_time = time.time()
            progress.sub_progress = 1.0
            progress.error_message = error_message
            
            # Record timing for learning
            self.time_estimator.record_actual_time(stage, progress.elapsed_seconds)
        
        self._update_display()
    
    def complete_research(self, success: bool = True, error_message: Optional[str] = None) -> None:
        """Complete the entire research process.
        
        Args:
            success: Whether research completed successfully
            error_message: Error message if research failed
        """
        # Complete current stage
        if self.current_stage:
            self.complete_stage(self.current_stage, success, error_message)
        
        # Update main progress
        if self.main_progress and self.main_task_id:
            if success:
                self.main_progress.update(self.main_task_id, completed=5)
            else:
                self.main_progress.update(
                    self.main_task_id,
                    description=f"[red]Failed: {error_message or 'Unknown error'}[/red]"
                )
        
        # Show final status
        self.current_activity = ActivityInfo(
            description="Research complete!" if success else f"Research failed: {error_message}",
            activity_type="completed" if success else "error"
        )
        
        self._update_display()
    
    def stop(self) -> None:
        """Stop the progress tracking and clean up."""
        if self.live_display:
            self.live_display.stop()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()