"""
Terminal-based progress indicator without Rich Live class.
Uses Rich Console with manual terminal control for perfect singleton behavior.
"""

import sys
import threading
import time
from typing import ClassVar, Optional

from rich.console import Console


class TerminalProgress:
    """
    Singleton progress indicator using Rich Console with manual terminal control.
    Completely avoids Rich's Live class to prevent duplicate displays.
    """

    _instance: ClassVar[Optional["TerminalProgress"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls) -> "TerminalProgress":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        # Only initialize once
        if hasattr(self, "_initialized"):
            return

        # Use stdout for progress to avoid conflicts with stderr logging
        self.console = Console(
            file=sys.stdout,
            force_terminal=sys.stdout.isatty(),  # Only force terminal if actually in TTY
            force_interactive=False,  # Disable interactive features to avoid conflicts
        )
        self._active = False
        self._message = ""
        self._start_time = 0.0
        self._update_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._display_lock = threading.Lock()

        # Spinner frames for manual animation
        self._spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._spinner_index = 0

        self._initialized = True

    def start(self, message: str = "Processing...") -> None:
        """Start the progress indicator with given message."""
        with self._display_lock:
            if self._active:
                self.stop()  # Stop any existing progress

            self._active = True
            self._message = message
            self._start_time = time.time()
            self._stop_event.clear()
            self._spinner_index = 0

            # Hide cursor and start update thread
            self.console.print("\033[?25l", end="")  # Hide cursor
            self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self._update_thread.start()

    def update_message(self, message: str) -> None:
        """Update the progress message."""
        with self._display_lock:
            if self._active:
                self._message = message

    def stop(self) -> None:
        """Stop the progress indicator and clean up display."""
        with self._display_lock:
            if not self._active:
                return

            self._active = False
            self._stop_event.set()

            if self._update_thread and self._update_thread.is_alive():
                self._update_thread.join(timeout=0.5)

            # Clear the progress line and show cursor
            self._clear_current_line()
            self.console.print("\033[?25h", end="")  # Show cursor
            self.console.file.flush()

    def _update_loop(self) -> None:
        """Background thread that updates the progress display."""
        while self._active and not self._stop_event.is_set():
            self._render_progress()
            time.sleep(0.1)  # 10 FPS for smooth spinner

    def _render_progress(self) -> None:
        """Render the current progress state to terminal."""
        if not self._active:
            return

        # Calculate elapsed time
        elapsed = time.time() - self._start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        elapsed_str = f"{minutes}:{seconds:02d}"

        # Get current spinner frame
        spinner_char = self._spinner_frames[self._spinner_index]
        self._spinner_index = (self._spinner_index + 1) % len(self._spinner_frames)

        # Only show progress updates if we're in an interactive terminal
        if self.console.is_terminal:
            # Use Rich markup for clean styling
            progress_line = (
                f"[cyan]🔍 {spinner_char}[/cyan]  "
                f"[bold blue]{self._message}[/bold blue]  •  "
                f"[yellow]{elapsed_str}[/yellow]"
            )

            # Write ANSI control directly to file, then use Rich for styled content
            self.console.file.write("\r\033[K")  # Move to start of line and clear
            self.console.print(progress_line, end="")
            self.console.file.flush()
        # If not in terminal (like during tests), stay silent until completion

    def _clear_current_line(self) -> None:
        """Clear the current terminal line."""
        # Simple single line clear - write ANSI directly
        self.console.file.write("\r\033[K")
        self.console.file.flush()

    def show_completion(self, elapsed_time: float) -> None:
        """Show completion message with elapsed time."""
        self.console.print(
            f"[bold green]✓[/bold green] Analysis completed in [cyan]{elapsed_time:.1f}s[/cyan]"
        )

    def __del__(self) -> None:
        """Ensure cleanup on deletion."""
        try:
            self.stop()
        except Exception:
            pass  # Ignore cleanup errors


# Global singleton instance
terminal_progress = TerminalProgress()
