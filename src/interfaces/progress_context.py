"""
Context manager interface for the terminal progress indicator.
"""

import time
from collections.abc import Generator
from contextlib import contextmanager

from .terminal_progress import terminal_progress


@contextmanager
def progress_context(message: str = "Processing...") -> Generator[None, None, None]:
    """
    Context manager for displaying progress with automatic cleanup.

    Usage:
        with progress_context("Loading data..."):
            # Your code here
            pass
    """
    start_time = time.time()
    terminal_progress.start(message)
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        terminal_progress.stop()
        terminal_progress.show_completion(elapsed)


class ProgressManager:
    """
    High-level interface for progress management.
    Ensures only one progress indicator is ever active.
    """

    def __init__(self) -> None:
        self._start_time: float | None = None

    def start(self, message: str = "Processing...") -> None:
        """Start progress with message."""
        self._start_time = time.time()
        terminal_progress.start(message)

    def update(self, message: str) -> None:
        """Update progress message."""
        terminal_progress.update_message(message)

    def stop(self) -> float:
        """Stop progress display and return elapsed time."""
        elapsed = time.time() - self._start_time if self._start_time else 0.0
        terminal_progress.stop()
        return elapsed

    def stop_and_complete(self) -> None:
        """Stop progress and show completion message."""
        elapsed = self.stop()
        terminal_progress.show_completion(elapsed)

    def __enter__(self) -> "ProgressManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop_and_complete()


# Convenience functions for simple usage
def start_progress(message: str = "Processing...") -> None:
    """Start global progress indicator."""
    terminal_progress.start(message)


def update_progress(message: str) -> None:
    """Update global progress message."""
    terminal_progress.update_message(message)


def stop_progress() -> None:
    """Stop global progress indicator."""
    terminal_progress.stop()
