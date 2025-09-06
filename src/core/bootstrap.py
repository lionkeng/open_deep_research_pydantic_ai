"""
CLI bootstrap module for early initialization of logging and event system.

This module ensures that logfire and event handlers are initialized before
any research operations begin, preventing the timing issue where the first
StageStartedEvent might not be logged properly.
"""

import sys
import threading

import logfire

# Global state for bootstrap lifecycle
_initialized = False
_init_lock = threading.Lock()


class BootstrapError(Exception):
    """Raised when CLI bootstrap fails."""


class CLIBootstrap:
    """
    Manages CLI initialization lifecycle.

    Ensures logfire and event handlers are set up before any research
    operations, preventing race conditions in event logging.
    """

    @classmethod
    async def initialize(cls, verbose: bool = False) -> None:
        """
        Initialize the CLI environment.

        This should be called once at CLI startup, before any commands execute.

        Args:
            verbose: Whether to enable verbose logging

        Raises:
            BootstrapError: If initialization fails
        """
        global _initialized

        # Thread-safe initialization check
        with _init_lock:
            if _initialized:
                return

            try:
                # 1. Initialize logging first - this must happen before any events
                from .logging import configure_logging

                configure_logging()
                logfire.info("CLI bootstrap: Logging initialized")

                # 2. Mark as initialized (event logging is handled by EventBus itself)
                _initialized = True
                logfire.info("CLI bootstrap: Initialization complete")

            except Exception as e:
                error_msg = f"Failed to initialize CLI environment: {e}"
                # Can't use logfire here since it might not be initialized
                print(f"ERROR: {error_msg}", file=sys.stderr)
                raise BootstrapError(error_msg) from e

    @classmethod
    def ensure_initialized(cls) -> None:
        """
        Ensure CLI is initialized (synchronous check).

        Raises:
            BootstrapError: If CLI is not initialized
        """
        if not _initialized:
            raise BootstrapError("CLI not initialized. Call CLIBootstrap.initialize() first.")

    @classmethod
    async def shutdown(cls) -> None:
        """Clean up CLI resources."""
        global _initialized

        if not _initialized:
            return

        logfire.info("CLI bootstrap: Shutting down")

        # Reset state
        with _init_lock:
            _initialized = False

        logfire.info("CLI bootstrap: Shutdown complete")
