"""Centralized logging configuration for the Deep Research application.

This module provides thread-safe, idempotent logfire configuration to ensure
logfire is configured exactly once per application lifecycle, preventing
multiple configuration warnings.

Usage:
    # At application entry points:
    from core.logging import configure_logging
    configure_logging()

    # Then use logfire normally:
    import logfire
    logfire.info("Application started")
"""

import sys
import threading

import logfire

_configured = False
_config_lock = threading.Lock()


def configure_logging(enable_console: bool = False) -> None:
    """Configure logfire logging if not already configured.

    This function ensures logfire is configured only once during the application's
    lifetime, preventing multiple configuration warnings and ensuring consistent
    logging behavior across CLI and API modes.

    Args:
        enable_console: Whether to enable console logging output. Defaults to False.

    Thread-safe implementation using double-checked locking pattern.
    """
    global _configured

    # Fast path - avoid lock if already configured
    if _configured:
        return

    # Double-checked locking pattern for thread safety
    with _config_lock:
        if not _configured:
            try:
                # Configure with console options if enabled, otherwise disable console
                if enable_console:
                    logfire.configure(console=logfire.ConsoleOptions(), min_level="debug")
                else:
                    logfire.configure(console=False, min_level="debug")
                _configured = True
            except Exception as e:
                # Log to stderr since logfire isn't configured yet
                print(f"Failed to configure logfire: {e}", file=sys.stderr)
                # Continue without raising to allow application to proceed


def is_configured() -> bool:
    """Check if logfire has been configured.

    Returns:
        bool: True if logfire has been configured, False otherwise.
    """
    return _configured
