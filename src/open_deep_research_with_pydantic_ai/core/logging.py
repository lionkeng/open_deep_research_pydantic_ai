"""Centralized logging configuration for the Deep Research application.

This module provides thread-safe, idempotent logfire configuration to ensure
logfire is configured exactly once per application lifecycle, preventing
multiple configuration warnings.

Usage:
    # At application entry points:
    from open_deep_research_with_pydantic_ai.core.logging import configure_logging
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


def configure_logging() -> None:
    """Configure logfire logging if not already configured.

    This function ensures logfire is configured only once during the application's
    lifetime, preventing multiple configuration warnings and ensuring consistent
    logging behavior across CLI and API modes.

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
                logfire.configure()
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
