"""Tests for centralized logging configuration."""

import threading
import time
from unittest.mock import patch

import logfire

from open_deep_research_with_pydantic_ai.core.logging import configure_logging, is_configured


def test_configure_logging_idempotent():
    """Test that configure_logging can be called multiple times safely."""
    # Should work without errors
    configure_logging()
    configure_logging()
    configure_logging()

    # Should be able to use logfire after configuration
    logfire.info("Test message")
    assert is_configured()


def test_configure_logging_enables_logfire():
    """Test that configure_logging enables logfire usage."""
    configure_logging()

    # This should not raise any warnings about unconfigured logfire
    logfire.info("Test logging after configuration")
    logfire.debug("Debug message")
    logfire.warning("Warning message")
    assert is_configured()


def test_configure_logging_thread_safety():
    """Test that concurrent calls to configure_logging are thread-safe."""
    # Reset state for test (accessing private module attributes for testing)
    import open_deep_research_with_pydantic_ai.core.logging as log_module
    original_configured = log_module._configured

    try:
        log_module._configured = False

        calls = []
        exceptions = []

        def worker():
            try:
                configure_logging()
                calls.append(threading.current_thread().name)
                # Verify logfire works after configuration
                logfire.info(f"Thread {threading.current_thread().name} configured logfire")
            except Exception as e:
                exceptions.append(e)

        # Start multiple threads concurrently
        threads = [threading.Thread(target=worker, name=f"worker-{i}") for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should succeed
        assert len(exceptions) == 0, f"Exceptions occurred: {exceptions}"
        assert len(calls) == 10
        assert is_configured()

        # Verify logfire still works after concurrent access
        logfire.info("Thread safety test completed")

    finally:
        # Restore original state
        log_module._configured = original_configured


def test_is_configured_reflects_state():
    """Test that is_configured correctly reflects configuration state."""
    configure_logging()
    assert is_configured()


@patch('open_deep_research_with_pydantic_ai.core.logging.logfire.configure')
def test_configure_logging_error_handling(mock_configure):
    """Test error handling when logfire.configure() fails."""
    # Reset state
    import open_deep_research_with_pydantic_ai.core.logging as log_module
    original_configured = log_module._configured

    try:
        log_module._configured = False
        mock_configure.side_effect = Exception("Configuration failed")

        # Should not raise exception, but print to stderr
        with patch('sys.stderr') as mock_stderr:
            configure_logging()

        # Should have printed error message
        mock_stderr.write.assert_called()

        # Configuration should remain False due to error
        assert not is_configured()

    finally:
        log_module._configured = original_configured
