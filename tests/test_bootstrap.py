"""
Tests for the CLI bootstrap system.

Tests ensure that logging is properly initialized before any research
operations begin. The EventBus itself handles event logging.
"""

import asyncio
from unittest.mock import patch

import pytest

from src.core.bootstrap import CLIBootstrap, BootstrapError


class TestCLIBootstrap:
    """Test the CLI bootstrap system."""

    @pytest.fixture(autouse=True)
    def reset_bootstrap(self) -> None:
        """Reset bootstrap state before each test."""
        # Import the global state variables
        from src.core import bootstrap

        # Reset the global state
        bootstrap._initialized = False
        yield
        # Clean up after test
        bootstrap._initialized = False

    @pytest.mark.asyncio
    async def test_initialize_success(self) -> None:
        """Test successful bootstrap initialization."""
        with patch("src.core.bootstrap.logfire") as mock_logfire:
            with patch("src.core.logging.configure_logging") as mock_configure_logging:
                await CLIBootstrap.initialize(verbose=True)

            from src.core import bootstrap

            assert bootstrap._initialized
            mock_configure_logging.assert_called_once()
            # Should log initialization and completion
            assert mock_logfire.info.call_count >= 2

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self) -> None:
        """Test that initialize can be called multiple times safely."""
        with patch("src.core.bootstrap.logfire"):
            with patch("src.core.logging.configure_logging") as mock_configure_logging:
                await CLIBootstrap.initialize()
                await CLIBootstrap.initialize()  # Should not initialize again

            from src.core import bootstrap

            assert bootstrap._initialized
            mock_configure_logging.assert_called_once()  # Only called once

    @pytest.mark.asyncio
    async def test_initialize_failure(self) -> None:
        """Test bootstrap initialization failure handling."""
        # Ensure we start with clean state
        from src.core import bootstrap

        bootstrap._initialized = False

        with (
            patch("src.core.logging.configure_logging", side_effect=Exception("Setup failed")),
            patch("builtins.print") as mock_print,
        ):
            with pytest.raises(BootstrapError, match="Failed to initialize CLI environment"):
                await CLIBootstrap.initialize()

            assert not bootstrap._initialized
            # Should print error to stderr
            mock_print.assert_called()

    def test_ensure_initialized_success(self) -> None:
        """Test ensure_initialized when CLI is initialized."""
        from src.core import bootstrap

        bootstrap._initialized = True
        CLIBootstrap.ensure_initialized()  # Should not raise

    def test_ensure_initialized_failure(self) -> None:
        """Test ensure_initialized when CLI is not initialized."""
        from src.core import bootstrap

        bootstrap._initialized = False
        with pytest.raises(BootstrapError, match="CLI not initialized"):
            CLIBootstrap.ensure_initialized()

    @pytest.mark.asyncio
    async def test_shutdown(self) -> None:
        """Test bootstrap shutdown."""
        # Initialize first
        with patch("src.core.logging.configure_logging"), patch("src.core.bootstrap.logfire"):
            await CLIBootstrap.initialize()

        from src.core import bootstrap

        assert bootstrap._initialized

        with patch("src.core.bootstrap.logfire") as mock_logfire:
            await CLIBootstrap.shutdown()
            # Should log shutdown messages
            assert mock_logfire.info.call_count >= 2

        assert not bootstrap._initialized

    @pytest.mark.asyncio
    async def test_shutdown_when_not_initialized(self) -> None:
        """Test shutdown when not initialized (should be safe)."""
        await CLIBootstrap.shutdown()  # Should not raise
        from src.core import bootstrap

        assert not bootstrap._initialized

    @pytest.mark.asyncio
    async def test_thread_safety(self) -> None:
        """Test that bootstrap initialization is thread-safe."""
        with patch("src.core.logging.configure_logging"), patch("src.core.bootstrap.logfire"):
            # Create multiple concurrent initialization tasks
            tasks = [CLIBootstrap.initialize() for _ in range(5)]

            # All should complete without error
            await asyncio.gather(*tasks)

            from src.core import bootstrap

            assert bootstrap._initialized

    @pytest.mark.asyncio
    async def test_bootstrap_error_inheritance(self) -> None:
        """Test that BootstrapError is properly configured."""
        error = BootstrapError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    @pytest.mark.asyncio
    async def test_verbose_flag_passed(self) -> None:
        """Test that verbose flag is handled during initialization."""
        with (
            patch("src.core.logging.configure_logging") as mock_configure_logging,
            patch("src.core.bootstrap.logfire"),
        ):
            await CLIBootstrap.initialize(verbose=True)
            mock_configure_logging.assert_called_once()

            # Reset for next test
            from src.core import bootstrap

            bootstrap._initialized = False

            await CLIBootstrap.initialize(verbose=False)
            # Should be called twice total now
            assert mock_configure_logging.call_count == 2

    @pytest.mark.asyncio
    async def test_logging_messages(self) -> None:
        """Test that appropriate log messages are generated during bootstrap."""
        with (
            patch("src.core.logging.configure_logging"),
            patch("src.core.bootstrap.logfire") as mock_logfire,
        ):
            await CLIBootstrap.initialize()

            # Check for expected log messages
            log_calls = [call[0][0] for call in mock_logfire.info.call_args_list]
            assert "CLI bootstrap: Logging initialized" in log_calls
            assert "CLI bootstrap: Initialization complete" in log_calls

    @pytest.mark.asyncio
    async def test_shutdown_logging_messages(self) -> None:
        """Test that appropriate log messages are generated during shutdown."""
        # Initialize first
        with patch("src.core.logging.configure_logging"), patch("src.core.bootstrap.logfire"):
            await CLIBootstrap.initialize()

        with patch("src.core.bootstrap.logfire") as mock_logfire:
            await CLIBootstrap.shutdown()

            # Check for expected log messages
            log_calls = [call[0][0] for call in mock_logfire.info.call_args_list]
            assert "CLI bootstrap: Shutting down" in log_calls
            assert "CLI bootstrap: Shutdown complete" in log_calls
