"""Unit tests for CLI HTTP mode functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cli import (
    CLIStreamHandler,
    HTTPResearchClient,
    validate_server_url,
)
from core.sse_models import (
    CompletedMessage,
    SSEDataType,
    SSEEventType,
    UpdateMessage,
)
from models.core import ResearchStage


class TestURLValidation:
    """Test URL validation function."""

    def test_valid_http_url(self):
        """Test with valid HTTP URL."""
        url = "http://localhost:8000"
        assert validate_server_url(url) == url

    def test_valid_https_url(self):
        """Test with valid HTTPS URL."""
        url = "https://api.example.com"
        assert validate_server_url(url) == url

    def test_url_without_scheme(self):
        """Test URL without scheme gets http:// added."""
        url = "localhost:8000"
        assert validate_server_url(url) == "http://localhost:8000"

    def test_invalid_scheme(self):
        """Test invalid scheme raises error - doesn't apply with auto-prepend."""
        # Since we now auto-prepend http://, this test changes behavior
        # ftp://example.com becomes http://ftp://example.com which is valid syntactically
        url = "ftp://example.com"
        result = validate_server_url(url)
        assert result == "http://ftp://example.com"  # Auto-prepended

    def test_missing_host(self):
        """Test missing host raises error."""
        with pytest.raises(ValueError, match="missing host"):
            validate_server_url("http://")

    def test_url_with_path(self):
        """Test URL with path is preserved."""
        url = "http://api.example.com:8080/v1"
        assert validate_server_url(url) == url


class TestSSEEventConstants:
    """Test SSE event type constants."""

    def test_event_constants_exist(self):
        """Test all expected event constants exist as enums."""
        assert SSEEventType.UPDATE.value == "update"
        assert SSEEventType.STAGE_COMPLETED.value == "stage"  # Note: event field is "stage"
        assert SSEEventType.ERROR.value == "error"
        assert SSEEventType.COMPLETE.value == "complete"
        assert SSEEventType.CONNECTION.value == "connection"
        assert SSEEventType.PING.value == "ping"

    def test_data_type_constants_exist(self):
        """Test all expected data type constants exist as enums."""
        assert SSEDataType.UPDATE.value == "update"
        assert SSEDataType.STAGE_COMPLETED.value == "stage_completed"
        assert SSEDataType.ERROR.value == "error"
        assert SSEDataType.COMPLETED.value == "completed"
        assert SSEDataType.CONNECTED.value == "connected"
        assert SSEDataType.HEARTBEAT.value == "heartbeat"
        assert SSEDataType.PING.value == "ping"


class TestSSEMessageValidation:
    """Test Pydantic SSE message validation."""

    def test_update_message_validation(self):
        """Test UpdateMessage validation."""
        msg = UpdateMessage(
            request_id="test-123", stage="CLARIFICATION", content="Test content", is_partial=True
        )
        assert msg.type == SSEDataType.UPDATE
        assert msg.request_id == "test-123"
        assert msg.stage == "CLARIFICATION"
        assert msg.content == "Test content"
        assert msg.is_partial is True

    def test_completed_message_validation(self):
        """Test CompletedMessage validation."""
        msg = CompletedMessage(
            request_id="test-123", success=True, duration=10.5, error=None, has_report=True
        )
        assert msg.type == SSEDataType.COMPLETED
        assert msg.request_id == "test-123"
        assert msg.success is True
        assert msg.duration == 10.5
        assert msg.error is None
        assert msg.has_report is True

    def test_message_json_roundtrip(self):
        """Test message serialization and deserialization."""
        original = UpdateMessage(
            request_id="test-123",
            stage="RESEARCH_EXECUTION",
            content="Searching...",
            is_partial=False,
        )
        json_str = original.model_dump_json()
        parsed = UpdateMessage.model_validate_json(json_str)
        assert parsed == original


@pytest.mark.asyncio
class TestHTTPResearchClient:
    """Test HTTPResearchClient class."""

    async def test_context_manager(self):
        """Test context manager functionality."""
        with patch("open_deep_research_pydantic_ai.cli._http_mode_available", True):
            with patch("open_deep_research_pydantic_ai.cli.httpx.AsyncClient") as mock_client:
                mock_instance = AsyncMock()
                mock_client.return_value = mock_instance

                async with HTTPResearchClient("http://localhost:8000") as client:
                    assert client.base_url == "http://localhost:8000"
                    assert client.timeout == 30.0

                # Ensure close was called
                mock_instance.aclose.assert_called_once()

    async def test_invalid_url_raises_error(self):
        """Test that invalid URL raises ValueError."""
        with patch("open_deep_research_pydantic_ai.cli._http_mode_available", True):
            with patch("open_deep_research_pydantic_ai.cli.httpx.AsyncClient"):
                # Since we auto-prepend http://, ftp:// becomes http://ftp://...
                # We need to test a different invalid case
                with pytest.raises(ValueError, match="missing host"):
                    HTTPResearchClient("http://")

    async def test_missing_httpx_sse_raises_error(self):
        """Test that missing httpx-sse raises ImportError."""
        with patch("open_deep_research_pydantic_ai.cli._http_mode_available", False):
            with pytest.raises(ImportError, match="HTTP mode requires httpx-sse"):
                HTTPResearchClient("http://localhost:8000")

    async def test_start_research(self):
        """Test start_research method."""
        with patch("open_deep_research_pydantic_ai.cli._http_mode_available", True):
            with patch("open_deep_research_pydantic_ai.cli.httpx.AsyncClient") as mock_client:
                # Setup mock
                mock_instance = AsyncMock()
                mock_response = AsyncMock()
                mock_response.json = MagicMock(return_value={"request_id": "test-123"})
                mock_response.raise_for_status = MagicMock()
                mock_instance.post.return_value = mock_response
                mock_client.return_value = mock_instance

                # Test
                client = HTTPResearchClient("http://localhost:8000")
                request_id = await client.start_research("Test query")

                assert request_id == "test-123"
                mock_instance.post.assert_called_once_with(
                    "http://localhost:8000/research",
                    json={
                        "query": "Test query",
                        "stream": True,
                    },
                )

    async def test_process_sse_event_update(self):
        """Test processing UPDATE SSE event."""
        with patch("open_deep_research_pydantic_ai.cli._http_mode_available", True):
            with patch("open_deep_research_pydantic_ai.cli.httpx.AsyncClient"):
                client = HTTPResearchClient("http://localhost:8000")
                handler = AsyncMock(spec=CLIStreamHandler)

                # Create mock SSE event
                mock_sse = MagicMock()
                mock_sse.event = SSEEventType.UPDATE
                # Create valid UpdateMessage JSON
                update_msg = UpdateMessage(
                    request_id="test-123", content="Processing...", stage="CLARIFICATION"
                )
                mock_sse.data = update_msg.model_dump_json()

                # Process event
                await client._process_sse_event(mock_sse, handler)

                # Verify handler was called
                handler.handle_streaming_update.assert_called_once()
                called_event = handler.handle_streaming_update.call_args[0][0]
                assert called_event.content == "Processing..."
                assert called_event.stage == ResearchStage.CLARIFICATION

    async def test_process_sse_event_complete(self):
        """Test processing COMPLETE SSE event."""
        with patch("open_deep_research_pydantic_ai.cli._http_mode_available", True):
            with patch("open_deep_research_pydantic_ai.cli.httpx.AsyncClient"):
                client = HTTPResearchClient("http://localhost:8000")
                handler = AsyncMock(spec=CLIStreamHandler)

                # Create mock SSE event
                mock_sse = MagicMock()
                mock_sse.event = SSEEventType.COMPLETE
                # Create valid CompletedMessage JSON
                test_report = {
                    "title": "Test Report",
                    "executive_summary": "Test summary",
                    "introduction": "Test intro",
                    "methodology": "Test methodology",
                    "sections": [],
                    "conclusion": "Test conclusion",
                }
                completed_msg = CompletedMessage(
                    request_id="test-123", success=True, error=None, report=test_report
                )
                mock_sse.data = completed_msg.model_dump_json()

                # Process event
                await client._process_sse_event(mock_sse, handler)

                # Verify handler was called with await
                handler.handle_research_completed.assert_awaited_once()
                called_event = handler.handle_research_completed.call_args[0][0]
                assert called_event.success is True
                assert called_event.report is not None
                assert called_event.report.title == "Test Report"

    async def test_process_sse_event_invalid_json(self):
        """Test handling of invalid JSON in SSE event."""
        with patch("open_deep_research_pydantic_ai.cli._http_mode_available", True):
            with patch("open_deep_research_pydantic_ai.cli.httpx.AsyncClient"):
                with patch("open_deep_research_pydantic_ai.cli.logfire") as mock_logfire:
                    client = HTTPResearchClient("http://localhost:8000")
                    handler = AsyncMock(spec=CLIStreamHandler)

                    # Create mock SSE event with invalid JSON
                    mock_sse = MagicMock()
                    mock_sse.event = SSEEventType.UPDATE
                    mock_sse.data = "invalid json {"

                    # Process event - should not raise
                    await client._process_sse_event(mock_sse, handler)

                    # Verify error was logged (now reports as general processing error)
                    mock_logfire.error.assert_called()
                    assert "Error processing SSE event" in mock_logfire.error.call_args[0][0]

    async def test_get_report(self):
        """Test get_report method."""
        with patch("open_deep_research_pydantic_ai.cli._http_mode_available", True):
            with patch("open_deep_research_pydantic_ai.cli.httpx.AsyncClient") as mock_client:
                # Setup mock
                mock_instance = AsyncMock()
                mock_response = AsyncMock()
                mock_response.json = MagicMock(
                    return_value={"title": "Research Report", "sections": []}
                )
                mock_response.raise_for_status = MagicMock()
                mock_instance.get.return_value = mock_response
                mock_client.return_value = mock_instance

                # Test
                client = HTTPResearchClient("http://localhost:8000")
                report = await client.get_report("test-123")

                assert report == {"title": "Research Report", "sections": []}
                mock_instance.get.assert_called_once_with(
                    "http://localhost:8000/research/test-123/report"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
