"""Command-line interface for the deep research system."""

import asyncio
import os
import sys
from types import TracebackType
from typing import Any, TypedDict, cast
from urllib.parse import urlparse

import click
import logfire
from pydantic import SecretStr, ValidationError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import TaskID
from rich.prompt import Prompt
from rich.table import Table

from core.workflow import ResearchWorkflow

# Try to import interactive selector for better UX
try:
    from interfaces.interactive_selector import interactive_select

    has_interactive_select = True
except ImportError:
    interactive_select = None
    has_interactive_select = False

# Optional imports for HTTP mode support
try:
    import httpx
    from httpx_sse import aconnect_sse

    _http_mode_available = True
except ImportError:
    httpx = None
    aconnect_sse = None
    _http_mode_available = False

from core.bootstrap import BootstrapError, CLIBootstrap
from core.events import (
    ErrorEvent,
    ResearchCompletedEvent,
    StageCompletedEvent,
    StageStartedEvent,
    StreamingUpdateEvent,
    research_event_bus,
)
from core.sse_models import (
    CompletedMessage,
    ConnectionMessage,
    ErrorMessage,
    HeartbeatMessage,
    PingMessage,
    SSEEventType,
    StageCompletedMessage,
    UpdateMessage,
    parse_sse_message,
)
from models.api_models import APIKeys
from models.clarification import ClarificationRequest, ClarificationResponse
from models.core import ResearchStage
from models.report_generator import ResearchReport

# Create console with force_terminal to ensure Live displays work correctly
console = Console(force_terminal=True)


# Type definitions for API response data
class SectionDict(TypedDict, total=False):
    """Type definition for section in API response."""

    title: str
    content: str
    subsections: list["SectionDict"]


class ReportDict(TypedDict, total=False):
    """Type definition for report in API response."""

    title: str
    executive_summary: str
    introduction: str
    methodology: str
    sections: list[SectionDict]
    conclusion: str
    recommendations: list[str]
    citations: list[str]
    generated_at: str


def validate_server_url(url: str) -> str:
    """Validate and normalize server URL.

    Args:
        url: Server URL to validate

    Returns:
        Validated URL

    Raises:
        ValueError: If URL is invalid
    """
    try:
        # If URL doesn't start with http:// or https://, add http://
        if not url.startswith(("http://", "https://")):
            url = f"http://{url}"

        parsed = urlparse(url)

        # Check scheme
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Invalid URL scheme: {parsed.scheme}. Only http/https are supported.")

        # Check host
        if not parsed.netloc:
            raise ValueError("Invalid URL: missing host")

        return url
    except Exception as e:
        raise ValueError(f"Invalid server URL: {e}") from e


class CLIStreamHandler:
    """Enhanced handler for streaming updates to the CLI with rich progress display."""

    def __init__(self, query: str = ""):
        """Initialize the enhanced CLI stream handler.

        Args:
            query: The research query being processed
        """
        from interfaces.progress_context import ProgressManager

        self.query = query
        self._research_started = False

        # Use new TerminalProgress system for all stages
        self.progress_manager = ProgressManager()
        self._clarification_active = False
        self._post_clarification_active = False  # Track non-clarification progress

        # Legacy progress for fallback - removed to prevent duplicate Live instances
        self.progress = None
        self.current_task = None
        self.stage_tasks: dict[ResearchStage, TaskID] = {}

    def start_research_tracking(self, query: str) -> None:
        """Start enhanced research tracking.

        Args:
            query: The research query being processed
        """
        if not self._research_started:
            self.query = query
            # Research tracking started (no enhanced progress anymore)
            self._research_started = True

    async def handle_streaming_update(self, event: StreamingUpdateEvent) -> None:
        """Handle streaming update events with enhanced progress display."""
        # Handle clarification stage with simple progress tracker
        if event.stage == ResearchStage.CLARIFICATION:
            # Start the simple progress tracker on first clarification event
            if not self._clarification_active:
                self._clarification_active = True
                self.progress_manager.start("Analyzing your query for clarity and scope...")
            # Update message if needed
            elif event.content:
                # Only update if it's a meaningful status update
                if "examining" in event.content.lower() or "analyzing" in event.content.lower():
                    self.progress_manager.update(event.content)
            # Don't print text during clarification - let progress bar handle it
            return

        # For non-clarification stages
        if self._clarification_active:
            return  # Don't process while clarification is running

        # Ensure research tracking is started
        if not self._research_started:
            self.start_research_tracking(self.query or "Research Query")

        # Start progress tracker ONLY if not already active
        if not self._post_clarification_active:
            self.progress_manager.start(f"Processing {event.stage.value}...")
            self._post_clarification_active = True
        else:
            # Just update the message for subsequent events
            if event.content:
                self.progress_manager.update(event.content[:80])

        # Update progress message
        if event.content:
            self.progress_manager.update(event.content[:80])

    async def handle_stage_started(self, event: StageStartedEvent) -> None:
        """Handle stage start events with enhanced display."""
        # Print to console so user can see it
        console.print(f"[cyan]{event.stage.value} started[/cyan]")
        logfire.info(f"Starting {event.stage.value} stage", request_id=event.request_id)

    async def handle_stage_completed(self, event: StageCompletedEvent) -> None:
        """Handle stage completion events with enhanced display."""
        # Handle clarification stage completion
        if event.stage == ResearchStage.CLARIFICATION:
            if self._clarification_active:
                # Stop the timer and show elapsed time
                self.progress_manager.stop_and_complete()
                self._clarification_active = False
                # Now it's safe for enhanced progress to start if needed
                # Add a small gap after completion message
                console.print()
            return

        # Also handle query transformation completion to ensure we're ready for research execution
        if event.stage == ResearchStage.QUERY_TRANSFORMATION:
            if self._clarification_active:
                # Query transformation is part of the two-phase clarification
                # Make sure we clear the flag so research execution updates can be shown
                self._clarification_active = False
            return

        # Handle FAILED stage - exit immediately
        if event.stage == ResearchStage.FAILED:
            if self._post_clarification_active:
                self.progress_manager.stop()
                self._post_clarification_active = False
            # Display failure message
            if not event.success and event.error_message:
                console.print(f"\n[red]✗ Research failed: {event.error_message}[/red]")
            else:
                console.print("\n[red]✗ Research failed[/red]")
            console.print("[dim]Exiting...[/dim]")
            sys.exit(1)

        # For non-clarification stages, DO NOT reset the post-clarification flag
        # Keep the progress manager running between stages

        # For non-clarification stages
        if not self._research_started:
            self.start_research_tracking(self.query or "Research Query")

        # Stage completed - keep progress running for next stage

    async def handle_error(self, event: ErrorEvent) -> None:
        """Handle error events with enhanced display."""
        if not self._research_started:
            self.start_research_tracking(self.query or "Research Query")

        # Show error in progress and console
        if event.error_message:
            self.progress_manager.update(f"Error: {event.error_message}")

            # Also print to console for visibility
            error_msg = event.error_message
            if event.stage == ResearchStage.FAILED:
                # Stop any active progress indicators
                if self._post_clarification_active:
                    self.progress_manager.stop()
                    self._post_clarification_active = False
                console.print(f"\n[red]✗ Research failed during {event.stage.value}[/red]")
                console.print(f"[red]  Error: {error_msg}[/red]")
                console.print("[dim]Exiting...[/dim]")
                sys.exit(1)
            elif event.recoverable:
                console.print(
                    f"[yellow]⚠ Recoverable error in {event.stage.value}: {error_msg}[/yellow]"
                )
            else:
                console.print(f"[red]✗ Error in {event.stage.value}: {error_msg}[/red]")

    async def handle_research_completed(self, event: ResearchCompletedEvent) -> None:
        """Handle research completion with enhanced display."""
        if not self._research_started:
            self.start_research_tracking(self.query or "Research Query")

        # Stop progress on completion
        if self._clarification_active:
            self.progress_manager.stop_and_complete()
            self._clarification_active = False

        # Also stop post-clarification progress if active
        if self._post_clarification_active:
            self.progress_manager.stop_and_complete()
            self._post_clarification_active = False

        # Keep display active for a moment to show completion
        import asyncio

        await asyncio.sleep(2)

        # Progress already stopped

    def _determine_activity_type(self, content: str) -> str:
        """Determine activity type based on content.

        Args:
            content: The activity content

        Returns:
            Activity type string
        """
        content_lower = content.lower()

        if any(word in content_lower for word in ["analyzing", "analyzing", "examining"]):
            return "analyzing"
        if any(word in content_lower for word in ["searching", "finding", "looking"]):
            return "searching"
        if any(word in content_lower for word in ["writing", "generating", "creating"]):
            return "writing"
        if any(word in content_lower for word in ["processing", "compressing", "organizing"]):
            return "synthesizing"
        if any(word in content_lower for word in ["validating", "checking", "verifying"]):
            return "validating"
        if any(word in content_lower for word in ["thinking", "considering", "evaluating"]):
            return "reasoning"
        return "thinking"

    def _extract_progress(self, content: str) -> float | None:
        """Extract progress information from content if available.

        Args:
            content: The activity content

        Returns:
            Progress value (0.0 to 1.0) or None
        """
        import re

        # Look for patterns like "3 of 10", "5/12", etc.
        patterns = [
            r"(\d+)\s+of\s+(\d+)",
            r"(\d+)/(\d+)",
            r"(\d+)\s+out\s+of\s+(\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                current = int(match.group(1))
                total = int(match.group(2))
                if total > 0:
                    return min(current / total, 1.0)

        # Look for percentage patterns
        percent_match = re.search(r"(\d+)%", content)
        if percent_match:
            return int(percent_match.group(1)) / 100.0

        return None

    def _extract_details(self, content: str) -> dict[str, str]:
        """Extract additional details from content.

        Args:
            content: The activity content

        Returns:
            Dictionary of detail key-value pairs
        """
        details = {}

        # Look for common patterns in research updates
        if "sources" in content.lower():
            if "processed" in content.lower():
                import re

                match = re.search(r"(\d+)\s+sources?\s+processed", content, re.IGNORECASE)
                if match:
                    details["Sources processed"] = match.group(1)

        if "confidence" in content.lower():
            import re

            match = re.search(r"confidence[:\s]+(\d+(?:\.\d+)?)", content, re.IGNORECASE)
            if match:
                details["Confidence"] = f"{match.group(1)}%"

        if "findings" in content.lower():
            import re

            match = re.search(r"(\d+)\s+findings?", content, re.IGNORECASE)
            if match:
                details["Findings"] = match.group(1)

        return details

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        if self._clarification_active:
            self.progress_manager.stop()
            self._clarification_active = False


class HTTPResearchClient:
    """HTTP client for research API with SSE support."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        """Initialize HTTP research client.

        Args:
            base_url: Base URL of the research API server
            timeout: Request timeout in seconds

        Raises:
            ImportError: If httpx-sse is not installed
            ValueError: If base_url is invalid

        Note:
            API tokens are expected to be configured on the server side; the CLI does
            not send secrets over the network.
        """
        if not _http_mode_available:
            raise ImportError("HTTP mode requires httpx-sse. Install with: uv add --optional cli")

        # Validate and normalize URL
        self.base_url = validate_server_url(base_url)
        self.timeout = timeout
        if httpx is None:
            raise ImportError("httpx not available")
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit - ensure client is closed."""
        _ = exc_type, exc_val, exc_tb  # Mark as intentionally unused
        await self.close()

    async def start_research(self, query: str) -> str:
        """Start research via HTTP POST.

        Args:
            query: Research query

        Returns:
            Research request ID
        """
        response = await self.client.post(
            f"{self.base_url}/research",
            json={
                "query": query,
                "stream": True,
            },
        )
        response.raise_for_status()
        data = response.json()  # httpx Response.json() is synchronous
        return data["request_id"]

    async def stream_events(self, request_id: str, handler: CLIStreamHandler) -> None:
        """Stream SSE events and convert to handler calls.

        Args:
            request_id: Research request ID
            handler: CLI stream handler
        """
        if aconnect_sse is None:
            raise ImportError("httpx-sse not available")
        async with aconnect_sse(
            self.client, "GET", f"{self.base_url}/research/{request_id}/stream"
        ) as event_source:
            async for sse in event_source.aiter_sse():
                await self._process_sse_event(sse, handler)

    async def _process_sse_event(self, sse: Any, handler: CLIStreamHandler) -> None:
        """Convert SSE event to handler method calls.

        Args:
            sse: Server-sent event
            handler: CLI stream handler
        """
        try:
            # Parse the message using Pydantic models
            msg = parse_sse_message(sse.data)

            if sse.event == SSEEventType.UPDATE and isinstance(msg, UpdateMessage):
                event = StreamingUpdateEvent(
                    _request_id=msg.request_id,
                    content=msg.content,
                    stage=ResearchStage[msg.stage],
                )
                await handler.handle_streaming_update(event)
            elif sse.event == SSEEventType.STAGE_COMPLETED and isinstance(
                msg, StageCompletedMessage
            ):
                event = StageCompletedEvent(
                    _request_id=msg.request_id,
                    stage=ResearchStage[msg.stage],
                    success=msg.success,
                    result=msg.result,
                )
                await handler.handle_stage_completed(event)
            elif sse.event == SSEEventType.ERROR and isinstance(msg, ErrorMessage):
                event = ErrorEvent(
                    _request_id=msg.request_id,
                    stage=ResearchStage[msg.stage],
                    error_message=msg.message,
                    error_type="research_error",
                )
                await handler.handle_error(event)
            elif sse.event == SSEEventType.COMPLETE and isinstance(msg, CompletedMessage):
                # Convert report dict to ResearchReport object if present
                report = None
                if msg.report:
                    try:
                        report = ResearchReport(**msg.report)
                    except (TypeError, ValidationError) as e:
                        logfire.warning(f"Failed to parse report in CompletedMessage: {e}")

                event = ResearchCompletedEvent(
                    _request_id=msg.request_id,
                    success=msg.success,
                    duration_seconds=msg.duration or 0.0,  # Default to 0 if not provided
                    error_message=msg.error,
                    report=report,
                )
                await handler.handle_research_completed(event)
            elif sse.event == SSEEventType.CONNECTION and isinstance(msg, ConnectionMessage):
                # Connection event, can be logged but not displayed
                logfire.info(f"SSE connection established: {msg.message}")
            elif sse.event == SSEEventType.PING and isinstance(msg, HeartbeatMessage | PingMessage):
                # Heartbeat ping, ignore
                pass
        except ValidationError as e:
            logfire.error(f"Failed to validate SSE data: {e}")
        except Exception as e:
            logfire.error(f"Error processing SSE event: {e}")

    async def stream_events_with_retry(
        self, request_id: str, handler: CLIStreamHandler, max_retries: int = 3
    ) -> None:
        """Stream events with automatic reconnection.

        Args:
            request_id: Research request ID
            handler: CLI stream handler
            max_retries: Maximum number of retry attempts
        """
        retry_count = 0

        while retry_count < max_retries:
            try:
                await self.stream_events(request_id, handler)
                break  # Success
            except Exception as e:
                if httpx and isinstance(e, httpx.ConnectError):
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise
                    msg = (
                        f"[yellow]Connection failed, "
                        f"retrying {retry_count}/{max_retries}...[/yellow]"
                    )
                    console.print(msg)
                    await asyncio.sleep(2**retry_count)  # Exponential backoff
                else:
                    console.print(f"[red]Stream error: {e}[/red]")
                    raise

    async def get_report(self, request_id: str) -> dict[str, Any]:
        """Fetch final report.

        Args:
            request_id: Research request ID

        Returns:
            Report data as dictionary
        """
        response = await self.client.get(f"{self.base_url}/research/{request_id}/report")
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()

    async def get_clarification(self, request_id: str) -> dict[str, Any]:
        """Fetch pending clarification status and questions.

        Args:
            request_id: Research request ID

        Returns:
            Dict with keys: awaiting_response (bool), clarification_request (dict|None),
            original_query (str|None), and additional metadata from the server.
            If no clarification is pending (404), returns {"awaiting_response": False}.
        """
        if httpx is None:
            raise ImportError("httpx not available")
        try:
            resp = await self.client.get(f"{self.base_url}/research/{request_id}/clarification")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 404:
                return {"awaiting_response": False}
            raise

    async def submit_clarification(
        self, request_id: str, response: ClarificationResponse | dict[str, Any]
    ) -> dict[str, Any]:
        """Submit clarification answers.

        Args:
            request_id: Research request ID
            response: ClarificationResponse model or equivalent dict

        Returns:
            Server response as dict
        """
        if httpx is None:
            raise ImportError("httpx not available")
        payload: dict[str, Any]
        if hasattr(response, "model_dump"):
            try:
                # Use JSON mode to ensure datetimes are ISO strings, etc.
                payload = response.model_dump(mode="json")  # type: ignore[assignment]
            except Exception:
                # Best effort fallback
                payload = dict(response)  # type: ignore[arg-type]
        else:
            payload = dict(response)

        resp = await self.client.post(
            f"{self.base_url}/research/{request_id}/clarification", json=payload
        )
        resp.raise_for_status()
        return resp.json()


def display_report_object(report: ResearchReport) -> None:
    """Display a ResearchReport object.

    Args:
        report: ResearchReport object to display
    """
    console.print("\n")

    # Display title and summary
    console.print(
        Panel(
            Markdown(f"# {report.title}\n\n{report.executive_summary}"),
            title="Research Report Summary",
            border_style="green",
        )
    )

    if report.metadata.source_summary:
        console.print("\n[bold magenta]Sources:[/bold magenta]")
        for source in report.metadata.source_summary[:10]:
            line = f"{source.get('id', '?')}: {source.get('title', '')}"
            if source.get("url"):
                line += f" ({source['url']})"
            console.print(f"  - {line}")

    # Display key sections (first 3)
    for section in report.sections[:3]:
        console.print(f"\n[bold cyan]{section.title}[/bold cyan]")
        content = section.content
        if len(content) > 500:
            content = content[:500] + "..."
        console.print(content)

    # Display recommendations
    if report.recommendations:
        console.print("\n[bold yellow]Recommendations:[/bold yellow]")
        for i, rec in enumerate(report.recommendations, 1):
            console.print(f"  {i}. {rec}")

    if report.references:
        console.print("\n[bold magenta]Footnotes:[/bold magenta]")
        for reference in report.references:
            console.print(f"  {reference}")


def display_report_dict(report_dict: dict[str, Any]) -> None:
    """Display a report from a dictionary.

    Args:
        report_dict: Report data as dictionary
    """
    console.print("\n")

    # Safely extract values with proper types
    title: str = str(report_dict.get("title", "Research Report"))
    summary: str = str(report_dict.get("executive_summary", ""))

    console.print(
        Panel(
            Markdown(f"# {title}\n\n{summary}"),
            title="Research Report Summary",
            border_style="green",
        )
    )

    metadata = report_dict.get("metadata")
    source_summary: list[Any] | None = None
    if isinstance(metadata, dict):
        raw_summary = metadata.get("source_summary")
        if isinstance(raw_summary, list):
            source_summary = raw_summary
    if source_summary:
        console.print("\n[bold magenta]Sources:[/bold magenta]")
        for entry in source_summary[:10]:
            if not isinstance(entry, dict):
                continue
            line = f"{entry.get('id', '?')}: {entry.get('title', '')}"
            url = entry.get("url")
            if url:
                line += f" ({url})"
            console.print(f"  - {line}")

    # Display key sections (first 3)
    sections_raw = report_dict.get("sections", [])
    # Cast to list[Any] after validation
    sections = cast(list[Any], sections_raw) if isinstance(sections_raw, list) else []

    for section_raw in sections[:3]:
        section_title: str = ""
        section_content: str = ""

        if isinstance(section_raw, dict):
            section = cast(dict[str, Any], section_raw)
            title_val = section.get("title")
            content_val = section.get("content")
            section_title = str(title_val) if title_val is not None else ""
            section_content = str(content_val) if content_val is not None else ""

        console.print(f"\n[bold cyan]{section_title}[/bold cyan]")

        # Truncate long content
        if len(section_content) > 500:
            display_content: str = section_content[:500] + "..."
        else:
            display_content = section_content
        console.print(display_content)

    # Display recommendations
    recommendations_raw = report_dict.get("recommendations", [])
    if isinstance(recommendations_raw, list) and recommendations_raw:
        recommendations = cast(list[Any], recommendations_raw)
        console.print("\n[bold yellow]Recommendations:[/bold yellow]")
        for i, rec in enumerate(recommendations, 1):
            rec_str = str(rec) if rec is not None else ""
            console.print(f"  {i}. {rec_str}")

    references_raw = report_dict.get("references", [])
    if isinstance(references_raw, list) and references_raw:
        references = cast(list[Any], references_raw)
        console.print("\n[bold magenta]Footnotes:[/bold magenta]")
        for reference in references:
            console.print(f"  {reference}")


def display_report(
    report_data: ResearchReport | dict[str, Any] | Any, is_dict: bool = False
) -> None:
    """Display report from either ResearchReport object or dictionary.

    Args:
        report_data: Report data as ResearchReport or dict
        is_dict: Whether report_data is already a dictionary
    """
    if isinstance(report_data, ResearchReport):
        display_report_object(report_data)
    elif isinstance(report_data, dict):
        # Cast to ensure type safety after isinstance check
        validated_dict = cast(dict[str, Any], report_data)
        display_report_dict(validated_dict)
    elif is_dict:
        # If marked as dict but not actually a dict, use empty dict
        display_report_dict({})
    elif hasattr(report_data, "model_dump"):
        # Handle other Pydantic models
        try:
            # Safely call model_dump and ensure it returns a dict
            dump_result = report_data.model_dump()
            validated_dict = cast(dict[str, Any], dump_result)
            display_report_dict(validated_dict)
        except (AttributeError, TypeError):
            # Fallback if model_dump fails
            display_report_dict({})
    else:
        # Fallback for unknown types
        display_report_dict({})


def display_http_report(report_data: dict[str, Any]) -> None:
    """Display report from HTTP response.

    Args:
        report_data: Report data dictionary
    """
    display_report_dict(report_data)


def save_http_report(report_data: dict[str, Any], filename: str) -> None:
    """Save HTTP report to file.

    Args:
        report_data: Report data dictionary
        filename: Output filename
    """
    content: list[str] = []

    # Safely extract and format as markdown
    title = str(report_data.get("title", "Research Report"))
    # Prefer top-level generated_at; otherwise use metadata.created_at
    generated_at_val = report_data.get("generated_at")
    if not generated_at_val:
        metadata = report_data.get("metadata")
        if isinstance(metadata, dict):
            generated_at_val = metadata.get("created_at")
    # Normalize to string
    generated_at = str(generated_at_val) if generated_at_val is not None else "N/A"
    executive_summary = str(report_data.get("executive_summary", ""))
    introduction = str(report_data.get("introduction", ""))
    methodology = str(report_data.get("methodology", ""))
    conclusion = str(report_data.get("conclusion", ""))

    content.append(f"# {title}\n")
    content.append(f"*Generated: {generated_at}*\n")
    content.append(f"\n## Executive Summary\n\n{executive_summary}\n")
    content.append(f"\n## Introduction\n\n{introduction}\n")
    content.append(f"\n## Methodology\n\n{methodology}\n")

    sections_raw = report_data.get("sections", [])
    if isinstance(sections_raw, list):
        sections = cast(list[Any], sections_raw)
        for section_raw in sections:
            if isinstance(section_raw, dict):
                section = cast(dict[str, Any], section_raw)
                sec_title = str(section.get("title", ""))
                sec_content = str(section.get("content", ""))
                content.append(f"\n## {sec_title}\n\n{sec_content}\n")

                subsections_raw = section.get("subsections")
                if isinstance(subsections_raw, list):
                    subsections = cast(list[Any], subsections_raw)
                    for subsection_raw in subsections:
                        if isinstance(subsection_raw, dict):
                            subsection = cast(dict[str, Any], subsection_raw)
                            sub_title = str(subsection.get("title", ""))
                            sub_content = str(subsection.get("content", ""))
                            content.append(f"\n### {sub_title}\n\n{sub_content}\n")

    content.append(f"\n## Conclusion\n\n{conclusion}\n")

    recommendations_raw = report_data.get("recommendations")
    if isinstance(recommendations_raw, list) and recommendations_raw:
        recommendations = cast(list[Any], recommendations_raw)
        content.append("\n## Recommendations\n")
        for rec in recommendations:
            content.append(f"- {str(rec)}\n")

    # Prefer references (ResearchReport model) but support legacy 'citations'
    references_raw = report_data.get("references")
    citations_raw = report_data.get("citations") if not references_raw else None
    entries_raw = references_raw if isinstance(references_raw, list) else citations_raw
    if isinstance(entries_raw, list) and entries_raw:
        citations = cast(list[Any], entries_raw)
        content.append("\n## References\n")
        for citation in citations:
            content.append(f"- {str(citation)}\n")

    with open(filename, "w") as f:
        f.write("\n".join(content))


async def handle_http_clarification_flow(
    client: "HTTPResearchClient",
    request_id: str,
    console: Console,
    stream_task: "asyncio.Task[None]",
    handler: "CLIStreamHandler",
) -> None:
    """Poll server for clarification, prompt user, and submit answers.

    Runs alongside SSE streaming. Exits when the stream completes or no
    clarification is pending. Designed to affect only HTTP mode.
    """
    # Try to import interactive clarification UI
    try:
        from interfaces.cli_multi_clarification import handle_multi_clarification_cli
    except ImportError:
        handle_multi_clarification_cli = None  # type: ignore[assignment]

    # If we cannot interactively prompt, just return (server may timeout)
    if handle_multi_clarification_cli is None:
        return

    backoff = 1.0
    max_backoff = 5.0
    handled_request_ids: set[str] = set()

    while not stream_task.done():
        try:
            status = await client.get_clarification(request_id)
        except Exception as e:  # Network or other error; keep trying while stream active
            logfire.debug(f"Clarification poll error: {e}")
            await asyncio.sleep(backoff)
            backoff = min(backoff + 1.0, max_backoff)
            continue

        awaiting = bool(status.get("awaiting_response"))
        if not awaiting:
            # No pending clarification – backoff and continue while stream active
            await asyncio.sleep(backoff)
            backoff = min(backoff + 1.0, max_backoff)
            continue

        # Reset backoff when we have work to do
        backoff = 1.0

        # Parse request payload
        req_data = status.get("clarification_request")
        original_query = str(status.get("original_query") or "")
        if not isinstance(req_data, dict):
            # Malformed payload – skip this cycle
            await asyncio.sleep(1.0)
            continue

        try:
            request_model = ClarificationRequest.model_validate(req_data)
        except Exception as e:
            logfire.warning(f"Failed to parse ClarificationRequest: {e}")
            await asyncio.sleep(1.0)
            continue

        # Deduplicate prompts: if we've already answered this request id,
        # wait for the server to resume the workflow instead of re-prompting.
        if request_model.id in handled_request_ids:
            await asyncio.sleep(1.0)
            continue

        # Telemetry: log the questions received from server (helps debug UI issues)
        try:
            question_summaries = [
                {
                    "id": q.id,
                    "order": q.order,
                    "required": q.is_required,
                    "type": q.question_type,
                    "text": (q.question[:120] if isinstance(q.question, str) else ""),
                }
                for q in request_model.questions
            ]
            logfire.info(
                "Clarification prompt received",
                request_id=request_id,
                clar_request_id=request_model.id,
                num_questions=len(request_model.questions),
                first_question=(
                    request_model.questions[0].question if request_model.questions else None
                ),
                questions=question_summaries,
            )
        except Exception:
            logfire.debug("Failed to log clarification questions", request_id=request_id)

        # Stop any active progress display so prompts render cleanly
        try:
            handler.progress_manager.stop()
        except Exception:
            pass

        # Prompt user interactively for answers
        try:
            response_model = await handle_multi_clarification_cli(
                request_model, original_query, console
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]Clarification cancelled by user[/yellow]")
            # Keep polling; user can respond later
            await asyncio.sleep(1.0)
            continue
        except Exception as e:
            console.print(f"[red]Clarification UI error: {e}[/red]")
            await asyncio.sleep(1.0)
            continue

        if response_model is None:
            # User chose not to answer now; keep polling
            await asyncio.sleep(1.0)
            continue

        # Submit answers
        try:
            _ = await client.submit_clarification(request_id, response_model)
            console.print("[green]Clarification submitted. Resuming research...[/green]")
            handled_request_ids.add(request_model.id)
            try:
                logfire.info(
                    "Clarification submitted",
                    request_id=request_id,
                    clar_request_id=request_model.id,
                    answers=len(response_model.answers),
                )
            except Exception:
                pass
        except Exception as e:
            if httpx and isinstance(e, httpx.HTTPStatusError):
                if e.response.status_code == 400:
                    try:
                        err = e.response.json()
                    except Exception:
                        err = {"detail": str(e)}
                    console.print(
                        f"[yellow]Validation error submitting clarification: {err}[/yellow]"
                    )
                    # Re-prompt next loop iteration
                elif e.response.status_code == 409:
                    console.print("[yellow]Clarification already processed. Continuing...[/yellow]")
                    handled_request_ids.add(request_model.id)
                else:
                    console.print(f"[red]Failed to submit clarification: {e}[/red]")
            else:
                console.print(f"[red]Failed to submit clarification: {e}[/red]")

        # After submit, poll until server clears pending or stream completes
        settle_backoff = 0.5
        for _ in range(10):  # up to ~5 seconds
            if stream_task.done():
                break
            try:
                post_status = await client.get_clarification(request_id)
                if not bool(post_status.get("awaiting_response")):
                    break
                # If a new request id appears, allow loop to handle it next
                new_req = post_status.get("clarification_request") or {}
                new_id = str(new_req.get("id") or "")
                if new_id and new_id != request_model.id:
                    break
            except Exception:
                pass
            await asyncio.sleep(settle_backoff)
            settle_backoff = min(settle_backoff + 0.5, 2.0)

        # Minimal change: after a successful submit + brief settle, stop polling for this run.
        # A new research run will start a fresh poller instance.
        return


async def run_research(
    query: str,
    api_keys: APIKeys | None = None,
    mode: str = "direct",
    server_url: str = "http://localhost:8000",
) -> None:
    """Run research with streaming updates.

    Args:
        query: Research query
        api_keys: Optional API keys
        mode: Execution mode ('direct' or 'http')
        server_url: Server URL for HTTP mode
    """
    # Create stream handler for UI updates only
    # Note: Global event handlers for logging are now set up by CLIBootstrap
    handler = CLIStreamHandler(query)

    if mode == "direct":
        # Subscribe to UI-related events only (logging events handled by bootstrap)
        await research_event_bus.subscribe(StreamingUpdateEvent, handler.handle_streaming_update)
        await research_event_bus.subscribe(StageStartedEvent, handler.handle_stage_started)
        await research_event_bus.subscribe(StageCompletedEvent, handler.handle_stage_completed)
        await research_event_bus.subscribe(ErrorEvent, handler.handle_error)
        await research_event_bus.subscribe(
            ResearchCompletedEvent, handler.handle_research_completed
        )

        # Start progress display
        with handler:
            # Execute research
            state = await ResearchWorkflow().run(
                user_query=query,
                api_keys=api_keys,
                stream_callback=True,
            )

        # Display results
        if state.final_report:
            display_report(state.final_report)

            # Save option with interactive selection
            save_choice = None
            if has_interactive_select and interactive_select:
                save_choice = interactive_select(
                    "Save full report to file?",
                    choices=["Yes", "No"],
                    default=1,  # Default to "No"
                    console=console,
                )
            else:
                resp = Prompt.ask("\nSave full report to file?", choices=["y", "n"], default="n")
                save_choice = "Yes" if resp == "y" else "No"

            if save_choice == "Yes":
                filename = Prompt.ask("Enter filename", default="research_report.md")
                save_report(state, filename)
                console.print(f"[green]Report saved to {filename}[/green]")

        elif state.error_message:
            console.print(
                Panel(
                    f"[red]Research failed: {state.error_message}[/red]",
                    title="Error",
                    border_style="red",
                )
            )

            # Show clarifying questions if available
            if state.metadata.pending_questions:
                console.print("\n[yellow]Clarifying questions needed:[/yellow]")
                for q in state.metadata.pending_questions:
                    console.print(f"  • {q.question}")

    else:  # HTTP mode
        if not _http_mode_available:
            console.print(
                "[red]HTTP mode requires additional dependencies.[/red]\n"
                + "Install with: uv add --optional cli"
            )
            sys.exit(1)

        async with HTTPResearchClient(server_url) as client:
            # Start research
            request_id = await client.start_research(query)
            console.print(f"[cyan]Research started with ID: {request_id}[/cyan]")

            # Start streaming in the background
            stream_task = asyncio.create_task(client.stream_events_with_retry(request_id, handler))

            # Handle clarification flow concurrently (HTTP-only)
            await handle_http_clarification_flow(client, request_id, console, stream_task, handler)

            # Ensure streaming completes
            await stream_task

            # Fetch and display final report
            try:
                report_data = await client.get_report(request_id)
                display_http_report(report_data)

                # Save option with interactive selection
                save_choice = None
                if has_interactive_select and interactive_select:
                    save_choice = interactive_select(
                        "Save full report to file?",
                        choices=["Yes", "No"],
                        default=1,  # Default to "No"
                        console=console,
                    )
                else:
                    save_prompt = Prompt.ask(
                        "\nSave full report to file?", choices=["y", "n"], default="n"
                    )
                    save_choice = "Yes" if save_prompt == "y" else "No"

                if save_choice == "Yes":
                    filename = Prompt.ask("Enter filename", default="research_report.md")
                    save_http_report(report_data, filename)
                    console.print(f"[green]Report saved to {filename}[/green]")

            except Exception as e:
                if httpx:
                    if isinstance(e, httpx.HTTPStatusError):
                        if e.response.status_code == 400:
                            console.print(
                                "[yellow]Report not yet available. "
                                + "Research may still be in progress.[/yellow]"
                            )
                        else:
                            console.print(f"[red]Failed to fetch report: {e}[/red]")
                    elif isinstance(e, httpx.ConnectError):
                        console.print(
                            f"[red]Failed to connect to server at {server_url}[/red]\n"
                            + "Ensure the server is running: "
                            + "uvicorn open_deep_research_pydantic_ai.api.main:app"
                        )
                    else:
                        console.print(f"[red]Failed to fetch report: {e}[/red]")
                else:
                    console.print(f"[red]Failed to fetch report: {e}[/red]")


def save_report_object(report: ResearchReport, filename: str) -> None:
    """Save ResearchReport object to file.

    Args:
        report: ResearchReport object
        filename: Output filename
    """
    content: list[str] = []

    # Format as markdown
    content.append(f"# {report.title}\n")
    content.append(f"*Generated: {report.metadata.created_at}*\n")
    content.append(f"\n## Executive Summary\n\n{report.executive_summary}\n")
    content.append(f"\n## Introduction\n\n{report.introduction}\n")

    for section in report.sections:
        content.append(f"\n## {section.title}\n\n{section.content}\n")
        if section.subsections:
            for subsection in section.subsections:
                content.append(f"\n### {subsection.title}\n\n{subsection.content}\n")

    content.append(f"\n## Conclusions\n\n{report.conclusions}\n")

    if report.recommendations:
        content.append("\n## Recommendations\n")
        for rec in report.recommendations:
            content.append(f"- {rec}\n")

    if report.references:
        content.append("\n## Footnotes\n")
        for reference in report.references:
            content.append(f"{reference}\n")

    with open(filename, "w") as f:
        f.write("\n".join(content))


def save_report(state: Any, filename: str) -> None:
    """Save research report to file.

    Args:
        state: Research state with report
        filename: Output filename
    """
    if not hasattr(state, "final_report") or not state.final_report:
        return

    report = state.final_report
    if isinstance(report, ResearchReport):
        save_report_object(report, filename)
    elif hasattr(report, "model_dump"):
        # Handle other Pydantic models by converting to dict first
        try:
            dump_result = report.model_dump()
            validated_dict = cast(dict[str, Any], dump_result)
            save_http_report(validated_dict, filename)
        except (AttributeError, TypeError):
            # Fallback if model_dump fails
            save_http_report({}, filename)


@click.group()
def cli():
    """Deep Research CLI - AI-powered research assistant."""
    pass


@cli.command()
@click.argument("query")
@click.option("--api-key", "-k", multiple=True, help="API key in format service:key")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["direct", "http"], case_sensitive=False),
    default="direct",
    help="Execution mode: direct (in-process) or http (client-server)",
)
@click.option(
    "--server-url",
    "-s",
    default="http://localhost:8000",
    help="Server URL for HTTP mode (default: http://localhost:8000)",
)
def research(query: str, api_key: tuple[str, ...], verbose: bool, mode: str, server_url: str):
    """Execute a research query.

    QUERY: The research question or topic to investigate.

    Examples:

        # Direct mode (default)
        deep-research "What is quantum computing?"

        # HTTP mode with local server
        deep-research "What is quantum computing?" --mode http

        # HTTP mode with remote server
        deep-research "What is quantum computing?" --mode http --server-url http://api.example.com:8000
    """

    async def _async_research():
        """Async wrapper for research execution with bootstrap initialization."""
        try:
            # Initialize bootstrap first - this ensures logfire is ready
            # and event handlers are set up before any research operations
            await CLIBootstrap.initialize(verbose=verbose)

            # Parse API keys from command line
            parsed_keys: dict[str, str] = {}
            for key_pair in api_key:
                if ":" in key_pair:
                    service, key_value = key_pair.split(":", 1)
                    parsed_keys[service] = key_value

            # Create APIKeys model with environment fallbacks
            openai_val = os.getenv("OPENAI_API_KEY")
            anthropic_val = os.getenv("ANTHROPIC_API_KEY")
            tavily_val = os.getenv("TAVILY_API_KEY")

            api_keys = APIKeys(
                openai=SecretStr(parsed_keys["openai"])
                if "openai" in parsed_keys
                else (SecretStr(openai_val) if openai_val else None),
                anthropic=SecretStr(parsed_keys["anthropic"])
                if "anthropic" in parsed_keys
                else (SecretStr(anthropic_val) if anthropic_val else None),
                tavily=SecretStr(parsed_keys["tavily"])
                if "tavily" in parsed_keys
                else (SecretStr(tavily_val) if tavily_val else None),
            )

            if mode == "http" and parsed_keys:
                console.print(
                    "[yellow]CLI-supplied API keys are ignored in HTTP mode."
                    " Configure keys on the server instead.[/yellow]"
                )

            api_keys_for_mode = api_keys if mode == "direct" else None

            console.print(
                Panel(
                    f"[bold cyan]Research Query:[/bold cyan] {query}\n"
                    + f"[bold cyan]Mode:[/bold cyan] {mode.upper()}"
                    + (f"\n[bold cyan]Server:[/bold cyan] {server_url}" if mode == "http" else ""),
                    title="Deep Research System",
                    border_style="cyan",
                )
            )

            # Run research with selected mode
            await run_research(query, api_keys_for_mode, mode=mode, server_url=server_url)

        except BootstrapError as e:
            console.print(f"[red]Failed to initialize CLI: {e}[/red]")
            sys.exit(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Research interrupted by user[/yellow]")
            sys.exit(0)
        except Exception as e:
            console.print(f"[red]Research failed: {e}[/red]")
            sys.exit(1)
        finally:
            # Clean up bootstrap resources
            await CLIBootstrap.shutdown()

    # Run the async research
    asyncio.run(_async_research())


@cli.command()
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["direct", "http"], case_sensitive=False),
    default="direct",
    help="Execution mode for interactive session",
)
@click.option(
    "--server-url",
    "-s",
    default="http://localhost:8000",
    help="Server URL for HTTP mode",
)
def interactive(mode: str, server_url: str):
    """Start an interactive research session."""

    async def _async_interactive():
        """Async wrapper for interactive session with bootstrap initialization."""
        try:
            # Initialize bootstrap once for the entire interactive session
            # This ensures logfire and event handlers are ready from the start
            await CLIBootstrap.initialize(verbose=False)

            console.print(
                Panel(
                    "[bold cyan]Deep Research Interactive Mode[/bold cyan]\n"
                    + f"Mode: {mode.upper()}"
                    + (f" | Server: {server_url}" if mode == "http" else "")
                    + "\n\nEnter your research queries, or type 'help' for commands.",
                    border_style="cyan",
                )
            )

            # Load API keys from environment
            openai_key = os.getenv("OPENAI_API_KEY")
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            tavily_key = os.getenv("TAVILY_API_KEY")

            api_keys = APIKeys(
                openai=SecretStr(openai_key) if openai_key else None,
                anthropic=SecretStr(anthropic_key) if anthropic_key else None,
                tavily=SecretStr(tavily_key) if tavily_key else None,
            )

            if mode == "http":
                console.print(
                    "[yellow]HTTP mode expects API keys to be configured on the server."
                    " Local keys will not be sent.[/yellow]"
                )

            api_keys_for_mode = api_keys if mode == "direct" else None

            while True:
                try:
                    query = Prompt.ask("\n[cyan]Research query[/cyan]")

                    if query.lower() in ["exit", "quit", "q"]:
                        console.print("[yellow]Goodbye![/yellow]")
                        break
                    if query.lower() == "help":
                        console.print("""
[bold]Available commands:[/bold]
  • Enter any research query to start research
  • 'help' - Show this help message
  • 'clear' - Clear the screen
  • 'exit', 'quit', 'q' - Exit the program
                        """)
                    elif query.lower() == "clear":
                        console.clear()
                    else:
                        await run_research(
                            query,
                            api_keys_for_mode,
                            mode=mode,
                            server_url=server_url,
                        )

                except KeyboardInterrupt:
                    console.print("\n[yellow]Use 'exit' to quit[/yellow]")
                except Exception as e:
                    console.print(f"[red]Error: {str(e)}[/red]")

        except BootstrapError as e:
            console.print(f"[red]Failed to initialize CLI: {e}[/red]")
            sys.exit(1)
        finally:
            # Clean up bootstrap resources when exiting interactive mode
            await CLIBootstrap.shutdown()

    # Run the async interactive session
    asyncio.run(_async_interactive())


@cli.command()
def version():
    """Show version information."""
    table = Table(title="Deep Research System", border_style="cyan")
    table.add_column("Component", style="cyan")
    table.add_column("Version", style="green")

    table.add_row("Deep Research", "1.0.0")
    table.add_row("Pydantic AI", "Latest")
    table.add_row(
        "Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    console.print(table)


if __name__ == "__main__":
    cli()
