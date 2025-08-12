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
from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from open_deep_research_with_pydantic_ai.core.events import (
    ErrorEvent,
    ResearchCompletedEvent,
    StageCompletedEvent,
    StreamingUpdateEvent,
    research_event_bus,
)
from open_deep_research_with_pydantic_ai.core.sse_models import (
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
from open_deep_research_with_pydantic_ai.core.workflow import workflow
from open_deep_research_with_pydantic_ai.models.api_models import APIKeys
from open_deep_research_with_pydantic_ai.models.research import ResearchReport, ResearchStage

# Try to import httpx-sse for HTTP mode support
try:
    import httpx
    from httpx_sse import aconnect_sse

    _http_mode_available = True
except ImportError:
    _http_mode_available = False
    httpx = None
    aconnect_sse = None

console = Console()


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
    """Handler for streaming updates to the CLI."""

    def __init__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        )
        self.current_task = None
        self.stage_tasks: dict[ResearchStage, TaskID] = {}

    async def handle_streaming_update(self, event: StreamingUpdateEvent) -> None:
        """Handle streaming update events."""
        stage_name = event.stage.value.replace("_", " ").title()

        if event.stage not in self.stage_tasks:
            self.stage_tasks[event.stage] = self.progress.add_task(
                f"[cyan]{stage_name}[/cyan]: {event.content[:50]}...",
                total=None,
            )
        else:
            self.progress.update(
                self.stage_tasks[event.stage],
                description=f"[cyan]{stage_name}[/cyan]: {event.content[:50]}...",
            )

    async def handle_stage_completed(self, event: StageCompletedEvent) -> None:
        """Handle stage completion events."""
        if event.stage in self.stage_tasks:
            stage_name = event.stage.value.replace("_", " ").title()
            status = "✓" if event.success else "✗"
            color = "green" if event.success else "red"

            self.progress.update(
                self.stage_tasks[event.stage],
                description=f"[{color}]{stage_name} {status}[/{color}]",
                completed=100,
            )

    async def handle_error(self, event: ErrorEvent) -> None:
        """Handle error events."""
        console.print(
            Panel(
                f"[red]Error in {event.stage.value}: {event.error_message}[/red]",
                title="Error",
                border_style="red",
            )
        )

    async def handle_research_completed(self, event: ResearchCompletedEvent) -> None:
        """Handle research completion."""
        if event.success:
            console.print("\n[green]✓ Research completed successfully![/green]")
        else:
            console.print(f"\n[red]✗ Research failed: {event.error_message}[/red]")


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

    async def start_research(self, query: str, api_keys: APIKeys | None) -> str:
        """Start research via HTTP POST.

        Args:
            query: Research query
            api_keys: Optional API keys

        Returns:
            Research request ID
        """
        response = await self.client.post(
            f"{self.base_url}/research",
            json={
                "query": query,
                "api_keys": api_keys.to_dict() if api_keys else None,
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
                    console.print(
                        f"[yellow]Connection failed, retrying "
                        f"{retry_count}/{max_retries}...[/yellow]"
                    )
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
    generated_at = str(report_data.get("generated_at", "N/A"))
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

    citations_raw = report_data.get("citations")
    if isinstance(citations_raw, list) and citations_raw:
        citations = cast(list[Any], citations_raw)
        content.append("\n## References\n")
        for citation in citations:
            content.append(f"- {str(citation)}\n")

    with open(filename, "w") as f:
        f.write("\n".join(content))


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
    # Create stream handler
    handler = CLIStreamHandler()

    if mode == "direct":
        # Current implementation - direct workflow execution
        await research_event_bus.subscribe(StreamingUpdateEvent, handler.handle_streaming_update)
        await research_event_bus.subscribe(StageCompletedEvent, handler.handle_stage_completed)
        await research_event_bus.subscribe(ErrorEvent, handler.handle_error)
        await research_event_bus.subscribe(
            ResearchCompletedEvent, handler.handle_research_completed
        )

        # Start progress display
        with handler.progress:
            # Execute research
            state = await workflow.execute_research(
                user_query=query,
                api_keys=api_keys,
                stream_callback=True,
            )

        # Display results
        if state.final_report:
            display_report(state.final_report)

            # Save option
            if Prompt.ask("\nSave full report to file?", choices=["y", "n"], default="n") == "y":
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
            if "clarifying_questions" in state.metadata:
                console.print("\n[yellow]Clarifying questions needed:[/yellow]")
                for q in state.metadata["clarifying_questions"]:
                    console.print(f"  • {q}")

    else:  # HTTP mode
        if not _http_mode_available:
            console.print(
                "[red]HTTP mode requires additional dependencies.[/red]\n"
                "Install with: uv add --optional cli"
            )
            sys.exit(1)

        async with HTTPResearchClient(server_url) as client:
            with handler.progress:
                # Start research
                request_id = await client.start_research(query, api_keys)
                console.print(f"[cyan]Research started with ID: {request_id}[/cyan]")

                # Stream events with retry logic
                await client.stream_events_with_retry(request_id, handler)

            # Fetch and display final report
            try:
                report_data = await client.get_report(request_id)
                display_http_report(report_data)

                # Save option
                save_prompt = Prompt.ask(
                    "\nSave full report to file?", choices=["y", "n"], default="n"
                )
                if save_prompt == "y":
                    filename = Prompt.ask("Enter filename", default="research_report.md")
                    save_http_report(report_data, filename)
                    console.print(f"[green]Report saved to {filename}[/green]")

            except Exception as e:
                if httpx:
                    if isinstance(e, httpx.HTTPStatusError):
                        if e.response.status_code == 400:
                            console.print(
                                "[yellow]Report not yet available. "
                                "Research may still be in progress.[/yellow]"
                            )
                        else:
                            console.print(f"[red]Failed to fetch report: {e}[/red]")
                    elif isinstance(e, httpx.ConnectError):
                        console.print(
                            f"[red]Failed to connect to server at {server_url}[/red]\n"
                            f"Ensure the server is running: "
                            f"uvicorn open_deep_research_with_pydantic_ai.api.main:app"
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
    content.append(f"*Generated: {report.generated_at}*\n")
    content.append(f"\n## Executive Summary\n\n{report.executive_summary}\n")
    content.append(f"\n## Introduction\n\n{report.introduction}\n")
    content.append(f"\n## Methodology\n\n{report.methodology}\n")

    for section in report.sections:
        content.append(f"\n## {section.title}\n\n{section.content}\n")
        if section.subsections:
            for subsection in section.subsections:
                content.append(f"\n### {subsection.title}\n\n{subsection.content}\n")

    content.append(f"\n## Conclusion\n\n{report.conclusion}\n")

    if report.recommendations:
        content.append("\n## Recommendations\n")
        for rec in report.recommendations:
            content.append(f"- {rec}\n")

    if report.citations:
        content.append("\n## References\n")
        for citation in report.citations:
            content.append(f"- {citation}\n")

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
    # Configure logging
    if verbose:
        logfire.configure()

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

    console.print(
        Panel(
            f"[bold cyan]Research Query:[/bold cyan] {query}\n"
            f"[bold cyan]Mode:[/bold cyan] {mode.upper()}"
            + (f"\n[bold cyan]Server:[/bold cyan] {server_url}" if mode == "http" else ""),
            title="Deep Research System",
            border_style="cyan",
        )
    )

    # Run research with selected mode
    try:
        asyncio.run(run_research(query, api_keys, mode=mode, server_url=server_url))
    except KeyboardInterrupt:
        console.print("\n[yellow]Research interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


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
    console.print(
        Panel(
            f"[bold cyan]Deep Research Interactive Mode[/bold cyan]\n"
            f"Mode: {mode.upper()}"
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

    while True:
        try:
            query = Prompt.ask("\n[cyan]Research query[/cyan]")

            if query.lower() in ["exit", "quit", "q"]:
                console.print("[yellow]Goodbye![/yellow]")
                break
            elif query.lower() == "help":
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
                asyncio.run(run_research(query, api_keys, mode=mode, server_url=server_url))

        except KeyboardInterrupt:
            console.print("\n[yellow]Use 'exit' to quit[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")


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
