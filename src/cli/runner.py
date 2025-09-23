"""Run coordinators for direct and HTTP modes."""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

import logfire
from pydantic import SecretStr
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from core.bootstrap import BootstrapError, CLIBootstrap
from core.config import config as global_config
from core.events import (
    ErrorEvent,
    ResearchCompletedEvent,
    StageCompletedEvent,
    StageStartedEvent,
    StreamingUpdateEvent,
    research_event_bus,
)
from core.workflow import ResearchWorkflow
from models.api_models import APIKeys

from .clarification_http import handle_http_clarification_flow
from .http_client import HTTPResearchClient
from .report_io import (
    display_http_report,
    display_report_dict,
    display_report_object,
    save_http_report,
    save_report_object,
)
from .stream import CLIStreamHandler
from .util import validate_server_url

console = Console(force_terminal=True)


def display_report(report: Any) -> None:
    if hasattr(report, "model_fields"):
        display_report_object(report)  # ResearchReport
    elif isinstance(report, dict):
        display_report_dict(report)
    else:
        try:
            display_report_dict(report.model_dump())  # type: ignore[attr-defined]
        except Exception:
            display_report_dict({})


def save_report(state: Any, filename: str) -> None:
    report = getattr(state, "final_report", None)
    if report is None:
        return
    if hasattr(report, "model_fields"):
        save_report_object(report, filename)
    else:
        try:
            save_http_report(report.model_dump(), filename)  # type: ignore[attr-defined]
        except Exception:
            save_http_report({}, filename)


async def run_direct(query: str, api_keys: APIKeys | None) -> None:
    # Log synthesis flags for direct mode startup
    logfire.info(
        "Direct mode start",
        embedding_similarity=global_config.enable_embedding_similarity,
        similarity_threshold=global_config.embedding_similarity_threshold,
        llm_clean_merge=global_config.enable_llm_clean_merge,
    )
    handler = CLIStreamHandler(query)
    await research_event_bus.subscribe(StreamingUpdateEvent, handler.handle_streaming_update)
    await research_event_bus.subscribe(StageStartedEvent, handler.handle_stage_started)
    await research_event_bus.subscribe(StageCompletedEvent, handler.handle_stage_completed)
    await research_event_bus.subscribe(ErrorEvent, handler.handle_error)
    await research_event_bus.subscribe(ResearchCompletedEvent, handler.handle_research_completed)

    with handler:
        state = await ResearchWorkflow().run(
            user_query=query, api_keys=api_keys, stream_callback=True
        )

    if state.final_report:
        display_report(state.final_report)
        resp = Prompt.ask("\nSave full report to file?", choices=["y", "n"], default="n")
        if resp == "y":
            filename = Prompt.ask("Enter filename", default="research_report.md")
            save_report(state, filename)
            console.print(f"[green]Report saved to {filename}[/green]")
    elif state.error_message:
        console.print(Panel(f"[red]Research failed: {state.error_message}[/red]", title="Error"))


async def run_http(query: str, server_url: str) -> None:
    handler = CLIStreamHandler(query)
    server_url = validate_server_url(server_url)

    async with HTTPResearchClient(server_url) as client:
        request_id = await client.start_research(query)
        console.print(f"[cyan]Research started with ID: {request_id}[/cyan]")
        stream_task = asyncio.create_task(client.stream_events_with_retry(request_id, handler))
        await handle_http_clarification_flow(client, request_id, console, stream_task, handler)
        await stream_task
        try:
            report_data = await client.get_report(request_id)
            display_http_report(report_data)
            save_prompt = Prompt.ask("\nSave full report to file?", choices=["y", "n"], default="n")
            if save_prompt == "y":
                filename = Prompt.ask("Enter filename", default="research_report.md")
                save_http_report(report_data, filename)
                console.print(f"[green]Report saved to {filename}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to fetch report: {e}[/red]")


def make_api_keys_from_env_and_cli(api_key: tuple[str, ...]) -> APIKeys:
    parsed: dict[str, str] = {}
    for key_pair in api_key:
        if ":" in key_pair:
            service, key_value = key_pair.split(":", 1)
            parsed[service] = key_value
    openai_val = os.getenv("OPENAI_API_KEY")
    anthropic_val = os.getenv("ANTHROPIC_API_KEY")
    tavily_val = os.getenv("TAVILY_API_KEY")
    return APIKeys(
        openai=(
            SecretStr(parsed.get("openai", openai_val or ""))
            if ("openai" in parsed or openai_val)
            else None
        ),
        anthropic=(
            SecretStr(parsed.get("anthropic", anthropic_val or ""))
            if ("anthropic" in parsed or anthropic_val)
            else None
        ),
        tavily=(
            SecretStr(parsed.get("tavily", tavily_val or ""))
            if ("tavily" in parsed or tavily_val)
            else None
        ),
    )


async def run_research(
    query: str, api_key: tuple[str, ...], mode: str, server_url: str, verbose: bool
) -> None:
    try:
        await CLIBootstrap.initialize(verbose=verbose)
        api_keys = make_api_keys_from_env_and_cli(api_key)
        console.print(
            Panel(
                f"[bold cyan]Research Query:[/bold cyan] {query}\n"
                + f"[bold cyan]Mode:[/bold cyan] {mode.upper()}"
                + (f"\n[bold cyan]Server:[/bold cyan] {server_url}" if mode == "http" else ""),
                title="Deep Research System",
                border_style="cyan",
            )
        )
        if mode == "http":
            await run_http(query, server_url)
        else:
            await run_direct(query, api_keys)
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
        await CLIBootstrap.shutdown()


async def interactive_loop(mode: str, server_url: str) -> None:
    try:
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
        while True:
            try:
                query = Prompt.ask("\n[cyan]Research query[/cyan]")
                if query.lower() in ["exit", "quit", "q"]:
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                if query.lower() == "help":
                    console.print(
                        """
[bold]Available commands:[/bold]
  • Enter any research query to start research
  • 'help' - Show this help message
  • 'clear' - Clear the screen
  • 'exit', 'quit', 'q' - Exit the program
                        """
                    )
                elif query.lower() == "clear":
                    console.clear()
                else:
                    await run_research(query, (), mode=mode, server_url=server_url, verbose=False)
            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' to quit[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")
    except BootstrapError as e:
        console.print(f"[red]Failed to initialize CLI: {e}[/red]")
        sys.exit(1)
    finally:
        await CLIBootstrap.shutdown()
