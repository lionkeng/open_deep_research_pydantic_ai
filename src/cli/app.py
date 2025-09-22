"""Click CLI entry points for deep research."""

from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console
from rich.table import Table

from .runner import interactive_loop, run_research


@click.group()
def cli() -> None:
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
def research(
    query: str,
    api_key: tuple[str, ...],
    verbose: bool,
    mode: str,
    server_url: str,
) -> None:
    """Execute a research query."""

    asyncio.run(run_research(query, api_key, mode=mode, server_url=server_url, verbose=verbose))


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
def interactive(mode: str, server_url: str) -> None:
    """Start an interactive research session."""

    asyncio.run(interactive_loop(mode, server_url))


@cli.command()
def version() -> None:
    """Show version information."""
    console = Console(force_terminal=True)
    table = Table(title="Deep Research System", border_style="cyan")
    table.add_column("Component", style="cyan")
    table.add_column("Version", style="green")
    table.add_row("Deep Research", "1.0.0")
    table.add_row("Pydantic AI", "Latest")
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    table.add_row("Python", pyver)
    console.print(table)
