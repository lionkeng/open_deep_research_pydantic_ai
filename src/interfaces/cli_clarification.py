"""Simple CLI interface for single question clarification."""

import sys

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text


def ask_single_clarification_question(question: str, console: Console | None = None) -> str | None:
    """Ask a single clarification question and get response.

    Args:
        question: The clarification question to ask
        console: Optional Rich console instance

    Returns:
        User's response string, or None if cancelled
    """
    if console is None:
        console = Console()

    # Check if we're in an interactive terminal
    if not sys.stdin.isatty():
        console.print(
            "[warning]Non-interactive environment detected, skipping clarification[/warning]"
        )
        return None

    try:
        # Display the question in a nice panel
        question_panel = Panel(
            Text(question, style="bold white"),
            title="🤔 Clarification Needed",
            border_style="yellow",
            padding=(1, 2),
        )

        console.print()
        console.print(question_panel)
        console.print()

        # Get user response
        response = Prompt.ask("Your response").strip()

        if not response:
            return None

        return response

    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Clarification cancelled by user[/yellow]")
        return None
    except Exception as e:
        console.print(f"[red]Error during clarification: {e}[/red]")
        return None


def display_clarification_complete(response: str, console: Console | None = None) -> None:
    """Display that clarification is complete.

    Args:
        response: User's response
        console: Optional Rich console instance
    """
    if console is None:
        console = Console()

    completion_panel = Panel(
        f"✅ Thank you for the clarification!\n\n[italic]Your response:[/italic] {response}",
        title="Clarification Complete",
        border_style="green",
        padding=(1, 2),
    )

    console.print()
    console.print(completion_panel)
    console.print("[dim]Proceeding with research...[/dim]")
    console.print()


# Maintain backward compatibility with old function signature
def run_cli_clarification(
    question: str, original_query: str, console: Console | None = None
) -> tuple[bool, str | None]:
    """Run a simple CLI clarification process.

    Args:
        question: The clarification question to ask
        original_query: Original user query (for context)
        console: Optional Rich console instance

    Returns:
        Tuple of (completed, user_response)
    """
    if console is None:
        console = Console()

    # Show context
    context_panel = Panel(
        f"[dim]Original query:[/dim] {original_query}",
        title="Research Context",
        border_style="dim",
        padding=(0, 2),
    )
    console.print(context_panel)

    # Ask the question
    response = ask_single_clarification_question(question, console)

    if response:
        display_clarification_complete(response, console)
        return True, response
    else:
        console.print("[yellow]Proceeding with original query...[/yellow]")
        return False, None
