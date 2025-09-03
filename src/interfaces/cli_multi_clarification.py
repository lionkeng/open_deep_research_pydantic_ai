"""Multi-question CLI interface for clarification system."""

import sys

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from src.models.clarification import (
    ClarificationAnswer,
    ClarificationQuestion,
    ClarificationRequest,
    ClarificationResponse,
)


def display_clarification_request(
    request: ClarificationRequest, original_query: str, console: Console | None = None
) -> None:
    """Display clarification request with all questions.

    Args:
        request: ClarificationRequest containing questions
        original_query: Original user query for context
        console: Optional Rich console instance
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

    # Display questions in a table
    table = Table(title="Clarification Questions", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Question", style="white")
    table.add_column("Type", style="cyan", width=12)
    table.add_column("Required", style="yellow", width=10)

    sorted_questions = request.get_sorted_questions()
    for idx, question in enumerate(sorted_questions, 1):
        table.add_row(
            str(idx),
            question.question,
            question.question_type,
            "Yes" if question.is_required else "No",
        )

    console.print()
    console.print(table)
    console.print()


def ask_text_question(question: ClarificationQuestion, console: Console) -> str | None:
    """Ask a text-based clarification question.

    Args:
        question: The question to ask
        console: Console instance

    Returns:
        User's response or None if skipped
    """
    if question.context:
        console.print(f"[dim]Context: {question.context}[/dim]")

    response = Prompt.ask(
        question.question,
        default="[skip]" if not question.is_required else ...,  # type: ignore
    )

    if response == "[skip]" and not question.is_required:
        return None
    return response


def ask_choice_question(question: ClarificationQuestion, console: Console) -> str | None:
    """Ask a single-choice clarification question.

    Args:
        question: The question with choices
        console: Console instance

    Returns:
        Selected choice or None if skipped
    """
    if not question.choices:
        return ask_text_question(question, console)  # Fallback

    console.print(f"\n[bold]{question.question}[/bold]")
    if question.context:
        console.print(f"[dim]Context: {question.context}[/dim]")

    # Display choices
    for idx, choice in enumerate(question.choices, 1):
        console.print(f"  {idx}. {choice}")

    if not question.is_required:
        console.print("  0. [Skip this question]")

    # Get selection
    max_choice = len(question.choices)
    min_choice = 0 if not question.is_required else 1

    choice_num = IntPrompt.ask(
        "Select option",
        choices=[str(i) for i in range(min_choice, max_choice + 1)],
        default=str(min_choice) if not question.is_required else None,
    )

    if choice_num == 0:
        return None
    return question.choices[choice_num - 1]


def ask_multi_choice_question(question: ClarificationQuestion, console: Console) -> str | None:
    """Ask a multi-choice clarification question.

    Args:
        question: The question with choices
        console: Console instance

    Returns:
        Comma-separated selected choices or None if skipped
    """
    if not question.choices:
        return ask_text_question(question, console)  # Fallback

    console.print(f"\n[bold]{question.question}[/bold]")
    console.print("[dim]Select multiple options (comma-separated numbers)[/dim]")
    if question.context:
        console.print(f"[dim]Context: {question.context}[/dim]")

    # Display choices
    for idx, choice in enumerate(question.choices, 1):
        console.print(f"  {idx}. {choice}")

    if not question.is_required:
        console.print("  0. [Skip this question]")

    # Get selections
    response = Prompt.ask("Select options (e.g., 1,3,4)")

    if response == "0" and not question.is_required:
        return None

    # Parse selections with proper input sanitization
    try:
        selections = []
        for x in response.split(","):
            x = x.strip()
            if x:  # Skip empty strings from malformed input like "1,,3"
                try:
                    selections.append(int(x))
                except ValueError:
                    console.print(f"[red]Invalid input '{x}' - please use numbers only[/red]")
                    return ask_multi_choice_question(question, console)  # Retry

        if not selections:
            console.print("[red]No selections made - please select at least one option[/red]")
            return ask_multi_choice_question(question, console)  # Retry

        if any(s < 1 or s > len(question.choices) for s in selections):
            console.print("[red]Invalid selection number(s) - out of range[/red]")
            return ask_multi_choice_question(question, console)  # Retry

        selected_choices = [question.choices[s - 1] for s in selections]
        return ", ".join(selected_choices)
    except ValueError:
        console.print("[red]Invalid input format[/red]")
        return ask_multi_choice_question(question, console)  # Retry


async def handle_multi_clarification_cli(
    request: ClarificationRequest,
    original_query: str,
    console: Console | None = None,
) -> ClarificationResponse | None:
    """Handle multi-question clarification via CLI.

    Args:
        request: ClarificationRequest with questions
        original_query: Original user query for context
        console: Optional Rich console instance

    Returns:
        ClarificationResponse with answers or None if cancelled
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
        # Display all questions first
        display_clarification_request(request, original_query, console)

        # Collect answers
        answers: list[ClarificationAnswer] = []
        sorted_questions = request.get_sorted_questions()

        console.print(
            Panel(
                "[bold]Please answer the following questions to help improve the research:[/bold]",
                border_style="green",
                padding=(1, 2),
            )
        )

        for idx, question in enumerate(sorted_questions, 1):
            console.print(f"\n[bold cyan]Question {idx} of {len(sorted_questions)}[/bold cyan]")

            # Ask based on question type
            answer_text: str | None = None
            if question.question_type == "choice":
                answer_text = ask_choice_question(question, console)
            elif question.question_type == "multi_choice":
                answer_text = ask_multi_choice_question(question, console)
            else:  # text
                answer_text = ask_text_question(question, console)

            # Create answer object
            if answer_text is not None:
                answer = ClarificationAnswer(
                    question_id=question.id,
                    answer=answer_text,
                    skipped=False,
                )
            else:
                answer = ClarificationAnswer(
                    question_id=question.id,
                    skipped=True,
                )
            answers.append(answer)

        # Create response
        response = ClarificationResponse(
            request_id=request.id,
            answers=answers,
        )

        # Validate response
        errors = response.validate_against_request(request)
        if errors:
            console.print("\n[red]Validation errors:[/red]")
            for error in errors:
                console.print(f"  - {error}")

            # Ask if they want to retry
            if Confirm.ask("Would you like to answer the missing required questions?"):
                return await handle_multi_clarification_cli(request, original_query, console)
            return None

        # Display completion
        console.print(
            Panel(
                "✅ Clarification complete! Thank you for your responses.",
                title="Success",
                border_style="green",
                padding=(1, 2),
            )
        )
        console.print("[dim]Proceeding with enhanced research...[/dim]\n")

        return response

    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Clarification cancelled by user[/yellow]")
        return None
    except Exception as e:
        console.print(f"[red]Error during clarification: {e}[/red]")
        return None
