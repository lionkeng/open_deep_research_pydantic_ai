"""Multi-question CLI interface for clarification system."""

import sys

import logfire
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from models.clarification import (
    ChoiceSelection,
    ClarificationAnswer,
    ClarificationQuestion,
    ClarificationRequest,
    ClarificationResponse,
)

# Import interactive selector for better UX
try:
    from .interactive_selector import interactive_select

    has_interactive = True
except ImportError:
    interactive_select = None
    has_interactive = False


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
    """Ask a text-based clarification question with improved UX.

    Args:
        question: The question to ask
        console: Console instance

    Returns:
        User's response or None if skipped
    """
    from rich.panel import Panel
    from rich.table import Table

    # Create a table for better layout
    table = Table(show_header=False, show_edge=False, box=None, padding=(0, 1))
    table.add_column(justify="left")

    # Add question with emphasis
    table.add_row(f"[bold white]{question.question}[/bold white]")
    table.add_row("")  # Spacing

    # Add context if available
    if question.context:
        table.add_row(f"[dim italic]💭 Hint: {question.context}[/dim italic]")
        table.add_row("")  # Spacing

    # Add input indicator box
    input_content = (
        "[bold cyan]✍️  Type your answer here[/bold cyan]\n"
        "[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]"
    )

    if question.is_required:
        subtitle_text = "[dim]Press Enter to submit[/dim]"
    else:
        subtitle_text = "[dim]Press Enter to submit • Type 'skip' to skip[/dim]"

    input_box = Panel(
        input_content,
        border_style="cyan",
        title="[bold]Text Input[/bold]",
        title_align="left",
        subtitle=subtitle_text,
        subtitle_align="right",
        padding=(0, 2),
    )
    table.add_row(input_box)

    # Add helper text
    if question.is_required:
        table.add_row("[yellow]⚠️  This question is required[/yellow]")
    else:
        table.add_row("[dim green]✓ Optional question • You can type 'skip' to skip[/dim green]")

    # Display the formatted question
    console.print(table)
    console.print()  # Add spacing before prompt

    # Create a more visible prompt with arrow indicator
    prompt_text = "[bold cyan]➤[/bold cyan] Your answer"

    # Handle the actual input
    try:
        if not question.is_required:
            response = Prompt.ask(
                prompt_text,
                default="",
                show_default=False,
            )

            # Check for skip variations
            if response.lower() in ["skip", "[skip]", ""] or response.strip() == "":
                console.print("[dim]↳ Skipping this question...[/dim]")
                return None
        else:
            # For required questions, keep asking until we get a non-empty response
            while True:
                response = Prompt.ask(prompt_text, default=...)

                # Response will be a string when a default of ... is used
                if isinstance(response, str) and response.strip():
                    break
                error_msg = (
                    "[red]⚠️  This question requires an answer. Please provide a response.[/red]"
                )
                console.print(error_msg)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Question cancelled by user[/yellow]")
        # Re-raise to terminate the workflow
        raise KeyboardInterrupt("User cancelled during text input") from None

    # Confirm receipt of answer
    console.print("[dim green]✓ Answer recorded[/dim green]")
    return response


def ask_choice_question(
    question: ClarificationQuestion, console: Console
) -> ChoiceSelection | None:
    """Ask a single-choice clarification question.

    Args:
        question: The question with choices
        console: Console instance

    Returns:
        Selected choice or None if skipped
    """
    if not question.choices:
        # Should not happen for choice type; treat as skipped
        return None

    # Use interactive selector if available
    if has_interactive and interactive_select:
        selected_label = interactive_select(
            question=question.question,
            choices=[c.label for c in question.choices],
            multiple=False,
            allow_skip=not question.is_required,
            context=question.context,
            console=console,
        )
        if selected_label is None:
            return None
        # Lookup choice by label
        selected = next((c for c in question.choices if c.label == selected_label), None)
        if selected is None:
            return None

        # Handle details when required or 'Other'
        if selected.is_other or selected.requires_details:
            while True:
                try:
                    details_prompt = selected.details_prompt or "Please provide details"
                    details = Prompt.ask(details_prompt)
                except (KeyboardInterrupt, EOFError):
                    raise KeyboardInterrupt("User cancelled during specify input") from None
                details = details.strip()
                if details:
                    return ChoiceSelection(id=selected.id, details=details)
                if question.is_required:
                    console.print("[red]Details are required for this selection[/red]")
                    continue
                # Optional, allow returning without details
                return ChoiceSelection(id=selected.id)

        return ChoiceSelection(id=selected.id)

    # Fallback to original numeric implementation
    console.print(f"\n[bold]{question.question}[/bold]")
    if question.context:
        console.print(f"[dim]Context: {question.context}[/dim]")

    # Display choices
    for idx, choice in enumerate(question.choices, 1):
        console.print(f"  {idx}. {choice.label}")

    if not question.is_required:
        console.print("  0. [Skip this question]")

    # Get selection
    max_choice = len(question.choices)
    min_choice = 0 if not question.is_required else 1

    try:
        choice_num = IntPrompt.ask(
            "Select option",
            choices=[str(i) for i in range(min_choice, max_choice + 1)],
            default=str(min_choice) if not question.is_required else None,
        )
    except (KeyboardInterrupt, EOFError):
        # Re-raise to terminate the workflow
        raise KeyboardInterrupt("User cancelled during choice selection") from None

    if choice_num == 0:
        return None
    selected = question.choices[int(choice_num) - 1]

    # Handle required details
    if selected.is_other or selected.requires_details:
        while True:
            try:
                details_prompt = selected.details_prompt or "Please provide details"
                details = Prompt.ask(details_prompt)
            except (KeyboardInterrupt, EOFError):
                raise KeyboardInterrupt("User cancelled during specify input") from None
            details = details.strip()
            if details:
                return ChoiceSelection(id=selected.id, details=details)
            if question.is_required:
                console.print("[red]Details are required for this selection[/red]")
                continue
            return ChoiceSelection(id=selected.id)

    return ChoiceSelection(id=selected.id)


def ask_multi_choice_question(
    question: ClarificationQuestion, console: Console
) -> list[ChoiceSelection] | None:
    """Ask a multi-choice clarification question.

    Args:
        question: The question with choices
        console: Console instance

    Returns:
        Pipe-separated selected choices or None if skipped
    """
    if not question.choices:
        # Should not happen for multi_choice type; treat as skipped
        return None

    # Use interactive selector if available
    if has_interactive and interactive_select:
        selected_labels = interactive_select(
            question=question.question,
            choices=[c.label for c in question.choices],
            multiple=True,
            allow_skip=not question.is_required,
            context=question.context,
            console=console,
        )
        if not selected_labels:
            return None
        selections: list[ChoiceSelection] = []
        for label in selected_labels:
            ch = next((c for c in question.choices if c.label == label), None)
            if ch is None:
                continue
            if ch.is_other or ch.requires_details:
                while True:
                    try:
                        prompt = ch.details_prompt or f"Specify for '{ch.label}'"
                        details = Prompt.ask(prompt)
                    except (KeyboardInterrupt, EOFError):
                        raise KeyboardInterrupt("User cancelled during specify input") from None
                    details = details.strip()
                    if details:
                        selections.append(ChoiceSelection(id=ch.id, details=details))
                        break
                    if question.is_required:
                        console.print("[red]Details are required for this selection[/red]")
                        continue
                    selections.append(ChoiceSelection(id=ch.id))
                    break
            else:
                selections.append(ChoiceSelection(id=ch.id))
        return selections if selections else None

    # Fallback to original implementation
    console.print(f"\n[bold]{question.question}[/bold]")
    console.print("[dim]Select multiple options (comma-separated numbers)[/dim]")
    if question.context:
        console.print(f"[dim]Context: {question.context}[/dim]")

    # Display choices
    for idx, choice in enumerate(question.choices, 1):
        console.print(f"  {idx}. {choice.label}")

    if not question.is_required:
        console.print("  0. [Skip this question]")

    # Get selections
    try:
        response = Prompt.ask("Select options (e.g., 1,3,4)")
    except (KeyboardInterrupt, EOFError):
        # Re-raise to terminate the workflow
        raise KeyboardInterrupt("User cancelled during multi-choice selection") from None

    if response == "0" and not question.is_required:
        return None

    # Parse selections with proper input sanitization
    try:
        index_selections: list[int] = []
        for x in response.split(","):
            x = x.strip()
            if x:  # Skip empty strings from malformed input like "1,,3"
                try:
                    index_selections.append(int(x))
                except ValueError:
                    console.print(f"[red]Invalid input '{x}' - please use numbers only[/red]")
                    return ask_multi_choice_question(question, console)  # Retry

        if not index_selections:
            console.print("[red]No selections made - please select at least one option[/red]")
            return ask_multi_choice_question(question, console)  # Retry

        if any(s < 1 or s > len(question.choices) for s in index_selections):
            console.print("[red]Invalid selection number(s) - out of range[/red]")
            return ask_multi_choice_question(question, console)  # Retry

        selections: list[ChoiceSelection] = []
        for s in index_selections:
            choice = question.choices[s - 1]
            if choice.is_other or choice.requires_details:
                while True:
                    try:
                        prompt = choice.details_prompt or f"Specify for '{choice.label}'"
                        details = Prompt.ask(prompt)
                    except (KeyboardInterrupt, EOFError):
                        raise KeyboardInterrupt("User cancelled during specify input") from None
                    details = details.strip()
                    if details:
                        selections.append(ChoiceSelection(id=choice.id, details=details))
                        break
                    if question.is_required:
                        console.print("[red]Details are required for this selection[/red]")
                        continue
                    selections.append(ChoiceSelection(id=choice.id))
                    break
            else:
                selections.append(ChoiceSelection(id=choice.id))

        return selections
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
            # Telemetry: log which question we are about to ask
            try:
                logfire.info(
                    "Asking clarification question",
                    question_id=question.id,
                    order=question.order,
                    required=question.is_required,
                    type=question.question_type,
                    preview=(question.question[:120] if isinstance(question.question, str) else ""),
                    index=idx,
                    total=len(sorted_questions),
                )
            except Exception:
                pass
            # Display question progress with enhanced formatting
            progress_bar = f"[{'█' * idx}{'░' * (len(sorted_questions) - idx)}]"
            if question.is_required:
                required_tag = "[red bold]REQUIRED[/red bold]"
            else:
                required_tag = "[green]Optional[/green]"

            progress_text = (
                f"[bold cyan]Question {idx} of {len(sorted_questions)}[/bold cyan]  "
                f"{progress_bar}  {required_tag}"
            )

            console.print(
                Panel(
                    progress_text,
                    border_style="blue",
                    padding=(0, 1),
                )
            )
            console.print()  # Add spacing

            # Ask based on question type and build structured answer
            if question.question_type == "choice":
                selection = ask_choice_question(question, console)
                if selection is not None:
                    answer = ClarificationAnswer(
                        question_id=question.id,
                        selection=selection,
                        skipped=False,
                    )
                else:
                    answer = ClarificationAnswer(question_id=question.id, skipped=True)
            elif question.question_type == "multi_choice":
                selections = ask_multi_choice_question(question, console)
                if selections is not None:
                    answer = ClarificationAnswer(
                        question_id=question.id,
                        selections=selections,
                        skipped=False,
                    )
                else:
                    answer = ClarificationAnswer(question_id=question.id, skipped=True)
            else:  # text
                text = ask_text_question(question, console)
                if text is not None:
                    answer = ClarificationAnswer(
                        question_id=question.id,
                        text=text,
                        skipped=False,
                    )
                else:
                    answer = ClarificationAnswer(question_id=question.id, skipped=True)
            answers.append(answer)

            # Telemetry: log answer outcome (no content for privacy)
            try:
                logfire.info(
                    "Clarification answer captured",
                    question_id=question.id,
                    skipped=answer.skipped,
                    answer_len=(
                        (len(answer.text) if answer.text else 0)
                        if question.question_type == "text"
                        else (
                            len(answer.selections or [])
                            if question.question_type == "multi_choice"
                            else 1
                        )
                    ),
                )
            except Exception:
                pass

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
            try:
                if Confirm.ask("Would you like to answer the missing required questions?"):
                    return await handle_multi_clarification_cli(request, original_query, console)
                return None
            except (KeyboardInterrupt, EOFError):
                # Re-raise to terminate the workflow
                raise KeyboardInterrupt("User cancelled during validation retry") from None

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

    except KeyboardInterrupt:
        console.print("\n[yellow]Clarification interrupted by user[/yellow]")
        raise  # Re-raise to propagate the interrupt
    except EOFError:
        console.print("\n[yellow]Clarification cancelled (EOF)[/yellow]")
        return None
    except Exception as e:
        console.print(f"[red]Error during clarification: {e}[/red]")
        return None
