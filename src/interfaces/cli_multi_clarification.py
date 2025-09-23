"""Multi-question CLI interface for clarification system."""

import sys

import logfire
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from models.clarification import (
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


def _is_other_option(choice: str) -> bool:
    choice_lower = choice.lower()
    return any(
        phrase in choice_lower
        for phrase in [
            "other (please specify",
            "other (specify",
            "other - please specify",
            "other (please describe",
            "other:",
            "something else",
            "none of the above (please specify",
        ]
    )


def _requires_specify_option(choice: str) -> bool:
    choice_lower = choice.lower()
    if "(" in choice_lower and ")" in choice_lower:
        try:
            hint = choice_lower[choice_lower.index("(") + 1 : choice_lower.rindex(")")]
        except ValueError:
            hint = choice_lower
    else:
        hint = choice_lower
    keywords = [
        "please specify",
        "specify",
        "describe",
        "enter",
        "provide",
        "input",
        "details",
        "detail",
        "name",
    ]
    return any(k in hint for k in keywords)


def _base_label(choice: str) -> str:
    label = choice.strip()
    if "(" in label and ")" in label:
        try:
            start = label.index("(")
            end = label.rindex(")")
            hint = label[start:end].lower()
            if any(
                k in hint
                for k in [
                    "please specify",
                    "specify",
                    "describe",
                    "enter",
                    "provide",
                    "input",
                    "details",
                    "detail",
                    "name",
                ]
            ):
                return (label[:start] + label[end + 1 :]).strip()
        except ValueError:
            return label
    return label


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

    # Use interactive selector if available
    if has_interactive and interactive_select:
        return interactive_select(
            question=question.question,
            choices=question.choices,
            multiple=False,
            allow_skip=not question.is_required,
            context=question.context,
            console=console,
        )

    # Fallback to original implementation
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

    # Handle follow-up input for 'Other' and '(please specify)' options
    if _is_other_option(selected):
        while True:
            try:
                details = Prompt.ask("Please specify")
            except (KeyboardInterrupt, EOFError):
                raise KeyboardInterrupt("User cancelled during specify input") from None
            details = details.strip()
            if details:
                return f"Other: {details}"
            if question.is_required:
                console.print("[red]This question requires a specific answer[/red]")
                continue
            return selected  # Optional with no details

    if _requires_specify_option(selected):
        base = _base_label(selected)
        while True:
            try:
                details = Prompt.ask("Please specify details")
            except (KeyboardInterrupt, EOFError):
                raise KeyboardInterrupt("User cancelled during specify input") from None
            details = details.strip()
            if details:
                return f"{base}: {details}"
            if question.is_required:
                console.print("[red]Details are required for this selection[/red]")
                continue
            return selected

    return selected


def ask_multi_choice_question(question: ClarificationQuestion, console: Console) -> str | None:
    """Ask a multi-choice clarification question.

    Args:
        question: The question with choices
        console: Console instance

    Returns:
        Pipe-separated selected choices or None if skipped
    """
    if not question.choices:
        return ask_text_question(question, console)  # Fallback

    # Use interactive selector if available
    if has_interactive and interactive_select:
        selected = interactive_select(
            question=question.question,
            choices=question.choices,
            multiple=True,
            allow_skip=not question.is_required,
            context=question.context,
            console=console,
        )
        if selected:
            # Use pipe separator to avoid conflicts with commas in choice text
            return " | ".join(selected)
        return None

    # Fallback to original implementation
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
    try:
        response = Prompt.ask("Select options (e.g., 1,3,4)")
    except (KeyboardInterrupt, EOFError):
        # Re-raise to terminate the workflow
        raise KeyboardInterrupt("User cancelled during multi-choice selection") from None

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

        processed: list[str] = []
        for s in selections:
            choice = question.choices[s - 1]
            if _is_other_option(choice):
                while True:
                    try:
                        details = Prompt.ask(f"Specify for '{choice}'")
                    except (KeyboardInterrupt, EOFError):
                        raise KeyboardInterrupt("User cancelled during specify input") from None
                    details = details.strip()
                    if details:
                        processed.append(f"Other: {details}")
                        break
                    if question.is_required:
                        console.print("[red]Details are required for this selection[/red]")
                        continue
                    processed.append(choice)
                    break
            elif _requires_specify_option(choice):
                base = _base_label(choice)
                while True:
                    try:
                        details = Prompt.ask(f"Specify for '{base}'")
                    except (KeyboardInterrupt, EOFError):
                        raise KeyboardInterrupt("User cancelled during specify input") from None
                    details = details.strip()
                    if details:
                        processed.append(f"{base}: {details}")
                        break
                    if question.is_required:
                        console.print("[red]Details are required for this selection[/red]")
                        continue
                    processed.append(choice)
                    break
            else:
                processed.append(choice)

        # Use pipe separator to avoid conflicts with commas in choice text
        return " | ".join(processed)
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

            # Telemetry: log answer outcome (no content for privacy)
            try:
                logfire.info(
                    "Clarification answer captured",
                    question_id=question.id,
                    skipped=answer.skipped,
                    answer_len=(len(answer.answer) if answer.answer else 0),
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
