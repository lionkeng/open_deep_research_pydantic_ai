"""Complete clarification flow with review step integration."""

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from models.clarification import (
    ClarificationAnswer,
    ClarificationQuestion,
    ClarificationRequest,
    ClarificationResponse,
)

from .cli_multi_clarification import handle_multi_clarification_cli
from .review_interface import handle_review_interface


async def handle_clarification_with_review(
    request: ClarificationRequest,
    original_query: str,
    console: Console | None = None,
    auto_review: bool = True,
    allow_skip_review: bool = True,
) -> ClarificationResponse | None:
    """Handle the complete clarification flow including review step.

    This function orchestrates the entire clarification process:
    1. Initial question answering
    2. Review interface (optional)
    3. Final confirmation

    Args:
        request: ClarificationRequest with questions
        original_query: Original user query for context
        console: Optional Rich console instance
        auto_review: Whether to automatically show review after initial answers
        allow_skip_review: Whether users can skip the review step

    Returns:
        Final ClarificationResponse with confirmed answers or None if cancelled
    """
    if console is None:
        console = Console()

    # Step 1: Initial clarification questions
    console.print(
        Panel(
            (
                "[bold cyan]Clarification Needed[/bold cyan]\n\n"
                "To provide the most relevant research results, I need to ask you "
                "a few questions about your requirements."
            ),
            title="Research Assistant",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    initial_response = await handle_multi_clarification_cli(
        request=request,
        original_query=original_query,
        console=console,
    )

    if initial_response is None:
        return None

    # Step 2: Review step (optional based on configuration)
    if auto_review:
        console.print()

        # Check if review is needed
        validation_errors = initial_response.validate_against_request(request)
        has_skipped_optional = any(
            answer.skipped
            for answer in initial_response.answers
            if (
                (question := request.get_question_by_id(answer.question_id))
                and not question.is_required
            )
        )

        # Determine if review should be offered
        should_offer_review = validation_errors or has_skipped_optional or auto_review

        if should_offer_review:
            review_message: list[str] = []

            if validation_errors:
                review_message.append("[yellow]Some required questions need attention.[/yellow]")
            elif has_skipped_optional:
                review_message.append(
                    "You've skipped some optional questions that might improve results."
                )
            else:
                review_message.append("Would you like to review and potentially edit your answers?")

            console.print(
                Panel(
                    "\n".join(review_message),
                    title="Review Your Answers",
                    border_style="yellow" if validation_errors else "cyan",
                    padding=(1, 2),
                )
            )

            # Ask if they want to review
            should_review = True
            if allow_skip_review and not validation_errors:
                try:
                    should_review = Confirm.ask(
                        "Would you like to review your answers before proceeding?", default=True
                    )
                except (KeyboardInterrupt, EOFError):
                    # Re-raise to terminate the workflow
                    raise KeyboardInterrupt("User cancelled during review confirmation") from None

            if should_review:
                reviewed_response = await handle_review_interface(
                    request=request,
                    response=initial_response,
                    original_query=original_query,
                    console=console,
                )

                if reviewed_response:
                    initial_response = reviewed_response
                else:
                    # User cancelled review, ask if they want to proceed with initial answers
                    try:
                        if Confirm.ask("Review cancelled. Use your initial answers?", default=True):
                            pass  # Keep initial_response
                        else:
                            return None
                    except (KeyboardInterrupt, EOFError):
                        # Re-raise to terminate the workflow
                        raise KeyboardInterrupt(
                            "User cancelled during final confirmation"
                        ) from None

    # Step 3: Final confirmation and summary
    console.print()
    display_final_summary(request, initial_response, console)

    return initial_response


def display_final_summary(
    request: ClarificationRequest, response: ClarificationResponse, console: Console
) -> None:
    """Display a final summary of the clarification process.

    Args:
        request: The clarification request
        response: The final response
        console: Console instance
    """
    from rich.table import Table

    # Create summary table
    summary = Table(
        title="Clarification Summary",
        show_header=True,
        header_style="bold cyan",
        border_style="green",
    )
    summary.add_column("Question", ratio=2)
    summary.add_column("Your Answer", ratio=3)
    summary.add_column("Type", width=10)

    def _format_answer(q: ClarificationQuestion, a: ClarificationAnswer | None) -> str:
        if a is None or a.skipped:
            return "[dim italic]Skipped[/dim italic]"
        if q.question_type == "text":
            txt = a.text or "—"
            return txt if len(txt) <= 50 else txt[:47] + "..."

        # Lookup helper
        def label(cid: str) -> str:
            for ch in q.choices or []:
                if ch.id == cid:
                    return ch.label
            return cid

        if q.question_type == "choice":
            if a.selection:
                lbl = label(a.selection.id)
                return f"{lbl}: {a.selection.details}" if a.selection.details else lbl
            return "[dim italic]Skipped[/dim italic]"
        if q.question_type == "multi_choice":
            items: list[str] = []
            for sel in a.selections or []:
                lbl = label(sel.id)
                items.append(f"{lbl}: {sel.details}" if sel.details else lbl)
            text = ", ".join(items) if items else "[dim italic]Skipped[/dim italic]"
            return text if len(text) <= 50 else text[:47] + "..."
        return ""

    for question in request.get_sorted_questions():
        answer = response.get_answer_for_question(question.id)
        answer_text = _format_answer(question, answer)

        # Determine type indicator
        type_text = "Required" if question.is_required else "Optional"
        type_style = "yellow" if question.is_required else "dim"

        summary.add_row(
            question.question[:40] + "..." if len(question.question) > 40 else question.question,
            answer_text,
            f"[{type_style}]{type_text}[/{type_style}]",
        )

    console.print(summary)
    console.print()

    # Statistics
    total_questions = len(request.questions)
    answered = sum(
        1
        for q in request.questions
        if ((answer := response.get_answer_for_question(q.id)) and not answer.skipped)
    )

    stats_panel = Panel(
        (
            f"[green]✓[/green] Answered: {answered}/{total_questions} questions\n"
            f"[blue]✎[/blue] Ready to proceed with enhanced research"
        ),
        title="Ready to Research",
        border_style="green",
        padding=(1, 2),
    )
    console.print(stats_panel)
