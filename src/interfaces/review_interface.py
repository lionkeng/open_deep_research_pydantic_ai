"""Review interface for clarification answers with navigation and editing capabilities."""

from dataclasses import dataclass
from enum import Enum

from rich.align import Align
from rich.box import ROUNDED
from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text

from models.clarification import (
    ClarificationAnswer,
    ClarificationQuestion,
    ClarificationRequest,
    ClarificationResponse,
)

from .cli_multi_clarification import (
    ask_choice_question,
    ask_multi_choice_question,
    ask_text_question,
)
from .interactive_selector import InteractiveSelector


class AnswerStatus(Enum):
    """Status of an answer in the review."""

    ANSWERED_REQUIRED = "answered_required"
    ANSWERED_OPTIONAL = "answered_optional"
    SKIPPED_OPTIONAL = "skipped_optional"
    UNANSWERED_REQUIRED = "unanswered_required"
    EDITED = "edited"


@dataclass
class ReviewState:
    """Current state of the review interface."""

    current_index: int = 0
    edited_questions: set[str] | None = None
    confirmed: bool = False
    recursion_depth: int = 0  # Track recursion to prevent infinite loops

    def __post_init__(self):
        if self.edited_questions is None:
            self.edited_questions = set()


class ReviewInterface:
    """Interactive review interface for clarification answers."""

    # Visual configuration
    STATUS_SYMBOLS = {
        AnswerStatus.ANSWERED_REQUIRED: ("✓", "green"),
        AnswerStatus.ANSWERED_OPTIONAL: ("○", "cyan"),
        AnswerStatus.SKIPPED_OPTIONAL: ("─", "dim white"),
        AnswerStatus.UNANSWERED_REQUIRED: ("⚠", "yellow"),
        AnswerStatus.EDITED: ("✎", "blue"),
    }

    BORDER_STYLES = {
        "required": "yellow",
        "optional": "dim white",
        "current": "bold green",
        "edited": "blue",
    }

    def __init__(self, console: Console | None = None):
        """Initialize the review interface.

        Args:
            console: Optional Rich console instance
        """
        self.console = console or Console()
        self.state = ReviewState()

    def get_answer_status(
        self,
        question: ClarificationQuestion,
        answer: ClarificationAnswer | None,
        is_edited: bool = False,
    ) -> AnswerStatus:
        """Determine the status of an answer.

        Args:
            question: The clarification question
            answer: The answer (if any)
            is_edited: Whether the answer has been edited

        Returns:
            AnswerStatus indicating the current state
        """
        if is_edited:
            return AnswerStatus.EDITED

        if answer is None or answer.skipped:
            if question.is_required:
                return AnswerStatus.UNANSWERED_REQUIRED
            return AnswerStatus.SKIPPED_OPTIONAL

        if question.is_required:
            return AnswerStatus.ANSWERED_REQUIRED
        return AnswerStatus.ANSWERED_OPTIONAL

    def format_answer_display(
        self,
        answer: ClarificationAnswer | None,
        question: ClarificationQuestion,
        max_length: int = 50,
    ) -> str:
        """Format an answer for display.

        Args:
            answer: The answer to format
            question: The corresponding question
            max_length: Maximum display length for text answers

        Returns:
            Formatted answer string
        """
        if answer is None or answer.skipped:
            return "[dim italic]Not answered[/dim italic]"

        answer_text = answer.answer or ""

        # Handle different question types
        if question.question_type == "multi_choice":
            # Format multi-choice as bullet list if long
            choices = [c.strip() for c in answer_text.split(",")]
            if len(choices) > 3:
                return f"{', '.join(choices[:3])}, ... (+{len(choices) - 3} more)"
            return answer_text

        # Truncate long text answers
        if len(answer_text) > max_length:
            return f"{answer_text[:max_length]}..."

        return answer_text

    def render_progress_bar(
        self, request: ClarificationRequest, response: ClarificationResponse
    ) -> Panel:
        """Render a progress bar showing completion status.

        Args:
            request: The clarification request
            response: Current response state

        Returns:
            Panel containing the progress bar
        """
        total = len(request.questions)
        answered = sum(
            1
            for q in request.questions
            if (answer := response.get_answer_for_question(q.id)) and not answer.skipped
        )

        progress_text = Text()
        progress_text.append("Progress: ", style="bold")

        # Create visual progress bar
        bar_width = 30
        filled = int((answered / total) * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)

        if answered == total:
            progress_text.append(bar, style="green")
        elif answered >= total * 0.8:
            progress_text.append(bar, style="yellow")
        else:
            progress_text.append(bar, style="dim white")

        progress_text.append(f" {answered}/{total} answered", style="dim")

        return Panel(
            Align.center(progress_text),
            title="REVIEW YOUR ANSWERS",
            border_style="cyan",
            padding=(0, 2),
        )

    def render_question_panel(
        self,
        question: ClarificationQuestion,
        answer: ClarificationAnswer | None,
        index: int,
        total: int,
        is_current: bool = False,
    ) -> Panel:
        """Render a single question panel.

        Args:
            question: The question to render
            answer: The corresponding answer
            index: Question index (1-based)
            total: Total number of questions
            is_current: Whether this is the currently selected question

        Returns:
            Panel containing the question and answer
        """
        # Determine status and styling
        is_edited = self.state.edited_questions and question.id in self.state.edited_questions
        status = self.get_answer_status(question, answer, is_edited)
        _, color = self.STATUS_SYMBOLS[status]

        # Build content
        content_parts = []

        # Question header
        header = Text()
        header.append(f"Q{index} ", style="bold")
        if question.is_required:
            header.append("[REQUIRED] ", style="yellow")
        else:
            header.append("[OPTIONAL] ", style="dim")
        header.append(f"• {question.question}", style="white")
        content_parts.append(header)

        # Separator
        content_parts.append(Text("─" * 50, style="dim"))

        # Question metadata
        meta = Text()
        meta.append("Type: ", style="dim")
        meta.append(question.question_type.replace("_", " ").title(), style="cyan")
        content_parts.append(meta)

        # Answer display
        answer_line = Text()
        answer_line.append("Your Answer: ", style="dim")
        answer_text = self.format_answer_display(answer, question)
        answer_line.append(answer_text, style=color if not is_edited else "blue")
        content_parts.append(answer_line)

        # Context if available
        if question.context:
            context_text = Text()
            context_text.append("Context: ", style="dim")
            context_text.append(question.context, style="dim italic")
            content_parts.append(Text())  # Empty line
            content_parts.append(context_text)

        # Navigation hints (only for current)
        if is_current:
            content_parts.append(Text())  # Empty line
            hints = Text()
            hints.append("[E]", style="bold yellow")
            hints.append(" Edit  ", style="dim")
            hints.append("[↑↓]", style="bold yellow")
            hints.append(" Navigate  ", style="dim")
            hints.append("[Tab]", style="bold yellow")
            hints.append(" Next Required", style="dim")
            content_parts.append(hints)

        # Determine border style
        if is_current:
            border_style = self.BORDER_STYLES["current"]
        elif is_edited:
            border_style = self.BORDER_STYLES["edited"]
        elif question.is_required:
            border_style = self.BORDER_STYLES["required"]
        else:
            border_style = self.BORDER_STYLES["optional"]

        return Panel(
            Group(*content_parts),
            border_style=border_style,
            box=ROUNDED,
            padding=(1, 2),
        )

    def render_navigation_panel(
        self, request: ClarificationRequest, response: ClarificationResponse
    ) -> Panel:
        """Render the quick navigation panel.

        Args:
            request: The clarification request
            response: Current response state

        Returns:
            Panel containing navigation overview
        """
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("", width=3)
        table.add_column("Question", ratio=2)
        table.add_column("Status", width=12)

        for idx, question in enumerate(request.get_sorted_questions(), 1):
            answer = response.get_answer_for_question(question.id)
            is_edited = self.state.edited_questions and question.id in self.state.edited_questions
            status = self.get_answer_status(question, answer, is_edited)
            symbol, color = self.STATUS_SYMBOLS[status]

            # Truncate question text
            q_text = (
                question.question[:30] + "..." if len(question.question) > 30 else question.question
            )

            # Highlight current question
            if idx - 1 == self.state.current_index:
                table.add_row(
                    Text("▸", style="bold green"),
                    Text(q_text, style="bold"),
                    Text(f"{symbol} {status.value.replace('_', ' ').title()}", style=color),
                )
            else:
                table.add_row(
                    Text(f"{idx}.", style="dim"),
                    Text(q_text, style="white"),
                    Text(symbol, style=color),
                )

        return Panel(
            table,
            title="Quick Navigation",
            border_style="dim",
            padding=(1, 1),
        )

    def render_summary_panel(
        self, request: ClarificationRequest, response: ClarificationResponse
    ) -> Panel:
        """Render the summary panel.

        Args:
            request: The clarification request
            response: Current response state

        Returns:
            Panel containing summary statistics
        """
        required_questions = request.get_required_questions()
        optional_questions = [q for q in request.questions if not q.is_required]

        required_answered = sum(
            1
            for q in required_questions
            if (answer := response.get_answer_for_question(q.id)) and not answer.skipped
        )

        optional_answered = sum(
            1
            for q in optional_questions
            if (answer := response.get_answer_for_question(q.id)) and not answer.skipped
        )

        content = Group(
            Text(
                f"✓ Required: {required_answered}/{len(required_questions)} answered",
                style="green" if required_answered == len(required_questions) else "yellow",
            ),
            Text(
                f"○ Optional: {optional_answered}/{len(optional_questions)} answered", style="cyan"
            ),
            Text(
                f"⚠ {len(optional_questions) - optional_answered} questions skipped",
                style="dim yellow" if optional_answered < len(optional_questions) else "dim",
            ),
            Text(
                f"✎ {len(self.state.edited_questions)} edited",
                style="blue" if self.state.edited_questions else "dim",
            ),
        )

        return Panel(
            content,
            title="SUMMARY",
            border_style="cyan",
            padding=(1, 2),
        )

    def edit_answer(
        self, question: ClarificationQuestion, current_answer: ClarificationAnswer | None
    ) -> ClarificationAnswer | None:
        """Edit an answer to a question.

        Args:
            question: The question to edit
            current_answer: The current answer (if any)

        Returns:
            New answer or None if cancelled
        """
        self.console.clear()

        # Show edit header
        edit_panel = Panel(
            f"[bold]Editing Answer for Question {self.state.current_index + 1}[/bold]\n\n"
            f"[dim]Original Question:[/dim] {question.question}\n"
            f"[dim]Current Answer:[/dim] {self.format_answer_display(current_answer, question)}",
            title="EDIT MODE",
            border_style="yellow",
            padding=(1, 2),
        )
        self.console.print(edit_panel)
        self.console.print()

        # Get new answer based on question type
        if question.question_type == "choice":
            answer_text = ask_choice_question(question, self.console)
        elif question.question_type == "multi_choice":
            answer_text = ask_multi_choice_question(question, self.console)
        else:
            answer_text = ask_text_question(question, self.console)

        # Create new answer if changed
        if answer_text is not None:
            return ClarificationAnswer(
                question_id=question.id,
                answer=answer_text,
                skipped=False,
            )
        if not question.is_required:
            return ClarificationAnswer(
                question_id=question.id,
                skipped=True,
            )

        return current_answer

    def handle_navigation(
        self,
        key: str,
        total_questions: int,
        request: ClarificationRequest,
        response: ClarificationResponse,
    ) -> bool:
        """Handle navigation input.

        Args:
            key: The pressed key
            total_questions: Total number of questions
            request: The clarification request
            response: The current response

        Returns:
            True if should continue, False if should exit
        """
        if key in ("q", "Q"):  # Quit with 'q'
            return False
        if key == "\x03":  # Ctrl+C
            raise KeyboardInterrupt("User interrupted with Ctrl+C")
        if key in ("\x1b[A", "k"):  # Up arrow
            self.state.current_index = (self.state.current_index - 1) % total_questions
        elif key in ("\x1b[B", "j"):  # Down arrow
            self.state.current_index = (self.state.current_index + 1) % total_questions
        elif key == "\t":  # Tab - jump to next required unanswered
            # Find next required unanswered question
            questions = request.get_sorted_questions()
            start_idx = self.state.current_index

            for offset in range(1, total_questions + 1):
                idx = (start_idx + offset) % total_questions
                question = questions[idx]
                answer = response.get_answer_for_question(question.id)

                # Check if this is a required unanswered question
                if question.is_required and (answer is None or answer.skipped):
                    self.state.current_index = idx
                    break
        elif key in ("c", "C"):  # Confirm
            self.state.confirmed = True
            return False

        return True

    async def review_answers(
        self, request: ClarificationRequest, response: ClarificationResponse, original_query: str
    ) -> ClarificationResponse | None:
        """Main review interface.

        Args:
            request: The original clarification request
            response: The initial response to review
            original_query: The original user query

        Returns:
            Final response after review or None if cancelled
        """
        import sys

        # Check if we're in an interactive terminal
        if not sys.stdin.isatty():
            self.console.print(
                "[warning]Non-interactive environment detected, skipping review[/warning]"
            )
            return response

        # Clone response for editing
        working_response = response.model_copy(deep=True)
        questions = request.get_sorted_questions()

        try:
            reviewing = True
            while reviewing:
                # Clear and render interface
                self.console.clear()

                # Progress bar
                self.console.print(self.render_progress_bar(request, working_response))
                self.console.print()

                # Current question
                current_question = questions[self.state.current_index]
                current_answer = working_response.get_answer_for_question(current_question.id)

                self.console.print(
                    self.render_question_panel(
                        current_question,
                        current_answer,
                        self.state.current_index + 1,
                        len(questions),
                        is_current=True,
                    )
                )

                # Navigation and summary in columns
                self.console.print()
                nav_panel = self.render_navigation_panel(request, working_response)
                summary_panel = self.render_summary_panel(request, working_response)

                self.console.print(Columns([nav_panel, summary_panel], padding=(0, 2)))

                # Get user input
                selector = InteractiveSelector(self.console)
                key = selector.get_key()

                # Handle input
                if key in ("e", "E", "\r", "\n"):  # Edit current
                    new_answer = self.edit_answer(current_question, current_answer)
                    if new_answer and new_answer != current_answer:
                        # Update the response
                        working_response.answers = [
                            a
                            for a in working_response.answers
                            if a.question_id != current_question.id
                        ]
                        working_response.answers.append(new_answer)
                        if self.state.edited_questions is not None:
                            self.state.edited_questions.add(current_question.id)
                        # Rebuild index
                        try:
                            working_response.model_post_init(None)
                        except AttributeError:
                            # If model_post_init doesn't exist, that's fine
                            pass

                elif not self.handle_navigation(key, len(questions), request, working_response):
                    reviewing = False

            if self.state.confirmed:
                # Final validation
                errors = working_response.validate_against_request(request)
                if errors:
                    self.console.print("\n[red]Cannot proceed - validation errors:[/red]")
                    for error in errors:
                        self.console.print(f"  • {error}")

                    if Confirm.ask("\nWould you like to fix these issues?"):
                        self.state.confirmed = False
                        # Check recursion depth to prevent infinite loops
                        if self.state.recursion_depth < 3:
                            self.state.recursion_depth += 1
                            return await self.review_answers(
                                request, working_response, original_query
                            )
                        self.console.print("[yellow]Maximum retry attempts reached.[/yellow]")
                    return None

                # Show confirmation
                self.console.print(
                    Panel(
                        "✅ Review complete! Your answers have been confirmed.",
                        title="SUCCESS",
                        border_style="green",
                        padding=(1, 2),
                    )
                )
                return working_response

            return None

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Review interrupted by user[/yellow]")
            raise  # Re-raise to propagate the interrupt
        except EOFError:
            self.console.print("\n[yellow]Review cancelled (EOF)[/yellow]")
            return None
        except Exception as e:
            self.console.print(f"[red]Error during review: {e}[/red]")
            return None


async def handle_review_interface(
    request: ClarificationRequest,
    response: ClarificationResponse,
    original_query: str,
    console: Console | None = None,
) -> ClarificationResponse | None:
    """Entry point for the review interface.

    Args:
        request: The original clarification request
        response: The initial response to review
        original_query: The original user query
        console: Optional Rich console instance

    Returns:
        Final response after review or None if cancelled
    """
    review = ReviewInterface(console)
    return await review.review_answers(request, response, original_query)
