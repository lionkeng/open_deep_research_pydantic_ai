"""Interactive selector with arrow key navigation for CLI choices."""

import sys
from typing import TYPE_CHECKING, Any

from rich.console import Console

# Initialize flags
_has_termios = False
_has_msvcrt = False

if TYPE_CHECKING:
    import msvcrt
    import termios
    import tty
else:
    # Check if we can use keyboard input
    try:
        import termios
        import tty

        _has_termios = True
    except ImportError:
        termios: Any = None
        tty: Any = None
        _has_termios = False

    # Try to import msvcrt for Windows
    try:
        import msvcrt

        _has_msvcrt = True
    except ImportError:
        msvcrt: Any = None
        _has_msvcrt = False


class InteractiveSelector:
    """Interactive selector with arrow key navigation."""

    def __init__(self, console: Console | None = None):
        """Initialize the selector.

        Args:
            console: Optional Rich console instance
        """
        self.console = console or Console()

    def get_key(self) -> str:
        """Get a single keypress from the user.

        Returns:
            Single key press as string

        Raises:
            KeyboardInterrupt: When Ctrl+C is pressed
        """
        try:
            if _has_termios and termios is not None:
                # Unix/Linux/Mac
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setraw(sys.stdin.fileno())
                    key = sys.stdin.read(1)
                    # Check for Ctrl+C immediately
                    if key == "\x03":
                        raise KeyboardInterrupt("User interrupted with Ctrl+C")
                    # Check for arrow keys (escape sequences)
                    if key == "\x1b":  # ESC
                        key += sys.stdin.read(2)
                    return key
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            elif _has_msvcrt and msvcrt is not None:
                # Windows
                getch = getattr(msvcrt, "getch", None)
                if getch:
                    key = getch()
                    # Check for Ctrl+C on Windows
                    if key == b"\x03":
                        raise KeyboardInterrupt("User interrupted with Ctrl+C")
                    if key in (b"\x00", b"\xe0"):  # Special keys (arrows, etc.)
                        key += getch()
                    return key.decode("utf-8", errors="ignore")
                return input()
            else:
                # Fallback - just read a line
                return input()
        except KeyboardInterrupt:
            # Make sure KeyboardInterrupt is always raised
            raise

    def select_single(
        self,
        question: str,
        choices: list[str],
        default: int | None = None,
        allow_skip: bool = False,
        context: str | None = None,
    ) -> str | None:
        """Select a single choice using arrow keys.

        Args:
            question: The question to ask
            choices: List of choices
            default: Default choice index (0-based)
            allow_skip: Whether to allow skipping
            context: Optional context to display

        Returns:
            Selected choice or None if skipped
        """
        if not choices:
            return None

        # Add skip option if allowed
        display_choices = choices.copy()
        if allow_skip:
            display_choices.append("[Skip this question]")

        current_index = default if default is not None else 0
        selected = False

        while not selected:
            # Clear previous display and show question/instructions once
            self.console.clear()
            self.console.print(f"\n[bold cyan]{question}[/bold cyan]")
            if context:
                self.console.print(f"[dim]{context}[/dim]")
            self.console.print(
                "[dim]Use ↑/↓ arrow keys to navigate, Enter to select, 'q' to quit[/dim]\n"
            )

            # Display choices with highlighting
            for idx, choice in enumerate(display_choices):
                if idx == current_index:
                    # Highlighted choice
                    self.console.print(f"[bold green]▸ {choice}[/bold green]")
                else:
                    self.console.print(f"  {choice}")

            # Get user input
            key = self.get_key()

            # Handle arrow keys
            if key in ("\x1b[A", "k"):  # Up arrow or 'k'
                current_index = (current_index - 1) % len(display_choices)
            elif key in ("\x1b[B", "j"):  # Down arrow or 'j'
                current_index = (current_index + 1) % len(display_choices)
            elif key in ("\r", "\n", " "):  # Enter or Space
                selected = True
            elif key in ("q", "Q"):  # 'q' to quit
                return None
            elif key == "\x03":  # Ctrl+C
                raise KeyboardInterrupt("User interrupted with Ctrl+C")

        # Handle selection
        if allow_skip and current_index == len(choices):
            return None  # Skip was selected

        selected_choice = choices[current_index]

        # Handle "Other (please specify)" options
        if self._is_other_option(selected_choice):
            return self._handle_other_option(selected_choice, question)

        return selected_choice

    def select_multiple(
        self,
        question: str,
        choices: list[str],
        default: list[int] | None = None,
        allow_skip: bool = False,
        context: str | None = None,
    ) -> list[str] | None:
        """Select multiple choices using arrow keys and space.

        Args:
            question: The question to ask
            choices: List of choices
            default: Default selected indices (0-based)
            allow_skip: Whether to allow skipping
            context: Optional context to display

        Returns:
            List of selected choices or None if skipped
        """
        if not choices:
            return None

        current_index = 0
        selected_indices = set(default) if default else set()
        done = False

        while not done:
            # Clear previous display
            self.console.clear()
            self.console.print(f"\n[bold cyan]{question}[/bold cyan]")
            if context:
                self.console.print(f"[dim]{context}[/dim]")
            self.console.print(
                "[dim]Use ↑/↓ to navigate, Space to select/deselect, "
                "Enter when done, 'q' to quit[/dim]\n"
            )

            # Display choices with highlighting and selection
            for idx, choice in enumerate(choices):
                selected_mark = "[✓]" if idx in selected_indices else "[ ]"
                if idx == current_index:
                    # Highlighted choice
                    color = "green" if idx in selected_indices else "yellow"
                    self.console.print(f"[bold {color}]▸ {selected_mark} {choice}[/bold {color}]")
                else:
                    if idx in selected_indices:
                        self.console.print(f"  [green]{selected_mark} {choice}[/green]")
                    else:
                        self.console.print(f"  {selected_mark} {choice}")

            # Show skip option if allowed and nothing selected
            if allow_skip and not selected_indices:
                skip_text = "[dim]Press Enter with no selections to skip[/dim]"
                self.console.print(f"\n{skip_text}")

            # Get user input
            key = self.get_key()

            # Handle arrow keys
            if key in ("\x1b[A", "k"):  # Up arrow or 'k'
                current_index = (current_index - 1) % len(choices)
            elif key in ("\x1b[B", "j"):  # Down arrow or 'j'
                current_index = (current_index + 1) % len(choices)
            elif key == " ":  # Space to toggle selection
                if current_index in selected_indices:
                    selected_indices.remove(current_index)
                else:
                    selected_indices.add(current_index)
            elif key in ("\r", "\n"):  # Enter to confirm
                done = True
            elif key in ("q", "Q"):  # 'q' to quit
                return None
            elif key == "\x03":  # Ctrl+C
                raise KeyboardInterrupt("User interrupted with Ctrl+C")

        # Return selected choices with "Other" option handling
        if not selected_indices and allow_skip:
            return None

        selected_choices = [choices[i] for i in sorted(selected_indices)]

        # Handle "Other (please specify)" options
        processed_choices = []
        for choice in selected_choices:
            if self._is_other_option(choice):
                custom_choice = self._handle_other_option(choice, question)
                processed_choices.append(custom_choice)
            else:
                processed_choices.append(choice)

        return processed_choices

    def _is_other_option(self, choice: str) -> bool:
        """Check if a choice is an 'Other (please specify)' type option.

        Args:
            choice: The choice text to check

        Returns:
            True if this is an "other" option requiring text input
        """
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

    def _handle_other_option(self, selected_choice: str, question: str) -> str:
        """Handle an 'Other' option by prompting for additional input.

        Args:
            selected_choice: The selected "Other" choice
            question: The original question for context

        Returns:
            The user's custom response
        """
        self.console.print(f"\n[yellow]You selected: {selected_choice}[/yellow]")
        self.console.print("[bold cyan]Please provide your specific answer:[/bold cyan]")

        from rich.prompt import Prompt

        # Prompt for custom input with validation
        while True:
            custom_input = Prompt.ask("Your answer").strip()

            if not custom_input:
                self.console.print("[red]Please provide a response or press 'q' to cancel[/red]")
                continue

            if custom_input.lower() in ["q", "quit", "cancel"]:
                return selected_choice  # Return original choice if cancelled

            # Confirm the input
            self.console.print(f"\n[dim]Your answer: {custom_input}[/dim]")
            confirm = Prompt.ask("Is this correct?", choices=["y", "n"], default="y")

            if confirm.lower() == "y":
                return f"Other: {custom_input}"

            # If not confirmed, loop back to ask again


def interactive_select(
    question: str,
    choices: list[str],
    multiple: bool = False,
    default: Any | None = None,
    allow_skip: bool = False,
    context: str | None = None,
    console: Console | None = None,
) -> Any | None:
    """Helper function for interactive selection.

    Args:
        question: The question to ask
        choices: List of choices
        multiple: Whether to allow multiple selections
        default: Default selection(s)
        allow_skip: Whether to allow skipping
        context: Optional context to display
        console: Optional Rich console instance

    Returns:
        Selected choice(s) or None if skipped
    """
    selector = InteractiveSelector(console)

    if multiple:
        return selector.select_multiple(question, choices, default, allow_skip, context)
    return selector.select_single(question, choices, default, allow_skip, context)
