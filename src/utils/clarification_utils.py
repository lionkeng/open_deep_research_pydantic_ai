"""Utility functions for clarification option processing.

This module provides shared utilities for handling clarification options,
including detecting when options require user input and extracting base labels.
"""

# Standardized list of keywords that indicate user input is required
SPECIFY_KEYWORDS = [
    "please specify",
    "specify",
    "describe",
    "enter",
    "provide",
    "input",
    "details",
    "detail",
    "name",
    "indicate",
    "tell us",
    "state which",
    "which ones",
]


def requires_specify_option(choice: str) -> bool:
    """Check if a choice option requires additional user input.

    This function uses three strategies to detect if an option needs follow-up input:
    1. Checks for keywords inside parentheses
    2. Checks for keywords after dashes
    3. Checks for keywords at the end of the text

    Special handling for "specify in 'Other'" pattern.

    Args:
        choice: The text of the choice option to analyze

    Returns:
        True if the option requires user input, False otherwise

    Examples:
        >>> requires_specify_option("Technology (please specify)")
        True
        >>> requires_specify_option("Python")
        False
        >>> requires_specify_option("Other - please describe")
        True
    """
    choice_lower = choice.lower()

    # Special case: "specify in 'other'" is misleading UX - it needs input here, not in Other option
    if "specify in 'other'" in choice_lower or 'specify in "other"' in choice_lower:
        return True

    # Extended keywords to check for
    keywords = SPECIFY_KEYWORDS

    # Strategy 1: Check inside parentheses (most reliable)
    if "(" in choice_lower and ")" in choice_lower:
        try:
            start = choice_lower.index("(")
            end = choice_lower.rindex(")")
            hint = choice_lower[start + 1 : end]
            if any(k in hint for k in keywords):
                return True
        except ValueError:
            pass

    # Strategy 2: Check after dash separators (—, –, -)
    for separator in ["—", "–", "-", ":"]:
        if separator in choice:
            parts = choice.split(separator)
            if len(parts) >= 2:
                # Check the part after the separator
                after_separator = parts[-1].lower().strip()
                if any(k in after_separator for k in keywords):
                    return True

    # Strategy 3: Check if keywords appear at the end of the option
    # This catches patterns like "United States, please specify"
    text_end = choice_lower[-30:] if len(choice_lower) > 30 else choice_lower
    if any(k in text_end for k in keywords):
        # Avoid false positives by checking it's not part of the main text
        # e.g., "Specific markets" shouldn't trigger
        if not any(choice_lower.startswith(k) for k in ["specific", "specified", "specify"]):
            return True

    return False


def extract_base_label(choice: str) -> str:
    """Extract the base label from a choice text by removing specification hints.

    This function removes parenthetical hints and dash-separated hints that
    contain keywords indicating user input is required.

    Args:
        choice: The full text of the choice option

    Returns:
        The base label with specification hints removed

    Examples:
        >>> extract_base_label("Technology (please specify)")
        'Technology'
        >>> extract_base_label("Python (programming language)")
        'Python (programming language)'
        >>> extract_base_label("Other - please describe")
        'Other'
    """
    label = choice.strip()

    # Strategy 1: Remove parenthetical hints if they contain keywords
    if "(" in label and ")" in label:
        try:
            start = label.index("(")
            end = label.rindex(")")
            hint = label[start:end].lower()
            if any(k in hint for k in SPECIFY_KEYWORDS):
                return (label[:start] + label[end + 1 :]).strip()
        except ValueError:
            pass

    # Strategy 2: Remove dash-separated hints if they contain keywords
    for separator in ["—", "–", "-", ":"]:
        if separator in label:
            parts = label.split(separator)
            if len(parts) >= 2:
                after_separator = parts[-1].lower().strip()
                if any(k in after_separator for k in SPECIFY_KEYWORDS):
                    return parts[0].strip()

    return label


def get_user_input_prompt(choice: str) -> str | None:
    """Generate an appropriate prompt for user input based on the choice text.

    Args:
        choice: The text of the choice that requires input

    Returns:
        A user-friendly prompt string, or None if no specific prompt is needed

    Examples:
        >>> get_user_input_prompt("Other (please specify)")
        'Please specify'
        >>> get_user_input_prompt("Technology - describe your choice")
        'Describe your choice'
    """
    choice_lower = choice.lower()

    # Try to extract prompt from parentheses
    if "(" in choice_lower and ")" in choice_lower:
        try:
            start = choice_lower.index("(")
            end = choice_lower.rindex(")")
            hint = choice_lower[start + 1 : end].strip()
            if any(k in hint for k in SPECIFY_KEYWORDS):
                # Return the hint itself as the prompt, properly capitalized
                return hint[0].upper() + hint[1:] if len(hint) > 1 else hint.upper()
        except ValueError:
            pass

    # Try to extract prompt after dash
    for separator in ["—", "–", "-", ":"]:
        if separator in choice:
            parts = choice.split(separator)
            if len(parts) >= 2:
                after_separator = parts[-1].strip()
                if any(k in after_separator.lower() for k in SPECIFY_KEYWORDS):
                    # Return the text after the separator as the prompt
                    if len(after_separator) > 1:
                        return after_separator[0].upper() + after_separator[1:]
                    return after_separator.upper()

    # Default prompt
    return None
