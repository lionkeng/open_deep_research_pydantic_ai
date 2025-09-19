"""
Single source of truth for priority ordering across the codebase.

This module defines the priority system used throughout the application to ensure
consistent priority handling across all components.
"""

from typing import Literal

from pydantic import Field


class Priority:
    """
    Single source of truth for priority ordering across the codebase.

    Priority is represented as an integer from 1 to 5 where:
    - 1 = HIGHEST priority (executed first)
    - 2 = HIGH priority
    - 3 = MEDIUM priority (default)
    - 4 = LOW priority
    - 5 = LOWEST priority (executed last)

    This ensures consistent priority handling across all components.
    """

    # Priority constants
    HIGHEST: Literal[1] = 1
    HIGH: Literal[2] = 2
    MEDIUM: Literal[3] = 3
    LOW: Literal[4] = 4
    LOWEST: Literal[5] = 5

    # Priority bounds
    MIN_PRIORITY: Literal[1] = 1  # Highest priority value
    MAX_PRIORITY: Literal[5] = 5  # Lowest priority value
    DEFAULT_PRIORITY: Literal[3] = 3  # Default to medium

    # Priority field definition for Pydantic models
    FIELD_DEFINITION = Field(
        default=3, ge=1, le=5, description="Priority level (1=highest, 5=lowest)"
    )

    @staticmethod
    def is_valid(priority: int) -> bool:
        """Check if a priority value is valid."""
        return Priority.MIN_PRIORITY <= priority <= Priority.MAX_PRIORITY

    @staticmethod
    def sort_key(item) -> int:
        """
        Sort key function for sorting by priority.
        Use with sorted() or list.sort() to sort items by priority.

        Example:
            sorted(items, key=Priority.sort_key)
        """
        if hasattr(item, "priority"):
            return item.priority
        elif isinstance(item, dict) and "priority" in item:
            return item["priority"]
        return Priority.DEFAULT_PRIORITY

    @staticmethod
    def should_execute_first(priority_a: int, priority_b: int) -> bool:
        """
        Compare two priorities to determine execution order.
        Returns True if priority_a should execute before priority_b.
        """
        return priority_a < priority_b

    @staticmethod
    def get_name(priority: int) -> str:
        """Get human-readable name for priority level."""
        names = {1: "HIGHEST", 2: "HIGH", 3: "MEDIUM", 4: "LOW", 5: "LOWEST"}
        return names.get(priority, f"UNKNOWN({priority})")

    @staticmethod
    def from_string(priority_str: str) -> int:
        """
        Convert string priority to integer.
        Accepts: 'highest', 'high', 'medium', 'low', 'lowest' (case-insensitive)
        """
        mapping = {"highest": 1, "high": 2, "medium": 3, "low": 4, "lowest": 5}
        return mapping.get(priority_str.lower(), Priority.DEFAULT_PRIORITY)
