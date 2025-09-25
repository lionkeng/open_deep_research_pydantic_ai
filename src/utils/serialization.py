"""Serialization utilities for clarification models."""

import json
from datetime import datetime
from typing import Any
from uuid import UUID


class ClarificationJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for clarification models.

    Handles special types like UUID and datetime that aren't natively
    JSON serializable.
    """

    def default(self, o: Any) -> Any:
        """Convert non-JSON-serializable objects.

        Args:
            o: Object to encode

        Returns:
            JSON-serializable representation
        """
        if isinstance(o, UUID):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        if hasattr(o, "model_dump"):
            # Handle Pydantic models
            return o.model_dump()
        if hasattr(o, "__dict__"):
            # Handle regular classes
            return o.__dict__

        return super().default(o)


def serialize_clarification_request(request: Any) -> str:
    """Serialize a ClarificationRequest to JSON.

    Args:
        request: ClarificationRequest object

    Returns:
        JSON string representation
    """
    return json.dumps(
        request.model_dump() if hasattr(request, "model_dump") else request,
        cls=ClarificationJSONEncoder,
        indent=2,
    )


def serialize_clarification_response(response: Any) -> str:
    """Serialize a ClarificationResponse to JSON.

    Args:
        response: ClarificationResponse object

    Returns:
        JSON string representation
    """
    return json.dumps(
        response.model_dump() if hasattr(response, "model_dump") else response,
        cls=ClarificationJSONEncoder,
        indent=2,
    )


def deserialize_clarification_request(json_str: str) -> dict[str, Any]:
    """Deserialize a ClarificationRequest from JSON.

    Args:
        json_str: JSON string

    Returns:
        Dictionary that can be used to create ClarificationRequest
    """
    return json.loads(json_str)


def deserialize_clarification_response(json_str: str) -> dict[str, Any]:
    """Deserialize a ClarificationResponse from JSON.

    Args:
        json_str: JSON string

    Returns:
        Dictionary that can be used to create ClarificationResponse
    """
    return json.loads(json_str)


def format_clarification_for_display(request: Any) -> str:
    """Format a ClarificationRequest for human-readable display.

    Args:
        request: ClarificationRequest object

    Returns:
        Formatted string for display
    """
    if not hasattr(request, "questions"):
        return str(request)

    lines = ["Clarification Questions:"]
    for idx, question in enumerate(request.get_sorted_questions(), 1):
        required = "[Required]" if question.is_required else "[Optional]"
        lines.append(f"{idx}. {required} {question.question}")

        if question.context:
            lines.append(f"   Context: {question.context}")

        if question.choices:
            lines.append("   Options:")
            for choice in question.choices:
                flags = []
                if choice.is_other:
                    flags.append("other")
                if choice.requires_details:
                    flags.append("requires details")
                flag_str = f" [{' & '.join(flags)}]" if flags else ""
                lines.append(f"   - {choice.label}{flag_str} (id={choice.id})")

    return "\n".join(lines)


def format_response_for_display(response: Any, request: Any) -> str:
    """Format a ClarificationResponse for human-readable display.

    Args:
        response: ClarificationResponse object
        request: Original ClarificationRequest for context

    Returns:
        Formatted string for display
    """
    if not hasattr(response, "answers"):
        return str(response)

    lines = ["Clarification Responses:"]

    for answer in response.answers:
        question = (
            request.get_question_by_id(answer.question_id)
            if hasattr(request, "get_question_by_id")
            else None
        )

        def _fmt(q, a) -> str:
            if a.skipped:
                return "[Skipped]"
            if q.question_type == "text":
                return a.text or ""
            if q.question_type == "choice":
                if not a.selection:
                    return ""
                ch = next((c for c in (q.choices or []) if c.id == a.selection.id), None)
                label = ch.label if ch else a.selection.id
                return f"{label}: {a.selection.details}" if a.selection.details else label
            if q.question_type == "multi_choice":
                parts: list[str] = []
                for sel in a.selections or []:
                    ch = next((c for c in (q.choices or []) if c.id == sel.id), None)
                    label = ch.label if ch else sel.id
                    parts.append(f"{label}: {sel.details}" if sel.details else label)
                return ", ".join(parts)
            return ""

        if question:
            lines.append(f"- {question.question}: {_fmt(question, answer)}")
        else:
            # Unknown question: best-effort formatting without label resolution
            if answer.skipped:
                lines.append(f"- Question {answer.question_id}: [Skipped]")
            elif answer.text is not None:
                lines.append(f"- Question {answer.question_id}: {answer.text}")
            elif answer.selection is not None:
                sel = answer.selection
                part = f"{sel.id}: {sel.details}" if sel.details else sel.id
                lines.append(f"- Question {answer.question_id}: {part}")
            elif answer.selections:
                parts = [f"{s.id}: {s.details}" if s.details else s.id for s in answer.selections]
                lines.append(f"- Question {answer.question_id}: {', '.join(parts)}")
            else:
                lines.append(f"- Question {answer.question_id}: ")

    return "\n".join(lines)
