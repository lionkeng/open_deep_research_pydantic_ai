"""LLM judge agent for comparing Control vs Treatment reports.

Uses a fixed model by default; can be overridden with EVAL_JUDGE_MODEL.
"""

from __future__ import annotations

import os

from pydantic_ai import Agent

from .rubric import JudgeOutput


def get_judge_model_name() -> str:
    """Return model name for the judge, overridable via EVAL_JUDGE_MODEL."""

    model = os.getenv("EVAL_JUDGE_MODEL")
    if model and model.strip():
        return model.strip()
    # Default to a reasoning-capable model; can be changed by env
    return "openai:gpt-4o-mini"


def get_judge_agent() -> Agent[None, JudgeOutput]:
    """Create and return the LLM judge agent instance."""

    system_prompt = (
        "You are a rigorous editorial evaluator. Compare two research reports strictly by the "
        "rubric. Do not invent facts. Provide per-criterion scores for CONTROL and TREATMENT, "
        "then choose a single preference (CONTROL or TREATMENT) unless they are truly equal (TIE). "
        "Be concise and cite concrete differences in your rationale."
    )

    judge = Agent(
        model=get_judge_model_name(),
        output_type=JudgeOutput,
        system_prompt=system_prompt,
    )
    return judge


__all__ = ["get_judge_agent", "get_judge_model_name"]
