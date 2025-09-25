"""Rubric models for LLM-as-judge evaluation of research reports."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CriterionScores(BaseModel):
    """Per-criterion 1–5 scores for a single report."""

    readability_flow: int = Field(ge=1, le=5, description="Readability and narrative flow")
    thematic_clarity: int = Field(ge=1, le=5, description="Thematic clarity and structure")
    evidence_integration: int = Field(ge=1, le=5, description="Use of evidence and citations")
    insightfulness: int = Field(ge=1, le=5, description="Insightfulness and actionability")
    contradiction_handling: int = Field(ge=1, le=5, description="Handling of contradictions/nuance")


class JudgeOutput(BaseModel):
    """Structured response from the LLM judge for a pairwise comparison."""

    scores_control: CriterionScores
    scores_treatment: CriterionScores
    preference: str = Field(
        description='"CONTROL" | "TREATMENT" | "TIE"',
        pattern=r"^(CONTROL|TREATMENT|TIE)$",
    )
    rationale: str = Field(description="2–4 sentence explanation citing specific differences")


__all__ = ["CriterionScores", "JudgeOutput"]
