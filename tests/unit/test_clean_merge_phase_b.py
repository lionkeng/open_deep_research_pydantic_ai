from __future__ import annotations

import types

import pytest

from agents.report_clean_merge import run_clean_merge
from models.report_generator import ReportSection, ResearchReport


class _Deps:
    pass


@pytest.mark.asyncio
async def test_clean_merge_core_narrative_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    # Fake Agent.run to avoid network: echo refined user content
    async def fake_run(self, *, deps=None, message_history=None, **kwargs):  # type: ignore[no-redef]
        class _Out:
            def __init__(self, value: str) -> None:
                self.value = value

        class _Res:
            def __init__(self, value: str) -> None:
                self.output = types.SimpleNamespace(value=value)

        user_text = message_history[-1]["content"] if message_history else ""
        return _Res(value=f"Refined: {user_text}")

    import pydantic_ai

    monkeypatch.setattr(pydantic_ai.Agent, "run", fake_run, raising=False)

    # Build a report with top-level fields and markers
    exec_text = (
        "This executive summary states an insight supported by [S1]. " + "x" * 80
    )
    intro_text = "Introduction provides context and cites [S2]. " + "y" * 85
    section_text = "Section content explains details with ref [S3]. " + "z" * 86
    concl_text = "We conclude and reference [S4]. " + "w" * 90

    report = ResearchReport(
        title="Test Report",
        executive_summary=exec_text,
        introduction=intro_text,
        sections=[ReportSection(title="A", content=section_text)],
        conclusions=concl_text,
        recommendations=[],
        references=[],
        appendices={},
        quality_score=0.5,
    )

    deps = _Deps()

    out, _ = await run_clean_merge(deps=deps, report=report)

    assert out.executive_summary.startswith("Refined: ")
    assert out.introduction.startswith("Refined: ")
    assert out.sections[0].content.startswith("Refined: ")
    assert out.conclusions.startswith("Refined: ")
