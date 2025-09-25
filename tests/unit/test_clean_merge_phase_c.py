from __future__ import annotations

import types

import pytest

from agents.report_clean_merge import run_clean_merge
from models.report_generator import ReportSection, ResearchReport


class _Deps:
    pass


@pytest.mark.asyncio
async def test_clean_merge_full_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    # Fake Agent.run to avoid network: echo refined user content
    async def fake_run(self, *, deps=None, message_history=None, **kwargs):  # type: ignore[no-redef]
        class _Res:
            def __init__(self, value: str) -> None:
                self.output = types.SimpleNamespace(value=value)

        user_text = message_history[-1]["content"] if message_history else ""
        return _Res(value=f"Refined: {user_text}")

    import pydantic_ai

    monkeypatch.setattr(pydantic_ai.Agent, "run", fake_run, raising=False)

    # Build a report with subsections, recommendations, and appendices
    exec_text = "Exec summary cites [S10]. " + ("x" * 120)
    intro_text = "Intro cites [S11]. " + ("y" * 120)
    section_text = "Section cites [S12]. " + ("z" * 120)
    sub_text = "Subsection cites [S13]. " + ("u" * 120)
    concl_text = "Conclusion cites [S14]. " + ("v" * 120)

    report = ResearchReport(
        title="Phase C Report",
        executive_summary=exec_text,
        introduction=intro_text,
        sections=[
            ReportSection(title="A", content=section_text, subsections=[ReportSection(title="A.1", content=sub_text)])
        ],
        conclusions=concl_text,
        recommendations=["Recommend step with ref [S15]. " + ("r" * 120)],
        references=[],
        appendices={"annex": "Annex text with ref [S16]. " + ("a" * 120)},
        quality_score=0.5,
    )

    deps = _Deps()

    out, _ = await run_clean_merge(deps=deps, report=report)

    # Narrative fields
    assert out.executive_summary.startswith("Refined: ")
    assert out.introduction.startswith("Refined: ")
    assert out.sections[0].content.startswith("Refined: ")
    assert out.sections[0].subsections[0].content.startswith("Refined: ")
    assert out.conclusions.startswith("Refined: ")

    # Post-narrative fields
    assert all(s.startswith("Refined: ") for s in out.recommendations)
    assert out.appendices["annex"].startswith("Refined: ")
