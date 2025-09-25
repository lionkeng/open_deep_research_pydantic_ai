from __future__ import annotations

import types

import pytest

from agents.report_clean_merge import marker_counts, run_clean_merge
from models.report_generator import ReportSection, ResearchReport


class _Deps:
    pass


@pytest.mark.asyncio
async def test_chunking_and_stitching_preserves_markers(monkeypatch: pytest.MonkeyPatch) -> None:
    # Fake Agent.run to avoid network: echo refined user content per chunk
    async def fake_run(self, *, deps=None, message_history=None, **kwargs):  # type: ignore[no-redef]
        class _Res:
            def __init__(self, value: str) -> None:
                self.output = types.SimpleNamespace(value=value)

        user_text = message_history[-1]["content"] if message_history else ""
        return _Res(value=f"Refined: {user_text}")

    import pydantic_ai

    monkeypatch.setattr(pydantic_ai.Agent, "run", fake_run, raising=False)

    # Build a section with 3 paragraphs to trigger chunking
    p1 = "Paragraph one cites [S21]." + (" a" * 60)
    p2 = "Paragraph two cites [S22]." + (" b" * 60)
    p3 = "Paragraph three cites [S23]." + (" c" * 60)
    content = "\n\n".join([p1, p2, p3])

    report = ResearchReport(
        title="Phase D Chunking",
        executive_summary="Summary cites [S20]." + (" s" * 60),
        introduction="Intro cites [S24]." + (" i" * 60),
        sections=[ReportSection(title="Chunked", content=content)],
        conclusions="Conclusion cites [S25]." + (" z" * 60),
        recommendations=[],
        references=[],
        appendices={},
        quality_score=0.5,
    )

    deps = _Deps()

    before_counts = marker_counts(report.sections[0].content)
    out, _ = await run_clean_merge(deps=deps, report=report)
    after_counts = marker_counts(out.sections[0].content)

    # Stitching kept markers and applied refinements per chunk
    assert out.sections[0].content.startswith("Refined: ")
    assert before_counts == after_counts
