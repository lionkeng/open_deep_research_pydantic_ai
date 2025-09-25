from __future__ import annotations

import types

import pytest

from agents.report_clean_merge import record_clean_merge_metrics, run_clean_merge
from models.report_generator import ReportSection, ResearchReport
from models.research_executor import ResearchResults, SynthesisMetadata


class _ResearchState:
    def __init__(self) -> None:
        self.research_results = ResearchResults(query="q", synthesis_metadata=SynthesisMetadata())


class _Deps:
    def __init__(self) -> None:
        self.research_state = _ResearchState()


@pytest.mark.asyncio
async def test_metrics_rollup(monkeypatch: pytest.MonkeyPatch) -> None:
    # Fake Agent.run to always produce a valid refined output
    async def fake_run(self, *, deps=None, message_history=None, **kwargs):  # type: ignore[no-redef]
        class _Res:
            def __init__(self, value: str) -> None:
                self.output = types.SimpleNamespace(value=value)

        user_text = message_history[-1]["content"] if message_history else ""
        return _Res(value=f"Refined: {user_text}")

    import pydantic_ai

    monkeypatch.setattr(pydantic_ai.Agent, "run", fake_run, raising=False)

    # Build report with multiple fields including chunking case
    chunk_content = "\n\n".join(
        [
            "P1 [S1]." + (" a" * 50),
            "P2 [S2]." + (" b" * 50),
            "P3 [S3]." + (" c" * 50),
        ]
    )
    report = ResearchReport(
        title="Phase E Metrics",
        executive_summary="Exec [S1]." + (" x" * 50),
        introduction="Intro [S2]." + (" y" * 50),
        sections=[ReportSection(title="A", content=chunk_content)],
        conclusions="Concl [S3]." + (" z" * 50),
        recommendations=["Do [S4]." + (" r" * 50)],
        references=[],
        appendices={"annex": "Annex [S5]." + (" a" * 50)},
        quality_score=0.5,
    )

    deps = _Deps()

    report, metrics = await run_clean_merge(deps=deps, report=report)
    record_clean_merge_metrics(metrics=metrics, deps=deps)

    qm = deps.research_state.research_results.synthesis_metadata.quality_metrics
    assert qm.get("clean_merge_fields_attempted", 0) >= 5
    assert qm.get("clean_merge_fields_applied", 0) >= 5
    assert qm.get("clean_merge_rejects_length", 0) >= 0
    assert qm.get("clean_merge_rejects_marker_mismatch", 0) >= 0
    # Chunking should have been applied for the section content
    assert qm.get("clean_merge_chunked_applied", 0) >= 1
