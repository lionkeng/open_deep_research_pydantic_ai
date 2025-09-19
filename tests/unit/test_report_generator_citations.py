"""Tests for citation enforcement in the report generator."""

from unittest.mock import MagicMock

from agents.base import ResearchDependencies
from agents.report_generator import ReportGeneratorAgent
from models.api_models import APIKeys
from models.core import ResearchMetadata, ResearchStage, ResearchState
from models.report_generator import ReportMetadata, ReportSection, ResearchReport
from models.research_executor import HierarchicalFinding, ResearchResults, ResearchSource


def test_postprocessing_converts_markers_to_footnotes() -> None:
    agent = ReportGeneratorAgent()
    report = ResearchReport(
        title="Test",
        executive_summary="Key takeaway supported by [S1].",
        introduction="Background details [S1] and [S2].",
        sections=[
            ReportSection(title="Findings", content="Insight [S2]", subsections=[]),
        ],
        conclusions="Summary paragraph.",
        metadata=ReportMetadata(),
        references=[],
        recommendations=[],
        appendices={},
        overall_quality_score=0.0,
    )
    sources = [
        ResearchSource(title="Source One", url="https://example.com/1", source_id="S1"),
        ResearchSource(title="Source Two", url="https://example.com/2", source_id="S2"),
    ]
    findings = [HierarchicalFinding(finding="Fact", source=sources[0])]
    research_results = ResearchResults(query="Test query", findings=findings, sources=sources)

    research_state = ResearchState(
        request_id="test",
        user_id="user",
        session_id=None,
        user_query="Test query",
        current_stage=ResearchStage.REPORT_GENERATION,
        metadata=ResearchMetadata(),
    )
    research_state.research_results = research_results

    deps = ResearchDependencies(
        http_client=MagicMock(),
        api_keys=APIKeys(),
        research_state=research_state,
    )

    processed = agent._apply_citation_postprocessing(report, deps)

    assert "[^1]" in processed.executive_summary
    assert processed.references
    assert processed.references[0].startswith("[^1]: Source One")
    assert processed.metadata.source_summary
    assert processed.metadata.citation_audit.get("status") == "pass"
    assert processed.metadata.citation_audit.get("contiguous") is True


def test_postprocessing_injects_missing_citations() -> None:
    agent = ReportGeneratorAgent()
    report = ResearchReport(
        title="Minimal",
        executive_summary="No citations yet.",
        introduction="",
        sections=[],
        conclusions="",
        metadata=ReportMetadata(),
        references=[],
        recommendations=[],
        appendices={},
        overall_quality_score=0.0,
    )
    sources = [
        ResearchSource(title="Alpha", url="https://alpha.com", source_id="S1"),
        ResearchSource(title="Beta", url="https://beta.com", source_id="S2"),
        ResearchSource(title="Gamma", url="https://gamma.com", source_id="S3"),
    ]
    research_results = ResearchResults(
        query="Minimal",
        findings=[],
        sources=sources,
    )

    research_state = ResearchState(
        request_id="test2",
        user_id="user",
        session_id=None,
        user_query="Minimal",
        current_stage=ResearchStage.REPORT_GENERATION,
        metadata=ResearchMetadata(),
    )
    research_state.research_results = research_results

    deps = ResearchDependencies(
        http_client=MagicMock(),
        api_keys=APIKeys(),
        research_state=research_state,
    )

    processed = agent._apply_citation_postprocessing(report, deps)

    assert "[^1]" in processed.conclusions or "[^2]" in processed.conclusions
    assert len(processed.references) >= 1
    audit = processed.metadata.citation_audit
    assert audit.get("status") in {"pass", "warn"}
    assert audit.get("cited_sources") >= 1
    assert audit.get("total_sources") == 3
