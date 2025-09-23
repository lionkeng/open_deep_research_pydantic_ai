"""Eval driver: Compare Control vs Treatment synthesis/report outputs.

Control:   ENABLE_EMBEDDING_SIMILARITY=0, ENABLE_LLM_CLEAN_MERGE=0
Treatment: ENABLE_EMBEDDING_SIMILARITY=1, ENABLE_LLM_CLEAN_MERGE=1

Artifacts are stored under eval_results/synthesis_features/<run_id>/.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from open_deep_research_pydantic_ai import ResearchWorkflow

from .judge import get_judge_agent

RUN_ID = time.strftime("%Y%m%d-%H%M%S")
BASE_OUT = Path("eval_results/synthesis_features") / RUN_ID


def _ensure_dirs() -> None:
    (BASE_OUT).mkdir(parents=True, exist_ok=True)


async def run_condition(query: str, enabled: bool) -> Any:
    """Execute workflow under a specific feature flag condition and return final report."""

    os.environ["ENABLE_EMBEDDING_SIMILARITY"] = "1" if enabled else "0"
    os.environ["ENABLE_LLM_CLEAN_MERGE"] = "1" if enabled else "0"
    if enabled and "EMBEDDING_SIMILARITY_THRESHOLD" not in os.environ:
        os.environ["EMBEDDING_SIMILARITY_THRESHOLD"] = "0.55"

    state = await ResearchWorkflow().run(user_query=query)
    return state.final_report


def _extract_for_judging(report: Any) -> dict[str, Any]:
    """Extract a comparable subset of report fields for judging."""

    if report is None:
        return {"error": "no_report"}

    sections = []
    for s in (report.sections or [])[:2]:
        content = getattr(s, "content", "") or ""
        sections.append({"title": getattr(s, "title", ""), "content": content[:3000]})

    return {
        "title": getattr(report, "title", ""),
        "executive_summary": getattr(report, "executive_summary", ""),
        "introduction": (getattr(report, "introduction", "") or "")[:3000],
        "sections": sections,
        "conclusions": (getattr(report, "conclusions", "") or "")[:1500],
        "recommendations": "\n".join(getattr(report, "recommendations", []) or [])[:1500],
    }


async def judge_pair(topic_id: str, query: str, control: Any, treatment: Any) -> Any:
    """Run LLM judge on a control/treatment pair and return structured output."""

    payload = {
        "topic_id": topic_id,
        "query": query,
        "control": _extract_for_judging(control),
        "treatment": _extract_for_judging(treatment),
    }
    judge = get_judge_agent()
    res = await judge.run(message_history=[{"role": "user", "content": json.dumps(payload)}])
    return res.output


async def main(
    topics_path: str = "tests/evals/evaluation_datasets/synthesis_topics.jsonl",
) -> None:
    _ensure_dirs()
    judgments_path = BASE_OUT / "judgments.jsonl"
    judgments_out: list[str] = []

    with open(topics_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            topic_id = str(item.get("id", ""))
            query = str(item.get("query", "")).strip()
            if not query:
                continue

            control = await run_condition(query, enabled=False)
            treatment = await run_condition(query, enabled=True)

            # Persist per-topic reports
            (BASE_OUT / f"{topic_id}_control.json").write_text(
                control.model_dump_json(indent=2) if control else "{}",
                encoding="utf-8",
            )
            (BASE_OUT / f"{topic_id}_treatment.json").write_text(
                treatment.model_dump_json(indent=2) if treatment else "{}",
                encoding="utf-8",
            )

            # Judge
            j = await judge_pair(topic_id, query, control, treatment)
            rec = {"id": topic_id, "query": query, "judgment": j.model_dump()}
            judgments_out.append(json.dumps(rec))

    judgments_path.write_text("\n".join(judgments_out), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
