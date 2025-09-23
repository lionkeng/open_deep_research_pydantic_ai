# Eval Plan: Do Embedding Similarity + Guardrailed Clean‑Merge Improve Reports?

This document specifies an eval using the Pydantic AI eval framework to judge, with an LLM, whether enabling two optional synthesis features improves the quality of generated research reports compared to the default (disabled) configuration.

- Feature A: Embedding‑based semantic grouping for convergence/themes
- Feature B: Guardrailed LLM clean‑merge of the executive summary

We test Control (A=off, B=off) vs Treatment (A=on, B=on).

## Goals & Hypotheses

- H1 (Readability): Treatment produces more readable, cohesive summaries while preserving citations.
- H2 (Thematic Quality): Treatment yields clearer groupings of findings and patterns.
- H3 (Overall Preference): An LLM judge prefers Treatment over Control on a majority of topics (≥60%).

## Design Overview

- Compare two conditions for the same set of topics:
  - Control: `ENABLE_EMBEDDING_SIMILARITY=0`, `ENABLE_LLM_CLEAN_MERGE=0`
  - Treatment: `ENABLE_EMBEDDING_SIMILARITY=1`, `ENABLE_LLM_CLEAN_MERGE=1`
- Generate a report per topic per condition with the same base model and settings.
- Use an LLM judge to rate both reports on a rubric and indicate a pairwise preference with rationale.
- Aggregate results and compute summary statistics and confidence intervals.

## Dataset

Use ~20–50 research prompts spanning varied domains:

- Tech: “Latest advances in vector databases for RAG”, “Edge vs cloud AI trade‑offs”
- Health: “AI in radiology triage”, “Wearables and early disease detection”
- Policy: “AI policy trends in the EU”, “Regulatory approaches to model transparency”
- Business: “Pricing strategies for SaaS in a downturn”, “Platform risk for marketplaces”
- Science: “CRISPR off‑target risk mitigation”, “Fusion energy commercialization timelines”

Store prompts in `tests/evals/evaluation_datasets/synthesis_topics.jsonl` (one JSON per line):

```json
{"id": "t001", "query": "Latest advances in vector databases for RAG"}
```

## Configuration & Controls

- Keep the same base generation model across runs (default configured in `APIConfig.default_model`).
- Fix temperature and other sampling params if applicable (use defaults; ensure consistency across conditions).
- Set `EMBEDDING_SIMILARITY_THRESHOLD` to a conservative default (e.g., 0.55) for Treatment.
- Ensure identical search/backends; avoid external changes mid‑eval (run in a short time window or cache inputs when possible).

## Judging Rubric (LLM‑as‑Judge)

For each topic, present the judge with Control and Treatment outputs and ask for:

1) Criterion scores (1–5):
   - Readability/Flow (RF)
   - Thematic Clarity/Structure (TC)
   - Evidence Integration & Citation Use (EI)
   - Insightfulness/Actionability (IA)
   - Contradiction Handling & Nuance (CH)

2) Pairwise Preference: CONTROL or TREATMENT (must choose one unless tie is unavoidable)

3) Rationale: 2–4 sentences citing specific differences; must not add new facts.

Guardrails for the judge prompt:

- Judge only based on provided text; do not imagine missing content.
- Consider `[Sx]` markers and whether they appear appropriate and consistent.
- Prefer outputs with better flow and structure if citation quality is similar.

Recommended judge model: a strong reasoning model such as `openai:gpt-4o-mini` or `openai:gpt-4o`. Keep the judge model fixed.

## Evaluation Procedure

1) For each topic `t` in the dataset:
   - Run Control pipeline → produce `report_control_{t}.json` (typed `ResearchReport`)
   - Run Treatment pipeline → produce `report_treatment_{t}.json`
2) Extract comparable fields for judging (limit length to fit context):
   - Executive summary (entire)
   - Introduction (first ~500 tokens)
   - 1–2 main sections (first ~800 tokens total)
   - Conclusions and recommendations (first ~300 tokens)
3) Provide both report excerpts to the judge with the rubric instructions.
4) Collect scores (RF, TC, EI, IA, CH), pairwise preference, and rationale.
5) Repeat 1–4 for all topics.

## Metrics & Analysis

- Preference rate: fraction of topics where Treatment is preferred.
- Average score deltas: Treatment − Control per criterion with 95% CIs (bootstrap).
- Tie handling: count ties, exclude from preference numerator (report separately).
- Optional robustness: multiple judge calls per topic (n=3) and majority vote.

Success criteria:

- Treatment preferred in ≥60% of topics, and
- Mean deltas positive for RF and TC with non‑overlapping CIs (or at least positive with narrow CIs), and
- No degradation in EI (citations) beyond 0.2 points on average.

## Implementation Outline (pydantic‑ai eval framework)

File layout:

```
tests/evals/
  synthesis/                     # synthesis eval package
    synthesis_features_eval.py   # driver & orchestration
    judge.py                     # LLM judge agent (pydantic-ai)
    rubric.py                    # Pydantic models for rubric and outputs
  evaluation_datasets/
    synthesis_topics.jsonl       # dataset of topics
eval_results/
  synthesis_features/<run_id>/   # artifacts (reports, judgments, summary)
```

Pydantic models (rubric.py):

```python
from pydantic import BaseModel, Field

class CriterionScores(BaseModel):
    readability_flow: int = Field(ge=1, le=5)
    thematic_clarity: int = Field(ge=1, le=5)
    evidence_integration: int = Field(ge=1, le=5)
    insightfulness: int = Field(ge=1, le=5)
    contradiction_handling: int = Field(ge=1, le=5)

class JudgeOutput(BaseModel):
    scores_control: CriterionScores
    scores_treatment: CriterionScores
    preference: str  # "CONTROL" | "TREATMENT" | "TIE"
    rationale: str
```

Judge agent (judge.py):

```python
from pydantic_ai import Agent
from .rubric import JudgeOutput

judge = Agent(
    model="openai:gpt-4o-mini",
    output_type=JudgeOutput,
    system_prompt=(
        "You are a rigorous editorial evaluator. Compare two research reports strictly "
        "by the rubric. Do not invent facts. Provide scores and a single preference."
    ),
)
```

Driver (synthesis_features_eval.py):

```python
import asyncio, os, json, time
from pathlib import Path
from open_deep_research_pydantic_ai import ResearchWorkflow
from .judge import judge

RUN_ID = time.strftime("%Y%m%d-%H%M%S")
OUT_DIR = Path("eval_results/synthesis_features") / RUN_ID

async def run_condition(query: str, enable: bool):
    os.environ["ENABLE_EMBEDDING_SIMILARITY"] = "1" if enable else "0"
    os.environ["ENABLE_LLM_CLEAN_MERGE"] = "1" if enable else "0"
    wf = ResearchWorkflow()
    state = await wf.run(user_query=query)
    return state.final_report

async def judge_pair(topic_id: str, query: str, control, treatment):
    # Truncate fields as needed for context
    def extract(rep):
        return {
            "executive_summary": rep.executive_summary,
            "introduction": rep.introduction[:3000],
            "sections": [
                {"title": s.title, "content": (s.content or "")[:3000]}
                for s in rep.sections[:2]
            ],
            "conclusions": rep.conclusions[:1500],
            "recommendations": "\n".join(rep.recommendations)[:1500],
        }
    payload = {
        "topic_id": topic_id,
        "query": query,
        "control": extract(control),
        "treatment": extract(treatment),
    }
    res = await judge.run(message_history=[{"role": "user", "content": json.dumps(payload)}])
    return res.output

async def main(topics_path: str = "tests/evals/evaluation_datasets/synthesis_topics.jsonl"):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    with open(topics_path, "r") as f:
        for line in f:
            item = json.loads(line)
            tid, query = item["id"], item["query"]
            control = await run_condition(query, enable=False)
            treatment = await run_condition(query, enable=True)
            # Save reports
            (OUT_DIR / f"{tid}_control.json").write_text(control.model_dump_json(indent=2))
            (OUT_DIR / f"{tid}_treatment.json").write_text(treatment.model_dump_json(indent=2))
            # Judge
            j = await judge_pair(tid, query, control, treatment)
            rec = {"id": tid, "query": query, "judgment": j.model_dump()}
            results.append(rec)
    (OUT_DIR / "judgments.jsonl").write_text("\n".join(json.dumps(r) for r in results))
    # Aggregate summaries can be computed in a follow-up step

if __name__ == "__main__":
    asyncio.run(main())
```

Run commands:

```bash
# Execute the eval
uv run python -m tests.evals.synthesis.synthesis_features_eval

# (Optional) Aggregate results
uv run python - << 'PY'
import json, statistics as st, sys, glob
from pathlib import Path
path = sorted(Path('eval_results/synthesis_features').glob('*/judgments.jsonl'))[-1]
recs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
prefs = [r['judgment']['preference'] for r in recs]
rate = sum(p == 'TREATMENT' for p in prefs)/len(prefs)
print('Treatment preference rate:', rate)
PY
```

## Observability & Artifacts

- Logs include synthesis feature flags per run (already implemented in the workflow).
- Save per‑topic artifacts in run folder:
  - Control/Treatment reports (typed JSON)
  - Judge outputs (scores, preference, rationale)
  - Optional: merged CSV with per‑criterion deltas.

## Risks & Mitigations

- LLM judge variance: Use a stronger model and/or multiple votes per topic.
- Cost/latency: Limit sections or number of topics; batch judge calls if needed.
- Non‑determinism: Keep model parameters constant and run Control/Treatment back‑to‑back per topic.

## Acceptance Criteria

- Treatment preferred ≥ 60% of the time and shows positive mean deltas for Readability and Thematic Clarity.
- No meaningful regression in Evidence Integration (citations).
- Evaluation executes end‑to‑end with artifacts stored and a summary script producing headline metrics.
