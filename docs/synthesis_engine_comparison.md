# Synthesis Engine Comparison

This document compares the “synthesis” approaches used by:
- This repository (Pydantic‑AI–based, linear workflow)
- LangChain’s Open Deep Research (LangGraph–based, supervisor/sub‑agent workflow)

The comparison is grounded in the code of both implementations.

## High‑Level Summary

- Pydantic‑AI
  - Deterministic, typed pipeline in a single pass within `ResearchExecutorAgent`.
  - Primarily deterministic utilities (no classical ML), now with optional embedding‑assisted similarity for convergence and contradiction gating. Pipeline extracts findings, clusters themes, detects contradictions, finds patterns, generates an executive summary, assesses quality, and assembles typed outputs.
  - References: `src/agents/research_executor.py`, `src/services/synthesis_tools.py`.

- LangChain Open Deep Research
  - LLM‑driven, graph‑orchestrated loops in LangGraph: a supervisor delegates to researcher subgraphs, researchers compress findings via an LLM, and a final LLM generates the report.
  - References: `open_deep_research/src/open_deep_research/deep_researcher.py`, `open_deep_research/src/open_deep_research/prompts.py`, `open_deep_research/src/open_deep_research/state.py`.

## Orchestration Model

- Pydantic‑AI: Single linear pass
  - Flow: Extract → Cluster → Contradictions → Patterns → Executive Summary → Quality → Typed `ResearchResults`.
  - Event‑driven progress via `core/events.py`.

- LangChain: Supervisor/researcher loops
  - Nodes: `write_research_brief` → `supervisor` (with tools: `ConductResearch`, `think_tool`, `ResearchComplete`) → researcher subgraphs → `compress_research` → `final_report_generation`.
  - Iterative planning and delegation controlled by the graph.

## What the “Engine” Does

- Pydantic‑AI (deterministic utilities + optional embeddings)
  - Similarity (token/Jaccard) with boosts for phrase/term matches: `src/services/synthesis_tools.py`.
  - Embedding‑assisted similarity for convergence via cosine + threshold clustering when enabled; caller precomputes vectors and passes them in for determinism and efficiency.
  - Specificity/recency heuristics (regex for numbers, dates, percentages, proper nouns; exponential time decay): `src/services/synthesis_tools.py`.
  - Pattern detection (temporal/causal/correlative) and contradiction checks (temporal/quantitative/negation). Contradiction detection can use precomputed vectors for topic‑gated pairing.
  - Assembles typed `ResearchResults` with sources, findings, clusters, patterns, contradictions, summary, quality metrics: `src/agents/research_executor.py`.
  - No classical ML models or training; embeddings are used only for vector similarity, not fine‑tuned models.

- LangChain (LLM prompts + tool loops)
  - “Compression” prompt turns researcher tool outputs and AI messages into cleaned, comprehensive notes with citations (LLM‑driven): `prompts.py` (compress prompts), `deep_researcher.py` (compress node).
  - Final report prompt synthesizes the report from the brief, all messages, and compressed findings: `prompts.py` (final report prompt), `deep_researcher.py` (final node).
  - Supervisor/researcher loops collect notes via tools and MCP/websearch integrations.

## Strengths

- Pydantic‑AI
  - Determinism & testability: same inputs yield the same outputs; easier unit testing and regression checks.
  - Strong typing: `ResearchResults` and `ResearchReport` enforce structure and invariants.
  - Robust citations: `[Sx]` markers → numbered footnotes + `metadata.source_summary`; explicit source usage tracking.
  - Transparent provenance: degraded source registration path and in‑memory repository make data lineage explicit.
  - Improved semantic grouping when embeddings are enabled, while keeping deterministic ordering by precomputing vectors and aligning them with sampled claims.

- LangChain
  - Semantic coverage: LLM “compression” and final report prompts can capture nuance and produce fluent, cohesive prose.
  - Iterative exploration: supervisor loops allow re‑planning, deeper searches, and redirection via tool calls.
  - Built‑in parallelism: multiple researcher subgraphs can run concurrently.

## Weaknesses

- Pydantic‑AI
  - Limited semantics: token/regex heuristics may miss subtle relationships, entailments, or nuanced causal structure.
  - Narrative cohesion is delegated to a later report agent; without an LLM “clean‑merge” pass, prose may be less cohesive.
  - Manual tuning: thresholds/regex patterns can be domain‑sensitive; no learned representation.

- LangChain
  - Non‑determinism: retries and context truncation for token limits can lead to run‑to‑run variation; harder to regression‑test.
  - Prompt fragility: citation behavior and coverage depend on prompt compliance; no structural enforcement.
  - Auditability: without a typed citation post‑processor, guarantees about consistent, numbered footnotes are weaker.

## Quality & Trust

- Pydantic‑AI
  - Strong provenance via explicit source IDs, usage, and deterministic footnote mapping; audit‑friendly.
  - Risk of under‑recall or missing semantic links compared to LLM text synthesis.

- LangChain
  - Likely improved topical coverage and narrative cohesion from LLM synthesis.
  - Greater risk of inconsistent citation behavior or subtle hallucinations if prompts are not perfectly adhered to.

## Performance & Cost

- Pydantic‑AI
  - Lower latency/cost after search; synthesis is CPU‑bound heuristics.
  - Embedding usage is bounded and efficient: vectors precomputed once where needed; heavy vector operations offloaded to background threads; convergence uses caps/sampling to avoid O(n^2) blowups.

- LangChain
  - More LLM tokens (supervisor + researchers + compression + final report), leading to higher latency/cost; retries for token limits add overhead.

## Operational Characteristics

- Pydantic‑AI
  - Clean eventing and typed artifacts for CI/deployment dashboards.
  - Circuit‑breaker wrappers and fallback logic around agents.
  - Async‑only convergence API with precomputed vectors; CPU‑heavy work offloaded with `asyncio.to_thread` to keep the event loop responsive.
  - HTTP/SSE resilience: server heartbeats every ~15s; client retries on SSE read timeouts with a helpful message; direct mode is in‑process and doesn’t require heartbeats.

- LangChain
  - Flexible control flow observable in LangGraph Studio.
  - Structured‑output retries present in graph (e.g., `.with_retry`).

## Fit & Use Cases

- Prefer Pydantic‑AI when you need:
  - Determinism, reproducibility, typed artifacts, strong citation auditing.
  - Lower inference cost and easier testing at scale.

- Prefer LangChain when you need:
  - Maximum semantic depth and richer prose from LLM‑driven synthesis.
  - Iterative research with supervisor planning and parallel branches.

## Practical Hybrid Options

- Add embedding‑based similarity to improve theme/pattern grouping in the deterministic pipeline.
  - Implemented: convergence and contradiction gating use embedding vectors when enabled, with precompute alignment and bounded runtime.
- Keep typed/cited backbone but add an optional, guardrailed LLM “clean‑merge” step to improve narrative flow.
- Retain deterministic post‑processing for citations even if an LLM writes spans (e.g., require emitting `[Sx]` markers and map them to footnotes programmatically).

## References

- Pydantic‑AI synthesis:
  - `src/agents/research_executor.py`
  - `src/services/synthesis_tools.py`
  - `src/agents/report_generator.py`
- LangChain Open Deep Research synthesis:
  - `open_deep_research/src/open_deep_research/deep_researcher.py`
  - `open_deep_research/src/open_deep_research/prompts.py`
  - `open_deep_research/src/open_deep_research/state.py`
