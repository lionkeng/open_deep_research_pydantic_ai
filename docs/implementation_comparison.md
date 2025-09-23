# Implementation Comparison: Pydantic‑AI Deep Research vs. LangChain Open Deep Research

This document summarizes concrete differences between the Pydantic‑AI‑based implementation and the LangChain Open Deep Research project.

## High‑Level Orchestration

- Pydantic‑AI implementation

  - Linear, 4‑stage pipeline orchestrated by `ResearchWorkflow` (src/core/workflow.py):
    1. Clarification → 2) Query Transformation → 3) Research Execution → 4) Report Generation.
  - Agents are created via `AgentFactory` (src/agents/factory.py) and run once per stage.
  - Stage progress and status are emitted as immutable events via `core/events.py` and consumed by the CLI and HTTP SSE server.

- LangChain implementation
  - Graph‑based orchestration using LangGraph (StateGraph) with supervisor/sub‑agents (open_deep_research/src/open_deep_research/deep_researcher.py).
  - Nodes include `clarify_with_user`, `write_research_brief`, `supervisor`, researcher tool loops, and terminal `ResearchComplete`.
  - Flow uses `Command(goto=...)` transitions between nodes and tools rather than an event bus.

## Clarification Flow

- Pydantic‑AI

  - Clarification is a two‑phase flow inside `ResearchWorkflow` (src/core/workflow.py): a Clarification agent produces a structured `ClarificationRequest` and the workflow either prompts the user (CLI) or pauses and exposes HTTP endpoints for answers (src/api/main.py, src/api/core/clarification/handler.py).
  - HTTP mode: pending prompts are available at `GET /research/{id}/clarification`; answers are posted to `POST /research/{id}/clarification`. The server resumes the workflow after answers.

- LangChain
  - `clarify_with_user` determines whether to ask a clarifying question using a structured output model `ClarifyWithUser` (open_deep_research/src/open_deep_research/state.py) and ends the node with a user question when needed. The rest of the flow proceeds to `write_research_brief` when clarification is not required.

## Query Transformation / Research Plan

- Pydantic‑AI

  - `QueryTransformationAgent` outputs a typed `TransformedQuery` that includes both a `SearchQueryBatch` and a `ResearchPlan` (src/agents/query_transformation.py; src/models/research_plan_models.py).
  - The workflow stores the transformed object on `research_state.metadata.query.transformed_query` and logs the number of plan objectives (src/core/workflow.py).

- LangChain
  - `write_research_brief` generates a structured `ResearchQuestion`/brief and initializes the supervisor with prompts (open_deep_research/src/open_deep_research/deep_researcher.py). A typed research plan model analogous to `ResearchPlan` is not evident; the brief is textual and used to guide supervisor planning.

## Research Execution & Synthesis

- Pydantic‑AI

  - `ResearchExecutorAgent` runs a synthesis pipeline over search results, producing typed `ResearchResults` with findings, clusters, contradictions, patterns, and quality metrics (src/agents/research_executor.py).
  - Search execution is coordinated by `SearchOrchestrator` with caching and retry (src/services/search_orchestrator.py) and `WebSearchService` (src/services/search.py).

- LangChain
  - Research is organized under a supervisor node that plans and delegates work using tools (`ConductResearch`, `think_tool`) to researcher loops (open_deep_research/src/open_deep_research/deep_researcher.py; state definitions in open_deep_research/src/open_deep_research/state.py). Tool definitions and their exact implementations are connected via utilities and LangChain tool initialization.

## Report Generation & Citations

- Pydantic‑AI

  - `ReportGeneratorAgent` composes a typed `ResearchReport` (src/agents/report_generator.py; src/models/report_generator.py). It collects `[Sx]` markers, converts them to numbered footnotes, populates `report.references`, and writes a `metadata.source_summary` with IDs/URLs. HTTP/CLI save paths render timestamps and footnotes.

- LangChain
  - The final report is composed within the graph using prompts (`final_report_generation_prompt`) and stored as text in state (open_deep_research/src/open_deep_research/deep_researcher.py; open_deep_research/src/open_deep_research/state.py). Footnote processing analogous to the Pydantic‑AI post‑processor is not evident from the inspected files.

## Eventing & Streaming

- Pydantic‑AI

  - Emits events (`StageStartedEvent`, `StageCompletedEvent`, `StreamingUpdateEvent`, `ResearchCompletedEvent`) via an async event bus (src/core/events.py). HTTP SSE server (src/api/sse_handler.py) translates events to SSE messages consumed by the CLI HTTP client.
  - CLI direct mode subscribes to the event bus and renders Rich progress (src/cli/stream.py). HTTP mode uses SSE client streaming (src/cli/http_client.py).

- LangChain
  - Execution and UI are integrated with LangGraph Studio; orchestration is observed through the LangGraph dev server (open_deep_research/README.md). An explicit event bus or SSE adapter is not shown in the core graph implementation.

## Source Handling

- Pydantic‑AI

  - Sources are registered in an `InMemorySourceRepository` and validated via `SourceValidationPipeline` with degraded fallback on failure (src/services/source_repository.py; src/services/source_validation.py). Findings carry `source_ids`; report generator maps markers to footnotes and updates usage.

- LangChain
  - The graph integrates search tools (e.g., Tavily; MCP/websearch paths are referenced in utilities). A centralized source repository and degraded registration path analogous to the Pydantic‑AI pipeline are not apparent from the inspected code.

## HTTP Interface & CLI

- Pydantic‑AI

  - FastAPI server exposes: `POST /research`, `GET /research/{id}/stream` (SSE), `GET /research/{id}/report`, and clarification endpoints (src/api/main.py). CLI provides direct and HTTP modes (src/cli).

- LangChain
  - The project runs with the LangGraph dev server and LangGraph Studio (`langgraph dev`), configurable via `langgraph.json` and `configuration.py` (open_deep_research/README.md). A separate FastAPI layer or CLI analogous to this repo’s is not part of the checked core implementation.

## Resilience & Metrics

- Pydantic‑AI

  - Per‑agent circuit breaker wrappers and fallbacks (src/core/workflow.py; src/core/circuit_breaker.py). Performance metrics fields exist on `BaseResearchAgent` (src/agents/base.py). Logging via Logfire throughout.

- LangChain
  - Uses LangGraph execution/runtime and LangChain model/tool abstractions. Checkpointing packages are present in dependencies (open_deep_research/uv.lock). The core graph shows retries via `.with_retry` on structured outputs.

## Summary

- Pydantic‑AI implementation favors a typed, linear pipeline with explicit eventing, HTTP/CLI integration, and strongly typed research artifacts (TransformedQuery, ResearchResults, ResearchReport).
- LangChain Open Deep Research uses a supervisor/sub‑agent graph with delegated tool loops, a textual research brief, and graph‑driven control flow managed by LangGraph. It integrates with LangGraph Studio for execution and observation.

This comparison focuses on what is directly visible in the code paths listed above and avoids speculation where implementation details are not present in the inspected files.
