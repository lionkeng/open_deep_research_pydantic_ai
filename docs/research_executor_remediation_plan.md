# Research Executor Remediation Plan

## Overview

This plan addresses the gaps identified in `src/agents/research_executor.py` relative to the hybrid architecture defined in `docs/research_executor_consolidated_plan.md` and `docs/research_executor_implementation_gaps_and_plan.md`. The focus is to align the agent with the pydantic-ai framework standards and enable deterministic access to supporting services.

## Phase 0: Clarify Open Questions ✅

**Structured Output Path**
- `ResearchWorkflow._execute_research_stages` assigns whatever the research executor returns to `research_state.findings` and immediately reads `len(findings.key_findings)` (`src/core/workflow.py:476-485`). There is no adapter for raw strings—downstream stages assume a structured object with `key_findings`, `executive_summary`, etc.
- Current typing mismatch (`ResearchState.findings` is `list[ResearchFinding]`) confirms we must update the model layer alongside the agent so the stored value becomes a `ResearchResults` instance rather than free-form text. No feature flag or compatibility layer exists today.

**Search Results Delivery**
- The workflow only enriches dependencies with the `SearchQueryBatch` (`src/core/workflow.py:190-211`); nothing fetches real search results before invoking the agent. `ResearchDependencies` exposes `search_queries` but not `search_results` (`src/agents/base.py:127-141`).
- Inside the agent module, `ResearchExecutorDependencies` expects a `search_results` list and builds services around it (`src/agents/research_executor.py:38-66`), but `execute_research()` is currently called with raw search results passed in directly and never receives the batch from the workflow.
- Conclusion: we must integrate the `SearchOrchestrator` (or another retrieval layer) inside the executor stage, populate `search_results`, and then inject those details into the model context via instructions/user messages. Until that plumbing exists the agent has nothing to synthesize.

*Owner: research executor primary contributor*

## Phase 1: Adopt Base Agent Infrastructure

1. **Class Refactor**
   - Convert the standalone `research_executor_agent` into a `BaseResearchAgent` subclass mirroring the pattern used by `QueryTransformationAgent` and `ReportGeneratorAgent`.
   - Ensure the constructor specifies `output_type=ResearchResults` (or `result_type` depending on API) so pydantic-ai enforces schema validation.
2. **Dependency Handling**
   - Define a concrete `ResearchExecutorDependencies` type compatible with `BaseResearchAgent` expectations (`ResearchDependencies` or equivalent).
   - Wire default services via dependency injection (cache, metrics) and allow overrides through config.
3. **Conversation & Configuration**
   - Use `AgentConfiguration` to load model, retries, and prompt templates consistently.
   - Register instruction hooks (`@self.agent.instructions`) to format query metadata, optimized prompt, and conversation history.

*Deliverable*: new agent class under `src/agents/research_executor.py`, retired module-level `Agent` instance.

## Phase 2: Tool Registration & Service Wiring

1. **Register Tools**
   - Decorate `extract_hierarchical_findings`, `identify_theme_clusters`, `detect_contradictions`, and `analyze_patterns` with `@self.agent.tool` (or equivalent) to expose them to the model.
   - Ensure function signatures conform to pydantic-ai expectations (typed arguments, dependency access via `RunContext`).
2. **Synthesis Engine Integration**
   - Provide cohesive fallbacks that return `HierarchicalFinding` objects directly (avoid dict handling). Implement missing service methods (e.g., `SynthesisEngine.extract_themes`) or adjust calls to match existing API.
3. **Executive Summary & Quality Tools**
   - Register `generate_executive_summary` and `assess_synthesis_quality` as tools or helper methods invoked post-run, consistent with design documents.

*Deliverable*: All core synthesis helpers accessible as tools with unit coverage ensuring tool registration.

## Phase 3: Prompt & Context Construction

1. **System Prompt**
   - Implement the comprehensive synthesis prompt (Tree of Thoughts, quality gates) as described in the consolidated plan.
2. **Dynamic Instructions**
   - Inject `search_results`, query metadata, and any research objectives into the conversation via instruction callback or initial user message chunking.
   - Apply caching/parallel execution hints from plan-phase docs.
3. **Usage of `search_results`**
   - Define a serialization strategy (e.g., summary digest) to keep context within token limits while conveying essential data to GPT.

*Deliverable*: Verified prompt content and context injection path aligned with doc requirements.

## Phase 4: Execution Orchestration

1. **`execute_research` Workflow**
   - Update the public `execute_research` function to instantiate the new agent class, set dependencies, and forward structured results.
   - Ensure it respects configuration (model selection, caching, metrics) and handles retries via the base agent.
2. **Search Orchestration Alignment**
   - Confirm compatibility with `SearchOrchestrator` / workflow expectations (`src/core/workflow.py`). Update to pass `search_results` and metadata explicitly.

*Deliverable*: Public API returning validated `ResearchResults` objects under integration tests.

## Phase 5: Testing & Validation

1. **Unit Tests**
   - Expand `tests/unit/agents/test_enhanced_research_executor.py` to cover:
     - Tool registration discovery
     - Structured output enforcement
     - Context injection logic
   - Add service-level tests for any newly implemented synthesis engine methods.
2. **Integration Tests**
   - Update `tests/integration/test_research_executor_integration.py` or create a new end-to-end test ensuring the agent processes real search result fixtures and returns populated models.
3. **Regression Safeguards**
   - Verify compatibility with existing workflow tests (`tests/unit/core/test_workflow_consolidation.py`).

*Deliverable*: Passing unit/integration suite; add regression for previously failing scenarios.

## Risk Management

- **Prompt Size**: Large system prompt + serialized search results may exceed token budgets. Mitigate via summarization and selective inclusion.
- **Service Contract Changes**: Adjusting tool signatures may ripple through mocks. Coordinate updates to test fixtures to match new method signatures.
- **Migration**: Introduce feature flags/config toggles if switching from free-form output to strict models risks breaking downstream consumers.

## Timeline & Ownership

1. Phase 0-1: 1-2 days (core agent refactor) — *Assigned to agent implementation owner*
2. Phase 2-3: 2-3 days (tools + prompt + context) — *Synthesis tooling engineer*
3. Phase 4-5: 2 days (orchestration + testing) — *Workflow maintainer*

Total estimated effort: 5-7 days with cross-team collaboration.

## Acceptance Criteria

- The research executor returns a validated `ResearchResults` instance for representative queries.
- All four synthesis tools operate through pydantic-ai with deterministic service integration.
- Integration tests demonstrate end-to-end synthesis over sample search results.
- Documentation updated to reflect new agent architecture and configuration.
