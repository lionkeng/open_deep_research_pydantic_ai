# HTTP Mode API Alignment Plan

## Background
The CLI currently runs reliably in direct (in-process) mode. When invoked with `--mode http`, it expects a FastAPI backend that mirrors the same streaming workflow, report lifecycle, and clarification handling that direct mode provides. The existing FastAPI app in `src/api/main.py` predates the latest workflow refactors (two-phase clarification, event bus progress, refined report models), so it no longer matches the CLI’s expectations. This plan documents the required updates so developers can restore feature parity.

## Objectives
- Reuse the same `ResearchWorkflow` orchestration in both CLI and HTTP code paths.
- Ensure the FastAPI service exposes endpoints and payloads that the CLI (and future clients) can rely on.
- Provide robust session/state management so SSE streams, status polling, and report retrieval stay in sync even under concurrent load.
- Support clarification pauses and resumptions end-to-end via HTTP.
- Deliver clear, validated responses and actionable error messages to API consumers.

## Current API Gaps
1. **Workflow Integration** (`src/api/main.py:16`, `:152-198`)
   - The server imports a non-existent `workflow` singleton and calls `execute_research`. The new workflow surface is `ResearchWorkflow().run(...)` and `resume_research(...)`. As a result, background tasks fail immediately, and the CLI never receives progress events.
2. **Session Storage & Concurrency** (`src/api/main.py:73-221`)
   - `active_sessions` is shared mutable state without consistent locking. Background tasks and HTTP requests can race, producing inconsistent views or partially mutated `ResearchState` instances.
   - Initial state never transitions beyond `PENDING`, so consumers cannot observe accurate stage progress.
3. **API Contracts** (`src/api/main.py:54-145` vs `src/models/api_models.py`)
   - Local `ResearchRequest`/`ResearchResponse` classes duplicate and drift from the canonical models. Responses omit the expected `stream_url`/`report_url`, and payload validation differs from CLI expectations.
   - Incoming API key blobs aren’t normalised into `APIKeys`, so secret validation and downstream services break.
4. **Clarification Flow** (`src/api/main.py:289-395`)
   - Clarification metadata is partially surfaced, but background execution does not pause correctly, and resumptions reuse outdated workflow calls. There is no restart of the workflow after storing clarification responses.
5. **Status & Report Endpoints** (`src/api/main.py:201-285`)
   - `/research/{id}` omits clarification status, completion timestamps, and fails to reflect `FAILED` terminal states properly.
   - `/research/{id}/report` always returns `200` and lacks retry hints when reports are still generating, leading to confusing CLI output.
6. **SSE Stream Coupling** (`src/api/sse_handler.py`)
   - SSE streaming largely matches the event bus, but relies on the background task updating the same `ResearchState` instance stored in `active_sessions`. Without shared mutation, SSE data becomes stale.

## Detailed Work Plan

### 1. Replace Legacy Workflow Usage
- Instantiate a single `ResearchWorkflow` at module load (e.g., `_workflow = ResearchWorkflow()`), and use it inside request handlers.
- In `execute_research_background`, call `_workflow.run(...)` with:
  - `user_query`, `api_keys`, and the generated `request_id`.
  - `stream_callback=True` to emit SSE-compatible events.
  - `conversation_history=None` initially, but pass stored messages when resuming.
- Maintain a reference to the same `ResearchState` object in `active_sessions` before the task starts, so the workflow mutates it in place.
- On completion, update `active_sessions[request_id]` to the final state; on error, ensure `state.set_error(...)` is called and stored.

### 2. Harden Session Storage & Lifecycle
- Guard all reads/writes to `active_sessions` with `_sessions_lock` (including status, report, clarification, and cancellation endpoints).
- When starting a session, call `state.start_research()` before saving so consumers see `CLARIFICATION` as soon as analysis begins.
- After completion or failure, optionally prune old sessions (configurable retention) to avoid unbounded growth. Emit logfire breadcrumbs on create/update/delete for observability.
- Ensure cancellation sets the state to `FAILED` and stores the final timestamp.

### 3. Align Request/Response Models
- Import `ResearchRequest` and `ResearchResponse` from `models.api_models` (or create dedicated server variants that wrap them) to avoid divergence.
- Adjust FastAPI route annotations to use these models, honouring stricter field validation (query length, optional `max_search_results`, etc.).
- Convert incoming `request.api_keys` dictionaries into an `APIKeys` instance, handling validation errors (return `422` with details when formats are wrong).
- Return `ResearchResponse(status="accepted", stream_url=..., report_url=...)` after kicking off the background task so clients immediately know where to connect/poll.
- Add consistent error responses using `ErrorResponse` for server failures.

### 4. Support Clarification Pause/Resume
- In `_workflow.run`, clarification may set `metadata.clarification.awaiting_clarification`. Detect this in `execute_research_background` and exit early without marking the research complete so the session remains in a waiting state.
- Expose clarification metadata in `/research/{id}` responses: include `awaiting_clarification`, outstanding questions, and any partial answers.
- When `POST /research/{id}/clarification` succeeds, spawn a new background task that calls `_workflow.resume_research(state, api_keys=..., stream_callback=True)`. Reuse the same `ResearchState` object to preserve stage history and event subscriptions.
- Capture and forward any resume errors to the client, updating the state to `FAILED` when necessary.

### 5. Enrich Status & Report Endpoints
- Extend the status payload to include:
  - `current_stage`, `is_completed`, `started_at`, `completed_at`, `error_message`.
  - `clarification_pending` flag and question summary when applicable.
  - Optional `progress` or recent event metadata if available (for future UI use).
- For `/research/{id}/report`:
  - If the report is not ready, return `202 Accepted` with a `Retry-After` header and a JSON body explaining the current stage.
  - When ready, return the full `ResearchReport.model_dump()` to match CLI display helpers.
  - Handle failed states by returning `409 Conflict` (or `400`) with the error message so clients can surface failures cleanly.

### 6. Testing & Validation Checklist
- **Unit / Integration**
  - Add or update tests in `tests/api/` to simulate full HTTP flows, including SSE subscription (can be mocked), clarification loops, and report retrieval.
  - Validate API key parsing paths with good/bad inputs.
- **Manual QA**
  - Run `uvicorn` locally and execute `uv run deep-research "<query>" --mode http` to exercise CLI streaming, clarification prompts, and save-to-file flow.
  - Hit `/research/{id}` and `/research/{id}/report` with `curl` during different stages to verify status codes and payloads.
- **Regression Checks**
  - Ensure direct CLI mode still passes `uv run pytest`, `uv run ruff check`, and `uv run pyright`.
  - Verify SSE remains responsive under slow clients (consider adding timeouts/backoff tests).

### 7. Deployment & Rollout Notes
- Update any API documentation (e.g., `docs/system_architecture.md`, `docs/api.md`) to reflect new endpoints and schema changes.
- Communicate dependency requirements (`uv add --optional cli` for HTTP mode) so developers have SSE dependencies installed.
- Plan for migrating any existing long-running sessions if the server is restarted mid-research; consider persisting `active_sessions` in Redis in future iterations.

## Acceptance Criteria
- CLI HTTP mode completes a full research run, including clarification, without errors using the updated FastAPI backend.
- SSE stream mirrors the same sequence of events as direct mode, and the CLI progress UI updates correctly.
- Status and report endpoints provide meaningful, accurate data for polling clients.
- Background task errors are surfaced via structured API responses and logfire logging.
- All quality gates (`uv run ruff check`, `uv run pyright`, `uv run pytest`) pass after the refactor.
