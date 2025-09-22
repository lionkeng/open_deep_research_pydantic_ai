# CLI HTTP Mode – Clarification Flow Implementation Plan

## Executive Summary
The CLI currently supports two modes:
- Direct mode (in‑process workflow) – already handles clarification via local interactive UI.
- HTTP mode (client/server) – streams SSE events but does not fetch, display, or submit clarification prompts.

Goal: Add end‑to‑end clarification handling for HTTP mode without changing direct mode behavior. We will poll the server for pending prompts, present questions in the terminal, submit answers, and resume streaming to completion.

## Constraints & Non‑Goals
- Preserve direct mode behavior and UX; do not modify direct‑mode clarification flow.
- Contain all new logic to the HTTP path in `src/cli.py` and small, HTTP‑specific helpers.
- Avoid server API changes (use existing endpoints):
  - `GET /research/{request_id}/clarification`
  - `POST /research/{request_id}/clarification`
- No new dependencies beyond the existing `cli` extra (httpx‑sse already required).

## User Flow (HTTP Mode)
1. CLI starts research via `POST /research` and begins streaming SSE.
2. In parallel, the CLI periodically checks `/clarification`.
3. If `awaiting_response=true`, the CLI renders the questions, collects answers, and posts them.
4. Streaming continues; upon completion, the CLI fetches the final report.

## High‑Level Design
- Concurrency: Run SSE streaming as an `asyncio.Task` while a companion loop polls for clarification status.
- Isolation: Introduce HTTP‑mode only helpers; leave direct mode code paths untouched.
- Reuse existing CLI question UI from `interfaces/cli_multi_clarification.py` for consistent UX.

## Files Touched
- `src/cli.py` (HTTP mode branch only)
- Tests (new): `tests/test_cli_http_clarification.py` (mocks)
- Docs (this file), plus brief updates to `docs/API_REFERENCE.md` and `README.md`

## Detailed Implementation Steps

### 1) HTTP client helpers (isolated to HTTP mode)
Add the following methods to `HTTPResearchClient` in `src/cli.py`:
- `async def get_clarification(self, request_id: str) -> dict[str, Any]`:
  - GET `${base_url}/research/{request_id}/clarification`
  - Returns a JSON dict compatible with `ClarificationStatusResponse`:
    - `{ request_id, state, awaiting_response, clarification_request, original_query }`
  - 404 → no pending clarification; treat as not awaiting
- `async def submit_clarification(self, request_id: str, response: ClarificationResponse | dict[str, Any]) -> dict[str, Any]`:
  - POST `${base_url}/research/{request_id}/clarification` with JSON body
  - Returns `ClarificationResumeResponse` on success
  - 400 with validation errors → surface to user and re‑prompt
  - 409 → already processed; continue streaming

Type notes:
- Accept a `ClarificationResponse` Pydantic model or a plain dict; call `.model_dump()` if available.

### 2) Clarification orchestration helper (HTTP‑only)
Add a helper in `src/cli.py`:
- `async def handle_http_clarification_flow(client: HTTPResearchClient, request_id: str, console: Console) -> None`:
  - Loop until stream completes or no longer pending:
    - Poll `get_clarification()` with a short delay/backoff (e.g., 1s → 2s → 3s, max 5s).
    - If `awaiting_response` is True:
      - Extract `clarification_request` and `original_query`.
      - Render and collect answers via `interfaces/cli_multi_clarification.handle_multi_clarification_cli(...)`.
      - If user cancels: show a message; continue polling (user can answer later). Optionally offer a future cancel action (non‑blocking for now).
      - Submit answers using `submit_clarification()`.
      - On 400 with errors: display, re‑prompt; on 409: treat as already handled.
  - Exit conditions:
    - SSE task has completed (tracked externally), or
    - `get_report()` succeeds, or
    - The server responds 404 (no pending) consistently while the stream continues.

Notes:
- Do NOT alter the global event subscriptions or handler logic used by direct mode.
- Keep progress UI minimal during prompting; pause status updates if needed.

### 3) Integrate into HTTP branch of `run_research()`
Modify only the HTTP code path in `src/cli.py`:
- Start SSE streaming with retry as a background task:
  - `stream_task = asyncio.create_task(client.stream_events_with_retry(request_id, handler))`
- In parallel, run `handle_http_clarification_flow(client, request_id, console)`.
- Await `stream_task` completion (or use `asyncio.wait` to detect whichever finishes first), then fetch and display the report (existing code).

Direct mode remains exactly as is.

### 4) Non‑interactive environments (graceful handling)
- Detect lack of TTY (optional); if interactive input is not possible:
  - Print instructions to answer via external client in the future (or re‑run in interactive terminal).
  - Continue streaming and allow the server to time out if no input is provided.

### 5) Error handling & edge cases
- GET `/clarification`:
  - 404 → treat as not pending; keep polling while stream is active.
- POST `/clarification`:
  - 400 → validation errors; display unified messages (from server’s error payload) and re‑prompt.
  - 409 → already processed; continue streaming.
- SSE disconnects:
  - Existing `stream_events_with_retry` handles reconnection; keep polling loop independent.
- Clarification timeout:
  - Server sets session state to `CLARIFICATION_TIMEOUT`; we show a warning and continue streaming.

### 6) Testing
- Unit tests with mocks for HTTP client:
  - `get_clarification()` returns `awaiting_response=true` → ensure prompting path invoked.
  - Validation error on POST → verify re‑prompt behavior.
  - 409 already processed → verify we continue.
  - 404 on GET → verify we do not prompt.
- “Integration‑style” test (mocked client + fake stream task):
  - Simulate: start stream → pending clarification → submit → resume → completion.
- Ensure direct mode tests still pass (no changes to direct path).

### 7) Documentation
- This plan (you are reading it): `docs/cli_http_mode_clarification_plan.md`.
- Update `docs/API_REFERENCE.md` to note CLI now actively polls `/clarification` and posts answers in HTTP mode.
- Update README “Web API / CLI” section: add short usage note for HTTP mode clarification behavior.

### 8) Rollout & Validation
- Manual validation sequence:
  1) Start server: `uv run deep-research-server`
  2) Run CLI (HTTP mode): `uv run deep-research research "<query>" --mode http`
  3) When clarification needed, answer prompts in terminal.
  4) Observe stream resuming and final report display.
- Quality checks (unchanged):
  - `uv run ruff check src tests`
  - `uv run pyright src`
  - `uv run pytest`

## Isolation & No‑Regression Strategy
- All changes confined to HTTP branch of `src/cli.py` and new HTTP client helpers.
- Direct mode code path, event subscriptions, and local interactive flow (`interfaces/clarification_flow.py`) remain unchanged.
- No shared state alterations that affect direct mode.
- Tests explicitly cover HTTP mode orchestration while keeping existing tests intact.

## Acceptance Criteria
- CLI (HTTP mode) successfully:
  - Detects pending clarification, renders questions, accepts answers, and submits to server.
  - Resumes streaming and completes the workflow.
  - Handles validation errors, idempotency (409), and timeouts gracefully.
- Direct mode behavior remains unchanged and tests continue to pass.

## Risks & Mitigations
- Race between stream completion and polling → gate by stream task completion; stop polling once `complete` arrives.
- Non‑interactive terminals → warn and proceed; consider a future `--clarification-file` option.
- Server API shape drift → we use the documented models; add light input validation and error messaging.

## Future Enhancements (Optional)
- SSE “clarification_pending” event to eliminate polling.
- `--clarification-file` for headless environments.
- Richer UI (e.g., multi‑select UX) for choice questions.
