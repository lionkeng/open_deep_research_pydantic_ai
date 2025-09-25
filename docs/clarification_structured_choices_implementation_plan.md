# Clarification Questions: Structured Choices Implementation Plan

This plan migrates our clarification UX and validators away from brittle, label-based parsing to a robust, structured, ID-based model. This is a breaking change with no backward compatibility or legacy support retained.

## Problem Statement

- LLM-generated labels are non-deterministic (punctuation, dashes, parentheses, commas).
- Current validators rely on string splitting and keyword detection to infer semantics (e.g., "Other (please specify)").
- Multi-select answers encoded as delimited label strings break when labels contain commas or formatting variations.

## Goals

- Stable, machine-readable selections using IDs, not labels.
- Explicit flags for choices requiring typed details (no label guessing).
- A clean UX: always prompt for details when required and attach them to the specific selection.
- Backward compatibility with existing payloads through a transition period.

## Non-Goals

- Overhauling unrelated parts of the research flow.

---

## High-Level Design

Introduce a structured choice model and structured answers. Remove all legacy string encodings and label-heuristics. Validators only accept structured answers; UI only emits structured answers.

### New/Extended Models

- ClarificationChoice
  - id: str (stable, unique within a question)
  - label: str (human-readable)
  - requires_details: bool = False
  - is_other: bool = False
  - details_prompt: str | None = None

- ClarificationQuestion (existing)
  - question_type: "text" | "choice" | "multi_choice"
  - choices: list[ClarificationChoice] | None (strings no longer accepted)

- ClarificationAnswer (extend)
  - For text questions: text: str
  - For choice questions: selection: { id: str, details?: str }
  - For multi_choice: selections: list[{ id: str, details?: str }]
  - Remove legacy `answer: str` usage for non-text questions

### Validation Logic

- Build a map[id -> ClarificationChoice] per question.
- For choice: validate `selection` (id exists; details present when required).
- For multi_choice: validate each entry in `selections` similarly.
- Reject any non-structured answers.

### UI (CLI) Changes

- ask_choice_question
  - Render `label` to the user; store `id` in the result.
  - If requires_details or is_other, prompt once with `details_prompt` (or a default) and store details.

- ask_multi_choice_question
  - Render checkable list by `label`; collect selected IDs.
  - For selections requiring details, prompt inline, one-by-one.
  - Return `selections: [{id, details?}]` (no delimited strings).

- review_interface
  - Look up labels by ID for display.
  - Show `details` inline beneath the selection when present.

### Agent Prompt Adjustments

- Update the clarification agent system prompt so the LLM emits structured choices with: id, label, requires_details, is_other.
- Add a strict validator for agent output; do not accept string-only choices.

---

## Rollout Plan (Breaking Change)

Single Release (No Compatibility Layer)
- Models & Validators
  - Add `ClarificationChoice` and require `choices: list[ClarificationChoice]` for non-text questions.
  - Add `text`, `selection`, `selections` fields to `ClarificationAnswer`. Remove `answer` for choice/multi_choice.
  - Remove all label-based heuristics and string splitting from validators.

- CLI UX
  - `ask_choice_question`/`ask_multi_choice_question` return structured `selection(s)` only.
  - Update `ClarificationResponse` creation to populate structured fields exclusively.

- Review UI
  - Display selections by ID→label lookup; show details inline.

- Agent
  - Enforce structured output in prompt. Validate output shape and fix IDs if missing by generating UUIDs.

- API
  - Accept and expect structured `ClarificationResponse` bodies. Reject legacy `answer` fields for non-text questions.
  - Update conversation logging to format from structured fields.

---

## Detailed Tasks by File

Models & Validation
- src/models/clarification.py
  - Add `ClarificationChoice` model.
  - Update `ClarificationQuestion` to require structured choices for non-text.
  - Extend `ClarificationAnswer` with `text`, `selection`, `selections`; disallow legacy `answer` for non-text.
  - Update `validate_against_request` to validate by IDs and required details only.
  - Add helpers: `needs_details(choice)`.

CLI UX
- src/interfaces/cli_multi_clarification.py
  - Update rendering to show labels, but store IDs.
  - Prompt for details when needed using `details_prompt or default`.
  - Return `selection`/`selections` in `ClarificationAnswer` creation.

Review UI
- src/interfaces/review_interface.py
  - Resolve labels by ID for display.
  - Show details inline; handle wrapping and truncation appropriately.

HTTP Client
- src/cli/http_client.py
  - Ensure `model_dump(mode="json")` includes structured fields. No endpoint changes expected.

Agent
- src/agents/clarification.py
  - Update system prompt to instruct structured choice output.
  - Add normalizer that converts legacy pattern labels into flags as a fallback.

Tests
- tests/unit/models/test_clarification_models.py
  - Add tests for structured choices and answers (choice and multi_choice), including details-required.
- tests/integration/test_multi_clarification.py
  - Verify CLI collects structured answers and validators pass.
- tests/acceptance/test_final_validation.py
  - Ensure end-to-end flows succeed with structured-only flow.

Docs
- Update README and developer docs to describe the new structure and the migration timeline.

---

## Data & Serialization Considerations

- JSON payload shape: `ClarificationAnswer` contains `text` (text type) or `selection`/`selections` (choice types).
- Remove legacy `answer` usage for non-text questions in both client and server.

---

## Error Handling & UX

- Clear error messages referencing choice IDs and labels (resolve IDs back to labels for user-facing errors).
- For details-required selections, the prompt should not allow continuing until non-empty details are provided (for required questions).
- Maintain skip semantics for optional questions.

---

## Telemetry

- Log structured selection counts per question type.
- Emit metrics for how often `requires_details` is triggered and completion rates.
- Track fallback to legacy parsing to know when it’s safe to remove it.

---

## Risks & Mitigations

- Inconsistent or missing IDs from LLM output
  - Mitigation: generate UUIDs for any missing IDs server-side in the Request before sending to clients.
- Downstream consumers assuming `answer: str`
  - Mitigation: Update all consumers to use structured fields; remove all uses of `answer: str` for non-text.

---

## Rollout Plan & Timeline

- Week 1: Implement structured models, validators, CLI, and agent prompt. Update tests.
- Week 2: Remove all legacy code paths, update API conversation formatting, and ship.

Success Criteria
- 0% validation failures due to label punctuation/format.
- No reliance on label keyword heuristics in core validation.
- Tests cover structured path thoroughly; legacy path remains green until removal.

---

## Example Payloads

Structured (choice):
```
{
  "question_id": "q1",
  "selection": { "id": "industry_bodies", "details": null },
  "skipped": false
}
```

Structured (multi_choice with details):
```
{
  "question_id": "q5",
  "selections": [
    { "id": "industry_bodies" },
    { "id": "other", "details": "Company disclosures" }
  ],
  "skipped": false
}
```

Legacy: Not supported. Submissions must use structured selections.

---

## Implementation Checklist

- [ ] Add ClarificationChoice model
- [ ] Extend ClarificationAnswer with structured fields; remove legacy non-text `answer`
- [ ] Implement structured-only validators
- [ ] Implement CLI selection by ID + details prompts
- [ ] Update review interface to display by ID + details
- [ ] Adjust agent prompt to emit structured choices
- [ ] Add/expand unit, integration, and acceptance tests
- [ ] Remove all legacy parsing and string heuristics

---

## Notes

- Keep line length and typing standards per project defaults.
- Do not change unrelated behavior; keep diffs minimal and focused.
