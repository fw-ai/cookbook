# Training skill journey telemetry schema

Bounded fields for welcome, research interview, configure path choice, and session
outcomes.

**Capture user responses** (which options they pick, how they answered, a short
redacted task summary). **Do not capture data artifacts** (datasets, file paths,
chat transcripts, training payloads).

Canonical transport:

1. **Phase 1 (now):** local run manifest (`fireworks-training-runs/<run-id>/run.md`)
2. **Phase 2:** `POST /v1/skill-journey-events` when API key is set and user has
   not opted out (see `telemetry/journey-api-spec.md`)

Show the customer notice once per session: `telemetry-notice.md`.

Join key: `skill_session_id` (same as `FIREWORKS_SESSION_ID`).

## Session envelope

| Field | Type | Notes |
|---|---|---|
| `skill_session_id` | UUID | Set at welcome or first skill touch |
| `skill_client_source` | string | `fireworks-training-skill/2.2.0` |
| `journey_schema_version` | int | `2` |
| `skill_version` | semver | From `.claude-plugin/plugin.json` |
| `entry_skill` | enum | `welcome` \| `research` \| `configure` \| `debug` (legacy: `discover`) |
| `agent_surface` | enum | `cursor` \| `claude_code` \| `codex` \| `unknown` |
| `account_id` | string | When authenticated; omit if unknown |
| `telemetry_opt_out` | bool | User declined server-side funnel events |

## User response log (per answer)

After **each** AskQuestion (or free-text routing), append one object to
`intake_responses[]` in the manifest and include the same object in the Phase 2
event `properties`.

| Field | Type | Notes |
|---|---|---|
| `question_id` | string | Stable ID, e.g. `welcome-entry`, `research-q1`, `configure-q-path` (legacy: `discover-q1`) |
| `user_choice` | string | Option ID from the question table (same as enum field value) |
| `response_source` | enum | `ask_question` \| `free_text` \| `inferred_from_message` |
| `answered_at_utc` | string | ISO-8601 UTC |
| `recommended_option` | string | Option ID marked recommended, if any |

When the user replies in free text instead of picking an option, set
`response_source: free_text` or `inferred_from_message` and still set
`user_choice` to the mapped option ID after you infer it.

## Task summary (bounded free text — responses, not data)

| Field | Type | Notes |
|---|---|---|
| `task_summary` | string | ≤200 chars; agent-redacted description of the **task** |

Rules for `task_summary`:

- Describe the problem shape in plain language ("parse receipts to JSON", "RAG
  returns wrong policy article").
- **No** file paths, dataset names, account IDs, model IDs, or pasted user text.
- Update once after the opening message; refresh only if the user corrects the task.

## Welcome (`welcome.md`)

| Field | Values |
|---|---|
| `welcome_choice` | `research` \| `configure` \| `debug` \| `inferred_from_text` (legacy: `discover` → `research`) |
| `welcome_skipped` | bool — user had unambiguous intent |

Event: `welcome_answered`

## Research interview (`research/references/interview-questions.md`)

Record after **each** AskQuestion answer (event: `research_intake_answered`; legacy:
`discover_intake_answered`) plus `intake_responses[]`. At readiness package (event:
`research_recommendation_presented`; legacy: `discover_recommendation_presented`).

| Field | Question | Enum values |
|---|---|---|
| `intake_q1_task_shape` | Q1 | `structured_output` \| `tone` \| `reasoning` \| `rag` \| `multitenancy` \| `unsure` |
| `intake_q1b_shape` | Q1b | `extraction` \| `classification` \| `template` \| `unsure` |
| `intake_q2_data` | Q2 | `labeled` \| `raw_only` \| `preference_pairs` \| `scored_prompts` \| `query_doc` \| `exploring` |
| `intake_q_eval` | Q-eval | `exact_match` \| `retrieval_metric` \| `win_rate` \| `verifier_score` \| `human_rubric` \| `unsure` |
| `intake_q3_path` | Q3 | `managed` \| `serverless` \| `dedicated` \| `no_preference` |
| `intake_q4_tried` | Q4 | `prompting` \| `rag` \| `other_vendor` \| `greenfield` |
| `domain_followup` | domain table | `sec_finance` \| `invoices_vision` \| `tone_prefs` \| `multilora` \| `none` |
| `cookbook_entry_tier` | readiness | `case_study` \| `example` \| `recipe` |
| `cookbook_entry_path` | readiness | string |
| `dataset_plan` | readiness | object (source, candidates) |
| `eval_plan` | readiness | object (metric, baseline_required, hook) |
| `questions_asked` | count | int |
| `questions_inferred_skipped` | count | int |
| `completion_gate_bypassed` | bool | skip questions / named slug |
| `matched_case_study` | slug | case study slug or `none` |
| `match_confidence` | enum | `high` \| `medium` \| `low` |
| `handoff_choice` | Handoff | `plan_configure` \| `readme_first` \| `labeling_help` \| `defer` |

Event: `research_handoff_answered` (legacy: `discover_handoff_answered`).

## Configure path (`configure/references/path-intake.md`)

| Field | Values |
|---|---|
| `workflow_path` | `managed_firectl` \| `managed_sdk` \| `serverless` \| `dedicated` |
| `execution_surface` | `firectl` \| `sdk` \| `training_api` |
| `inherited_from_research` | bool (legacy: `inherited_from_discover`) |
| `method` | `sft` \| `dpo` \| `orpo` \| `rft` \| `embedding` |

Event: `configure_path_answered`

## Session outcome (terminal)

| Field | Values |
|---|---|
| `session_outcome` | `research_only` \| `research_to_configure` \| `configure_plan_only` \| `job_created` \| `job_completed` \| `debug_triage` (legacy: `discover_*`) |
| `terminal_at_utc` | timestamp |

Event: `session_terminal`

## Manifest block (Phase 1)

Append to `fireworks-training-runs/<run-id>/run.md`:

```yaml
## Journey telemetry

journey_schema_version: 2
telemetry_opt_out: false
task_summary:   # ≤200 chars, redacted task shape only

welcome_choice:
welcome_skipped:

intake_responses:
  - question_id: discover-q1
    user_choice: rag
    response_source: ask_question
    answered_at_utc: 2026-09-03T20:00:00Z
    recommended_option:

intake_q1_task_shape:
intake_q1b_shape:
intake_q2_data:
intake_q3_path:
intake_q4_tried:
domain_followup:
questions_asked:
questions_inferred_skipped:
completion_gate_bypassed:
matched_case_study:
match_confidence:
handoff_choice:
workflow_path:
execution_surface:
inherited_from_discover:
method:
session_outcome:
```

Update fields after each AskQuestion. Omit unset keys.

## Phase 2 API call (when `FIREWORKS_API_KEY` is set and not opted out)

```bash
curl -sS -X POST "${FIREWORKS_API_BASE:-https://api.fireworks.ai}/v1/skill-journey-events" \
  -H "Authorization: Bearer ${FIREWORKS_API_KEY}" \
  -H "Content-Type: application/json" \
  -H "X-Fireworks-Client-Source: ${FIREWORKS_CLIENT_SOURCE}" \
  -H "X-Fireworks-Client-Session-Id: ${FIREWORKS_SESSION_ID}" \
  -d '{
    "event": "discover_intake_answered",
    "journey_schema_version": 2,
    "properties": {
      "question_id": "discover-q1",
      "user_choice": "rag",
      "response_source": "ask_question",
      "task_summary": "support search returns wrong governing article"
    }
  }'
```

Degraded: manifest-only when API key is unset or `telemetry_opt_out: true`.

SDK helper: `telemetry/sdk-firectl-helpers.md`. Server contract:
`telemetry/journey-api-spec.md`. Jarvis tiles: `telemetry/jarvis-funnel-tiles.md`.

## Privacy boundary

| In scope (responses) | Out of scope (data) |
|---|---|
| Option IDs, question IDs, response source | Dataset files, JSONL rows, labels |
| Redacted `task_summary` ≤200 chars | Full chat transcript |
| `intake_responses[]` audit trail | Local paths, buckets, secrets |
| Session enums and funnel outcomes | Eval payloads, error strings, API keys |

`customer_goal` prose may live in the **local** manifest for orchestration when
the user approves; do **not** copy it into server journey events.
