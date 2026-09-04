# Skill journey events API (Phase 2 spec)

**Status:** design — implement in `fw-ai/fireworks` control plane.  
**Owner:** training skill / DevEx. **Consumer:** Jarvis `training_skill_funnel`.

## Endpoint

```
POST /v1/skill-journey-events
```

Authenticated with the same API key as `firectl` and the Python SDK.

### Request headers

| Header | Value |
|---|---|
| `Authorization` | `Bearer <api_key>` |
| `Content-Type` | `application/json` |
| `X-Fireworks-Client-Source` | `fireworks-training-skill/<version>` |
| `X-Fireworks-Client-Session-Id` | UUID (must match body `skill_session_id`) |

### Request body

```json
{
  "event": "discover_intake_answered",
  "journey_schema_version": 2,
  "skill_session_id": "8f3c2a1b-4d5e-6f7a-9b0c-1d2e3f4a5b6c",
  "skill_client_source": "fireworks-training-skill/2.1.0",
  "entry_skill": "discover",
  "agent_surface": "cursor",
  "properties": {
    "question_id": "discover-q1",
    "user_choice": "rag",
    "response_source": "ask_question",
    "task_summary": "support search returns wrong governing article"
  }
}
```

### Event types

| `event` | When |
|---|---|
| `welcome_answered` | Entry AskQuestion answered or welcome skipped |
| `discover_intake_answered` | Each discover AskQuestion answer |
| `discover_recommendation_presented` | Completion gate passed; case study shown |
| `discover_handoff_answered` | Handoff AskQuestion answered |
| `configure_path_answered` | Q-path or Q-method answered |
| `session_terminal` | Session ends (including `discover_only`) |

### `properties` schema

Enum fields only — see [`../telemetry-schema.md`](../telemetry-schema.md).
Reject requests with unknown keys. Allow `task_summary` ≤200 chars;
`intake_responses` entries must use registered `question_id` values. Reject file
paths, dataset references, and strings matching path/secret patterns.

### Response

`201 Created` with `{ "event_id": "<uuid>" }`.

### Errors

| Code | Meaning |
|---|---|
| `400` | Invalid enum, schema version mismatch, session ID format |
| `401` | Missing or invalid API key |
| `429` | Rate limit (suggest 60 events/session, 10 sessions/min/account) |

## Privacy and audit

- Do **not** log request body to raw audit logs.
- Curated view `analytics.training_skill_journey_events` exposes enums + account_id
  + timestamps only (mirror `training_skill_events` privacy model).
- No prompts, datasets, paths, or error strings.

## BigQuery view

```sql
CREATE OR REPLACE VIEW `fw-ai-cp-prod.analytics.training_skill_journey_events` AS
SELECT
  event_timestamp,
  account_id,
  user_id,
  client_session_id AS skill_session_id,
  client_source AS skill_client_source,
  event_name,
  journey_schema_version,
  entry_skill,
  agent_surface,
  -- enum property columns (flattened from properties JSON)
  JSON_VALUE(properties, '$.intake_q1_task_shape') AS intake_q1_task_shape,
  JSON_VALUE(properties, '$.intake_q2_data') AS intake_q2_data,
  JSON_VALUE(properties, '$.matched_case_study') AS matched_case_study,
  JSON_VALUE(properties, '$.handoff_choice') AS handoff_choice,
  JSON_VALUE(properties, '$.workflow_path') AS workflow_path,
  JSON_VALUE(properties, '$.session_outcome') AS session_outcome
FROM `fw-ai-cp-prod.analytics.skill_journey_events_raw`
WHERE client_source LIKE 'fireworks-training-skill/%';
```

Join to `analytics.training_skill_events` on `client_session_id`.

## Rollout

1. Ship endpoint behind internal flag.
2. Dogfood from cookbook skills with `FIREWORKS_API_KEY`.
3. Enable Jarvis funnel provider (`telemetry/jarvis-funnel-tiles.md`).
4. PostHog mirror for 14–30 day parity window.
5. Document in public API reference when stable.
