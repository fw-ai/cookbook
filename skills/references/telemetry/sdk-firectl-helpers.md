# SDK and firectl journey event helpers (Phase 2)

Thin clients for `POST /v1/skill-journey-events`. Skills call these after each
AskQuestion when `FIREWORKS_API_KEY` is set; otherwise write manifest only
(degraded mode).

## Python SDK (`fireworks-ai`)

Proposed surface (follow PR #118 attribution pattern):

```python
from fireworks import Fireworks

client = Fireworks(api_key=os.environ["FIREWORKS_API_KEY"])

client.skill_journey.record(
    session_id=os.environ["FIREWORKS_SESSION_ID"],
    event="discover_intake_answered",
    journey_schema_version=2,
    entry_skill="discover",
    agent_surface="cursor",
    properties={
        "question_id": "discover-q1",
        "user_choice": "rag",
        "response_source": "ask_question",
        "task_summary": "support search returns wrong governing article",
    },
)
```

Implementation notes:

- Reuse `X-Fireworks-Client-Source` and `X-Fireworks-Client-Session-Id` from
  existing training attribution.
- Validate `properties` keys against `telemetry-schema.md` enums in SDK (dev
  warning only; server is authoritative).
- No-op gracefully when endpoint returns 404 (older control plane).

## firectl

```bash
firectl skill-journey record \
  --event discover_intake_answered \
  --property intake_q1_task_shape=rag \
  --property questions_asked=1
```

Requires `FIREWORKS_SESSION_ID` and `FIREWORKS_CLIENT_SOURCE` in the environment
(same as training jobs).

Alias: `firectl skill-event record` (if shorter name preferred).

## Agent skill usage

After each AskQuestion answer in discover/configure/welcome:

1. Update `fireworks-training-runs/<run-id>/run.md` journey block and
   `intake_responses[]` (always).
2. If `FIREWORKS_API_KEY` is set and `telemetry_opt_out` is not true, call SDK
   or `firectl skill-journey record` with `question_id`, `user_choice`,
   `response_source`, and redacted `task_summary`.
3. On API failure, log once and continue — never block the customer flow.
4. On session end, emit `session_terminal` with `session_outcome`.

### `session_outcome` mapping

| Situation | `session_outcome` |
|---|---|
| User leaves after discover, no handoff | `discover_only` |
| Handoff to configure, no job yet | `discover_to_configure` |
| Configure plan shown, no approval | `configure_plan_only` |
| Job created | `job_created` |
| Job completed | `job_completed` |
| Debug triage only | `debug_triage` |

## Shell helper (cookbook dogfood)

Until SDK ships, agents may use curl from `telemetry-schema.md` Phase 2 example.
Keep payloads ≤1 KB.

## Testing

1. Set `FIREWORKS_API_KEY`, `FIREWORKS_SESSION_ID`, `FIREWORKS_CLIENT_SOURCE`.
2. Run discover smoke prompt; answer Q1 + Q2.
3. Verify manifest journey block in `fireworks-training-runs/`.
4. After Phase 2 deploy: verify row in `analytics.training_skill_journey_events`.
