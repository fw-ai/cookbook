# Jarvis funnel tiles — discover intake (Phase 2)

**Status:** design — add to Company Dashboard **Training agent skill** section or
new `training_skill_funnel` API.

**Depends on:** `analytics.training_skill_journey_events` (see
`journey-api-spec.md`).

## API (proposed)

```
GET https://company-dashboard.fireworks-ai.cloud/api/training_skill_funnel?days=30
```

Valid windows: 7, 30, 90 (match `training_skill_insights`).

## Tiles

| Tile ID | Label | Definition |
|---|---|---|
| `discover_sessions` | Discover sessions / week | Distinct `skill_session_id` with `entry_skill=discover` or any `discover_*` event |
| `intake_completion_rate` | Intake completion rate | Sessions with `intake_q1_task_shape` AND `intake_q2_data` ÷ discover sessions started |
| `case_study_mix` | Case study recommendations | Count by `matched_case_study` where event = `discover_recommendation_presented` |
| `handoff_rate` | Handoff to configure | `handoff_choice=plan_configure` ÷ `discover_recommendation_presented` |
| `discover_to_job` | Discover → job conversion | Sessions with `session_outcome` in (`job_created`, `job_completed`) ÷ discover sessions (7d attribution window) |
| `path_preference` | Configure path mix | Count by `workflow_path` where event = `configure_path_answered` |
| `drop_off_step` | Last intake step | Mode of last non-null `intake_*` field before `session_outcome=discover_only` |

## Health fields

| Field | Meaning |
|---|---|
| `journey_data_available` | BQ journey view reachable |
| `observable_journey_sessions` | Distinct sessions with ≥1 journey event |
| `discover_only_sessions` | Terminal `discover_only` count |
| `joined_to_api_sessions` | Journey sessions also in `training_skill_events` |

## Example HogQL / BigQuery

**Intake funnel by task shape:**

```sql
SELECT intake_q1_task_shape,
       COUNT(DISTINCT client_session_id) AS sessions,
       COUNTIF(handoff_choice = 'plan_configure') AS handed_off
FROM `fw-ai-cp-prod.analytics.training_skill_journey_events`
WHERE event_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
GROUP BY 1
ORDER BY sessions DESC;
```

**Discover-only vs converted:**

```sql
SELECT session_outcome, COUNT(*) AS sessions
FROM `fw-ai-cp-prod.analytics.training_skill_journey_events`
WHERE event_name = 'session_terminal'
  AND event_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
GROUP BY 1;
```

## Dashboard placement

Add a **Discover funnel** subsection below existing **Training agent skill**
tiles in Training Analytics → Overall. Link to Jarvis Connection Guide.

**Owners:** Terry (spec) · Charlie/Arkhan (dashboard) · control-plane (BQ view).
