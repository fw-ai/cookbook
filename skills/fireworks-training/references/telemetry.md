# Training skill journey telemetry

Training skill interaction telemetry uses one remote analytics path:

```text
agent
  → private run manifest
  → authenticated Fireworks journey endpoint
  → PostHog
```

BigQuery is not part of journey event delivery. Fireworks may separately join
the session UUID to authoritative training job and billing outcomes.

## Data contract

PostHog receives three event names:

| Event | Purpose |
|---|---|
| `training skill question answered` | Question ID and bounded option ID |
| `training skill milestone reached` | Skill funnel progression |
| `training skill session ended` | Terminal session outcome |

Every event includes a stable event UUID, session UUID, skill source and
version, agent surface, entry skill, schema version, and occurrence time.
Identity and environment are derived by Fireworks after authentication.

Question and answer IDs map to analyst-facing prompts and labels in the
versioned contract. Question events may include an agent-written task summary.
It is limited to 200 characters and omitted when it resembles a path, URL,
email, credential, or multiline text. Raw customer messages are never sent.

## Record a question answer

Show [`telemetry-notice.md`](telemetry-notice.md) once before the first
structured question. Check command availability once:

```bash
firectl skill-journey record --help
```

When available, call after the answer:

```bash
EVENT_ID="$(python3 -c 'import uuid; print(uuid.uuid4())')"
OCCURRED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
firectl skill-journey record \
  --event-id "$EVENT_ID" \
  --occurred-at "$OCCURRED_AT" \
  --event-type question_answered \
  --entry-skill research \
  --agent-surface cursor \
  --question-id research-q1 \
  --answer-id rag \
  --response-source ask_question \
  --task-summary "RAG retrieval returns the wrong policy article"
```

Set `FIREWORKS_SESSION_ID` and `FIREWORKS_CLIENT_SOURCE` for the whole run.
Record each event ID and occurrence time in the private manifest before sending
it. A retry must reuse both values.
The normal `firectl` account and API key configuration supplies authenticated
identity. The command never receives a PostHog credential.

For a free text reply, map it to the nearest bounded option ID and set
`response_source` to `free_text` or `inferred_from_message`. Do not send the
original prose.

## Record milestones and session outcome

```bash
firectl skill-journey record \
  --event-id "<event-uuid>" \
  --occurred-at "<RFC3339-UTC>" \
  --event-type milestone_reached \
  --entry-skill configure \
  --agent-surface cursor \
  --milestone configure_plan_approved
```

```bash
firectl skill-journey record \
  --event-id "<event-uuid>" \
  --occurred-at "<RFC3339-UTC>" \
  --event-type session_ended \
  --entry-skill configure \
  --agent-surface cursor \
  --session-outcome job_created
```

## Failure behavior

The command retries temporary remote errors three times and returns nonzero
when Fireworks does not accept the event. Telemetry failure must not block
training. If the user opts out, set `TELEMETRY_OPT_OUT=true`; the command sends
nothing.

Use `--dry-run` to validate and print the bounded payload without sending it.
If the installed `firectl` predates `skill-journey`, recommend upgrading it.
Until it is available, record the bounded decision in the private run manifest
and continue without remote telemetry.

The command normalizes `FIREWORKS_API_BASE` to its origin, so values ending in
`/inference` or `/inference/v1` still reach the journey endpoint.

## Local manifest

The private run manifest remains the session resume record and degraded copy.
Do not create a separate telemetry file. Do not commit or share a manifest
without the user's approval.

For local fallback:

1. Reject a manifest path that is a symlink.
2. Create `fireworks-training-runs/` with mode `0700`.
3. Create manifests atomically with mode `0600`.
4. Ensure `fireworks-training-runs/.gitignore` preserves existing rules and
   contains `*` plus `!.gitignore`.
5. If private local state cannot be written safely, skip recording and continue
   the customer workflow.
