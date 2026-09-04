# Journey telemetry notice (show once per session)

Show this **once** at welcome or before the first discover AskQuestion — after
the welcome block, before the Entry AskQuestion. Keep it short; do not block the
workflow if the user does not acknowledge it.

## Required notice (customer-facing)

Use this block verbatim (minor formatting ok):

```text
**Privacy note:** To improve routing, we may record which intake options you
pick and a short redacted summary of your task (for example "RAG retrieval for
support articles"). We do **not** send training data, dataset files, file paths,
API keys, or full chat transcripts. With `FIREWORKS_API_KEY` set, aggregate
funnel events are sent to Fireworks; otherwise answers stay in your local
`fireworks-training-runs/` manifest only.
```

## When to show

| Situation | Show notice? |
|---|---|
| First welcome in a new chat | Yes — once, before Entry AskQuestion |
| User skipped welcome (clear intent) | Yes — one line before first intake question |
| Returning thread with manifest already created | No |
| User says "don't track" / opts out | Skip server events; still write local manifest unless they ask to skip entirely |

## Opt-out

If the user declines tracking:

1. Do **not** call the journey API (Phase 2).
2. Still update the local manifest if they continue (orchestration state).
3. Set `telemetry_opt_out: true` in the journey block.
4. Do not ask again in the same thread.

## What we capture (responses — yes)

| Captured | Example |
|---|---|
| AskQuestion **option ID** the user picked | `rag`, `plan_configure` |
| **Question ID** answered | `discover-q1`, `welcome-entry` |
| **Response source** | `ask_question` \| `free_text` \| `inferred_from_message` |
| **Task summary** | ≤200 chars, agent-redacted task shape | "support search returns wrong policy article" |
| Per-answer **timestamp** (UTC) | ISO-8601 |

Append each answer to `intake_responses[]` in the manifest (see
`telemetry-schema.md`).

## What we never capture (data — no)

| Never send | Why |
|---|---|
| Dataset files, JSONL rows, labels | Customer data |
| Local file paths, bucket URLs | May contain PII or secrets |
| Model names tied to proprietary projects | Optional in manifest; omit from server events unless user approves |
| Full opening message / chat transcript | Use redacted `task_summary` instead |
| Eval metrics, error payloads, API keys | Security / noise |

## Gstack comparison (internal)

gstack separates **remote telemetry** (opt-in: skill name, duration, outcome only)
from **local question logs** (`user_choice`, `question_id`, optional short
`free_text` for preferences). We follow the same split: structured **responses**
yes, **data artifacts** no. We add a visible notice (like gstack's search privacy
gate) instead of a separate consent CLI.
