---
name: debug
description: >-
  Diagnose Fireworks training and deployment issues, including stuck or failed
  jobs, error messages, poor quality, checkpoint and resume problems, and
  deployments that are ready but not serving. Use for systematic read-only
  triage before retrying. Use configure for a new or approved retry run; use
  research when the starting method, data, or cookbook example is unclear.
---

# Debug

`Debug`: triage a stuck or failed Fireworks training run.

Debug is read-only by default. Do not create jobs, upload data, or spend without
an explicit handoff to `configure` and a new approved plan.

## Entry routing

| Signal | Route |
|---|---|
| New training or approved retry | `configure` |
| Unclear method, data, or starting example | `research` |
| Failure, stall, bad quality, resume, or serving issue | stay in `debug` |

Research owns the `welcome-entry` question and its privacy notice. Route a
vague first message there instead of asking a second welcome question.

## Attribution and privacy

Reuse the run's `FIREWORKS_SESSION_ID`. If none exists, create one UUID. Set:

```bash
export FIREWORKS_CLIENT_SOURCE="fireworks-training-skill/2.2.0"
```

Record `entry_skill: debug` in the private run manifest. Before the first
structured question, show
`../fireworks-training/references/telemetry-notice.md`.

Check `firectl skill-journey record --help` once. When available, record only
registered IDs through `../fireworks-training/references/telemetry.md`. If it is
unavailable or the user opts out, keep local run state and continue.

Never send raw errors, customer prose, credentials, datasets, or paths as
journey telemetry.

## Triage

1. Read `references/triage-paths.md`.
2. Ask one category question: job state, error, quality, resume/checkpoint, or
   deploy/serving.
3. Stop and wait for the answer.
4. Follow the ordered read-only checks for that category.
5. Use the detailed carrier references listed below.

## Three-strike rule

After three failed hypotheses on the same issue:

1. Stop guessing.
2. Build an escalation bundle with UTC timestamps, resource IDs, model and
   shape, CLI and SDK versions, cookbook commit, retry history, evidence, and
   what was ruled out.
3. Redact credentials and customer data.
4. State what the evidence supports and what remains unknown.

## Handoff

1. Route to `configure` only when the user wants a new or retry run.
2. Route to `research` when the starting example or method is wrong.
3. On resolution without new spend, record `debug_resolved` and stop.

## Progressive references

| Need | Reference |
|---|---|
| Triage categories | `references/triage-paths.md` |
| Error catalog | `../fireworks-training/references/error-reference.md` |
| Resume and idempotency | `../fireworks-training/references/run-state-and-reporting.md` |
| Deployment proof | `../fireworks-training/references/deploy-and-troubleshoot.md` |
| Journey telemetry | `../fireworks-training/references/telemetry.md` |

## Interaction rules

1. Use AskQuestion one question at a time, then stop.
2. Use plain customer language.
3. The category question uses its five registered options.
4. Do not retry or create resources from debug.
