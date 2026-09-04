---
name: debug
description: >-
  Diagnose Fireworks training and deployment issues — stuck jobs, failed runs,
  error messages, poor quality after training, resume and checkpoint problems,
  and deployment not serving. Use when the user reports training is slow, hanging,
  failed, errored, or producing bad outputs; when they ask if Fireworks is down;
  or when they need systematic triage before retrying. For planning and running
  new training use configure; for exploring method and data use research.
---

# Debug

> **New here?** Customers start at [`../GETTING-STARTED.md`](../GETTING-STARTED.md).
> For failures, they describe the symptom in a new chat — the agent routes here.

Systematic triage for Fireworks training failures. **Read-only by default** —
do not create jobs, upload data, or spend without explicit user approval to
retry in **configure**.

## Entry routing

| Signal | Route to |
|---|---|
| User wants to start or continue a new training plan | **configure** |
| User unsure which cookbook entry or dataset fits | **research** |
| Failed, stuck, error, bad quality, resume, deploy issue | stay in **debug** |

## Workflow

### 0. First turn — welcome

If `../references/welcome.md` applies (vague first message, no entry chosen yet),
show the welcome block and **Entry** AskQuestion. STOP. Do not start triage until
the user picks **Debug** or describes a clear failure symptom.

### 1. Attribution

Reuse `FIREWORKS_SESSION_ID` from the run under investigation when known.
Otherwise generate a new UUID. Always set:

```bash
export FIREWORKS_CLIENT_SOURCE="fireworks-training-skill/2.2.0"
```

If `firectl whoami` fails, give the user
[`../references/api-key-setup.md`](../references/api-key-setup.md) before
read-only triage.

Record `entry_skill: debug` in the run manifest.

### 2. Triage category

Read `references/triage-paths.md`. Fire **one AskQuestion** to pick the
problem category (job state, error, quality, resume, deploy). STOP and wait.

### 3. Follow the path

Execute the ordered steps for that category in `references/triage-paths.md`.
For error strings and symptom tables, read **configure** → `error-reference.md`.
Same for resume (`run-state-and-reporting.md`) and deploy (`deploy-and-troubleshoot.md`).

### 4. Three-strike rule

After three failed hypotheses, stop and build an escalation bundle (see
triage-paths). Do not loop indefinitely.

### 5. Hand off

- **configure** — only when the user explicitly approves a new or retry run.
- **research** — when the root issue is "wrong starting example" not a bug.

## Progressive references

| Need | Reference |
|---|---|
| Triage categories and ordered steps | `references/triage-paths.md` |
| Full error catalog and escalation checklist | configure skill `error-reference.md` |
| Resume and idempotency | configure skill `run-state-and-reporting.md` |
| Deploy and serving proof | configure skill `deploy-and-troubleshoot.md` |
| First-turn welcome (shared) | `../references/welcome.md` |

## GUI rules

- Use **AskQuestion** — one question at a time, then STOP.
- Plain customer language — no internal jargon.
- Max 4 options per question.
