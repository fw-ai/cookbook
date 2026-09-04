---
name: research
description: >-
  Plan Fireworks post-training from a task-shaped goal — interview the user,
  scan the full cookbook (case studies, examples, recipes), propose data and
  eval plans, and hand off to configure when ready to train. Use when the user
  describes a problem in plain language, is unsure which method or dataset fits,
  needs public dataset ideas, or wants to explore before committing. Read-only —
  does not create jobs or upload data. For explicit train requests with known
  model and data, use configure; for failures use debug.
---

# Research

> **New here?** Customers should read [`../GETTING-STARTED.md`](../GETTING-STARTED.md)
> after install — open a new chat and describe the goal.

Help the customer **understand what to train, on what data, and how to measure
it** before they spend GPU. This skill is the interview-driven front door (like
Tinker `research`). Execution lives in **configure**; failures live in **debug**.

## What research finds (and what it does not)

| Research answers | Configure answers (later) |
|---|---|
| **Implied method** (SFT, DPO, RFT, embedding) from supervision signal | Exact method confirmation + hyperparameters |
| **Cookbook entry** — case study, example, or recipe path | Workflow surface (`firectl` vs SDK vs Training API) |
| **Dataset plan** — local, bundled, HF candidates, labeling schema | Upload, column mapping, job create |
| **Eval plan** — metric class, baseline, cookbook eval hook | Run baseline eval, wire evaluators |
| **Suggested path** (managed, serverless, dedicated) — coarse | Cost, model choice, monitor, deploy |

Research does **not** pick a final base model, **estimate cost**, or create jobs. If
the user already named model + JSONL + method, skip research and go to
**configure**.

**Order:** research → configure → (debug if needed).

## Research methodology

Follow `references/methodology.md`. In short:

1. **Inspect the cookbook first** — read `references/cookbook-catalog.md` and
   open the closest README before asking questions the catalog can answer.
2. **Interview one question at a time** — `references/interview-questions.md`;
   propose options grounded in what you read; let the user revise.
3. **Eval before train** — every readiness package includes how success will be
   measured and whether a baseline run is required.
4. **Propose, then approve** — present one recommendation; user can ask for
   alternatives or stay in research for labeling/data help.
5. **Hand off only when ready** — configure starts after explicit handoff choice.

## What this skill does

1. Show the **Research** skill banner every turn.
2. Scan the full cookbook catalog (not only case studies).
3. Run the **interview** — one **AskQuestion** per turn until the completion
   gate in `references/interview-questions.md` passes.
4. Optional **Hugging Face dataset search** (privacy gate) when data is missing.
5. Write a **readiness package** to the run manifest and hand off to
   **configure** via AskQuestion.

## What this skill does not do

- Create training jobs, upload datasets, or run mutating `firectl`.
- Choose hyperparameters or final model (configure owns that).
- Debug failed runs (use **debug**).

## Workflow

### 0. First turn — welcome

If `../references/welcome.md` applies, show the welcome block and **Entry**
AskQuestion. STOP. Do not open the research banner or interview until the user
picks **Research** or describes a task to explore.

### 1. Attribution

```bash
export FIREWORKS_SESSION_ID="$(python -c 'import uuid; print(uuid.uuid4())')"
export FIREWORKS_CLIENT_SOURCE="fireworks-training-skill/2.2.0"
```

Record `entry_skill: research` when writing a run manifest.

Optional: if the user will run `firectl` catalog reads or wants journey
telemetry, ensure `FIREWORKS_API_KEY` is set via
[`../references/api-key-setup.md`](../references/api-key-setup.md) — never in
chat.

### 1b. Journey telemetry

After **each** AskQuestion answer, update the journey block in
`fireworks-training-runs/<run-id>/run.md` using option IDs from
`references/interview-questions.md`. Schema:
[`../references/telemetry-schema.md`](../references/telemetry-schema.md).

- Append to `intake_responses[]` with `question_id`, `user_choice`,
  `response_source`, `answered_at_utc`.
- Set `task_summary` once (≤200 chars, redacted).
- Emit `research_intake_answered` when API key set and not opted out (legacy
  event name `discover_intake_answered` still accepted server-side).
- On recommendation: `research_recommendation_presented` (legacy:
  `discover_recommendation_presented`).
- On handoff: `research_handoff_answered` (legacy: `discover_handoff_answered`).
- If user defers: `session_outcome: research_only` (legacy: `discover_only`).
- Show `../references/telemetry-notice.md` before the first interview question
  when welcome was skipped.

### 2. Skill banner

First line of every response (see `references/output-template.md`):

`**Research** — exploring method, data, eval, and cookbook starting points.`

### 3. Inspect, then interview

Read `references/cookbook-catalog.md`. Open the best candidate README or example
README. Run `references/interview-questions.md` — **one** AskQuestion per turn.
STOP and wait. Do not recommend until the **completion gate** passes.

### 4. Present the readiness package

Use `references/output-template.md`:

- Best cookbook entry (tier + path) and why.
- Dataset plan (local / bundled / HF / labeling).
- Eval plan (metric, baseline, notebook eval hook or gap).
- Runner-up only when genuinely close.

### 5. Hand off to configure

Fire the **Handoff** AskQuestion. On `plan_configure`, write the handoff block
from `references/cookbook-catalog.md` into `fireworks-training-runs/<run-id>/run.md`,
then continue in **configure**.

Never tell the user to paste a canned configure prompt.

## Progressive references

| Need | Reference |
|---|---|
| Full cookbook index | `references/cookbook-catalog.md` |
| Interview scripts and completion gate | `references/interview-questions.md` |
| Methodology (inspect → interview → eval) | `references/methodology.md` |
| Turn shapes and readiness package | `references/output-template.md` |
| HF dataset search (gated) | `references/external-dataset-discovery-draft.md` |
| Case study slugs (subset) | `references/case-studies.md` |
| Design notes (v2.2 reframe) | `references/reframe-v2-draft.md` |
| Welcome (shared) | `../references/welcome.md` |
| Telemetry | `../references/telemetry-schema.md` |

## Cross-skill routing

| Signal | Route to |
|---|---|
| User wants to train, deploy, or resume with known plan | **configure** |
| User reports failure, stuck job, or error | **debug** |
| User still exploring data, eval, or cookbook | stay in **research** |
