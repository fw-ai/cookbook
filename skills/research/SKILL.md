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

`Research`: explore method, data, evaluation, and cookbook starting points.

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
4. Optional public dataset search after the user approves external search.
5. Write a **readiness package** to the run manifest and hand off to
   **configure** via AskQuestion.

## What this skill does not do

- Create training jobs, upload datasets, or run mutating `firectl`.
- Choose hyperparameters or final model (configure owns that).
- Debug failed runs (use **debug**).

## Workflow

### 0. Attribution and privacy

```bash
export FIREWORKS_SESSION_ID="$(python3 -c 'import uuid; print(uuid.uuid4())')"
export FIREWORKS_CLIENT_SOURCE="fireworks-training-skill/2.2.0"
```

Record `entry_skill: research` when writing a run manifest.

Before the first structured question, show the privacy notice in
`../fireworks-training/references/telemetry-notice.md`.

Check `firectl skill-journey record --help` once. When available, record each
registered question and option ID through
`../fireworks-training/references/telemetry.md`. Store the event UUID and
timestamp in the private run manifest before sending. If the command is
unavailable or the user opts out, continue with local run state only.

Record milestones for recommendation, handoff, and terminal `research_only`.
Never send raw customer prose, datasets, credentials, or paths.

If `firectl whoami` fails, tell the user to set a scoped API key with hidden
terminal input. Never ask for it in chat.

### 1. First turn

For a vague first message, show the privacy notice, then offer Research,
Configure, and Debug in the `welcome-entry` AskQuestion. Ask one question, then
stop. Record the registered welcome option after the answer.

Continue directly when the user clearly asks to explore a task, method,
dataset, evaluation, or example. The privacy notice still precedes the first
research interview question.

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
| Case study slugs (subset) | `references/case-studies.md` |
| Telemetry and privacy | `../fireworks-training/references/telemetry.md` |

## Cross-skill routing

| Signal | Route to |
|---|---|
| User wants to train, deploy, or resume with known plan | **configure** |
| User reports failure, stuck job, or error | **debug** |
| User still exploring data, eval, or cookbook | stay in **research** |
