# Orchestrating a full training run from your coding agent (replaces Pilot)

*Source of truth: [firectl](https://docs.fireworks.ai/tools-sdks/firectl/firectl.md) · [Fine-tuning intro](https://docs.fireworks.ai/fine-tuning/finetuning-intro.md). Defer to the live docs for current commands/flags.*

> **Companion skill.** For SDK-driven runs (custom training loops, RL recipes, hotload, distillation), use the separately installed [`fireworks-training` skill](https://github.com/fw-ai/cookbook/blob/main/skills/dev/SKILL.md). This file covers the managed `firectl` lifecycle.

Fireworks is moving off the server-side **Pilot agent** (`firectl session ...`) and onto a **thin-harness / fat-skill** model: your coding agent (Claude Code, Cursor) reads this skill and drives the `firectl` primitives directly. The CLI stays a dumb 1:1 wrapper over the REST endpoints; the orchestration intelligence lives here in the skill, disclosed progressively (this file loads only when you actually need to run a job end to end).

## Why this replaces Pilot (and what is actually redundant)

`firectl session create/update/delete/cancel/events/get/list` are the **Pilot** verbs (note the `--scope optimize` "Pilot scope" flag). Pilot is a separate **orchestration endpoint**: you hand it a plain-English instruction, its server-side agent plans the run, and it then calls the *same* training and inference REST resources that the direct CLI commands call. So:

- `firectl session *` maps to the **Pilot session resource**, not to the training endpoints.
- `firectl sftj / dpo-job / reinforcement-fine-tuning-job / rlor-trainer-job / dataset / deployment / model` map **directly** to the training and inference resources.
- Both surfaces sit behind the same API host + auth. Pilot just fans out to the training resources one layer down.

When we decommission Pilot, what becomes redundant is the **Pilot orchestration service and the `session` verbs**, not the training endpoints. Your coding agent + this skill now *is* the orchestration layer, calling the training primitives directly.

### What replaced what

| Pilot verb | What it did server-side | What the agent + skill does now |
|---|---|---|
| `session create --instruction "fine-tune ..."` | Planned + launched the run from NL | Runs the Preflight -> Launch loop below (`firectl sftj create ...`) |
| `session events [--wait]` | Streamed execution events | Polls `firectl sftj get <id>` (and `deployment get`); watches the loss curve |
| `session update` | Responded to a waiting session (e.g. approved a cost gate) | The agent decides inline, or asks the user. No server round-trip |
| `session get` / `list` | Session status | `firectl sftj get` / `sftj list`, `deployment get` / `list` |
| `session cancel` / `delete` | Stopped/removed the session | `firectl sftj cancel <id>`; delete artifacts via their own resource commands |

## The managed lifecycle loop

Rollout order per the product direction: **managed jobs first, Training API next (private preview), inference/deployment CLI throughout.**

**1. Read-only + local preflight** — fail fast without creating a resource or spending money.
```bash
firectl whoami                                   # right account
firectl quota list                               # GPU ceiling + spend limit headroom
firectl model get -a fireworks <MODEL_ID>        # confirm Tunable: true
```
Validate the local JSONL without uploading it (`references/choose-method.md`). Inspect row count, message roles, output or preference schema, label distribution, train/eval leakage, and the fields the selected evaluator or inline reward reads. Base model not `Tunable` -> pick another from the live catalog. No quota/spend headroom -> that is the blocker, not a platform bug (`references/deploy-and-troubleshoot.md`).

**1.5 Plan + confirm — REQUIRED before any upload, spend, or resource creation.** Dataset upload, training jobs, evaluator registration, inference for preference-pair generation/evaluation, and deployments are **protected work**. Do not run them until you have shown the user a concrete plan and they have said go for *this* run. Show a terse preview, not a generic yes/no:

- **Account** (from `whoami`) — confirm it's the intended one, not a shared account by accident.
- **Method + base model** — and *why this model* (say it; don't silently pick). Offer the alternative if there's a reasonable one.
- **Dataset** — name + example count.
- **Stable resource IDs** — choose the dataset name, job ID, output-model ID, and deployment ID before creation. Record them in the run manifest so a lost response can be reconciled without creating duplicates.
- **Full resolved config** — echo the *complete* configuration for this run, both the values the user set and the defaults you are silently taking, so nothing is invisible. At minimum: base model, dataset, LoRA rank (default 8) or `--full-parameter`, epochs (default 1), learning rate + scheduler, max context length, `--batch-size-samples` + `--gradient-accumulation-steps`, optimizer weight decay, training shape (auto-mapped), and for DPO the `--dpo-beta`. Mark each value as `(set)` or `(default)`. Call out any user-set knob with a known footgun (for example `--batch-size-samples` on DPO). Do not show only rank/epochs/LR/shape and hide the rest.
- **Cost driver + ceiling** — the training job (per training-token) and any deployment (per GPU-second, bills until torn down). Size the estimate and gate by it: under ~$5 (a small smoke) can proceed if the user already gave a one-shot go this conversation; ~$5-50, confirm; over ~$50, confirm and itemize the drivers (model size, rows, epochs, sweep breadth), a high estimate is usually a broad sweep grid that can be narrowed.
- **Teardown** — that you'll delete/`scale_to_zero` the deployment after.

Then let the user confirm. Deployment (step 5) gets its **own** confirm, since it's the ongoing-spend action. Only skip the gate if the user already said "run it" for this specific work in this conversation.

**2. Upload + launch** (only after confirmation). Upload the already-validated dataset, preserve its full resource name, then use the command for the selected method:
```bash
firectl dataset create <name> /path/data.jsonl

# SFT
firectl sftj create \
  --job-id <run-id> \
  --base-model accounts/fireworks/models/<model> \
  --dataset <name> \
  --output-model <tuned-name>

# DPO
firectl dpo-job create \
  --job-id <run-id> \
  --base-model accounts/fireworks/models/<model> \
  --dataset <name> \
  --output-model <tuned-name>

# Managed RFT
firectl rftj create \
  --job-id <run-id> \
  --base-model accounts/fireworks/models/<model> \
  --dataset <name> \
  --evaluator accounts/<acct>/evaluators/<id> \
  --output-model <tuned-name>
```
Run the selected command only, not all three. Use a stable, valid `<run-id>` and record it before executing the command. Use `-o json` when the installed command supports it and persist the full job resource. If the response is lost or create reports `AlreadyExists`, run the matching `get <run-id> -o json` and compare its dataset, base model, output model, and config with the approved plan. Reuse only an exact match; otherwise stop and surface the collision. Change one thing at a time from defaults; let the platform map GPU -> model via the training shape. Before launch, run each selected command with `--help`; the installed CLI is the command contract.

**3. Monitor** (this is the `session events --wait` replacement). Use the resource family that created the job:

| Method | State | Progress signal | Cancel |
|---|---|---|---|
| SFT | `firectl sftj get <JOB_ID> -o json` | State fields plus linked W&B metrics when enabled. The current CLI has no `sftj export-metrics` command. | `firectl sftj cancel <JOB_ID>` |
| DPO / ORPO | `firectl dpo-job get <JOB_ID> -o json` | `firectl dpo-job export-metrics <JOB_ID>` plus W&B when enabled. | `firectl dpo-job cancel <JOB_ID>` |
| Managed RFT | `firectl rftj get <JOB_ID> -o json` | State fields, evaluator/rollout status, and linked W&B metrics when enabled. The current CLI has no `rftj export-metrics` command. | `firectl rftj cancel <JOB_ID>` |

Do not substitute `sftj` commands for DPO or RFT. Run `<resource> --help` and `<resource> get --help` first because CLI capabilities evolve.

Loop until `COMPLETED`/`FAILED`, watching the strongest available progress signal advance. **A silent failure can leave state at `RUNNING` with no error**, so state alone can hide a dead job. Set a bounded no-progress timeout. If the job remains `RUNNING`/`CREATING` without a new step, rollout, or W&B metric for several minutes, treat it as a suspected stall, then use the triage below and escalate with context. Never poll state indefinitely.

**4. Handle stalls / failures** — do not just report "stuck"; classify it. See the **Common issues (field-observed)** table in `references/deploy-and-troubleshoot.md` (stuck-at-0% / readiness timeout, `RESOURCE_EXHAUSTED`, quota 429, warm-start 400, masked "Internal error", reused-deployment ledger, and more). First decide platform-side (retry / escalate) vs user-side (fix dataset/config).

**5. Deploy on-demand + smoke test.** Use the deployment ID recorded in the approved plan:
```bash
firectl deployment create accounts/<acct>/models/<tuned-name> \
  --deployment-id <run-id>-deploy
# send one real completion, confirm sane output
```
If the response is lost or create reports `AlreadyExists`, read `firectl deployment get <run-id>-deploy -o json` and verify that it serves the intended output model and approved shape before reusing it.

**6. Tear down** the moment you are done. On-demand bills by GPU-second even when idle.
```bash
firectl deployment delete <DEPLOYMENT_ID>
```

**7. Persist + report.** Update the run manifest after every transition, then produce the standard final report with config, resources, evaluation, estimated and actual cost, exact replay commands, and teardown state. This is what makes the flow resumable after the coding-agent chat ends. Read `references/run-state-and-reporting.md`.

## Method-specific sweep + promotion gate

Pilot ran automatic sweeps; here the coding agent runs them with the resource family selected in the approved plan. Do not apply an SFT grid to DPO or RFT.

1. **Subsample for the search.** For datasets over ~1,000 rows, search on a fixed 1,000-row sample (seed it) to bound cost; train the winner on full data.
2. **Choose a method-specific grid:**
   - SFT: the 3-cell LoRA-rank and learning-rate grid from `references/choose-method.md`; launch each with `sftj create --job-id <stable-id>`.
   - DPO: use `dpo-job create --job-id <stable-id>` and vary only approved DPO knobs such as LoRA rank, learning rate, or `--dpo-beta`.
   - ORPO: use `dpo-job create --loss-method ORPO --job-id <stable-id>` and vary only approved ORPO knobs such as `--orpo-lambda`.
   - Managed RFT: use `rftj create --job-id <stable-id>` with the same evaluator. Start with one small validation run; sweep only when the evaluator discriminates and the user explicitly approves the GPU-hour cost. Vary one approved RFT knob at a time.
3. **Launch candidates as separate jobs.** Give every candidate a stable ID and separate cost line. Cap concurrency (typically 3–6 active jobs, lower for RFT) to respect quota.
4. **Evaluate each candidate on the same held-out split.** For classification, report per-label accuracy; for open-ended work, use the reviewed evaluator or rubric.
5. **Promotion gate — pause and ask.** Present the candidate scoreboard and let the user confirm the winner before the full-data run. This is a protected decision; do not auto-pick.
6. **Full-data final run** with the same resource family on the winning config, then deploy + benchmark base versus fine-tuned (`references/deploy-and-troubleshoot.md`).

## Rules specific to agent-driven orchestration

- **Do not use `firectl session *` in new work.** It is the deprecated Pilot path and is being decommissioned. Drive the resource commands above directly.
- **Poll, decide, act, do not block.** Replace "stream and wait" with a bounded poll loop plus an explicit decision at each transition.
- **Every run is reproducible.** Emit the exact `firectl` commands you ran so the user (or a rerun) can replay them. This is the payoff of the CLI being a thin 1:1 map to the API.
- **Every run is resumable.** Persist the run manifest before and after protected actions. On resume, read it and query recorded resources; never replay a create command just because chat context was lost.
- **Prefer a scoped service-account key**, never a personal admin key, for anything the agent runs unattended.
