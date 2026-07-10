# Orchestrating a full training run from your coding agent (replaces Pilot)

*Source of truth: [firectl](https://docs.fireworks.ai/tools-sdks/firectl/firectl.md) · [Fine-tuning intro](https://docs.fireworks.ai/fine-tuning/finetuning-intro.md). Defer to the live docs for current commands/flags.*

> **Companion skill.** For SDK-driven runs (custom training loops, RL recipes, hotload, distillation), use [`../../dev/SKILL.md`](../../dev/SKILL.md) (the `fireworks-training` skill). This file covers the managed `firectl` lifecycle.

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

## The managed lifecycle loop (SFT; DPO/RFT are the same shape)

Rollout order per the product direction: **managed jobs first, Training API next (private preview), inference/deployment CLI throughout.**

**1. Preflight** — fail fast before spending a GPU-second.
```bash
firectl whoami                                   # right account
firectl quota list                               # GPU ceiling + spend limit headroom
firectl model get -a fireworks <MODEL_ID>        # confirm Tunable: true
firectl dataset create <name> /path/data.jsonl   # validates schema on upload
```
Decision: base model not `Tunable` -> pick another from the live catalog. Dataset rejected -> fix format (`references/choose-method.md`) before anything else. No quota/spend headroom -> that is the blocker, not a platform bug (`references/deploy-and-troubleshoot.md`).

**1.5 Plan + confirm — REQUIRED before any spend or resource creation.** Dataset upload, training jobs, and deployments are **protected work**. Do not run them until you have shown the user a concrete plan and they have said go for *this* run. Show a terse preview, not a generic yes/no:

- **Account** (from `whoami`) — confirm it's the intended one, not a shared account by accident.
- **Method + base model** — and *why this model* (say it; don't silently pick). Offer the alternative if there's a reasonable one.
- **Dataset** — name + example count.
- **Full resolved config** — echo the *complete* configuration for this run, both the values the user set and the defaults you are silently taking, so nothing is invisible. At minimum: base model, dataset, LoRA rank (default 8) or `--full-parameter`, epochs (default 1), learning rate + scheduler, max context length, `--batch-size-samples` + `--gradient-accumulation-steps`, optimizer weight decay, training shape (auto-mapped), and for DPO the `--dpo-beta`. Mark each value as `(set)` or `(default)`. Call out any user-set knob with a known footgun (for example `--batch-size-samples` on DPO). Do not show only rank/epochs/LR/shape and hide the rest.
- **Cost driver + ceiling** — the training job (per training-token) and any deployment (per GPU-second, bills until torn down). Size the estimate and gate by it: under ~$5 (a small smoke) can proceed if the user already gave a one-shot go this conversation; ~$5-50, confirm; over ~$50, confirm and itemize the drivers (model size, rows, epochs, sweep breadth), a high estimate is usually a broad sweep grid that can be narrowed.
- **Teardown** — that you'll delete/`scale_to_zero` the deployment after.

Then let the user confirm. Deployment (step 5) gets its **own** confirm, since it's the ongoing-spend action. Only skip the gate if the user already said "run it" for this specific work in this conversation.

**2. Launch** (after confirmation).
```bash
firectl sftj create \
  --base-model accounts/fireworks/models/<model> \
  --dataset <name> \
  --output-model <tuned-name>
```
Change one thing at a time from defaults; let the platform map GPU -> model via the training shape.

**3. Monitor** (this is the `session events --wait` replacement). Poll both state and *progress*, and do not trust state alone.
```bash
firectl sftj get <JOB_ID>                        # state
firectl sftj export-metrics <JOB_ID>             # loss/step series — the real progress signal
```
Loop until `COMPLETED`/`FAILED`, watching that loss/step actually advance. **A silent failure can leave state at `RUNNING` with no error** (the backend crashed but the control plane never transitioned), so state alone can hide a dead job. Set a bounded no-progress timeout: if state is `RUNNING`/`CREATING` but `export-metrics` shows no new step for several minutes, treat it as a suspected stall, not as healthy. On a suspected stall, pull the real signal, the linked **W&B run** (if enabled) or the exported metrics, to de-mask the cause, then use the stall triage below and escalate with context (job id, elapsed, last step, config). Never poll `State` indefinitely.

**4. Handle stalls / failures** — do not just report "stuck"; classify it. See the **Common issues (field-observed)** table in `references/deploy-and-troubleshoot.md` (stuck-at-0% / readiness timeout, `RESOURCE_EXHAUSTED`, quota 429, warm-start 400, masked "Internal error", reused-deployment ledger, and more). First decide platform-side (retry / escalate) vs user-side (fix dataset/config).

**5. Deploy on-demand + smoke test.**
```bash
firectl deployment create --model accounts/<acct>/models/<tuned-name>   # LoRA is on-demand only, never serverless
# send one real completion, confirm sane output
```

**6. Tear down** the moment you are done. On-demand bills by GPU-second even when idle.
```bash
firectl deployment delete <DEPLOYMENT_ID>
```

## Rules specific to agent-driven orchestration

- **Do not use `firectl session *` in new work.** It is the deprecated Pilot path and is being decommissioned. Drive the resource commands above directly.
- **Poll, decide, act, do not block.** Replace "stream and wait" with a bounded poll loop plus an explicit decision at each transition.
- **Every run is reproducible.** Emit the exact `firectl` commands you ran so the user (or a rerun) can replay them. This is the payoff of the CLI being a thin 1:1 map to the API.
- **Prefer a scoped service-account key**, never a personal admin key, for anything the agent runs unattended.
