---
name: fireworks-fine-tuning
description: >-
  Managed fine-tuning on Fireworks, driven from your coding agent via firectl: choosing a
  method (SFT/DPO/RFT), preparing and validating datasets, launching + monitoring jobs,
  picking training shapes + models + context/LoRA rank, estimating cost (GPU-hour vs
  per-token), and deploying + troubleshooting fine-tuned models. This is the managed
  SFT/DPO/RFT router and the successor to the Pilot agent (`firectl session`). Use when
  fine-tuning, SFT-ing, DPO-ing, or RFT-ing a Fireworks-hosted model, preparing a dataset,
  choosing a training shape, deploying a fine-tuned model, or debugging a failed training
  job. For custom Python training loops + SDK recipe internals, see the companion
  `fireworks-training` skill (`../dev/SKILL.md`).
---

Fireworks lets you fine-tune open models and serve them on one platform. **Managed fine-tuning (SFT/DPO/RFT) is GA**; the **Training API (custom Python loops) is private preview** — request access at https://fireworks.ai/contact-training . Always use the latest model + the correct training shape from the live catalog, and prefer a **scoped/service-account API key** over a personal admin key.

**Companion skill.** This skill is the *managed* fine-tuning router (drive `firectl` SFT/DPO/RFT end to end). For the *SDK power-user* path — forking a cookbook recipe, writing a custom training loop, RL recipe internals, hotload, checkpoint promotion, distillation — use [`../dev/SKILL.md`](../dev/SKILL.md) (the `fireworks-training` skill). They complement each other: this skill for launching + operating managed jobs, `dev` for the reference implementation of `fireworks.training.sdk`.

## Before you build

Install + auth: `firectl` + `firectl signin` -> `references/getting-started.md`.

**If the Fireworks training MCP is available, call `training_planner` first** (describe the task -> it returns the recommended method + shape + recipe, from live data). Otherwise use the routing table below.

The references **link the live `.md` docs**; for any specific value (training shapes, models, context limits, prices, API params) **defer to the linked doc** — the references are durable guidance, not a snapshot to trust over the docs.

## Routing — read the matching reference before answering or writing code

| Task | Reference |
|---|---|
| Set up, install `firectl`, first job, quota | `references/getting-started.md` |
| Pick SFT vs DPO vs RFT; dataset format; LoRA vs full-param | `references/choose-method.md` |
| Custom training loop, cookbook recipes, RL/GRPO, custom loss/reward, numerics | `references/training-api.md` |
| Pick a training shape / model; context length; GPU class; what it costs | `references/models-shapes-and-cost.md` |
| Deploy a fine-tuned model; evaluate a model cheaply (preemptible); tear down; fix a failed job / quota / suspension | `references/deploy-and-troubleshoot.md` |
| Orchestrate a full run from your coding agent (replaces Pilot); managed lifecycle loop; `firectl session` deprecation | `references/orchestrate-from-agent.md` |

**Not covered above?** The full machine-readable doc index is at <https://docs.fireworks.ai/llms.txt> — find the relevant page and fetch its `.md` version (every page has one). The routing table covers the common paths; `llms.txt` + the live docs are the complete, always-current source.

## Orchestrating from your coding agent (replaces Pilot)

This skill *is* the training-agent layer now. Instead of the server-side Pilot agent (`firectl session ...`), your coding agent reads this skill and drives the `firectl` primitives directly: **thin harness (the CLI, a 1:1 map to the REST endpoints), fat skill (the orchestration logic here), progressive disclosure (references load on demand)**. To run or debug a job end to end, read `references/orchestrate-from-agent.md`.

**`firectl session *` is deprecated.** Those are the Pilot verbs (`--scope optimize`); Pilot is a separate orchestration endpoint that fans out to the same training resources the CLI already hits directly. Do not build on it. Drive `sftj` / `dpo-job` / `reinforcement-fine-tuning-job` / `dataset` / `deployment` directly. (The old `fireworks-agent` skill documented the Pilot path; it is deprecated in favor of this skill.)

## Critical rules

- **Plan + confirm before any spend.** Dataset upload, training jobs, and deployments are protected work. Before running them, show the user a concrete plan — account, method + base model (and *why*), dataset, the **full resolved config** (every value you set and every default you apply, each marked set/default, not just rank/epochs/LR/shape), cost driver, teardown — and let them confirm. Never silently pick a model or launch a job. Deployment gets its own confirm (it's the ongoing-spend action). Read-only commands (`whoami`, `get`, `list`, `quota`) run freely. See `references/orchestrate-from-agent.md` step 1.5.
- **Start with SFT + LoRA + defaults** unless you have preference pairs (DPO) or a reward/verifiable task (RFT). Change one thing at a time; watch loss/reward curves.
- **Validate dataset format before uploading** — JSONL, one object per line, correct schema per method.
- **Let the platform map GPU -> model** via the training shape; use the model's full context. Check the live catalog — don't hardcode shapes/models.
- **Fine-tuned LoRA deploys on-demand only** (dedicated live-merge or multi-LoRA); serverless per-token serving of your own fine-tuned LoRA is not available yet. **Tear down the deployment when done** — on-demand bills by GPU-second even when idle.
- **Align numerics** (precision + logprob divergence; Router Replay/R3 for MoE) before trusting RL signal.
- **Before assuming a Fireworks bug, check quota (a GPU ceiling) and billing (suspension/spending-limit is billing-side)** — different controls.
- SFT/DPO bill **per training token**; full-param/RFT trainers + deployments bill **per GPU-hour**. Saturate the GPU.

## Docs

Latest + machine-readable: https://docs.fireworks.ai/fine-tuning/finetuning-intro.md · prefer the `.md` URL form of any doc page. Training shapes catalog + pricing are generated live — link them, don't memorize.
