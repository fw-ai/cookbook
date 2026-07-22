---
name: fireworks-training
description: >-
  Train and fine-tune models on Fireworks from a coding agent. Covers managed SFT,
  DPO, ORPO, and RFT through firectl; Training API serverless and dedicated
  workflows; cookbook recipes and custom Python loops; dataset preparation and
  evaluators; model and shape choice; complete parameter and cost confirmation;
  monitoring, checkpoints, deployment, resume, teardown, and troubleshooting.
  Use whenever the user asks to fine-tune, post-train, SFT, DPO, ORPO, RFT, RL,
  distill, train with custom losses or rollouts, deploy a tuned model, resume a
  training run, or debug Fireworks training. Also use for implementing or
  verifying cookbook renderers and for extending custom RL losses; these are
  progressive references within this single skill, not separate skills.
---

# Fireworks training

This is the single Fireworks training skill. The coding agent is the thin
harness. This skill owns planning and orchestration, and the cookbook provides
the executable recipes and tested runtime.

## Source precedence

Use the most current source for each kind of fact:

1. **Installed `firectl ... --help`** for available managed CLI commands and
   flags.
2. **Live Fireworks docs** for models, shapes, prices, limits, permissions, and
   API parameters. Start at <https://docs.fireworks.ai/llms.txt> and prefer each
   page's `.md` URL.
3. **Cookbook code at the recorded commit** for Training API implementation,
   recipe behavior, checkpointing, resume, and cleanup.
4. **This skill** for durable routing, safety, and workflow rules.

Never copy a volatile catalog or price into an answer when it can be read live.
Record the docs URLs, cookbook commit, SDK version, and CLI version used in the
run manifest and final report.

### Degraded or offline sources

When a higher-priority source is unreachable (locked-down network, docs or
pricing site blocked, GitHub blocked), do not fall back to hardcoded values —
degrade explicitly:

- **Live docs unreachable:** substitute read-only `firectl` catalog reads
  (`model get`, `training-shape list`) and a `--dry-run -o json` to resolve
  shapes and defaults; label anything still unresolved as unknown.
- **Pricing page unreachable:** present the cost *formula* and ask the user for
  the current per-unit rate rather than guessing a number.
- **GitHub blocked:** the Training API / cookbook path (which requires cloning
  the cookbook) is unavailable; prefer managed training, and say so.

## Privacy and feedback

The public skill does not write or transmit usage telemetry and does not
automatically collect issues. Run manifests are customer-private local files and
must not contain keys, raw environment dumps, or secret-bearing output. Share
feedback or manifests only when the user explicitly chooses to do so; no
telemetry opt-out is required because collection is off.

If a user pastes a secret (API key, token) into the conversation, do not repeat
it back, treat the transcript itself as an exposure, and advise the user to
rotate or revoke that key and re-issue a scoped service-account key.

## Cookbook checkout

The standalone skill package does not vendor the cookbook. For Training API
work, clone the current public cookbook, record its commit, and pin that checkout
for the run before opening a recipe:

```bash
git clone https://github.com/fw-ai/cookbook
cd cookbook
git rev-parse HEAD
pip install -e ./training
```

Read the SDK constraint from `training/pyproject.toml`. Install the cookbook
package rather than upgrading the SDK outside that constraint. Record the
actual commit and installed SDK version in the run manifest.

## Choose the training path

First choose the training workflow. Then, only for Training API work, choose
the infrastructure.

| Need | Choose | Why |
|---|---|---|
| Standard SFT, DPO, ORPO, or RFT with supported configuration | **Managed training** | Declarative job, platform-managed lifecycle, least code |
| Custom loss, reward, rollout, trajectory, per-step logic, distillation, or research loop | **Training API** | Python control over the loop |

For Training API:

| Infrastructure | Use when | Key constraint |
|---|---|---|
| **Serverless training** | Fast LoRA SFT or RL experiments on supported models, shared pooled compute, per-token billing | Private preview, LoRA only, supported model set, no dedicated trainer/deployment lifecycle |
| **Dedicated training** | Full-parameter work, DPO, larger or unsupported serverless models, provisioned run resources, sustained high utilization, explicit checkpoint/resume/deployment control | Provisions trainer and deployment resources billed by time, subject to quota and availability |

The coding agent, UI, CLI, REST API, and Python SDK are **interaction
surfaces**, not separate training products. The coding agent can drive managed,
serverless, or dedicated workflows.

Live docs:

- Training overview: <https://docs.fireworks.ai/fine-tuning/finetuning-intro.md>
- Managed training: <https://docs.fireworks.ai/fine-tuning/managed-finetuning-intro.md>
- Training API: <https://docs.fireworks.ai/fine-tuning/training-api/introduction.md>
- Serverless training: <https://docs.fireworks.ai/fine-tuning/training-api/serverless.md>
- Dedicated training lifecycle: <https://docs.fireworks.ai/fine-tuning/training-api/training-and-sampling.md>

## Mandatory final-plan confirmation

Before **any** dataset upload, evaluator registration, paid inference, trainer
or job creation, checkpoint promotion, deployment, or other mutation:

1. Perform local validation and read-only account checks.
2. Resolve the configuration before asking:
   - run the selected managed command with `--help`;
   - build the exact create command and run its `--dry-run -o json` form when
     supported;
   - read current defaults from installed CLI help and live `.md` docs;
   - for Training API work, resolve the recipe config, cookbook commit, SDK
     version, training profile, and linked deployment shape without provisioning;
   - if a backend default cannot be known before creation, either set it
     explicitly or label it **platform-resolved, unknown before create**. Do not
     imply a value.
3. Show the user one complete final plan:
   - account;
   - managed, Training API serverless, or Training API dedicated path;
   - method and why it matches the available signal;
   - base model and why;
   - dataset, row counts, split, and schema;
   - evaluator, reward, or loss contract;
   - stable resource IDs;
   - every parameter the user set, marked **set**;
   - any preemptible trainer scheduling request, marked **admin-only**;
   - every default the agent or platform will apply, marked **default**;
   - resolved model, training shape, deployment shape, and context when relevant;
   - cost model, estimate or ceiling, and unknown cost lines;
   - success metric, evaluation plan, resume plan, and teardown.
4. Ask the user to confirm that exact resolved plan, including any explicitly
   labeled platform-resolved unknown.

Do not skip this gate because the run is small or because the user supplied
some parameters. A prior “run it” counts only when it approved the same complete
resolved plan. Any change to method, model, parameters, sweep breadth, or cost
ceiling requires renewed confirmation. Promotion and deployment each require a
separate confirmation.

Treat these as independent approval stages when present: paid pair generation or
evaluation, evaluator registration, dataset upload plus training, expanded
sweep breadth, promotion, and deployment. Approval for one stage does not
authorize a later stage.

Read-only commands such as `whoami`, `get`, `list`, `quota`, catalog reads,
local parsing, and offline evaluator tests do not require confirmation.

## Agent execution boundary

`firectl` can block mutating commands when it detects Claude Code, Cursor,
Codex, or another AI-agent environment. This is a platform safety control, not
an authentication error.

- Never unset agent-detection variables, set safe-account overrides, switch
  tools, or otherwise work around the guard.
- After the user approves a protected action, attempt it only through the
  documented command. If `firectl` returns `BLOCKED: mutating command ...`,
  surface the exact reconstructed command and ask the user to run it manually
  in their terminal.
- The guard also blocks the **`--dry-run`** form of a mutating command (it is
  classified as mutating). The confirmation-gate step "resolve config via
  `--dry-run -o json`" must therefore also be run by the user, not the agent;
  ask them to paste the dry-run output.
- After the user runs the command, continue with read-only `get`, `list`,
  monitoring, evaluation, and reporting.
- Execute a mutation inside the agent only when the installed CLI itself allows
  it through an agent-safe command or a safe-account policy that the user or
  administrator configured before the session. The skill must never configure
  that policy.

This handoff is identical across Claude Code, Cursor, and Codex.

## Method and recipe routing

| Task | Managed path | Cookbook implementation | Read |
|---|---|---|---|
| Managed SFT | `firectl sftj` | Not applicable | `references/choose-method.md` |
| Managed DPO | `firectl dpo-job create --loss-method DPO` | Not applicable | `references/choose-method.md` |
| Managed ORPO | `firectl dpo-job create --loss-method ORPO` | Not applicable | `references/choose-method.md` |
| Managed RFT | `firectl rftj create --evaluator <resource>` | Not applicable | `references/preference-data-and-evaluators.md`, `references/training-api.md` |
| Training API SFT | Not applicable | [`training/recipes/sft_loop.py`](https://github.com/fw-ai/cookbook/blob/main/training/recipes/sft_loop.py) | `references/sdk-recipes.md` |
| Training API DPO | Not applicable | [`training/recipes/dpo_loop.py`](https://github.com/fw-ai/cookbook/blob/main/training/recipes/dpo_loop.py) | `references/sdk-recipes.md` |
| Training API ORPO | Not applicable | [`training/recipes/orpo_loop.py`](https://github.com/fw-ai/cookbook/blob/main/training/recipes/orpo_loop.py) | `references/sdk-recipes.md` |
| Training API RL | Not applicable | [`training/recipes/rl_loop.py`](https://github.com/fw-ai/cookbook/blob/main/training/recipes/rl_loop.py) | `references/training-api.md`, `references/rl-loss-paths.md` |
| Async or agentic RL | Not applicable | [`training/recipes/async_rl_loop.py`](https://github.com/fw-ai/cookbook/blob/main/training/recipes/async_rl_loop.py) | `references/rl-async.md` |
| IGPO | Not applicable | [`training/recipes/igpo_loop.py`](https://github.com/fw-ai/cookbook/blob/main/training/recipes/igpo_loop.py) | `references/sdk-recipes.md` |
| Distillation | Not applicable | [`training/recipes/distillation_loop.py`](https://github.com/fw-ai/cookbook/blob/main/training/recipes/distillation_loop.py) | `references/sdk-distillation.md` |
| Serverless RL example | Not applicable | [`training/examples/serverless_rl/`](https://github.com/fw-ai/cookbook/tree/main/training/examples/serverless_rl) | Live serverless docs |
| Custom RL loss or research algorithm | Not applicable | Fork the closest maintained RL recipe and replace its documented loss call | `references/rl-custom-loss.md` |
| New or changed renderer | Not applicable | [`training/renderer/`](https://github.com/fw-ai/cookbook/tree/main/training/renderer) | `references/renderer.md`, `references/renderer-verification.md` |

**Cookbook first.** Inspect and fork the closest maintained recipe before
writing a loop. Change the loss, reward, rollout, data, or config needed by the
task. Do not reimplement trainer provisioning, weight sync, checkpoint,
deployment, reconnect, or cleanup plumbing.

## Common workflow

### 1. Local and read-only preflight

Confirm:

- `firectl version`, `firectl whoami`, quota, billing readiness, and account;
- the installed `fireworks-ai[training]` version satisfies
  `training/pyproject.toml` for Training API work;
- model support and live training shape availability;
- dataset format, row count, roles, preference schema, labels, leakage, token
  lengths, and evaluator/reward fields;
- held-out evaluation data and success metric.

Do not upload during preflight.

### 2. Present and confirm the final plan

Use the mandatory gate above. Persist the approved plan and exact approval quote
in `fireworks-training-runs/<run-id>/run.md`. Read
`references/run-state-and-reporting.md`.

### 3. Create resources with stable IDs

For managed jobs, upload the validated dataset and run only the selected method:

```bash
# SFT
firectl sftj create --job-id <run-id> \
  --base-model accounts/fireworks/models/<model> \
  --dataset <dataset-id> --output-model <output-model-id>

# DPO or ORPO
firectl dpo-job create --job-id <run-id> \
  --loss-method <DPO-or-ORPO> \
  --base-model accounts/fireworks/models/<model> \
  --dataset <dataset-id> --output-model <output-model-id>

# Managed RFT
firectl rftj create --job-id <run-id> \
  --base-model accounts/fireworks/models/<model> \
  --dataset <dataset-id> --evaluator accounts/<acct>/evaluators/<id> \
  --output-model <output-model-id>
```

Before launch, read the selected command's `--help`; the installed CLI is the
command contract.

If the approved create command is blocked by the agent guard, present it
verbatim for manual terminal execution and wait. Do not substitute another
mutation path. Resume with a read-only `get` on the stable ID.

For Training API work, record the cookbook commit and fork the routed recipe.
Use the serverless endpoint only when the serverless choice criteria pass.
Otherwise use dedicated provisioning through the recipe and SDK-managed service.

If a create response is lost or returns `AlreadyExists`, query the planned ID
and reuse only an exact config match. Never create a replacement ID before
reconciliation.

### 4. Monitor the right signal

| Method | State | Progress |
|---|---|---|
| Managed SFT | `firectl sftj get <id> -o json` | Job fields and linked W&B when enabled |
| Managed DPO / ORPO | `firectl dpo-job get <id> -o json` | `dpo-job export-metrics` and linked W&B |
| Managed RFT | `firectl rftj get <id> -o json` | Job, evaluator, rollout, and linked W&B signals |
| Training API serverless | Session/run IDs and recipe metrics | Forward/backward, optimizer, reward, and snapshot progress |
| Training API dedicated | RLOR trainer, deployment, checkpoints, and recipe metrics | Steps, rollouts, snapshots, W&B, and runner artifacts |

State alone is not progress. Put a numeric no-progress timeout in the approved
plan: default to 10 minutes for a small smoke run unless live docs or the
selected shape justify a different startup window. On timeout, gather evidence
before classifying. Do not launch a replacement until the old job is cancelled
or terminal, its final state is confirmed, and the user approves replacement
spend. Use `references/error-reference.md`; do not poll indefinitely.

### 5. Evaluate and promote

Compare base and tuned behavior on the same held-out set. Use the reviewed
evaluator or rubric and record failures. For sweeps, show the candidate
scoreboard and receive promotion confirmation before the full-data run or
checkpoint promotion.

### 6. Deploy and prove serving

Deployment has its own approval. Fine-tuned LoRA serving uses an on-demand
deployment; do not claim that a user's adapter is available through serverless
per-token inference.

```bash
firectl deployment create accounts/<acct>/models/<output-model-id> \
  --deployment-id <run-id>-deploy \
  --deployment-shape accounts/fireworks/deploymentShapes/<resolved-shape>
```

`READY` is not serving proof. Send one real request and require a successful,
sensible response.

### 7. Teardown and report

Delete or scale to zero all billable trainers and deployments according to the
approved plan. Read final resource state. Produce the report contract in
`references/run-state-and-reporting.md`.

## Progressive references

Read only what the task requires:

| Need | Reference |
|---|---|
| Installation, auth, quota, first job | `references/getting-started.md` |
| Method choice, schemas, classification, LoRA/full parameter | `references/choose-method.md` |
| Preference generation and evaluator authoring | `references/preference-data-and-evaluators.md` |
| Managed versus Training API RFT | `references/training-api.md` |
| Models, contexts, shapes, and costs | `references/models-shapes-and-cost.md` |
| Deployment, evaluation, and teardown | `references/deploy-and-troubleshoot.md` |
| Failure classification and escalation | `references/error-reference.md` |
| Resume, idempotency, progress, and final report | `references/run-state-and-reporting.md` |
| Cookbook setup and examples | `references/sdk-setup.md`, `references/sdk-examples.md` |
| Cookbook recipes | `references/sdk-recipes.md` |
| Training API shapes and migration | `references/sdk-shapes.md`, `references/sdk-migrate.md` |
| Checkpoints and tools | `references/sdk-checkpoints.md`, `references/sdk-tools.md` |
| Distillation | `references/sdk-distillation.md` |
| RL built-in/client losses and normalization | `references/rl-loss-paths.md`, `references/rl-custom-loss.md`, `references/rl-gradient-accumulation.md` |
| Async RL, concurrency, and filtering | `references/rl-async.md`, `references/rl-concurrency.md`, `references/rl-dynamic-filter.md` |
| Hotload and sampler failures | `references/rl-hotload.md`, `references/rl-sampling-timeouts.md` |
| Renderer implementation and training-token invariants | `references/renderer.md` |
| Renderer parity, live probes, and verifier UI | `references/renderer-verification.md` |

## Non-negotiables

- Validate locally before upload.
- Prefer managed training for standard supported jobs.
- Prefer cookbook recipes over blank Training API loops.
- Use live docs and catalog data instead of stale snapshots.
- Let training shapes own infrastructure; do not hand-set shape-owned fields.
- For RL, align trainer and inference numerics; use Router Replay for MoE when
  required.
- Separate quota, billing, scheduler capacity, user configuration, and platform
  failures.
- Never expose API keys, raw environment dumps, customer data, or private paths
  in reports or shared escalation channels.
