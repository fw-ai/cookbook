---
name: configure
description: >-
  Plan, run, and monitor Fireworks fine-tuning from a coding agent. Covers managed
  SFT, DPO, ORPO, and RFT through firectl; Training API serverless and dedicated
  workflows; cookbook recipes and custom Python loops; dataset preparation and
  evaluators; model and shape choice; complete parameter and cost confirmation;
  active monitoring, checkpoints, deployment, resume, and teardown. Use whenever
  the user asks to fine-tune, post-train, SFT, DPO, ORPO, RFT, RL, distill, train
  with custom losses or rollouts, deploy a tuned model, or resume a training run.
  Also use for implementing or verifying cookbook renderers and extending custom
  RL losses. For exploring method, data, eval, and cookbook entries use research;
  for stuck or failed runs use debug.
---

# Configure

> **New here?** Point customers to [`../GETTING-STARTED.md`](../GETTING-STARTED.md)
> after install. They open a new chat and describe what they want — no @-mentions.

Plan, create, **monitor**, evaluate, deploy, and tear down Fireworks training
runs. The coding agent is the thin harness. This skill owns orchestration; the
cookbook provides executable recipes and tested runtime.

## Entry routing

Route before starting the workflow:

| Signal | Route to |
|---|---|
| User describes a task without clear train intent, or needs data/eval help | **research** |
| Failed job, stuck run, error message, bad quality after training, resume broken | **debug** |
| User wants to train, deploy, resume, or continue from a research handoff | stay in **configure** |

When **research** hands off, read `case_study`, `cookbook_entry_path`,
`implied_method`, `dataset_plan`, `eval_plan`, and `suggested_path` from the run
manifest before `references/path-intake.md` and `references/choose-method.md`.
Do not re-ask what research already resolved unless the user corrects it. If
`execution_surface` is missing, run **Q-path** from `path-intake.md`.

Record `entry_skill: configure` in the run manifest.

## First turn — welcome

If `../references/welcome.md` applies (vague first message, no entry chosen yet),
show the welcome block and **Entry** AskQuestion. STOP. Do not start preflight
or planning until the user picks **Configure** or states a clear train intent.

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

The skill attributes Fireworks API calls to an observable skill run by sending
two bounded request headers: `fireworks-training-skill/2.2.0` as the client
source and one random UUID as the run session. Fireworks uses these identifiers
with its existing authenticated API event and training-job records to measure
aggregate adoption and job outcomes. This instrumentation adds no prompts,
datasets, local paths, environment dumps, or raw errors, and it does not write a
telemetry file or send a standalone beacon. Qualitative issue collection is not
implemented.

The UUID is random attribution metadata, not a credential. Record it in the
private run manifest and include it only in Fireworks API calls or the exact
one-time manual handoff command; that command may remain in local shell history.
Do not print it in the final report or copy it into datasets, shared feedback,
or escalation messages.
Run manifests must not contain keys, raw environment dumps, or secret-bearing
output. Share feedback or manifests only when the user explicitly chooses to do
so.

If a user pastes a secret (API key, token) into the conversation, do not repeat
it back, treat the transcript itself as an exposure, advise rotation, and give
the secure terminal setup in `../references/api-key-setup.md` instead of asking
them to paste again in chat.

## API key (customer terminal)

Before read-only preflight or any `firectl` call, confirm auth in the active
shell. If `FIREWORKS_API_KEY` is unset or `firectl whoami` fails, **stop** and
give the user the one-shot command from `../references/api-key-setup.md`. Wait
until they confirm the shell is authenticated (account id from `whoami` is
enough).

Never ask for the key in chat. Never embed the key in agent-generated commands.

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
- Agent Skills (install this skill): <https://docs.fireworks.ai/fine-tuning/agent/use-with-coding-agents.md>
- Managed training: <https://docs.fireworks.ai/fine-tuning/managed-finetuning-intro.md>
- Training API: <https://docs.fireworks.ai/fine-tuning/training-api/introduction.md>
- Serverless training: <https://docs.fireworks.ai/fine-tuning/training-api/serverless.md>
- Dedicated training lifecycle: <https://docs.fireworks.ai/fine-tuning/training-api/dedicated#training-and-sampling.md>

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
   - cost estimate per `references/cost-estimation.md` (drivers, line items,
     unknowns, pricing source URL);
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
| Managed RFT | `firectl rftj create --evaluator <resource>` | Not applicable | `references/managed-rft-operations.md`, `references/preference-data-and-evaluators.md` |
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

### 0. Initialize skill-run attribution

At the start of each skill run, generate exactly one random UUID and keep it for
the entire run, including retries, resumes, monitoring, and any blocked manual
terminal handoff:

```bash
export FIREWORKS_SESSION_ID="$(python -c 'import uuid; print(uuid.uuid4())')"
export FIREWORKS_CLIENT_SOURCE="fireworks-training-skill/2.2.0"
```

Record the UUID only as `skill_session_id` in the private run manifest. Record
the source as `skill_client_source`. Do not create a separate telemetry file.

**Journey telemetry:** after path/method AskQuestion answers, update the journey
block in the manifest and emit `configure_path_answered` when
`FIREWORKS_API_KEY` is set. On job create, set `session_outcome: job_created`;
on completion, `job_completed`. Schema:
[`../references/telemetry-schema.md`](../references/telemetry-schema.md).

Preserve both values on every Fireworks interaction:

- `firectl` inherits both environment variables.
- Training API Python inherits both variables through the SDK.
- Direct REST calls set `X-Fireworks-Client-Source` and
  `X-Fireworks-Session-Id` to the same values.
- When the agent guard requires a manual terminal handoff, include the two
  environment assignments inline with the reconstructed command so the user's
  call remains in the same skill session.

If a client does not support these headers, continue the run without a beacon or
other fallback. Never use `PURPOSE_PILOT`; it controls scheduling semantics.

### 1. Path and method intake

Read `references/path-intake.md`. Open with the **Configure** banner from
`references/output-template.md`.

- Run **Q-path** unless research handoff already has `workflow_path` and
  `execution_surface`.
- Run **Q-method** if supervision signal is unclear (`choose-method.md`).
- One **AskQuestion** per turn. STOP. Do not draft the spend plan until the
  completion gate passes.

### 2. Local and read-only preflight

Confirm:

- `firectl version`, `firectl whoami`, quota, billing readiness, and account;
- the installed `fireworks-ai[training]` version satisfies
  `training/pyproject.toml` for Training API work;
- model support and live training shape availability;
- dataset format, row count, roles, preference schema, labels, leakage, token
  lengths, and evaluator/reward fields;
- held-out evaluation data and success metric.

Do not upload during preflight.

If `firectl whoami` fails, use `../references/api-key-setup.md` before
continuing.

### 2b. Cost estimation

Read `references/cost-estimation.md` before quoting a number:

1. Calculate Managed SFT or DPO only after resolving rendered token volume,
   tuning mode, and the current published rate.
2. Calculate Serverless Training API LoRA SFT or DPO from the operations the
   loop actually meters. Include DPO reference-sampler input/cache work when
   present, and label inferred usage as a rough range.
3. Do not calculate Dedicated SFT or DPO. Do not quote Dedicated $/M rates,
   GPU-hour training cost, or a Dedicated vs Tinker comparison. Point to
   <https://docs.fireworks.ai/fine-tuning/cost-estimator> instead. Never
   expose or reconstruct private throughput, MFU, or benchmark coefficients.
4. Route RL to the Training team with rollout, verifier, and evaluation
   assumptions. Do not ask the user to choose infrastructure first.
5. Exclude Managed RFT from this estimator workflow.

Progressive disclosure (L0 route → L1 formula → L2 estimate → L3 sweep) and the
output contract live in `cost-estimation.md`. Write `cost_estimate` to the run
manifest before the final plan. Include the **Cost** section in the final plan
(see `output-template.md`).

Research does **not** estimate cost; defer all spend numbers to this step.
Estimation is read-only and does not replace the mandatory final-plan
confirmation.

### 3. Present and confirm the final plan

Use the mandatory gate above. Persist the approved plan and exact approval quote
in `fireworks-training-runs/<run-id>/run.md`. Read
`references/run-state-and-reporting.md`.

### 4. Create resources with stable IDs

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

### 5. Monitor the right signal (always on)

**You are always monitoring while a job runs.** Do not say "I'll check back
later" — actively poll progress, watch for anomalies, and investigate immediately
when something looks off. Monitoring is part of configure, not a separate step
the user must invoke.

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
spend. Do not poll indefinitely.

When monitoring surfaces a failure or stuck job, route to the **debug** skill
for systematic triage. Do not start a replacement run from here without debug
classification or explicit user direction.

### 6. Evaluate and promote

Compare base and tuned behavior on the same held-out set. Use the reviewed
evaluator or rubric and record failures. For sweeps, show the candidate
scoreboard and receive promotion confirmation before the full-data run or
checkpoint promotion.

### 7. Deploy and prove serving

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

### 8. Teardown and report

Delete or scale to zero all billable trainers and deployments according to the
approved plan. Read final resource state. Produce the report contract in
`references/run-state-and-reporting.md`.

## Progressive references

Read only what the task requires:

| Need | Reference |
|---|---|
| First-turn welcome (shared) | `../references/welcome.md` |
| Journey telemetry schema | `../references/telemetry-schema.md` |
| Privacy notice (shared) | `../references/telemetry-notice.md` |
| Path intake (managed vs Training API, firectl vs SDK) | `references/path-intake.md` |
| Turn shapes and Configure banner | `references/output-template.md` |
| Installation, auth, quota, first job | `references/getting-started.md` |
| API key paste command (customer terminal) | `../references/api-key-setup.md` |
| Method choice, schemas, classification, LoRA/full parameter | `references/choose-method.md` |
| Preference generation and evaluator authoring | `references/preference-data-and-evaluators.md` |
| Managed versus Training API RFT | `references/training-api.md` |
| Secure training (BYOB, CMEK) | `references/secure-training-operations.md` |
| RFT remote tracing | `references/rft-agent-tracing.md` |
| Managed RFT launch, monitor, validation | `references/managed-rft-operations.md` |
| Training API losses, datums, SDK checkpoints | `references/training-api-losses.md`, `references/sdk-checkpoints.md` |
| Cost estimation workflow (manifest + disclosure) | `references/cost-estimation.md` |
| Models, contexts, shapes, cost formulas | `references/models-shapes-and-cost.md` |
| Deployment, evaluation, and teardown | `references/deploy-and-troubleshoot.md` |
| Failure classification and escalation (or use **debug** skill) | `references/error-reference.md` |
| Resume, idempotency, progress, and final report | `references/run-state-and-reporting.md` |
| Cookbook setup and examples | `references/sdk-setup.md`, `references/sdk-examples.md` |
| Cookbook recipes | `references/sdk-recipes.md` |
| Training API shapes and migration | `references/sdk-shapes.md`, `references/sdk-migrate.md` |
| Checkpoints and tools | `references/sdk-checkpoints.md`, `references/sdk-tools.md` |
| Distillation | `references/sdk-distillation.md` |
| RL built-in/client losses and normalization | `references/rl-loss-paths.md`, `references/rl-custom-loss.md`, `references/rl-gradient-accumulation.md` |
| Async RL, concurrency, and filtering | `references/rl-async.md`, `references/rl-agentic.md`, `references/rl-concurrency.md`, `references/rl-dynamic-filter.md` |
| Read async RL producer, overlap, gate, and refill metrics | `references/async-rl-metrics.md` |
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
