---
name: configure
description: >-
  Plan, run, and monitor Fireworks training from a coding agent. Covers managed
  SFT, DPO, ORPO, and RFT plus Training API serverless and dedicated workflows,
  complete cost and parameter confirmation, active monitoring, checkpoints,
  deployment, resume, and teardown. Use for explicit train, fine-tune, deploy,
  or resume requests. Use research when method, data, or evaluation is unclear;
  use debug for a stuck or failed run.
---

# Configure

`Configure`: plan, run, and monitor Fireworks training.

This skill owns the execution workflow. The installed `fireworks-training`
compatibility skill carries the detailed references; the public cookbook carries
the executable recipes.

## Entry routing

| Signal | Route |
|---|---|
| Task-shaped goal without clear method, data, or evaluation | `research` |
| Failed, stuck, errored, or unexpectedly low-quality run | `debug` |
| Explicit train, deploy, resume, or research handoff | stay in `configure` |

When a research handoff exists, reuse its `case_study` when present,
`cookbook_entry_tier`, `cookbook_entry_path`, `notebook`, `readme`,
`implied_method`, `dataset_plan`, `eval_plan`, and `suggested_path`. Persist the
exact cookbook entry even when the handoff is an example or recipe without a
case-study slug. Do not ask again unless the user corrects it.

Research owns the `welcome-entry` question and its privacy notice. Route a
vague first message there instead of asking a second welcome question.

## Attribution and privacy

Use one UUID for the whole run:

```bash
export FIREWORKS_SESSION_ID="$(python3 -c 'import uuid; print(uuid.uuid4())')"
export FIREWORKS_CLIENT_SOURCE="fireworks-training-skill/2.2.0"
```

Record `entry_skill: configure` in the private run manifest. Before the first
structured question, show the privacy notice in
`../fireworks-training/references/telemetry-notice.md`.

Check `firectl skill-journey record --help` once. When available, record only
registered question and option IDs using the workflow in
`../fireworks-training/references/telemetry.md`. If unavailable, recommend a
firectl upgrade and continue with local run state. Telemetry must never block
training.

## Source precedence

1. Installed `firectl ... --help` for commands and flags.
2. Live Fireworks docs from <https://docs.fireworks.ai/llms.txt>.
3. Public cookbook code at the recorded commit.
4. This skill and its installed references for stable workflow rules.

Never guess current models, shapes, prices, limits, or defaults.

## Authentication

Before any `firectl` call, run `firectl whoami`. If authentication is missing,
tell the user to enter a scoped API key in their terminal with hidden input:

```bash
read -s FIREWORKS_API_KEY
echo
export FIREWORKS_API_KEY
firectl whoami
```

Never ask for an API key in chat or echo it.

## Path and method intake

Read `references/path-intake.md`.

1. Run Q-path even when research recommends a coarse path. Skip it only for a
   recorded `question_id: configure-q-path` answer; `research-q3` never counts.
2. Run Q-method when the supervision signal is unclear.
3. Use one AskQuestion per turn.
4. Do not present a spend plan until the path and method completion gate passes.

## Read-only preflight

Confirm:

1. Account, `firectl` version, authentication, quota, and billing readiness.
2. Model support and live training shape availability.
3. Dataset format, row count, split, schema, leakage, and token lengths.
4. Evaluator, reward, or preference-data contract.
5. Held-out evaluation data and success metric.
6. Cookbook commit and installed SDK version for Training API work.

Do not upload or create resources during preflight.

## Cost

Read `references/cost-estimation.md`.

1. Calculate Managed SFT or DPO only after resolving rendered token volume,
   tuning mode, exact path context, and the current published rate.
2. Calculate Serverless SFT from trainer tokens. For DPO, return an unpadded
   baseline with policy train tokens plus one-time all-uncached reference
   prefill; list padding and optional sampling as excluded work.
3. Do not calculate vision SFT or DPO until model-specific visual token
   accounting is available.
4. Do not calculate Dedicated SFT or DPO. Direct the user to
   <https://docs.fireworks.ai/fine-tuning/cost-estimator>.
5. Do not calculate Managed ORPO until its billing contract is documented.
6. Route RL, embedding, IGPO, and distillation estimates to the Training team
   with method-specific assumptions.
7. Put the structured estimate and unknowns in the private run manifest.

Cost estimation does not replace the mandatory final-plan confirmation.

## Mandatory final-plan gate

Before any upload, evaluator registration, paid inference, training creation,
checkpoint promotion, or deployment, show one complete plan:

1. Account, method, workflow path, and execution surface.
2. Base model, dataset, split, schema, and row counts.
3. Evaluator, reward, loss, and success metric.
4. Stable resource IDs.
5. Every user-set parameter and every platform default that can be resolved
   before creation. Label an unknowable backend default `platform-resolved,
   unknown before create`; never guess or stall the plan.
6. Model, training shape, deployment shape, and context when relevant.
7. Cost line items, range, assumptions, unknowns, and pricing source.
8. Monitoring, no-progress timeout, resume, evaluation, and teardown.

Ask the user to approve that exact plan. Any change to method, model, parameters,
sweep breadth, or cost ceiling requires renewed confirmation. Promotion and
deployment require separate confirmation.

## Agent execution boundary

Read-only `get`, `list`, `whoami`, quota, and catalog commands are allowed.

After approval, attempt the documented mutation. If firectl returns
`BLOCKED: mutating command`, show the exact reconstructed command and ask the
user to run it. Never unset agent-detection variables, configure a safe-account
override, or switch tools to bypass the guard.

The guard can also block a mutating command's dry-run form. In that case the
user runs the dry-run and returns its output before final approval.

## Execution workflow

1. Persist the approved plan and approval quote in
   `fireworks-training-runs/<run-id>/run.md`.
2. Create resources with stable IDs through the selected managed command or
   recorded cookbook recipe.
3. If a response is lost or reports `AlreadyExists`, query the planned ID and
   reuse only an exact configuration match.
4. Stay actively monitoring while a run is in progress. State alone is not
   progress; poll the method's numeric progress signal.
5. On the approved no-progress timeout, gather evidence and route failures to
   `debug`. Never launch a replacement before reconciling the prior run.
6. Evaluate base and tuned behavior on the same held-out set.
7. Deploy only after separate approval, then prove serving with a real request.
8. Tear down billable resources and report final state.

## Progressive references

| Need | Reference |
|---|---|
| Path intake | `references/path-intake.md` |
| Cost contract | `references/cost-estimation.md` |
| Configure response shapes | `references/output-template.md` |
| Installation and authentication | `../fireworks-training/references/getting-started.md` |
| Method and data selection | `../fireworks-training/references/choose-method.md` |
| Preference data and evaluators | `../fireworks-training/references/preference-data-and-evaluators.md` |
| Managed RFT | `../fireworks-training/references/managed-rft-operations.md` |
| RFT tracing | `../fireworks-training/references/rft-agent-tracing.md` |
| Training API | `../fireworks-training/references/training-api.md` |
| Training API losses | `../fireworks-training/references/training-api-losses.md` |
| Secure training | `../fireworks-training/references/secure-training-operations.md` |
| Models and shapes | `../fireworks-training/references/models-shapes-and-cost.md` |
| Run state and resume | `../fireworks-training/references/run-state-and-reporting.md` |
| Deployment and teardown | `../fireworks-training/references/deploy-and-troubleshoot.md` |
| Error catalog | `../fireworks-training/references/error-reference.md` |
| SDK and cookbook recipes | `../fireworks-training/references/sdk-recipes.md` |
| SDK setup and examples | `../fireworks-training/references/sdk-setup.md`, `../fireworks-training/references/sdk-examples.md` |
| SDK migration and shapes | `../fireworks-training/references/sdk-migrate.md`, `../fireworks-training/references/sdk-shapes.md` |
| Checkpoints and tools | `../fireworks-training/references/sdk-checkpoints.md`, `../fireworks-training/references/sdk-tools.md` |
| Distillation | `../fireworks-training/references/sdk-distillation.md` |
| RL losses | `../fireworks-training/references/rl-loss-paths.md`, `../fireworks-training/references/rl-custom-loss.md`, `../fireworks-training/references/rl-gradient-accumulation.md` |
| Async and agentic RL | `../fireworks-training/references/rl-async.md`, `../fireworks-training/references/rl-agentic.md`, `../fireworks-training/references/rl-concurrency.md`, `../fireworks-training/references/rl-dynamic-filter.md` |
| Async RL metrics | `../fireworks-training/references/async-rl-metrics.md` |
| Hotload and sampling | `../fireworks-training/references/rl-hotload.md`, `../fireworks-training/references/rl-sampling-timeouts.md` |
| Renderer work | `../fireworks-training/references/renderer.md` |
| Renderer verification | `../fireworks-training/references/renderer-verification.md` |
| Journey telemetry | `../fireworks-training/references/telemetry.md` |
| Telemetry notice | `../fireworks-training/references/telemetry-notice.md` |

## Non-negotiables

1. Validate locally before upload.
2. Prefer managed training for standard supported jobs.
3. Prefer maintained cookbook recipes over blank custom loops.
4. Keep quota, billing, capacity, user configuration, and platform failures
   distinct.
5. Never expose credentials, customer data, raw answers, or private paths.
