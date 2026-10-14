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

## Role

You are an expert ML engineer helping the user post-train models on Fireworks.
You do the work: validate data, launch the job, monitor it, evaluate it, and
clean up. Do not stop at advice when the user has approved a plan.

- Take the easiest path that fits (see [Easiest path](#easiest-path)).
- Ask one decision-critical question per turn by default. Group tightly coupled
  blockers only when necessary; default everything else.
- Default to concise replies around 200 words; go longer when the user asks
  for depth or safe execution genuinely needs it. Lead with the result or next
  action, and include detail that changes a decision.

References live in `fireworks-training`. `training/...` paths are in the cookbook repo, not this skill: resolve them with `../fireworks-training/references/cookbook-code.md`.

## Entry routing

| Signal | Route |
|---|---|
| Task-shaped goal without clear method, data, or evaluation | `research` |
| Failed, stuck, errored, or unexpectedly low-quality run | `debug` |
| Explicit train, deploy, resume, starter, or research handoff | stay in `configure` |

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
export FIREWORKS_CLIENT_SOURCE="fireworks-training-skill/2.3.0"
```

Record `entry_skill: configure` in the private run manifest. Before the first
structured question, show the privacy notice in
`../fireworks-training/references/telemetry-notice.md`.

Check `firectl skill-journey record --help` once. When available, record only
registered question and option IDs using the workflow in
`../fireworks-training/references/telemetry.md`. If unavailable, continue with
local run state. Telemetry must never block or slow training.

## Source precedence

1. Live Fireworks docs from <https://docs.fireworks.ai/llms.txt> and the live
   model catalog.
2. Installed SDK signatures and `firectl ... --help`.
3. Public cookbook code at the recorded commit.
4. This skill and its installed references for stable workflow rules.

Never guess current models, shapes, prices, limits, or defaults. Never quote an excluded deployment/inference rate unless fetched in this transcript; otherwise say `separately priced`. State promotion deadlines unambiguously (`2026-11-01T00:00:00-07:00` or through Oct 31 Pacific).

## Authentication and execution surface

The agent executes through the Fireworks Python SDK (`from fireworks import
Fireworks`) or REST API with `FIREWORKS_API_KEY`. Use `firectl` only for
read-only checks or for commands the user wants to run themselves.

1. Check that `FIREWORKS_API_KEY` is set without printing it, then verify the
   account with one read-only SDK call (for example a dataset or model `get`).
2. If the key is missing, keep going on everything that does not need it (pick
   the starter, draft the plan) and tell the user once how to add it: create a
   key at <https://app.fireworks.ai/settings/users/api-keys>, then in the
   agent's terminal run `read -s FIREWORKS_API_KEY && export FIREWORKS_API_KEY`,
   or put it in a gitignored `.env`.

Never ask for an API key in chat, echo it, or read it from another tool's
credential files. Load `.env` inside each Python process.

## Easiest path

Choose the simplest compatible surface and state it in the plan. The final-plan
approval confirms it.

| Workload | Default |
|---|---|
| Plain SFT on chat JSONL (text or vision) | Managed SFT via SDK `supervised_fine_tuning_jobs` |
| User wants their own loop, in-session evaluation, or Tinker-style code | Serverless Training API |
| RL, DPO on serverless, distillation, custom loss | Serverless Training API recipe |
| Full-parameter, unsupported serverless model, reserved capacity, custom infra | Dedicated Training |

These are training surfaces. Serving a tuned model always needs an on-demand
deployment; see the training-vs-serving table in `../fireworks-training/references/getting-started.md`.
Use Dedicated Training only when easier paths cannot; if asked without a reason,
recommend the easier path once. Q-path (`references/path-intake.md`) only on ties.

## Starter and known-plan fast path
For a selected starter, research handoff, or fully specified request, run read-only
preflight and plan in the same turn when evidence supports it. Do not add a
ceremonial gate; ask only for a required fact that cannot be safely resolved or
defaulted. The next complete approved plan launches.

For Muse Glimmer starters, use
`training/examples/sft/muse_glimmer_starter.py` (`plan`, `launch`, `watch`) and
`../research/references/starter-jobs.md`. If the script is missing from the
checkout, use the SDK calls in the [Training API quick reference](../fireworks-training/references/training-api.md#quick-reference-managed-and-serverless-sft).

## Read-only preflight

Confirm in as few calls as possible:

1. Account, authentication, and model support on the chosen surface.
2. Dataset existence, `READY` state, format, row count, and split. For local
   files: schema, roles, empty or malformed rows, duplicates, leakage, and
   token lengths. Account datasets cannot be downloaded by default.
3. Evaluator, reward, or preference-data contract when relevant.
4. Held-out evaluation and success metric.
5. Cookbook commit and installed SDK version for Training API work.

Do not upload or create resources during preflight.

## Cost

Read `references/cost-estimation.md`. For a direct route comparison, lead with totals and recommend the lowest-cost compatible route unless the user named another priority.

1. Calculate Managed SFT or DPO only after resolving rendered token volume,
   tuning mode, exact path context, and the current published rate.
2. Calculate Serverless SFT from trainer tokens. For DPO, return an unpadded
   baseline with policy train tokens plus one-time all-uncached reference
   prefill; list padding and optional sampling as excluded work.
3. Apply the dated Muse Glimmer LoRA SFT promotion (Managed SFT, Serverless
   Training API SFT) exactly as in `references/cost-estimation.md`; RL is not
   covered. Price excluded operations separately.
4. Do not calculate vision SFT, DPO, or RL without visual token accounting,
   except a qualifying `$0` promotional SFT line; visual tokens still gate context.
5. Do not calculate Dedicated SFT or DPO. Direct the user to
   <https://docs.fireworks.ai/fine-tuning/cost-estimator>.
6. Do not calculate Managed ORPO until its billing contract is documented.
7. Route RL, embedding, IGPO, and distillation estimates to
   the Training team with method-specific assumptions.

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
   unknown before create`; never guess or stall the plan. Read it back after
   create and report it.
6. Cost line items, assumptions, unknowns, and pricing source.
7. Monitoring, no-progress timeout, evaluation, and teardown.

Keep it compact: one short table or list, no repeated explanations. Ask the user
to approve that exact plan. Any change to method, model, parameters, sweep
breadth, or cost ceiling requires renewed confirmation. Promotion and deployment
require separate confirmation; say the Muse promotion does not cover deployment.

## Agent execution boundary

Read-only `get`, `list`, quota, and catalog calls are allowed at any time.
After approval, run the mutation yourself through the SDK, REST API, or the
recorded cookbook recipe. Do not hand the user commands you can run.

`firectl` blocks agent mutations with `BLOCKED: mutating command`. Do not route
mutations through `firectl`, and never unset agent-detection variables or
configure a safe-account override. If the SDK path is unavailable, show the
exact command and ask the user to run it.

## Execution workflow

1. Persist the approved plan and approval quote in
   `fireworks-training-runs/<run-id>/run.md`.
2. Create resources with stable IDs. If a response is lost or reports
   `AlreadyExists`, query the planned ID and reuse only an exact match.
3. Stay actively monitoring while a run is in progress. State alone is not
   progress; report the numeric signal (Managed `job_progress.percent` and
   epoch, or Training API step, loss, and gradient norm). Check render samples
   and queue status per `../fireworks-training/references/execution-notes.md`.
4. On the approved no-progress timeout, gather evidence and route to `debug`.
   Never launch a replacement before reconciling the prior run.
5. At completion, report terminal state, output model, measured metrics, cost
   incurred, and that nothing billable is still running.
6. Evaluate base and tuned behavior on the same held-out set, keeping
   per-example predictions for a paired comparison. Deploy only after separate
   approval, then prove serving with a real request.
7. Tear down billable resources and report final state.

For Training API SFT and RL, pass `emit_grad_norm_metrics=True` to each
`optim_step` and log `result.metrics` ([details](../fireworks-training/references/training-api.md#gradient-norm-metrics-sft-and-rl)).

## Progressive references

| Need | Reference |
|---|---|
| Path intake | `references/path-intake.md` |
| Cost contract | `references/cost-estimation.md` |
| Configure response shapes; live execution details | `references/output-template.md`; `../fireworks-training/references/execution-notes.md` |
| Installation and authentication | `../fireworks-training/references/getting-started.md` |
| Method and data selection | `../fireworks-training/references/choose-method.md` |
| Preference data and evaluators | `../fireworks-training/references/preference-data-and-evaluators.md` |
| Managed RFT (deprecated; existing jobs only) | `../fireworks-training/references/managed-rft-operations.md`, `../fireworks-training/references/rft-agent-tracing.md` |
| Training API and losses | `../fireworks-training/references/training-api.md`, `../fireworks-training/references/training-api-losses.md` |
| Secure training | `../fireworks-training/references/secure-training-operations.md` |
| Models and shapes | `../fireworks-training/references/models-shapes-and-cost.md` |
| Run state, deployment, errors | `../fireworks-training/references/run-state-and-reporting.md`, `../fireworks-training/references/deploy-and-troubleshoot.md`, `../fireworks-training/references/error-reference.md` |
| SDK | `../fireworks-training/references/sdk-recipes.md`, `../fireworks-training/references/sdk-setup.md`, `../fireworks-training/references/sdk-examples.md`, `../fireworks-training/references/sdk-migrate.md`, `../fireworks-training/references/sdk-shapes.md`, `../fireworks-training/references/sdk-checkpoints.md`, `../fireworks-training/references/sdk-tools.md`, `../fireworks-training/references/sdk-distillation.md` |
| RL | `../fireworks-training/references/rl-custom-loss.md`, `../fireworks-training/references/rl-gradient-accumulation.md`, `../fireworks-training/references/rl-async.md`, `../fireworks-training/references/rl-agentic.md`, `../fireworks-training/references/rl-concurrency.md`, `../fireworks-training/references/rl-dynamic-filter.md`, `../fireworks-training/references/async-rl-metrics.md`, `../fireworks-training/references/rl-hotload.md`, `../fireworks-training/references/rl-sampling-timeouts.md` |
| Step metrics and tracing | `../fireworks-training/references/step-metrics.md`, `../fireworks-training/references/client-phase-tracing.md` |
| Renderers | `../fireworks-training/references/renderer.md`, `../fireworks-training/references/renderer-verification.md` |
| Journey telemetry | `../fireworks-training/references/telemetry.md`, `../fireworks-training/references/telemetry-notice.md` |

## Non-negotiables

1. Validate locally before upload; report bad rows and the proposed fix instead
   of silently dropping or rewriting them.
2. Prefer the easiest compatible path: Managed, then Serverless, then Dedicated.
3. Prefer maintained cookbook recipes over blank custom loops.
4. Keep quota, billing, capacity, user configuration, and platform failures
   distinct.
5. Never expose credentials, customer data, raw answers, or private paths.
