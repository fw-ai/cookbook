# Run state, resume, progress, and final reporting

*Use this reference whenever a run may outlive the current coding-agent conversation, when resuming work, or when preparing the final result.*

The coding-agent flow preserves its plan, approvals, resources, progress, and report in a small local run manifest. The manifest stores orchestration state only; Fireworks remains the source of truth for datasets, jobs, models, deployments, and billing.

## Create one manifest per run

At the start of the skill run, generate one random UUID and export it as
`FIREWORKS_SESSION_ID`; export
`FIREWORKS_CLIENT_SOURCE=fireworks-training-skill/2.2.0`. Preserve those values
for all firectl, Training API Python, direct REST, retry, resume, and blocked
manual terminal handoff calls in this run. Do not use `PURPOSE_PILOT`.

Before the first protected action, create `fireworks-training-runs/<run-id>/run.md` in the current workspace. Use a stable, human-readable `<run-id>` such as `<method>-<output-model>-<UTC timestamp>`. Keep datasets and generated evaluation files beside it unless the user chooses another location. Record the UUID only in this private manifest. Do not create a telemetry file or standalone beacon.

Do not write API keys, environment values, raw customer data, or secret-bearing command output into the manifest. Resource names, configs, commands, metrics, and explicit user decisions are allowed, but treat the file as customer-private.

**Journey telemetry:** update enum fields and `intake_responses[]` after each
AskQuestion. Never store raw user messages in server events — use option IDs plus
redacted `task_summary` (≤200 chars). See `skills/references/telemetry-schema.md`
and `telemetry-notice.md`. When `FIREWORKS_API_KEY` is set and user has not opted
out, emit journey events per `skills/references/telemetry/sdk-firectl-helpers.md`.

````markdown
# Fireworks training run

status: planned
phase: awaiting_plan_approval
updated_at_utc:
skill_session_id:
skill_client_source: fireworks-training-skill/2.2.0
journey_schema_version: 2

## Journey telemetry
telemetry_opt_out: false
task_summary:
welcome_choice:
welcome_skipped:
intake_responses: []
intake_q1_task_shape:
intake_q1b_shape:
intake_q2_data:
intake_q3_path:
intake_q4_tried:
domain_followup:
questions_asked:
questions_inferred_skipped:
completion_gate_bypassed:
matched_case_study:
match_confidence:
handoff_choice:
workflow_path:
execution_surface:
inherited_from_research:   # legacy: inherited_from_discover
method:
session_outcome:

## Intent
task:
success_metric:
method: sft | dpo | managed-rft | training-api-serverless | training-api-dedicated

## Inputs
account:
firectl_version:
docs_urls:
cookbook_commit:
sdk_version:
local_dataset:
fireworks_dataset:
base_model:
evaluator_or_reward:
train_rows:
eval_rows:

## Approved plan
approved_at_utc:
approval_quote:
resolved_config:
cost_estimate:          # structured block — see cost-estimation.md
  level:
  route:
  pricing_source:
  pricing_fetched_at_utc:
  recommended_path:
  cost_range:
  rate_certainty:
  usage_certainty:
  supplied_inputs: {}
  inferred_inputs: []
  assumptions: []
  lines: []
  unknowns: []
  total_low_usd:
  total_high_usd:
  dominant_line:
  dedicated_estimator_url:   # when route is dedicated-*
  notes:
teardown_plan:

## Resources
planned_dataset_id:
planned_evaluator_name:
evaluator_source_sha256:
evaluator_account:
evaluator_registration_started_at_utc:
evaluator_registration_ended_at_utc:
planned_job_id:
planned_output_model_id:
planned_deployment_id:
evaluator:
job:
output_model:
deployment:
training_api_recipe_path_and_commit:
trainer_job:
rollout_deployments:
latest_checkpoint:
wandb_or_metrics:

## Progress
last_state:
last_step_or_rollout:
last_progress_at_utc:
next_action:

## Evaluation
base_result:
tuned_result:
promotion_decision:

## Commands
```text
# exact commands, in execution order
```

## Teardown
deployment_state:
other_resources:
````

## State machine

Use these phases consistently:

```text
local_preflight
-> awaiting_plan_approval
-> dataset_uploaded
-> job_running
-> awaiting_promotion_approval        # only for a sweep
-> training_completed
-> awaiting_deployment_approval
-> deployment_ready
-> evaluated
-> torn_down
-> reported
```

Use `partial_failure_cleanup` when training succeeds but evaluation, deployment,
serving proof, or teardown fails. Reconcile and clean up every resource before
reporting the partial outcome. Also use `failed` or `cancelled` as terminal
states. Record the failed phase, exact platform state, last progress signal, and
next safe action.

Update the manifest:

1. Before and after every protected action.
2. Whenever a resource ID becomes known.
3. Whenever the user approves, changes, or rejects a plan.
4. After each meaningful state transition, not every poll.
5. Before the agent stops or hands work to another person.

## Resume safely

When asked to resume:

1. Read the manifest before running any command.
2. Confirm the account with `firectl whoami`.
3. Query every recorded resource with the matching resource family:
   - Dataset: `firectl dataset get <DATASET_ID> -o json`
   - SFT: `firectl sftj get <JOB_ID> -o json`
   - DPO / ORPO: `firectl dpo-job get <JOB_ID> -o json`
   - Managed RFT: `firectl rftj get <JOB_ID> -o json`
   - Training API dedicated trainer: `firectl rlor-trainer-job get <TRAINER_JOB_ID> -o json`
   - Training API dedicated checkpoints: `firectl rlor-trainer-job list-checkpoints <TRAINER_JOB_ID>`
   - Deployment: `firectl deployment get <DEPLOYMENT_ID> -o json`
   - Output model: `firectl model get <OUTPUT_MODEL_ID> -o json`
4. Reconcile the manifest with Fireworks state and update it.
5. Continue from `next_action`.

Never replay dataset, evaluator, job, model, or deployment creation merely because the previous chat ended. Every create must use the stable ID recorded before execution:

- dataset: the positional dataset name;
- evaluator: planned display name plus the reviewed source SHA-256 when the current evaluator API supports naming;
- managed job: `--job-id <planned_job_id>`;
- output model: `--output-model <planned_output_model_id>`;
- deployment: `--deployment-id <planned_deployment_id>`.

If the manifest says creation was attempted but has no returned resource, call the exact matching `get` command with the planned ID. If it exists, compare its immutable inputs and resolved config with the approved plan. Reuse only an exact match. Evaluator registration may not expose a caller-selected ID: if its response is lost, locate it through Eval Protocol output, the account UI, or the current evaluator API by matching the planned name and source hash; never register it again while state is ambiguous. If a resource does not exist, ask before retrying when the original action may have incurred spend. Never generate a second ID as a recovery shortcut.

### Resume a Training API dedicated run

Training API dedicated recipes may provision several resources rather than one managed job. Before launch, record the recipe path, cookbook commit, full config, local working directory, planned output-model ID, and deployment IDs. Immediately after provisioning, persist every returned RLOR trainer job and rollout deployment ID.

On resume:

1. Confirm the same cookbook commit and SDK version are available.
2. Read the recorded RLOR trainer with `firectl rlor-trainer-job get`.
3. List its checkpoints with `firectl rlor-trainer-job list-checkpoints`.
4. Read every rollout or evaluation deployment and the promoted model, if present.
5. Reconcile the last checkpoint, optimizer step, deployment attachment, and promotion state with the manifest.
6. Use the recipe's documented reconnect path only when it supports reconnecting to that trainer. Do not start the recipe from the beginning.

If the process exited before the trainer ID was recorded, do not relaunch automatically. List RLOR trainer jobs and deployments created in the narrow UTC launch window, then match account, base model, shape, and planned output-model ID. Continue only when one exact resource set is identified; otherwise ask the user or support to reconcile possible orphaned resources.

## Method-neutral progress updates

Normalize the strongest method-specific signal into a concise progress update:

```text
Phase: job_running
Method: managed-rft
State: RUNNING
Elapsed: 18m
Progress: rollout batch 4 completed at 23:42 UTC
Cost: unavailable from this surface
Next check: 23:47 UTC
```

Always distinguish:

- **State** from the job resource.
- **Progress** from steps, metrics, rollouts, checkpoints, or W&B.
- **Cost** from authoritative usage or billing evidence. If unavailable, say unavailable.
- **ETA** from an explicit platform estimate. Do not invent one from elapsed time.

For method-specific commands use the common workflow in `SKILL.md`; for no-progress handling use `references/error-reference.md`.

## Required final report

Every completed, failed, or cancelled run gets a report. Keep it concise but complete:

````markdown
# Result

## Outcome
status:
method:
base_model:
output_model:
job:
deployment:

## Provenance
firectl_version:
docs_urls:
cookbook_commit:
sdk_version:

## Evidence
success_metric:
base_result:
tuned_result:
training_metrics:
smoke_test:

## Cost
estimated:
actual:
source:

## Reproduce
```bash
# exact commands used, with secret values referenced by environment variable
```

## Teardown
deployment:
remaining_billable_resources:

## Issues and next step
user_side:
platform_side:
recommended_next_action:
````

Rules:

- Include full resource names, not only display names.
- Report base versus tuned evidence on a held-out split whenever evaluation was in scope.
- Report estimated versus actual cost only from real evidence. Label unavailable values.
- Include the exact final config, including defaults the agent applied.
- Include CLI version, live documentation URLs, and, for Training API runs, cookbook commit and SDK version.
- Include one copy-paste inference example when a deployment remains available.
- Confirm teardown state. Never say teardown succeeded without reading the final resource state.
- For a sweep, include every candidate, metric, cost line, and the user's promotion decision.
