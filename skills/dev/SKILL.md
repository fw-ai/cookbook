---
name: fireworks-training
description: Train models on Fireworks via the cookbook. Covers greenfield work (pick a recipe, fork it, resolve training + deployment shape from a profile) and user-level recovery (promote a checkpoint, list promotable checkpoints on a trainer, self-check a `WeightSyncScope.PER_TRAINER` vs `PER_DEPLOYMENT` bucket-scope mix-up). The cookbook is the reference implementation of `fireworks.training.sdk`; fork a recipe or run an example instead of reimplementing. Trigger when the user wants to start, resume, promote, or do a first-line diagnosis on a training run; for deeper recovery the skill routes users to Fireworks support.
---

# Fireworks training

The cookbook is the reference implementation of the Fireworks Training SDK. Fork a recipe, run an example, use the standalone tools in [`references/tools.md`](references/tools.md). Use **shapes** for both trainer and deployment — never hand-set `accelerator_type` / `node_count` / `custom_image_tag`.

---

## Task → reference

| Task or signal | Reference |
|----------------|-----------|
| "How do I set up / install the cookbook?" | [`references/setup.md`](references/setup.md) |
| "I want to run something out of the box" | [`references/examples.md`](references/examples.md) |
| "I want to fork a recipe and edit the Config" | [`references/recipes.md`](references/recipes.md) |
| "How do I set the training / deployment shape?" | [`references/shapes.md`](references/shapes.md) |
| `RuntimeError: Failed to resolve latest validated training shape` | [`references/shapes.md`](references/shapes.md#when-resolve_training_profile-raises-failed-to-resolve-latest-validated-training-shape) — don't pin a version; retry or reach out |
| "Can I run two deployments off one trainer (sampler + eval)?" | [`references/rl/hotload.md`](references/rl/hotload.md#two-deployments-one-trainer) |
| "How does RL dispatch server-side vs client-side loss? What's the cost?" | [`references/rl/loss-paths.md`](references/rl/loss-paths.md) |
| "How does gradient accumulation work at `optim_step`? What normalization does RL use?" | [`references/rl/gradient-accumulation.md`](references/rl/gradient-accumulation.md) |
| "Why are some RL samples being filtered?" | [`references/rl/dynamic-filter.md`](references/rl/dynamic-filter.md) |
| "Custom loss for RL" | [`references/rl/custom-loss.md`](references/rl/custom-loss.md) |
| "RL hotload / weight sync cadence, on-policy vs off-policy, `weight_sync_timeout`" | [`references/rl/hotload.md`](references/rl/hotload.md) |
| "Concurrency control for RL rollouts — adaptive vs fixed?" | [`references/rl/concurrency.md`](references/rl/concurrency.md) |
| "How do I promote a checkpoint?" | [`references/tools.md`](references/tools.md#promote_checkpointpy) |
| "Which checkpoints does the server know about / are promotable?" | [`references/tools.md`](references/tools.md#listing-checkpoints-fireworksclientlist_checkpoints) — `FireworksClient.list_checkpoints(job_id)` |
| "How do I reconnect a training **client** to a running trainer?" | [`references/tools.md`](references/tools.md#reconnect_and_adjust_lrpy) |
| "Hotload keeps failing — is this a `PER_TRAINER` / `PER_DEPLOYMENT` scope mix-up?" | [`references/rl/hotload.md`](references/rl/hotload.md#self-check-when-hotload-fails) — self-check and reach out to Fireworks support |
| "How do I verify train vs inference logprobs?" | [`references/tools.md`](references/tools.md#verify_logprobspy) |
| "Where does checkpoint state live?" / CheckpointKind / `checkpoints.jsonl` | [`references/checkpoints.md`](references/checkpoints.md) |
| "Continue LoRA training from a prior adapter" / `warm_start_from_adapter` | [`references/checkpoints.md`](references/checkpoints.md#warm-start-from-a-promoted-adapter-lora-only) |
| Error: `checkpoint "<name>" not found in GCS` | [`references/checkpoints.md`](references/checkpoints.md#when-promote-fails) — validate `output_model_id` first; reach out to Fireworks support if still failing |
| Error: `Hotload failed for snapshot ...` | [`references/rl/hotload.md`](references/rl/hotload.md#self-check-when-hotload-fails) |
| Error: `hotload flow mismatch: trainer wants deployment-first ... but deployment ... is trainer-first` | [`references/rl/hotload.md`](references/rl/hotload.md#server-side-validation) — the server still emits the old "trainer-first / deployment-first" wording; it maps to `PER_TRAINER` / `PER_DEPLOYMENT` bucket scope. Scopes crossed at `CreateRlorTrainerJob`; pick one scope. |
| Error: `hotload flow mismatch: trainer T is deployment-first-keyed for deployment D` | [`references/rl/hotload.md`](references/rl/hotload.md#server-side-validation) — trainer is keyed to a different deployment's bucket (`PER_DEPLOYMENT`); use a `PER_TRAINER`-scope trainer |
| Error: `hot_load_bucket_url %q conflicts with hot_load_trainer_job %s; set exactly one` | [`references/rl/hotload.md`](references/rl/hotload.md#server-side-validation) — create-time: drop whichever field is wrong |
| Error: `hot_load_bucket_url %q conflicts with hot_load_trainer_job %s; update hot_load_trainer_job instead, or clear it first` | [`references/rl/hotload.md`](references/rl/hotload.md#server-side-validation) — update-time: clear bucket URL first, then PATCH trainer job |
| Error: `invalid FW_HOSTED hot_load_bucket_url` / `must use gs:// scheme` / `path must start with rl-checkpoints/` | [`references/rl/hotload.md`](references/rl/hotload.md#server-side-validation) — structural validation on FW_HOSTED URL at create/update |
| Error: `configured FW_HOSTED hot_load bucket is not reachable` / `control plane lacks permission` | [`references/rl/hotload.md`](references/rl/hotload.md#server-side-validation) — account's `ModelBucket` misprovisioned; reach out to Fireworks support |
| Error: `cannot cancel job in state: JOB_STATE_DELETED` | [`references/rl/hotload.md`](references/rl/hotload.md#trainer-deletion-and-retention) — trainer is tombstoned during the retention window; no action needed |
| `list_checkpoints` / `promote_checkpoint` returns NOT_FOUND > 30 days after delete | [`references/rl/hotload.md`](references/rl/hotload.md#trainer-deletion-and-retention) — past retention, expected |
| HTTP 400 on `output_model_id` | [`references/tools.md`](references/tools.md#promote_checkpointpy) — validate before calling |
| "Is this a `PER_TRAINER` or `PER_DEPLOYMENT` bucket scope?" | [`references/rl/hotload.md`](references/rl/hotload.md#weight-sync-scope-per_trainer-vs-per_deployment) |
| Manual `accelerator_type` / `node_count` set on `Config` | [`references/shapes.md`](references/shapes.md) — drop them, the profile owns infra |

---

## First debug step — always

Before assuming the platform is broken, confirm the user's installed `fireworks-ai` satisfies the cookbook's SDK requirement. A stale SDK produces errors that masquerade as server bugs: missing keyword arguments, "unknown field", silent no-ops on new config fields, or `promote_checkpoint` behaviour that doesn't match the code.

The requirement lives in the cookbook's `training/pyproject.toml` — look for the `fireworks-ai[training]` pin:

```bash
grep 'fireworks-ai\[training\]' cookbook/training/pyproject.toml
# e.g. "fireworks-ai[training]>=1.0.0a62,<2"

pip show fireworks-ai | grep -i version
```

If the installed version doesn't satisfy the pin, upgrade first and retry. Only after the SDK meets the requirement should you start triaging the actual symptom. Users do **not** need to sync the cookbook to upstream `main` — whatever cookbook commit they're on declares its own SDK requirement, and matching that is what matters.

## Non-negotiables

1. **Shape first.** `cfg.infra.training_shape_id` is required. The deployment shape comes from the profile. Manual infra fields are a mistake; the backend will reject or ignore them. See [`references/shapes.md`](references/shapes.md).
2. **`WeightSyncScope.PER_TRAINER` is the default.** Set `DeployConfig(weight_sync_scope=WeightSyncScope.PER_TRAINER)` (the default). Do not combine it with `hot_load_deployment_id` — that field belongs to `PER_DEPLOYMENT`. Pick one bucket scope. See [`references/rl/hotload.md`](references/rl/hotload.md#weight-sync-scope-per_trainer-vs-per_deployment).
3. **Fork, don't reinvent.** Training loop plumbing lives in `training/recipes/`. Fork the file that matches the task; do not rewire `FiretitanTrainingClient` / `DeploymentManager` / `WeightSyncer` from scratch.
4. **Validate `output_model_id` before promote.** Server cap is 63 chars, charset `[a-z0-9-]`. A rejected promote orphans the sampler blob; the same `checkpoint_id` returns "not found in GCS" after GC. See [`references/checkpoints.md`](references/checkpoints.md#output_model_id-validation).

---

## SDK surface

The training SDK lives at <https://github.com/stainless-sdks/fireworks-ai-python> under `src/fireworks/training/sdk/`. Code outside that directory in that repo is auto-generated — ignore. For any SDK call an agent needs, read the cookbook recipe that already makes it: recipe files are listed in [`references/recipes.md`](references/recipes.md).
