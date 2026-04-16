---
name: fireworks-training
description: Train models on Fireworks via the cookbook. Covers greenfield work (pick a recipe, fork it, resolve training + deployment shape from a profile) and user-level recovery (promote a checkpoint, list promotable checkpoints on a trainer, self-check a trainer-first vs legacy deployment-first mix-up). The cookbook is the reference implementation of `fireworks.training.sdk`; fork a recipe or run an example instead of reimplementing. Trigger when the user wants to start, resume, promote, or do a first-line diagnosis on a training run; for deeper recovery the skill routes users to Fireworks support.
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
| "Hotload keeps failing — is this a trainer-first / deployment-first mix-up?" | [`references/rl/hotload.md`](references/rl/hotload.md#self-check-when-hotload-fails) — self-check and reach out to Fireworks support |
| "How do I verify train vs inference logprobs?" | [`references/tools.md`](references/tools.md#verify_logprobspy) |
| "Where does checkpoint state live?" / CheckpointKind / `checkpoints.jsonl` | [`references/checkpoints.md`](references/checkpoints.md) |
| Error: `checkpoint "<name>" not found in GCS` | [`references/checkpoints.md`](references/checkpoints.md#when-promote-fails) — validate `output_model_id` first; reach out to Fireworks support if still failing |
| Error: `Hotload failed for snapshot ...` | [`references/rl/hotload.md`](references/rl/hotload.md#self-check-when-hotload-fails) |
| HTTP 400 on `output_model_id` | [`references/tools.md`](references/tools.md#promote_checkpointpy) — validate before calling |
| "Is this trainer-first or deployment-first?" | [`references/rl/hotload.md`](references/rl/hotload.md#trainer-first-vs-deployment-first-two-creation-orders) |
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
2. **Trainer-first for new work.** Create the trainer, then create the deployment with `hot_load_trainer_job=<trainer>`. Do not mix in a legacy `hot_load_deployment_id`. See [`references/rl/hotload.md`](references/rl/hotload.md#trainer-first-vs-deployment-first-two-creation-orders).
3. **Fork, don't reinvent.** Training loop plumbing lives in `training/recipes/`. Fork the file that matches the task; do not rewire `FiretitanTrainingClient` / `DeploymentManager` / `WeightSyncer` from scratch.
4. **Validate `output_model_id` before promote.** Server cap is 63 chars, charset `[a-z0-9-]`. A rejected promote orphans the sampler blob; the same `checkpoint_id` returns "not found in GCS" after GC. See [`references/checkpoints.md`](references/checkpoints.md#output_model_id-validation).

---

## SDK surface

The training SDK lives at <https://github.com/stainless-sdks/fireworks-ai-python> under `src/fireworks/training/sdk/`. Code outside that directory in that repo is auto-generated — ignore. For any SDK call an agent needs, read the cookbook recipe that already makes it: recipe files are listed in [`references/recipes.md`](references/recipes.md).
