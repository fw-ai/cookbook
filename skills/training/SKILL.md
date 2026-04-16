---
name: fireworks-training
description: Train models on Fireworks via the cookbook. Covers both greenfield work (pick a recipe, fork it, resolve training + deployment shape from a profile) and recovery (promote a checkpoint, re-attach a deployment, diagnose "checkpoint not found in GCS", triage trainer-first vs legacy deployment-first flows). The cookbook is the reference implementation of `fireworks.training.sdk`; fork a recipe or run an example instead of reimplementing. Trigger when the user wants to start, resume, promote, re-attach, or debug a training run on Fireworks.
---

# Fireworks training

The cookbook is the reference implementation of the Fireworks Training SDK. Fork a recipe, run an example, use the three standalone tools. Use **shapes** for both trainer and deployment — never hand-set `accelerator_type` / `node_count` / `custom_image_tag`.

---

## Task → reference

| Task or signal | Reference |
|----------------|-----------|
| "I want to run something out of the box" | [`references/examples.md`](references/examples.md) |
| "I want to fork a recipe and edit the Config" | [`references/recipes.md`](references/recipes.md) |
| "How do I set the training / deployment shape?" | [`references/shapes.md`](references/shapes.md) |
| "How do I promote a checkpoint?" | [`references/tools.md`](references/tools.md#promote_checkpointpy) |
| "How do I re-attach a deployment to a new trainer?" | [`references/tools.md`](references/tools.md#reconnect_and_adjust_lrpy) |
| "How do I verify train vs inference logprobs?" | [`references/tools.md`](references/tools.md#verify_logprobspy) |
| "Where does checkpoint state live?" / CheckpointKind / `checkpoints.jsonl` | [`references/checkpoints.md`](references/checkpoints.md) |
| Error: `checkpoint "<name>" not found in GCS` | [`references/checkpoints.md`](references/checkpoints.md#recovery-sampler-orphan) |
| Error: `Hotload failed for snapshot ...` | [`references/trainer-first-vs-deployment-first.md`](references/trainer-first-vs-deployment-first.md) |
| HTTP 400 on `output_model_id` | [`references/tools.md`](references/tools.md#promote_checkpointpy) — validate before calling |
| "Is this trainer-first or deployment-first?" | [`references/trainer-first-vs-deployment-first.md`](references/trainer-first-vs-deployment-first.md) |
| Manual `accelerator_type` / `node_count` set on `Config` | [`references/shapes.md`](references/shapes.md) — drop them, the profile owns infra |

---

## Non-negotiables

1. **Shape first.** `cfg.infra.training_shape_id` is required. The deployment shape comes from the profile. Manual infra fields are a mistake; the backend will reject or ignore them. See [`references/shapes.md`](references/shapes.md).
2. **Trainer-first for new work.** Create the trainer, then create the deployment with `hot_load_trainer_job=<trainer>`. Do not mix in a legacy `hot_load_deployment_id`. See [`references/trainer-first-vs-deployment-first.md`](references/trainer-first-vs-deployment-first.md).
3. **Fork, don't reinvent.** Training loop plumbing lives in `training/recipes/`. Fork the file that matches the task; do not rewire `FiretitanTrainingClient` / `DeploymentManager` / `WeightSyncer` from scratch.
4. **Validate `output_model_id` before promote.** Server cap is 63 chars, charset `[a-z0-9-]`. A rejected promote orphans the sampler blob; the same `checkpoint_id` returns "not found in GCS" after GC. See [`references/checkpoints.md`](references/checkpoints.md#output_model_id-validation).

---

## SDK surface

The training SDK lives at <https://github.com/stainless-sdks/fireworks-ai-python> under `src/fireworks/training/sdk/`. Code outside that directory in that repo is auto-generated — ignore. For any SDK call an agent needs, read the cookbook recipe that already makes it: recipe files are listed in [`references/recipes.md`](references/recipes.md).
