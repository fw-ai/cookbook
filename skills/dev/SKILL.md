---
name: fireworks-training-dev
description: Develop training runs on Fireworks. Use the cookbook as the reference implementation of the Training SDK. Covers picking a recipe to fork, running out-of-the-box examples, resolving training shape + deployment shape from a single profile, running the promote and re-attach tools, and where checkpoint state lives. Point the agent at specific files rather than explaining from scratch. For fixing a broken run, use the sibling `debug` skill.
---

# Fireworks training — dev

The cookbook is the reference implementation of the Fireworks Training SDK. Do not reimplement training loop plumbing — fork the recipe that matches the task, or run an example straight from `training/examples/`.

Use **shapes** for both trainer and deployment. Resolve from a single profile; never hand-set `accelerator_type` / `node_count` / image tags.

---

## I want to …

### Run something out of the box

Point the agent at `training/examples/`:

| Task | File |
|------|------|
| Minimal SFT | `training/examples/sft_getting_started/` |
| SFT (real datasets) | `training/examples/sft/` |
| DPO | `training/examples/dpo/` |
| ORPO | `training/examples/orpo/` |
| RL (GRPO, deepmath) | `training/examples/deepmath_rl/` |
| RL (tool use, frozen lake) | `training/examples/frozen_lake/` |
| RL (multi-hop QA) | `training/examples/multihop_qa/` |

Examples import from the recipes and ship with ready-to-run `Config`.

### Fork a recipe

Agent: read the loop you need, copy it, edit the `Config` at the top.

| Recipe | File |
|--------|------|
| SFT | `training/recipes/sft_loop.py` |
| DPO | `training/recipes/dpo_loop.py` |
| ORPO | `training/recipes/orpo_loop.py` |
| Importance-weighted GRPO | `training/recipes/igpo_loop.py` |
| RL loop (generic GRPO scaffold) | `training/recipes/rl_loop.py` |

The "reference loop" means these files: they are the canonical wiring of `FiretitanTrainingClient` + `DeploymentManager` + `WeightSyncer`. Fork, do not reinvent.

### Use a training shape (required)

Agent: call `resolve_training_profile` on the trainer manager and copy the fields.

```python
profile = rlor_mgr.resolve_training_profile(cfg.infra.training_shape_id)
# profile.max_supported_context_length, profile.training_shape_version,
# profile.deployment_shape_version -- ready to pass through.
```

Reference: `training/recipes/rl_loop.py` lines ~336–347 and `training/utils/training_shapes.py`.

### Use a deployment shape (required)

Every recipe auto-sets `cfg.deployment.deployment_shape = profile.deployment_shape_version` when it's not explicitly set. Do not pass `deployment_accelerator_type` or any other manual infra field — the profile pins them.

Reference: `training/utils/config.py` — `DeployConfig.to_deployment_config` (manual path is only for the skip-validations superuser case).

### Promote a checkpoint

```bash
python training/examples/snippets/promote_checkpoint.py \
    --checkpoints-jsonl <path>
```

Source: `training/examples/snippets/promote_checkpoint.py`. Reads the latest `sampler_path` row from `checkpoints.jsonl` and calls the promote API. Use `--step N` to pin a specific step, `--output-model-id` to override the generated ID.

Agent: before running, `validate_output_model_id(output_model_id)` from the SDK — the server cap is 63 chars.

### Re-attach a deployment to a new trainer

```bash
python training/examples/snippets/reconnect_and_adjust_lr.py ...
```

Source: `training/examples/snippets/reconnect_and_adjust_lr.py` wraps `setup_or_reattach_deployment` from `training/utils/infra.py`.

After a re-attach, call `syncer.reset_delta_chain()` before the next `save_and_hotload` — otherwise the next delta references a base that is not in the new bucket.

### Verify logprobs across train/inference

Source: `training/examples/snippets/verify_logprobs.py`.

---

## Checkpoint state — where things live

| Thing | Location |
|-------|----------|
| Per-run metadata (one line per checkpoint) | `{log_path}/checkpoints.jsonl` |
| DCP state (resume, optimizer + weights) | `state_path` field on each row |
| Sampler blob (promotable, HF format) | `sampler_path` field on each row |
| `CheckpointKind` enum + save helpers | `training/utils/checkpoint_utils.py` |
| Which kinds get saved when | `CheckpointKind` docstring + recipe `dcp_save_interval` + final save |

Rule: a row with `sampler_path` is promotable; a row with only `state_path` is resumable. See `training/utils/checkpoint_utils.py` for the enum.

---

## Where the SDK lives

The training SDK is in the stainless-generated repo: <https://github.com/stainless-sdks/fireworks-ai-python> under `src/fireworks/training/sdk/`. The cookbook imports from it as `fireworks.training.sdk`.

Code outside `src/fireworks/training/sdk/` in that repo is auto-generated and should not be referenced. For the SDK surface agents typically need (`TrainerJobManager`, `DeploymentManager`, `WeightSyncer`, `FireworksClient.promote_checkpoint`, `validate_output_model_id`), read the cookbook recipe files above — they exercise every call an agent needs.

---

For recovery flows (promote fails, hotload fails, deployment drifted), switch to [`../debug/SKILL.md`](../debug/SKILL.md).
