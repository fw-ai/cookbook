# Utils: what to call, what to pass

Shared utilities under `training/utils/`. Recipes import from here -- do not reinvent. This page lists each module's public entry points with the arguments that matter.

## `utils/config.py` -- dataclasses

| Config | Key fields |
|--------|-----------|
| `InfraConfig` | `training_shape_id`, `ref_training_shape_id`, `region`, `accelerator_type/count`, `custom_image_tag`, `trainer_timeout_s`, `extra_args`, `skip_validations` |
| `DeployConfig` | `deployment_id`, `deployment_shape`, `hot_load_trainer_job`, `tokenizer_model` (required for RL), `deployment_timeout_s`, `replica_count`, `extra_args`, `extra_values` |
| `WeightSyncConfig` | `weight_sync_interval` (hotload every N steps), `weight_sync_before_training`, `first_checkpoint_type` (`"base"` \| `"delta"`), `weight_sync_timeout`, `dcp_save_interval` |
| `WandBConfig` | `entity`, `project`, `run_name`, `tags` |
| `RunnerConfig` | `log_path`, `metrics_file`, `status_file_cadence_s` |
| `ConcurrencyConfig` | `mode` (`"fixed"` \| `"adaptive"`), `max_concurrency` (fixed), `initial_window`/`min_window`/`max_window`/`prefill_queue_target` (adaptive) |
| `ResumeConfig` | `resume_from_checkpoint`, `resume_step`, `resume_lr_factor` |
| `ISConfig` | `ratio_log_cap`, `clip_high`, `clip_low` (RL importance-sampling clip caps) |

Reuse semantics: see [`recipes.md`](recipes.md).

## `utils/infra.py` -- orchestration

### `create_trainer_job(rlor_mgr, *, base_model, infra, profile=None, job_id=None, forward_only=False, ...) -> TrainerServiceEndpoint`

Create or reuse a trainer.

- `job_id=None` → new trainer (shape path via `profile` or manual path via `infra` fields)
- `job_id="..."` → reuse existing trainer (resumes if FAILED/CANCELLED/PAUSED)
- `job_id="..."` + `base_url_override="..."` → skip all health polling; return endpoint pointing at the URL directly (smoke-test shortcut)
- `forward_only=True` → reference trainer for DPO (different shape required via `ref_training_shape_id`)
- `on_status=callback` → invoked with progress strings (`"creating trainer job"`, etc.)

### `setup_deployment(deploy_mgr, deploy_cfg, base_model, infra) -> DeploymentInfo`

Idempotent on `deploy_cfg.deployment_id`:

- ID unset → auto-generates `<model_short>-<timestamp>`, creates fresh
- ID set, deployment exists → fetches and waits for READY/UPDATING
- ID set, no deployment → creates with that ID

Does NOT re-attach to a new trainer. Use `setup_or_reattach_deployment` for that.

### `setup_or_reattach_deployment(deploy_mgr, deploy_cfg, base_model, infra, trainer_job_name, weight_syncer=None, reattach_settle_timeout_s=600) -> DeploymentInfo`

The recommended entry point for RL. See [`reattach.md`](reattach.md) for the decision logic.

## `utils/client.py` -- `ReconnectableClient`

```python
policy = ReconnectableClient(
    rlor_mgr=trainer_mgr,
    job_id=endpoint.job_id,
    base_model=base_model,
    lora_rank=0,
    fw_api_key=api_key,
)
policy.inner             # underlying FiretitanTrainingClient
policy.close()           # clean teardown
```

Wraps `FiretitanTrainingClient` with dispatch/wait logic and reconnection (`TrainerJobManager.wait_for_existing`). Always close in a `finally:` block.

## `utils/data.py` -- `RLPromptDataset`

```python
dataset = RLPromptDataset(
    dataset_path="my_prompts.jsonl",       # local path or gs:// URL
    max_rows=1000,
    shuffle=True,
    seed=42,
)
batch = dataset.get_batch(step_idx, batch_size)
# batch: list[dict] with "prompt" (str) or "messages" (chat format)
```

## `utils/losses.py` -- SL / preference losses

| Function | Returns |
|----------|---------|
| `make_sft_loss_fn(...)` | Response-only CE, compatible with `forward_backward_custom` |
| `make_dpo_loss_fn(beta, ...)` | DPO loss; requires reference logprobs in `loss_fn_inputs` |
| `make_orpo_loss_fn(beta, sft_weight, ...)` | Combined SFT + odds-ratio loss |

RL losses in `utils/rl/` -- see [`rl.md`](rl.md).

## `utils/checkpoint_utils.py`

### `save_checkpoint(trainer_client, log_path, step, snapshot_name, *, kind=CheckpointKind.BOTH, base_model=None, training_shape=None)`

- `CheckpointKind.STATE` → optimizer + weights (DCP)
- `CheckpointKind.SAMPLER` → weights only (for inference hotload)
- `CheckpointKind.BOTH` → default; writes both and appends to `checkpoints.jsonl`

`base_model` and `training_shape` are persisted into the entry so `tools/promote_checkpoint.py` can auto-detect them.

### `resolve_resume(policy_client, log_path) -> ResumeInfo | None`

Reads `checkpoints.jsonl` to find the latest state. Returns `None` if no checkpoints.

### `CHECKPOINTS_BASE_NAME`

Default base name for DCP checkpoints within a session. Don't change unless you know why.

## `utils/fileio.py`

Transparent local / GCS I/O. All functions accept both local paths and `gs://` URLs.

```python
from training.utils.fileio import read_jsonl, write_jsonl, exists, read_text, write_text

data = read_jsonl("gs://my-bucket/dataset.jsonl")
write_jsonl(data, "local_output.jsonl")
if exists("gs://my-bucket/checkpoint/"): ...
```

## `utils/logging.py`

```python
wandb_init(config, log_path)     # offline-mode fallback if no API key; safe to call unconditionally
wandb_log(metrics, step)         # also appends to metrics.jsonl in log_path
wandb_finish()                   # call at end
```

## `utils/timer.py`

```python
with timer("forward_backward"):
    tc.forward_backward(...)
timer.get_summary()              # {"forward_backward": 2.1, ...}
```

Auto-logged to WandB when `wandb_log` is called.

## `utils/runner.py`

Orchestration-compatible status writer. Emits `status.json` (protojson-compatible state: `RUNNING`/`COMPLETED`/`FAILED`), `metadata.json` (trainer/deployment IDs, config snapshot), `metrics.jsonl`.

Used under job orchestrators (Temporal). Standalone runs work fine with defaults.

## `utils/training_shapes.py` -- auto-selection

`resolve_shape_profile(rlor_mgr, base_model, infra, *, forward_only=False) -> TrainingShapeProfile | None` -- queries control-plane for validated shapes. Returns `None` if none match (fall back to manual path).

`TrainingShapeProfile` fields: `training_shape_version`, `region`, `accelerator_type`, `trainer_image_tag`, `max_supported_context_length`, `deployment_shape_version`, `deployment_image_tag`.

## `utils/validation.py`

### `validate_preflight(config)`

Run at the top of your recipe's `main()`. Fails fast on:

- `FIREWORKS_API_KEY` unset
- `base_model` not found via API
- `dataset` not readable
- WandB entity invalid (if set)

## `utils/supervised.py` -- SL rendering

Used by SFT/DPO/ORPO to produce `Datum` objects with token-level masks.

| Function | Use when |
|----------|----------|
| `conversation_to_datum(messages, renderer, max_length, train_on_what)` | Full pipeline; most callers |
| `build_supervised_datum(messages, renderer, max_length, train_on)` | Same, alternate name |
| `datum_from_model_input_weights(model_input, weights, max_length)` | You already have `(ModelInput, weights)` from a custom renderer |

`train_on_what`: `"assistant_only"` (default), `"all"`, or `"last_turn"`.

Renderers live in `training/renderer/` -- see [`renderers.md`](renderers.md).
