# Checkpoints — one remote source of truth

`TrainingCheckpoints` in `training/utils/checkpoints.py` is the cookbook boundary for resume, promotion, and sampler weight sync. Checkpoint identity and metadata live on the Fireworks control plane. Recipes discover them through `list_checkpoints`; they do not maintain a second checkpoint registry.

The only local checkpoint state is the dataset position that the trainer cannot infer:

```json
{
  "trainer-job-a": {"10": 40, "50": 200},
  "trainer-job-b": {"8": 32}
}
```

This is `{log_path}/dataloader.json`, a bounded KV mapping from trainer job id and checkpoint step to the next raw row cursor. It contains no paths, checkpoint types, timestamps, or promotion metadata. Each trainer keeps its newest 20 cursor entries. The former flat `{"step-10": 40}` shape is migrated when it is read.

## Save and sync APIs

`TrainingCheckpoints.save(step, *, resumable, promotable, row_cursor)` keeps the two save capabilities independent:

- `resumable=True` writes DCP weights and optimizer state and records `(trainer_job_id, actual_saved_step) -> row_cursor` locally.
- `promotable=True` writes a complete sampler export that can be promoted.
- Both perform both writes under the same `step-N` logical name.

Periodic saves use `resumable=True, promotable=False`; the final save usually uses both. `row_cursor` is required for every resumable save. If `dcp_save_interval=0` (the default), there are no intermediate resume points.

RL recipes call `TrainingCheckpoints.sync_weights(step, hotload)`. They never pass `checkpoint_type` or branch on LoRA versus full-parameter training. The SDK saves a complete LoRA adapter for LoRA runs and manages the full/base/delta chain for full-parameter runs. The checkpoint manager also hides the complete-export choice required for promotion.

`checkpoint_type="merged_base"` remains an explicit, specialized LoRA export operation: it folds an adapter into its base model to produce a standalone `HF_BASE_MODEL`. Use `training/examples/tools/merge_lora_and_promote.py` for that workflow; ordinary recipes should not choose checkpoint formats.

## Resume

`TrainingCheckpoints.list()` returns the authoritative RPC rows. A user may pass any of these forms as `init_from_checkpoint`:

- the row dictionary returned by `ckpt.list()` or `FireworksClient.list_checkpoints(job_id)`;
- the row's full checkpoint resource name;
- `job_id:step-N`;
- a bare `step-N` for the current trainer.

The checkpoint row determines the trainer state and step. The local KV mapping supplies only its row cursor. If the mapping has no entry, the cursor is `0`.

Every recipe also exposes `dataloader_cursor`. When it is explicitly set, that exact cursor is used and the local mapping is not read. Use this for a deliberate data-position override or when resuming a remote checkpoint without its original `dataloader.json`.

Resume priority is:

1. `init_from_checkpoint` — load the explicit resumable checkpoint row/resource.
2. Newest resumable RPC row for the current trainer — auto-resume.
3. `warm_start_from_adapter` — load LoRA weights only and start at step/cursor zero.
4. Fresh start.

RPC list failures propagate instead of silently starting fresh.

## Warm-start from a promoted adapter (LoRA only)

To continue LoRA training from a previously-trained adapter — typically a promoted Fireworks Model or any HF PEFT directory — set `warm_start_from_adapter` on the recipe `Config`:

```python
cfg = Config(
    base_model="accounts/fireworks/models/qwen3-8b",
    lora_rank=16,
    warm_start_from_adapter="gs://bucket/path/to/adapter-dir",
    ...
)
```

Semantics: weights-only load — LoRA A/B matrices initialize from the adapter; optimizer, LR schedule, step, and row cursor start fresh.

**Constraints (enforced by `validate_warm_start_config`):**

- `warm_start_from_adapter` and `init_from_checkpoint` are mutually exclusive.
- Requires `lora_rank > 0`. Full-parameter continue-training uses `base_model` instead.

Supported in all recipe loops: `sft_loop`, `dpo_loop`, `orpo_loop`, `rl_loop`, `async_rl_loop`, `igpo_loop`.

## Cross-run resume

Auto-resume is scoped to the current trainer job. To continue a prior trainer, pin `cfg.trainer.job_id`; its RPC rows and local row cursors will line up automatically.

To resume into another trainer, pass the prior RPC row (preferred), its full resource name, or `f"{prior_job_id}:step-N"`. Cross-job resume preserves checkpoint step `N`. The cursor is looked up under the source trainer id, or taken directly from `dataloader_cursor` when provided.

---

## `output_model_id` validation

**Cap: 63 chars, charset `[a-z0-9-]`.** A longer or otherwise-invalid ID is rejected server-side with HTTP 400. Once rejected, that same `checkpoint_id` can later return "not found in GCS" (the staged blob is no longer usable).

**Validate client-side before every promote:**

```python
from fireworks.training.sdk import validate_output_model_id

errors = validate_output_model_id(output_model_id)
if errors:
    raise ValueError("\n".join(errors))
```

When deriving `output_model_id` from a long policy / run name, trim the prefix before the step suffix:

```python
# Bad:  "<long-workflow>-<long-policy>-<long-base-model>-step-14"  # 70+ chars → HTTP 400
# Good: "my-policy-step-14"
output_model_id = f"{short_policy}-step-{step}"[:63].rstrip("-")
```

---

## When promote fails

If `promote_checkpoint` returns `checkpoint "<name>" not found in GCS`:

1. Call `FireworksClient.list_checkpoints(job_id)` (or run the `list_checkpoints.py` CLI wrapper) to see which rows the server will actually accept. See [`tools.md`](tools.md#listing-checkpoints-fireworksclientlist_checkpoints).
2. Make sure your `output_model_id` passes `validate_output_model_id` and retry `promote_checkpoint.py` against one of those rows.
3. If every promotable row still fails, **reach out to Fireworks support** — some recoveries (re-staging a sampler blob from surviving DCP state, looking up a pre-migration `PER_DEPLOYMENT` bucket) require server-side access.

The modern promote API takes a single `name=` (4-segment resource path from `list_checkpoints` output) and works identically for `PER_TRAINER` and `PER_DEPLOYMENT`. The legacy positional `(job_id, checkpoint_id, ...)` form still works but emits a `DeprecationWarning`. See [`rl/hotload.md#promoting-a-checkpoint`](rl/hotload.md#promoting-a-checkpoint) and `tools.md` for the CLI flow.
