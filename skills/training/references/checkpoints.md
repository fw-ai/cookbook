# Checkpoints — where state lives

## Two layers

| Layer | Purpose | Metadata field |
|-------|---------|---------------|
| **DCP** (Distributed Checkpoint) | Resume training with optimizer + weights | `state_path` |
| **Sampler** (HF safetensors) | Promotable inference blob | `sampler_path` |

A row in `checkpoints.jsonl` can carry either or both, depending on the `CheckpointKind` the recipe used:

| Kind | Fields present | Resumable | Promotable |
|------|----------------|-----------|------------|
| `STATE` | `state_path` | Yes | No |
| `SAMPLER` | `sampler_path` | No | Yes |
| `BOTH` | both | Yes | Yes |

The enum and save helpers live in `training/utils/checkpoint_utils.py`.

## `checkpoints.jsonl`

Written to `{log_path}/checkpoints.jsonl` during the run. One JSON object per save:

```json
{"name": "step-10", "step": 10, "data_consumed": 40, "state_path": "cross_job://job-abc/step-10", "source_job_id": "job-abc", "base_model": "accounts/fireworks/models/qwen3-8b"}
{"name": "step-50", "step": 50, "data_consumed": 200, "state_path": "cross_job://job-abc/step-50", "sampler_path": "step-50-a1b2c3d4", "source_job_id": "job-abc", "base_model": "accounts/fireworks/models/qwen3-8b"}
```

Fields the promote tool reads: `name`, `sampler_path`, `source_job_id`, `base_model`.

## When each kind is used

- Mid-training saves (`cfg.dcp_save_interval`) → `STATE`.
- Final save at end of training → `BOTH`.
- Explicit promote-friendly save → call the recipe's `save_checkpoint(..., kind=SAMPLER)` or `BOTH` from inside a custom loop.

If `dcp_save_interval` is `0` (the default), mid-training saves are off — training cannot be resumed from intermediate steps. Set it in the `Config`.

## Raw SDK saves (outside the cookbook)

`training_client.save_weights_for_sampler_ext(...)` writes a sampler blob to GCS but does **not** write a `checkpoints.jsonl` row. The promote tool then needs the full coordinates — grab them from the returned `snapshot_name` and the client's `job_id`.

On the trainer side, firetitan logs `[save_weights_for_sampler] promote_ready snapshot_name=<id> base_model=<m> result_path=gs://...` after every successful save. That line is the authoritative source for all three coordinates.

## Delta chain

For full-parameter training, `save_weights_for_sampler_ext(checkpoint_type="delta")` saves an XOR diff over the previous base. Deltas are not promotable on their own — only `base` saves are. LoRA always saves the full adapter regardless of `checkpoint_type`, so every LoRA checkpoint is promotable.

`WeightSyncer` manages the base-then-delta pattern automatically. Call `reset_delta_chain()` whenever the deployment's bucket changes (e.g. after a re-attach).

---

## Listing checkpoints on a trainer

`checkpoints.jsonl` only covers rows the cookbook wrote. For the authoritative list on a trainer (including checkpoints inherited from a predecessor via hotload), call the server:

```bash
curl "https://api.fireworks.ai/v1/accounts/<account>/rlorTrainerJobs/<job>/checkpoints?pageSize=200" \
  -H "Authorization: Bearer $FIREWORKS_API_KEY"
```

Each entry includes `name`, `createTime`, `updateTime`, `checkpointType` (`INFERENCE_BASE`, `INFERENCE_LORA`, `INFERENCE_ARC_V2`), and `promotable`. Pick the **latest `createTime` with `promotable: true`** — step numbers mislead when a trainer inherits from a predecessor.

---

## `output_model_id` validation

**Cap: 63 chars, charset `[a-z0-9-]`.** A longer or otherwise-invalid ID is rejected server-side with HTTP 400, but the sampler blob staged for the failed promote is **not rolled back** — it lingers in GCS until GC, after which the same `checkpoint_id` returns "not found in GCS".

Validate client-side before every promote:

```python
from fireworks.training.sdk import validate_output_model_id

errors = validate_output_model_id(output_model_id)
if errors:
    raise ValueError("\n".join(errors))
```

When deriving `output_model_id` from a long policy / run name, trim the prefix before the step suffix:

```python
# Bad: "<long-workflow>-<long-policy>-<long-base-model>-step-14"  # 70+ chars → HTTP 400
# Good: "my-policy-step-14"
output_model_id = f"{short_policy}-step-{step}"[:63].rstrip("-")
```

---

## Recovery: sampler orphan

Symptom: `promote_checkpoint` returns `checkpoint "<name>" not found in GCS`, but the list endpoint above still shows the record.

Root cause (almost always): an earlier `promote_checkpoint` for the same `checkpoint_id` was rejected for `output_model_id` length or charset. The staged sampler blob was never anchored to a Model, and GC reclaimed the bytes. The record remains.

Recovery:

1. List checkpoints (above). Pick a different entry with `promotable: true`, preferring the latest `createTime`.
2. Validate your `output_model_id` with `validate_output_model_id`.
3. Retry `promote_checkpoint.py` against the new checkpoint.

If every remaining row is `promotable: false` but a DCP `state_path` survives for the step you want: spin up a fresh trainer, `load_state_with_optimizer(state_path)`, `save_weights_for_sampler_ext(checkpoint_type="base")`, then promote the fresh blob.

Legacy deployment-first: see [`trainer-first-vs-deployment-first.md`](trainer-first-vs-deployment-first.md) — the recovery path needs `--hot-load-deployment-id`.
