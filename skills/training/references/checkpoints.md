# Checkpoints — where state lives

Concept SOT: <https://docs.fireworks.ai/fine-tuning/training-api/cookbook/checkpoints#checkpoint-kinds> covers the `CheckpointKind` matrix (`STATE` / `SAMPLER` / `BOTH`) and the three-layer "type" story. This file is the practical user-level guide: what ends up in `checkpoints.jsonl`, how to validate an `output_model_id` before promoting, and what to do when a promote fails.

## Two layers in one line

- **DCP** (`state_path`) — resume training (optimizer + weights). Not promotable.
- **Sampler** (`sampler_path`) — HF safetensors blob. Promotable (with one exception: full-param `delta` saves).

A `checkpoints.jsonl` row carries either or both.

## `checkpoints.jsonl`

Written to `{log_path}/checkpoints.jsonl` during the run. One JSON object per save:

```json
{"name": "step-10", "step": 10, "data_consumed": 40, "state_path": "cross_job://job-abc/step-10", "source_job_id": "job-abc", "base_model": "accounts/fireworks/models/qwen3-8b"}
{"name": "step-50", "step": 50, "data_consumed": 200, "state_path": "cross_job://job-abc/step-50", "sampler_path": "step-50-a1b2c3d4", "source_job_id": "job-abc", "base_model": "accounts/fireworks/models/qwen3-8b"}
```

`promote_checkpoint.py` reads `name`, `sampler_path`, `source_job_id`, `base_model` from this file — users don't need to touch anything else.

## When each kind is used

- Mid-training saves (`cfg.dcp_save_interval`) → `STATE`.
- Final save at end of training → `BOTH`.
- Want a promotable mid-training checkpoint → call the recipe's `save_checkpoint(..., kind=SAMPLER)` or `BOTH` from inside a custom loop.

If `dcp_save_interval` is `0` (the default), mid-training saves are off — training cannot be resumed from intermediate steps. Set it in the `Config`.

## Delta chain

For full-parameter training, only `base` sampler saves are promotable; `delta` saves are not. LoRA always saves the full adapter, so every LoRA sampler checkpoint is promotable. `WeightSyncer` manages the base-then-delta pattern automatically in the recipes — users don't drive it by hand.

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

1. Run [`list_checkpoints.py`](../../../training/examples/snippets/list_checkpoints.py) (see [`tools.md`](tools.md#list_checkpointspy)) with `--promotable-only` to see which rows the server will actually accept, newest first.
2. Make sure your `output_model_id` passes `validate_output_model_id` and retry `promote_checkpoint.py` against one of those rows.
3. If every promotable row still fails, **reach out to Fireworks support** — some recoveries (re-staging a sampler blob from surviving DCP state, looking up a legacy deployment-first bucket) require server-side access.

For the separate legacy-deployment-first recovery path (`--hot-load-deployment-id`), see [`trainer-first-vs-deployment-first.md`](trainer-first-vs-deployment-first.md).
