# Checkpoints — where state lives

The cookbook's checkpoint manager is `TrainingCheckpoints` in `training/utils/checkpoints.py`. The control plane is the source of truth for what checkpoints exist; the only cookbook-side state is `dataloader.json`, which maps each saved checkpoint name to a `data_consumed` counter (one int per row).

## Two axes

`TrainingCheckpoints.save(name, *, resumable, promotable, data_consumed=None)` — pick capabilities independently:

- `resumable=True` → DCP write (weights + optimizer). Training can continue from this.
- `promotable=True` → sampler write (HF safetensors). Eligible for `promote_checkpoint`.
- Both → DCP + sampler in one call.

Periodic mid-training saves are usually `resumable=True, promotable=False`. The final save is `resumable=True, promotable=True`. RL weight sync saves sampler checkpoints with `save_weights_for_sampler_ext` and hotloads the returned snapshot identity; those sampler rows are separate from DCP resume saves.

### Snapshot type is part of the contract

The path returned by `save_weights_for_sampler()` is an HF/PEFT sampler
snapshot, not a resumable training checkpoint. It may contain files such as
`adapter_config.json` and `adapter_model.safetensors`, but it has no DCP trainer
state or optimizer state. Do not pass that path to `load_state()` or
`create_training_client_from_state()`.

- Exact continuation: call `save_state()` after a successful `optim_step()` and
  resume from the returned DCP path. This restores weights and optimizer state.
- Weights-only LoRA initialization: use `load_adapter()` (or recipe
  `warm_start_from_adapter`) with the HF/PEFT adapter. The optimizer, schedule,
  recipe step, and data cursor start fresh.
- Sampling/hotload: use the identity returned by
  `save_weights_for_sampler()` with the sampler deployment.

The trainer API rejects an existing non-DCP directory before distributed
checkpoint loading, so a sampler/DCP type mismatch is returned as a
request-level user error rather than a trainer-wide failure.

## `dataloader.json`

Written to `{log_path}/dataloader.json`. Single int per checkpoint name:

```json
{"step-10": 40, "step-50": 200}
```

Bounded to the newest 20 entries. There is no `checkpoints.jsonl` — never has been, in the new model. The control plane (`FireworksClient.list_checkpoints(job_id)`) is queried at resume / promote time for everything else.

## When each axis is used

- `cfg.dcp_save_interval` > 0 → recipe calls `ckpt.save(resumable=True, promotable=False, ...)` every N steps.
- End of training → recipe calls `ckpt.save(resumable=True, promotable=True, ...)`.
- `cfg.output_model_id` set → recipe also calls `ckpt.promote_latest(output_model_id, base_model)`.

If `dcp_save_interval` is `0` (the default), mid-training saves are off — training cannot be resumed from intermediate steps. Set it in the recipe's `Config`.

## Delta chain (sampler `checkpoint_type`)

This is an SDK-level detail, surfaced when you call `save_weights_for_sampler_ext` directly. For full-parameter training, only `base` sampler saves are promotable; `delta` saves are not. LoRA always saves the full adapter, so every LoRA sampler checkpoint is promotable. The SDK-managed sampler backend records the base-then-delta chain for recipe hotload; recipe-level `ckpt.save(promotable=True)` always saves `base`.

`checkpoint_type="merged_base"` is a third, LoRA-only value. Instead of saving the standalone adapter, the trainer folds the active adapter into the base weights (`W <- W + scaling*(B@A)`), strips the adapter metadata, and exports a full base checkpoint that promotes as `INFERENCE_BASE` / `HF_BASE_MODEL`. The session must have a non-trivial adapter — either trained in this run, or loaded via `load_adapter` / the recipe `warm_start_from_adapter`; a fresh LoRA session has zero-delta weights and would export base-identical output. It is a full standalone checkpoint, so — like `base` — it never participates in the delta chain. Note this client-side adapter load is the supported path; control-plane `warmStartFrom` of a LoRA addon is **not** effective and is rejected for service-mode RLOR jobs.

Two ways to use it:

- **During a LoRA training run** — to emit a full `HF_BASE_MODEL` instead of a `HF_PEFT_ADDON`, save the final promotable checkpoint directly with `client.save_weights_for_sampler("final-merged", checkpoint_type="merged_base")` rather than relying on `TrainingCheckpoints.save(promotable=True)` (which always saves `base`). The adapter is already loaded in-session, so no separate merge step is needed.
- **Merge an existing adapter** (no further training) — the standalone `training/examples/tools/merge_lora_and_promote.py` script drives it end to end (provision → `load_adapter` → `merged_base` save → promote).

---

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

Semantics: weights-only load — LoRA A/B matrices initialize from the adapter; optimizer, LR schedule, and data cursor start fresh.

Priority inside `TrainingCheckpoints.resume` (highest first):

1. `init_from_checkpoint` — explicit DCP load. A dedicated
   `<current_job_id>:<checkpoint>` or serverless current-run bare reference
   restores weights, optimizer, recipe step, and the local dataset cursor.
   Dedicated bare/path/cross-job and serverless cross-run references restore
   weights and optimizer but start a new recipe position at step/cursor 0.
2. Newest resumable row on the control plane for the current trainer — auto-resume.
3. `warm_start_from_adapter` — fresh start with adapter weights.
4. None — fresh start from `base_model`.

**Constraints (enforced by `validate_warm_start_config`):**

- `warm_start_from_adapter` and `init_from_checkpoint` are mutually exclusive.
- Requires `lora_rank > 0`. Full-parameter continue-training uses `base_model` instead.

Exposed by `sft_loop`, `dpo_loop`, `orpo_loop`, and `igpo_loop`. The generic
GRPO recipes keep the smaller shared resume surface and use
`init_from_checkpoint` instead.

## Cross-run resume

Auto-resume (priority 2) is **scoped to one trainer job**. If you re-run with the same `log_path` but provision a fresh trainer, the new trainer's `list_checkpoints` is empty and resume falls through to fresh start.

For a full continuation, pin both invocations to the same trainer via
`cfg.trainer.job_id`, keep the same `log_path`, and use auto-resume. An explicit
dedicated full resume uses `"<current_job_id>:<checkpoint>"`; an explicit
serverless current-run full resume uses a bare checkpoint name. The local
`dataloader.json` is cookbook-owned; it must still be present to recover the
dataset cursor.

To initialize a new recipe run from another DCP checkpoint:

- Dedicated: use `init_from_checkpoint=f"{prior_job_id}:step-N"`, an opaque DCP
  URI such as `tinker://` or `gs://`, an absolute/relative DCP path, or a bare
  checkpoint name. A bare dedicated name deliberately keeps the existing
  step-0 initialization behavior used for phase handoffs on one trainer.
- Serverless: use the fully-qualified
  `"<account>/run-<32 lowercase hex>/<checkpoint>"` logical name. Bare names
  refer to the current run. Raw URI (including `gs://`, `tinker://`, and
  `cross_job://`) and absolute-path references are not supported by pooled
  trainers, and `<other-session>:<checkpoint>` is rejected rather than silently
  redirected to the current run.

Cross-job/cross-run DCP initialization restores both weights and optimizer
state, but resets cookbook-owned step and dataset cursor to 0. Use
`warm_start_from_adapter` when you specifically want LoRA weights only and a
fresh optimizer.

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

1. Call `FireworksClient.list_checkpoints(job_id)` (or run the `list_checkpoints.py` CLI wrapper) to see which rows the server will actually accept. See [`sdk-tools.md`](sdk-tools.md#listing-checkpoints-fireworksclientlist_checkpoints).
2. Make sure your `output_model_id` passes `validate_output_model_id` and retry `promote_checkpoint.py` against one of those rows.
3. If every promotable row still fails, **reach out to Fireworks support** — some recoveries (re-staging a sampler blob from surviving DCP state, looking up a pre-migration `PER_DEPLOYMENT` bucket) require server-side access.

The modern promote API takes a single `name=` (4-segment resource path from `list_checkpoints` output) and works identically for `PER_TRAINER` and `PER_DEPLOYMENT`. The legacy positional `(job_id, checkpoint_id, ...)` form still works but emits a `DeprecationWarning`. See [`rl-hotload.md#promoting-a-checkpoint`](rl-hotload.md#promoting-a-checkpoint) and `sdk-tools.md` for the CLI flow.
