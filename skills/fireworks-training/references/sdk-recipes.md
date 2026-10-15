# Recipes — fork, don't reinvent

Each recipe is a single Python file in `training/recipes/` that wires the Training API SDK. Copy, edit the `Config` at the top, run.

| Task | File |
|------|------|
| SFT | `training/recipes/sft_loop.py` |
| DPO | `training/recipes/dpo_loop.py` |
| ORPO | `training/recipes/orpo_loop.py` |
| **RL (primary)** — write a rollout function; recipe owns the loop. Async by default; strict on-policy scheduling via `max_head_offpolicy_versions=0` | `training/recipes/async_rl_loop.py` — see [`rl-async.md`](rl-async.md) |
| Async RL on the shared serverless pool (experimental) | `training/recipes/experiment/async_rl_loop_serverless.py` — same rollout contract; serverless snapshot publication |
| RL (simpler, synchronous GRPO reference loop) | `training/recipes/rl_loop.py` |
| SAO / token-level PPO with a separate value-head critic (experimental; not validated for every model) | `training/recipes/experiment/ppo_value_head_loop.py` — runnable Qwen3-4B example in `training/examples/rl/experiment/sao/qwen3_4b_sao.py` |
| Information Gain-based Policy Optimization (IGPO) | `training/recipes/rl_loop.py` with `prepare_batch_fn=igpo.prepare_batch(gamma=...)` — strict on-policy; each run sets `metadata[TURN_SPANS]` and `metadata[ANSWER_TOKENS]`; multi-turn example in `training/examples/rl/multihop_qa/` |
| Distillation / OPD | `training/recipes/distillation_loop.py` — see [`sdk-distillation.md`](sdk-distillation.md) |
| SDFT | [Customer guide](../../../training/docs/sdft.md); maintained in the external Fireworks Tinker-Cookbook fork |

## "Reference loop" means these files

They are the canonical wiring of `FiretitanServiceClient` + `FiretitanTrainingClient` + `TrainingCheckpoints` + deployment sampler hotload. Do not rewrite — fork.

## What to fill in on `Config`

Always required on `Config` (with `trainer=TrainerConfig(...)`):

- `base_model` — `accounts/fireworks/models/<name>`
- `dataset` — path to JSONL
- `dataset_loading_mode` — optional, default `"stage"`. Leave this default for
  normal recipe runs. `"stream"` is for Fireworks-managed SFT/DPO jobs whose
  dataset is a `gs://` JSONL object or prefix and is validated/read by bounded
  GCS range reads instead of staging the payload onto the orchestrator disk.
  Stream mode does not support local paths, CMEK datasets, or
  `group_by_length`.
- `tokenizer_model` — HF model name
- `log_path` — directory for `dataloader.json` and logs
- `trainer.training_shape_id` — optional override; leave unset for auto-selection. `accelerator_type` and `accelerator_count` are unsupported; do not set manual `node_count` (see [`sdk-shapes.md`](sdk-shapes.md))
- `trainer.use_reservation` — optional, default `True`. Tries reservation
  capacity first; set `False` to use shared trainer capacity. A
  full-parameter DPO reference trainer inherits the option and tries
  independently; an existing `trainer.job_id` is reused as-is.

The SAO/PPO recipe uses `actor_trainer` and `critic_trainer` instead of one
`trainer`; configure the two shapes and resume IDs independently.

RL-specific: use `async_rl_loop.py` for the primary rollout-function workflow,
`rl_loop.py` for a simpler synchronous loop, and
`recipes/experiment/ppo_value_head_loop.py` when the method needs a separately trained critic. Projection-head SAO is experimental and is not validated for every model.
See [`rl-async.md`](rl-async.md) for the async workflow.

Both GRPO loops train the trainer's built-in GRPO loss against the rollout
logprobs, with no reference model or KL. Their only objective fields are
`eps_clip` and `eps_clip_high`. Other objectives, reference KL, a recomputed
trainer anchor and sampling-support replay are forks of the train step; see
[`rl-custom-loss.md`](rl-custom-loss.md).

Forward/backward metrics retain Tinker's reducer suffixes after SDK request
chunking. Runtime input-sharding telemetry therefore appears as
`train/dp_sharded_counts:min`, `train/dp_sharded_counts:max`, and
`train/local_input_sequences:sum`. Consumers that need proof of the executed
path must require min and max to agree, then use the summed rank-local input
count. Do not change these to `:last`: Tinker does not support that reducer for
chunked forward/backward results and silently drops those metrics.

Distillation-specific: use `distillation_loop.py` for sampled-token OPD and the [external SDFT cookbook](https://github.com/fw-ai-external/tinker-cookbook) for top-K SDFT. Open [`sdk-distillation.md`](sdk-distillation.md) before changing its config, dataset format, or teacher scoring.

## Resume

Auto-resume is scoped to one trainer. Pin both runs to the same trainer via `cfg.trainer.job_id` (all recipes; the reference trainer is SDK-managed, so there is no separate reference job id to pin), keep the same `log_path`, and rerun. `TrainingCheckpoints.resume()` lists the trainer's checkpoints on the control plane, picks the newest resumable row, and restores the rollout cursor from `dataloader.json`. See [`sdk-checkpoints.md`](sdk-checkpoints.md) for the full priority order and constraints.

## Initialize from another job

```python
config = Config(
    log_path="./new_run",
    init_from_checkpoint="i44pvd4syzg8hjfk:step-4",  # job_id:checkpoint_name
    ...
)
```

Loads DCP weights and optimizer state from the other job, then starts the new
recipe at step/cursor 0. For a full continuation, reattach to the same trainer,
keep the same `log_path`, and use auto-resume or
`"<current_job_id>:<checkpoint>"`. See
[`sdk-checkpoints.md`](sdk-checkpoints.md#cross-run-resume) for the serverless
cross-run reference format.

## Optional SFT generation comparison

For serverless SFT, set `generation_eval=True` (managed SFT: `generation_eval`,
`firectl --generation-eval`). The recipe uses `evaluation_dataset`, or a seeded
20% holdout excluded from training, and samples up to `max_eval_seqs` text-only
conversations. The last assistant turn selected by `train_on_what` and the
normalized per-message flags is the reference; earlier turns
are the prompt. After training, the base model and the final LoRA weights answer
each prompt with temperature 0 and at most 1024 generated tokens.
`log_path/generation_eval.json` (managed: `generation_eval_signed_url`) stores
the prompts, references, answers and status. Its `checkpoints` are
`["Original", "Your model"]`; each `examples` entry has string `prompt` and
`reference` fields and two ordered `cells` with `output`, `tokens`, and
`stop_reason`. Prompts include all preceding turns with role labels; structured
message content is JSON-encoded. Cells without a generated answer have only
`output: ""`. There is no automatic scoring: `isCorrect` and `reason` are omitted,
so consumers must treat them as optional and hide correctness badges when absent.
Generation prompts are validated before training. Unsupported or invalid prompts
fail validation; an inference failure stops evaluation and saves partial answers
without failing the training job.
Managed jobs enter `JOB_STATE_EVALUATION` during generation and complete only
afterwards, so session cleanup follows evaluation; `state_timestamps` keeps the
evaluation entry even when it finishes between watcher polls.

## RL specifics

RL details stay in this skill. Open only the relevant reference when working with the RL loops:

- [`rl-custom-loss.md`](rl-custom-loss.md) — what the loops train; swap the objective, add reference KL, recompute the anchor, replay sampling support
- [`rl-gradient-accumulation.md`](rl-gradient-accumulation.md) — `optim_step` normalization; the trap custom losses fall into
- [`rl-dynamic-filter.md`](rl-dynamic-filter.md) — `should_accept`, why zero-variance groups get dropped
- [`rl-hotload.md`](rl-hotload.md) — strict sync hotload, `weight_sync_timeout`, on-policy vs off-policy, base/delta chain
- [`rl-concurrency.md`](rl-concurrency.md) — sync batch concurrency vs async sample-level admission
- [`rl-sampling-timeouts.md`](rl-sampling-timeouts.md) — diagnose `DeploymentSamplerTimeoutError` from request shape and serving metrics
- [`rl-async.md`](rl-async.md) — `async_rl_loop.py` overlap recipe: sample-level cap, off-policy budget, complete-group submission
- [`client-phase-tracing.md`](client-phase-tracing.md) — export client-loop and custom harness spans to Perfetto, with an optional OpenTelemetry mirror

SFT / DPO / ORPO users do not need these.
