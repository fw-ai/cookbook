# RL: async recipe (`async_rl_loop.py`)

> ⚠️ **EXPERIMENTAL — under active development.** API surface and config
> field names may change without backward-compat shims.  Pin to a specific
> commit if you depend on the current shape.  The recipe also emits a
> runtime `WARNING` at `main()` start.

The async recipe runs rollout sampling and training through one pipelined
scheduler loop behind a gate that controls **submission headroom**.  Be precise
about the separate ideas:

- **Scheduler pipeline chunking** means ready GRPO prompt groups are dispatched
  to the trainer as soon as they arrive, but the trainer runs one `optim_step`
  only after the full `prompt_groups_per_step` global batch is accumulated.
  `pipeline_chunks_per_step` controls the maximum scheduler chunk size; trainer
  continuous batching owns execution-level coalescing/microbatching.
- **On-policy training** means the behavior policy that generated the trained
  tokens matches the policy being updated.  AReaL tracks this with per-token
  `versions` in rollout outputs.  Our async recipe currently tracks the row's
  **submit version** (`async/version_offset_*`), which is only a proxy for the
  behavior-policy version.
- `max_head_offpolicy_versions=O` is an **admission budget**, not a completion
  guarantee.  It controls whether the runner may submit more rows before future
  weight syncs happen.  Once a request is submitted, the network/model runtime
  can still make it finish before or after later syncs.

For strict on-policy training, use `max_head_offpolicy_versions=0`.  Raising
`O` permits more head-of-line submission and can create useful overlap, but
`async/version_offset_*` remains only a submission-lag proxy. The stock path
uses rollout behavior logprobs for TIS but does not emit a direct
behavior-policy divergence metric. Add that metric explicitly in a research
fork if the experiment needs it.

`rl_loop.py` is the older synchronous recipe (always strict on-policy, drains
rollouts before each step).  The async recipe is the recommended starting
point for new RL work because it supports both strict on-policy scheduling
(`O=0`) and off-policy overlap (`O>0`) without a separate mode flag.

`prompt_groups_per_step` is the global optimizer batch size.
`pipeline_chunks_per_step` caps the number of scheduler chunks inside that
global batch.  Internally the runner passes each dispatch a
`run_optimizer_step` control flag; intermediate chunks run forward/backward
only, while the optimizer step runs at the global batch boundary.  Do not force
`max_concurrency_rollout_sample` to the global batch size for on-policy runs:
with `max_head_offpolicy_versions=0`, the staleness gate is already the
on-policy admission cap.  Keep the recipe default or any explicit user-provided
concurrency cap.

**Two-file layout.** A run is typically a `rollout.py` (the rollout function,
`make_rollout_fn(setup) -> rollout_fn`) plus a `train.py` (the `Config` —
training/deployment shapes and reward wiring — and the
`main(cfg, rollout_fn_factory=..., rows=...)` call). The reward is computed
inside the rollout and set on each `RolloutSample.reward` segment inside a
`RolloutRun` (often factored into a `reward.py` the rollout imports). The
recipe owns everything between.

## What you customize (and what you don't)

The recipe is intentionally minimal-surface: **the only thing most users need
to write is the rollout function**.  Everything else — gate, advantage
computation, optional reference and old-policy forwards, weight sync, TIS, pipeline
chunking, checkpoint plumbing — is handled by `recipes/async_rl_loop.py::main`.

| You write | The recipe handles |
|---|---|
| `rollout_fn_factory(setup) -> rollout_fn` (one trajectory per call) | Async fan-out / GroupAssembler / off-policy gate |
| (optional) `dynamic_filter_fn(pg)` for batch-level filtering | Reference/old-policy forwards, KL, and TIS correction |
| Dataset rows (or pass `rows=` to `main()`) | Weight sync cadence, checkpoint save/promote, WandB axes |
| `Config(...)` knobs (LR, gate sizes, deployment shape) | Pipeline chunking, gradient accumulation, advantage z-score |

This is the design intent — extend the rollout, not the loop.  If you find
yourself forking the recipe, file an issue first; it usually means a knob
should exist on `Config`.

| | `rl_loop.py` (sync) | `async_rl_loop.py` (async) |
|---|---|---|
| Sampler/trainer overlap | None — drains rollouts before each step | Concurrent by default; ready prompt groups can overlap rollout with trainer fwd/bwd while delaying `optim_step` to the global batch boundary |
| Rollout API | `make_rollout_fn(rl_cfg, deploy_mgr) -> rollout_fn` | `rollout_fn_factory(setup) -> rollout_fn` |
| Per-call signature | `rollout_fn(prompt, n)` returns N samples | `rollout_fn(sample_prompt) -> RolloutRun \| None` (one trajectory per call) |
| Concurrency | `ConcurrencyConfig` (adaptive AIMD) | Sample-level cap on the async runner (`max_concurrency_rollout_sample`) |
| On-/off-policy | Strict on-policy only | `async/version_offset_*` measures submit-version lag; true behavior-policy staleness requires generation-version metadata |
| Optimizer boundary | Can run multiple inner optimizer steps | One optimizer step per rollout batch. Pipeline: `pipeline_chunks_per_step` fwd/bwd chunks + one `optim_step` |

Prefer the async recipe for new work.  Use `max_head_offpolicy_versions=0` for
strict on-policy scheduling, and raise it only when you want off-policy overlap.
The sync recipe stays for users who already depend on it.

## The rollout API

The user supplies one trajectory per call:

```python
async def rollout_fn(sample_prompt: dict) -> RolloutRun | None: ...
```

`sample_prompt` is the dataset row's dict, renamed once it crosses the
dataset/sampling seam (the dataset emits a "row"; the sampler treats it as a
prompt to draw a trajectory against).  Return `None` to drop that trajectory
draw (counts as one lost run within the row's group).

`RolloutRun` is one trajectory. It contains one or more trainable
`RolloutSample` segments that share the trajectory reward:

```python
@dataclass
class RolloutRun:
    segments: list[RolloutSample]
```

Each `RolloutSample` segment is three parallel lists plus a scalar reward:

```python
@dataclass
class RolloutSample:
    tokens: list[int]
    logprobs: list[float]   # 0.0 on non-generated positions
    loss_mask: list[int]    # 1 on assistant tokens, 0 elsewhere
    reward: float
    routing_matrices: list[str] | None = None  # MoE R3 replay
    finish_reason: str = "stop"
    text: str = ""
```

Multi-turn rollouts flatten into the same shape: turn boundaries are implicit
in `loss_mask` transitions (0 on prompts/user/tool, 1 on assistant).  Per-token
mask alignment is the contract — the trainer relies on it to mask non-generated
positions out of the loss, TIS weight, and entropy metric.

For the common one-segment case, wrap the segment:

```python
return RolloutRun(segments=[sample])
```

Multi-segment runs are for trajectories that produce multiple disjoint
trainable traces under one trajectory reward. The recipe computes advantage at
the run level and broadcasts it to each segment.

The factory pattern keeps per-rollout dependencies (sampler, tokenizer, sample
kwargs, custom state) out of the per-call signature:

```python
def make_rollout_fn(setup: RolloutSetup) -> RolloutFn:
    sampler = DeploymentSampler(setup.inference_base_url, setup.model, setup.api_key, setup.tokenizer)
    async def rollout_fn(sample_prompt: dict) -> RolloutRun | None:
        ...
    return rollout_fn
```

`RolloutSetup.extras` carries arbitrary caller state (replaces the older
`RolloutContext.ctx_extras` setattr injection).

## Black-box coding-agent example

Use `training/examples/rl/coding_agent/` when the user wants to train on an
unmodified coding-agent harness.  The public example intentionally keeps one
path: NVIDIA ProRL-Agent-Server SWE-Gym parity with local Docker runtimes,
public SWE-Gym/SWE-bench images, and fresh-runtime SWE-bench grading.

The flow is:

1. `async_rl_loop` calls `rollout_fn(sample_prompt)` once per trajectory draw.
2. The rollout opens a shim session and launches the real agent CLI in the
   SWE-Gym Docker runtime named by the row metadata.
3. The agent sends Anthropic `/v1/messages` traffic to the shim.  The shim
   renders each turn, calls Fireworks `DeploymentSampler` token-in/token-out,
   records prompt/output tokens plus per-token logprobs, and translates the
   response back to Anthropic format.
4. The rollout captures the produced `git diff`, grades it in a fresh copy of
   the same SWE-Gym runtime, drains the session, and returns one `RolloutRun`
   whose segments share the run reward.

Comparison target checked for this example:

- `NVIDIA-NeMo/ProRL-Agent-Server` `examples/swegym_slime_grpo/sample_tasks.py`
  and `prepare_data.py` for the public `NovaSky-AI/SkyRL-v0-293-data` split,
  prompt row shape, and image naming.
- `examples/swegym_slime_grpo/polar_config.yaml` for SWE-bench harness grading,
  patch filtering, and runtime layout.
- `examples/swegym_slime_grpo/run.sh` for GRPO task/batch sizing.

Fireworks-specific substitutions are intentional: `async_rl_loop` owns the
trainer loop, Fireworks `DeploymentSampler` provides TITO sampling/logprobs,
and the deployment tokenizer/renderer path is used instead of the ProRL
Polar/Slime/SGLang serving stack.  Do not describe the cookbook example as
infra-identical to ProRL; parity here means the public task, image, grader, and
batch recipe.

Public-facing guardrails:

- Do not expose internal template IDs, team IDs, GCS service-account material,
  private backend paths, or checked-in credentials.
- Keep setup examples to user-provided `FIREWORKS_API_KEY`,
  user-provided `WANDB_API_KEY` when needed, public Docker images, and public
  dataset names.
- For multi-turn cache reuse, forward the per-trajectory session id as the
  completions `user` field and leave prompt caching enabled.
- Keep docs and tests centered on the ProRL SWE-Gym path.

## The off-policy gate (sample-level)

Bookkeeping is in samples (LLM calls), the same unit the deployment's
`max_batch_size` gates on, but admission is **row-atomic**: the runner
submits whole rows (`cpp` samples each) so every row in the assembler has
the same submit version.  A row is admitted iff *both* caps have at least
`cpp` sample slots free:

```
staleness_slots  = (max_head_offpolicy_versions + version + 1) * batch_size_samples
                   - (samples_in_flight + samples_accepted)
concurrency_slots = max_concurrency_rollout_sample - samples_in_flight   (∞ if unset)

admit one row iff min(staleness_slots, concurrency_slots) >= completions_per_prompt
```

In the runner this is `slots = capacity() // completions_per_prompt`; that
many rows are submitted in the current admission tick.

| Knob | Unit | Meaning |
|---|---|---|
| `prompt_groups_per_step` | prompts | `B_p`: how many prompts make one optimizer step |
| `completions_per_prompt` | samples/prompt | `cpp`: GRPO group size per prompt |
| `max_head_offpolicy_versions` | weight-sync versions | `O`: headroom for submitting rows ahead of future sampler versions.  This is an admission budget, not a guarantee that every request finishes within `O` versions |
| `max_concurrency_rollout_sample` | samples | `C_s`: hard cap on in-flight LLM calls; map to `deployment.max_batch_size`.  Must be `>= cpp` or the gate deadlocks |
| `pipeline_chunks_per_step` | scheduler chunks | Caps how many fwd/bwd chunks feed one optimizer step; these are not trainer/executor microbatches |
| `anchor_logp` | logprob source | `"old_policy"` (default) snapshots trainer logprobs and applies TIS; `"rollout"` skips that forward and makes TIS identity |

Derived: `B_s = B_p × cpp` (samples per outer batch), `R = C_s / B_s` (active set
relative to one batch).

**Version semantics.** `version` increments once per `weight_sync_fn` call
(once per optimizer batch when `weight_sync_interval=1`, which the recipe
pins).  Each row records the version at submission time; the train step reports
`current_version - submit_version` as
`async/version_offset_*`.

That metric is **submit-version lag**, not token-generation lag.  AReaL's
stronger contract is per-token: workflows return a `versions` tensor whose
entries are the weight version used when each generated token was produced, and
staleness metrics compare those behavior-policy versions to the training
version.  Fireworks async RL does not currently have equivalent per-token
generation-version metadata in `RolloutSample`, so do not describe
`version_offset=0` as a proof of on-policy generation.  It means the row was
submitted under the same sampler-version counter the trainer currently sees.

**Admission is not completion.** The gate decides whether to submit another
row by looking at current in-flight and accepted sample counts.  It cannot
promise that a submitted request will finish before a future weight sync, nor
can it prove which model version served every generated token.  A slow rollout
may come back after the trainer has advanced the sampler version, so monitor
the submission lag with:

- `async/version_offset_mean`
- `async/version_offset_max`

Treat the `async/version_offset_*` values as admission/submission diagnostics.
They are not direct policy-divergence measurements. The stock client loss uses
rollout behavior logprobs for TIS. If lag is too high, lower
`max_head_offpolicy_versions`, reduce
`max_concurrency_rollout_sample`, or set `max_head_offpolicy_versions=0` for
the strict on-policy baseline.

**Sizing rule of thumb.**  For sustained overlap you usually need `O >= R - 1`.
Treat this as an admission-sizing heuristic, not an on-policy guarantee.  At
`O = 0`, the runner stops submitting new rows once the current version's batch
budget is full.  Pipeline mode can still dispatch trainer fwd/bwd chunks before
the whole global batch has landed, but it does not advance weights until the
batch boundary.  This is why strict on-policy pipeline runs do not need a
separate concurrency override to the global batch size;
`max_concurrency_rollout_sample` remains only an optional serving/backpressure
cap.  AReaL's GSM8K example uses `R=1` with `O=2`; we've tested
`R=4` with `O=4` (one margin step over the minimum) and found it healthy.  See
`## Metrics` below for tuning from a live run.

## Metrics: tuning staleness and the trainer/sampler GPU split

`rl_loop.py`, `async_rl_loop.py`, and held-out eval send metrics through the
shared `training.utils.log_metrics` path. The producer passes a plain metric
dictionary and an explicit business `step`; the logger never infers semantics
from metric names. It makes values finite and JSON-compatible, appends the
result to the configured `COOKBOOK_METRICS_FILE` JSONL ledger, then sends that
exact dictionary to W&B. Async aggregates chunk metrics into one record per
optimizer step and logs post-dispatch overlap telemetry separately. When a
ledger is configured, its write is required and failures propagate. It works
when W&B is disabled or unavailable and is the source CI uses for validation,
plots, and artifacts.

W&B uses its implicit, unique `_step` only as transport. Dashboard axes
come from `train/step` and `rollout/step` through `define_metric`; never pass a
repeated business step as W&B's transport step because chunk, overlap, rollout,
and eval log calls can legitimately share it. JSONL line order preserves those
calls without an event envelope. The ledger can later be replayed into other
sinks such as MLflow. RL metrics never depend on
`Config.runner` or `RunnerIO`; do not reintroduce runner state to produce
`metrics.jsonl`. The CI wrapper may still use `RunnerIO` for the distinct
status and resource-metadata lifecycle files.

The target regime is **sampler-bound with minimal trainer wait**: the trainer
finishes its step + sync just before the next batch is ready, so it idles only
briefly waiting on the sampler.  The four core wall-time metrics tell you
where you sit on that frontier and what to change.

| Metric | Means | Healthy value |
|---|---|---|
| `perf/trainer_wait_for_sampler_time` | Trainer idle, waiting for the next batch to assemble | **Small but >0** (sampler-bound, minimal slack) |
| `perf/sampler_wait_for_trainer_time` | Sampler blocked on the staleness gate, waiting for a weight sync to release budget | **≈0** for off-policy; expected `>0` at `O=0` |
| `perf/wait_time_ratio` | `(trainer_wait + sampler_wait) / step_time` | < ~0.1 in the off-policy regime |
| `perf/overlap_ratio` | Fraction of step wall-time the two sides ran concurrently | → 1.0 in the off-policy regime |

In off-policy steady state the two wait metrics are mutually exclusive: exactly
one side waits per step.  Read the two ratios first to triage, then drop into
the wait pair to decide what to change.

> At `max_head_offpolicy_versions=0`, `sampler_wait_for_trainer_time` can be
> structurally non-zero because the gate refuses to admit the next batch once
> the current version's submission budget is full.  That is correct behavior,
> not a bug.  In strict on-policy runs, this wait reflects the budget while
> trainer fwd/bwd chunks may still overlap with outstanding rollouts from the
> same global batch.

### Reading a run (off-policy regime, `O > 0`)

1. **`sampler_wait_for_trainer_time` >> 0** → samplers are blocked on the
   staleness budget.  The trainer is the slow side; admit more off-policyness
   or reduce trainer load.
   - First lever: raise `max_head_offpolicy_versions` by 1 (`O += 1`).  Cheap
     and reversible; TIS and PPO clipping correct or bound modest extra drift.
   - If `O` is already at the sizing rule (`O ≥ R−1`) and the wait persists,
     the trainer step itself is too long.  Add training replicas /
     pipeline-parallel ranks, or reduce the optimizer batch size.
2. **`trainer_wait_for_sampler_time` >> 0 and `sampler_wait` ≈ 0** → the
   intended sampler-bound regime.  Check whether the wait is *minimized*:
   - If the wait is small and `wait_time_ratio < ~0.1`, you're done.
   - If the wait is large, the sampler is the slow side.  Raise inference
     replicas, raise `max_concurrency_rollout_sample` toward the deployment's
     `max_batch_size`, or shrink the per-sample work (lower
     `max_completion_tokens`, tighter retry budget in the rollout).
3. **Both waits >0** → the gate is mis-sized; one side is starving the other
   on alternating steps.  Usually `R = C_s / (B_p·cpp)` is high but `O` is too
   low: each batch admits fast, but no off-policy slack means the next batch
   stalls until the sync.  Bump `O` until `sampler_wait` collapses to ~0.
4. **Both waits ≈ 0, low `overlap_ratio`** → step times are dominated by
   non-overlapped phases (weight sync, checkpoint save).  Re-check
   `weight_sync_interval=1` is in effect; if the sync itself is the long pole,
   investigate the deployment configuration separately.

### Choosing the trainer/sampler GPU split

Same metrics, in aggregate:

- **Sampler-bound run, large `trainer_wait_for_sampler`** → shift GPUs from
  trainer to inference: more replicas, larger TP, or a larger
  `max_concurrency_rollout_sample`.
- **Trainer-bound run, large `sampler_wait_for_trainer` after maxing out `O`**
  → shift GPUs from inference to trainer: more training replicas, raise
  pipeline parallelism, or reduce the optimizer batch size.

In the off-policy regime, tune until `sampler_wait_for_trainer_time ≈ 0` *and*
`trainer_wait_for_sampler_time` is small — the trainer is the marginal-cost
resource and is fully utilized; the sampler always has work queued
just-in-time.  In strict on-policy pipeline mode, sampler wait can remain
non-zero because the gate cannot admit the next policy version's rows until
the current global batch reaches its optimizer boundary.

### Other useful metrics

- `train/ppo_kl` — KL between the current policy and the `old_policy_logprobs`
  snapshot for the current optimizer batch.  In async pipeline mode it helps
  check whether delayed optimizer boundaries are seeing meaningful policy
  movement.
- `rollout/entropy` — averaged over `loss_mask>0` only.  Sudden collapse is
  the usual mode-collapse signal.

## Configuration cheatsheet

```python
from training.recipes.async_rl_loop import Config, main

cfg = Config(
    log_path="./logs",
    base_model="accounts/fireworks/models/qwen3-1p5b-instruct",
    learning_rate=1.7e-5,
    completions_per_prompt=8,
    prompt_groups_per_step=8,            # B_s = 64 samples per batch
    max_head_offpolicy_versions=4,       # O = 4 weight-sync versions of slack
    max_concurrency_rollout_sample=256,  # C_s = 256 -> R = 4x batch in flight
    max_completion_tokens=16384,
    deployment=DeployConfig(tokenizer_model="Qwen/Qwen2.5-1.5B-Instruct"),
    trainer=TrainerConfig(training_shape_id="..."),
)

main(cfg, rollout_fn_factory=make_rollout_fn, rows=rows)
```

For strict on-policy training, set `max_head_offpolicy_versions=0`.
`pipeline_chunks_per_step` controls how many forward/backward chunks feed one
global optimizer step; it does not change the optimizer batch size.

## Loss path

Async snapshots old-policy logprobs and passes a direct
`make_grpo_loss_fn(...)` closure to `forward_backward_custom(...)` by default.
Set `anchor_logp="rollout"` to reuse aligned rollout behavior logprobs instead.
That closure applies PPO clipping, TIS, optional reference KL, and
observability-only raw-inference drift metrics. There is no loss selector or
fallback. Fork the documented `fwd_bwd_batch` function and follow
[customize RL loss skill](https://github.com/fw-ai/fireworks/blob/main/public-repos/cookbook/skills/customize-rl-loss/SKILL.md) to
switch to a trainer built-in or another client loss.

## TIS correction

`utils/rl/common.py::run_loss_loop` computes TIS weights over **active
positions only** (`loss_mask>0`). Including masked bridge/user/tool
tokens biases the geometric-mean TIS toward 1. This
matches slime's `tis_level="geometric"` `masked_mean` and AReaL's
`masked_mean(log_diff, mask, expand=True)` semantics. Training TIS uses
`rollout_logprobs` and reports the TIS metrics returned by the client loss.
When raw inference logprobs are available, the same direct builder reports
`train/inference_diff`, `train/inference_k1`, and `train/inference_kld`; those
metrics do not feed PPO or TIS.
The SDK keeps a backward-compatibility fallback for legacy completions routes
that return only `logprob`: raw values may stand in for `rollout_logprobs` only
when sampling is full-distribution (`temperature=1.0`, `top_p=1.0`,
`top_k=0`), where raw and behavior-policy logprobs are equivalent.

The `rollout/entropy` metric in `metrics.py::compute_step_metrics` is also
masked (`-logprob` averaged over `loss_mask>0`).

## Examples

Two minimal examples ship under `training/examples/rl/`:

- `single_turn_token_in/` — pre-tokenized rows; `rollout_fn` makes one
  `/v1/completions` token-in/token-out call per invocation (the recipe
  invokes it `completions_per_prompt` times per dataset row).
- `multi_turn_message_in/` — OpenAI-style messages; `rollout_fn` runs a retry
  loop using `MessageTrajectoryAssembler` to keep prior assistant tokens
  exact across turns.  Ports AReaL's `examples/multi_turn_math/` to this recipe.

Each example exposes a `rollout_fn_factory(setup)` and a `train.py` that wires
the dataset and factory into `recipes.async_rl_loop.main`.

## Related

- [`recipes.md`](../recipes.md) — sync `rl_loop.py` overview
- [`hotload.md`](hotload.md) — weight sync internals (the version counter that
  `max_head_offpolicy_versions` budgets against)
- [`sampling-timeouts.md`](sampling-timeouts.md) — diagnose sampler timeout
  errors from hard evidence before changing capacity or concurrency
- [`gradient-accumulation.md`](gradient-accumulation.md) — optimizer-step
  gradient normalization
- [`dynamic-filter.md`](dynamic-filter.md) — async runner accepts the same
  `dynamic_filter_fn` signature
