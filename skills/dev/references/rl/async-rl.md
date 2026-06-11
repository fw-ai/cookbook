# RL: async recipe (`async_rl_loop.py`)

> ⚠️ **EXPERIMENTAL — under active development.** API surface and config
> field names may change without backward-compat shims.  Pin to a specific
> commit if you depend on the current shape.  The recipe also emits a
> runtime `WARNING` at `main()` start.

The async recipe runs rollout sampling and training as concurrent tasks behind a
gate that controls **submission headroom**.  Be precise about the two separate
ideas:

- **Fully synchronous execution** means `synchronous_training=True`: once a
  train batch is ready, the runner drains outstanding rollout tasks before
  entering `train_step`.  This removes rollout/trainer wall-time overlap.
- **On-policy training** means the behavior policy that generated the trained
  tokens matches the policy being updated.  AReaL tracks this with per-token
  `versions` in rollout outputs.  Our async recipe currently tracks the row's
  **submit version** (`async/version_offset_*`), which is only a proxy for the
  behavior-policy version.
- `max_head_offpolicy_versions=O` is an **admission budget**, not a completion
  guarantee.  It controls whether the runner may submit more rows before future
  weight syncs happen.  Once a request is submitted, the network/model runtime
  can still make it finish before or after later syncs.

For a conservative synchronous baseline, use `synchronous_training=True` and
`max_head_offpolicy_versions=0`.  Raising `O` permits more head-of-line
submission and can create useful overlap, but actual policy staleness must be
validated with behavior-policy metadata when available and policy-drift metrics
such as `train/inference_kld` / `train/inference_diff`.

`rl_loop.py` is the older synchronous recipe (always strict on-policy, drains
rollouts before each step).  The async recipe is a strict superset of that
behavior and is the recommended starting point for new RL work.

**Sync RL is the same recipe.** Set `synchronous_training=True` to force the
drain-before-train behavior.  Do not use `max_head_offpolicy_versions=0` as a
synonym for sync mode: `O=0` constrains new submissions, while
`synchronous_training=True` changes when the runner is allowed to train.

**Two-file layout.** A run is typically a `rollout.py` (the rollout function,
`make_rollout_fn(setup) -> rollout_fn`) plus a `train.py` (the `Config` —
training/deployment shapes, `policy_loss`, reward wiring — and the
`main(cfg, rollout_fn_factory=..., rows=...)` call). The reward is computed
inside the rollout and set on each `RolloutSample.reward` segment inside a
`RolloutRun` (often factored into a `reward.py` the rollout imports). The
recipe owns everything between.

## What you customize (and what you don't)

The recipe is intentionally minimal-surface: **the only thing most users need
to write is the rollout function**.  Everything else — gate, advantage
computation, reference-model forwards, weight sync, KL/TIS metrics, PPO inner
loop, checkpoint plumbing — is handled by `recipes/async_rl_loop.py::main`.

| You write | The recipe handles |
|---|---|
| `rollout_fn_factory(setup) -> rollout_fn` (one trajectory per call) | Async fan-out / GroupAssembler / off-policy gate |
| (optional) `dynamic_filter_fn(pg)` for batch-level filtering | Reference model forwards, KL, TIS, drift metrics |
| Dataset rows (or pass `rows=` to `main()`) | Weight sync cadence, checkpoint save/promote, WandB axes |
| `Config(...)` knobs (LR, gate sizes, deployment shape) | PPO inner minibatching, gradient accumulation, advantage z-score |

This is the design intent — extend the rollout, not the loop.  If you find
yourself forking the recipe, file an issue first; it usually means a knob
should exist on `Config`.

| | `rl_loop.py` (sync) | `async_rl_loop.py` (async) |
|---|---|---|
| Sampler/trainer overlap | None — drains rollouts before each step | Concurrent by default; `synchronous_training=True` disables train/rollout overlap |
| Rollout API | `make_rollout_fn(rl_cfg, deploy_mgr) -> rollout_fn` | `rollout_fn_factory(setup) -> rollout_fn` |
| Per-call signature | `rollout_fn(prompt, n)` returns N samples | `rollout_fn(sample_prompt) -> RolloutRun \| None` (one trajectory per call) |
| Concurrency | `ConcurrencyConfig` (adaptive AIMD) | Sample-level cap on the async runner (`max_concurrency_rollout_sample`) |
| On-/off-policy | Strict on-policy only | `async/version_offset_*` measures submit-version lag; true behavior-policy staleness requires generation-version metadata |
| PPO inner steps | One `fwd_bwd + optim_step` per rollout | `ppo_n_minibatches` × inner steps per rollout |

Prefer the async recipe for new work.  Use `synchronous_training=True` for a
fully synchronous baseline, and tune `max_head_offpolicy_versions` separately
when you want overlap.  The sync recipe stays for users who already depend on
it.

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
| `ppo_n_minibatches` | minibatches | `K`: inner PPO steps per rollout batch (each with `old_policy_logprobs` snapshot reused) |

Derived: `B_s = B_p × cpp` (samples per outer batch), `R = C_s / B_s` (active set
relative to one batch).

**Version semantics.** `version` increments once per `weight_sync_fn` call
(once per outer rollout batch when `weight_sync_interval=1`, which the recipe
pins).  It is **not** an optimizer-step counter — with `K>1` the same version
spans `K` optim steps.  Each row records the version at submission time; the
train step reports `current_version - submit_version` as
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
may come back after the trainer has advanced the sampler version, so validate
policy drift with:

- `async/version_offset_mean`
- `async/version_offset_max`
- `train/inference_kld`
- `train/inference_diff`

Treat the `async/version_offset_*` values as admission/submission diagnostics.
Treat `train/inference_kld` and `train/inference_diff` as the policy-drift
checks.  If drift is too high, lower `max_head_offpolicy_versions`, reduce
`max_concurrency_rollout_sample`, or force `synchronous_training=True` for the
baseline.

**Sizing rule of thumb.**  For sustained overlap you usually need `O >= R - 1`.
Treat this as an admission-sizing heuristic, not an on-policy guarantee.  At
`O = 0`, the runner stops submitting new rows once the current version's batch
budget is full, but only `synchronous_training=True` explicitly drains
outstanding rollout tasks before training.  AReaL's GSM8K example uses `R=1`
with `O=2`; we've tested `R=4` with `O=4` (one margin step over the minimum)
and found it healthy.  See `## Metrics` below for tuning from a live run.

## Metrics: tuning staleness and the trainer/sampler GPU split

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
> not a bug.  For the true fully synchronous baseline, also set
> `synchronous_training=True`; then sampler wait includes the explicit
> drain-before-train period.

### Reading a run (off-policy regime, `O > 0`)

1. **`sampler_wait_for_trainer_time` >> 0** → samplers are blocked on the
   staleness budget.  The trainer is the slow side; admit more off-policyness
   or reduce trainer load.
   - First lever: raise `max_head_offpolicy_versions` by 1 (`O += 1`).  Cheap
     and reversible; KL/PPO-clip will absorb modest extra drift.
   - If `O` is already at the sizing rule (`O ≥ R−1`) and the wait persists,
     the trainer step itself is too long.  Add training replicas /
     pipeline-parallel ranks, or lower `ppo_n_minibatches`.
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
  pipeline parallelism, or accept lower `ppo_n_minibatches`.

In the off-policy regime, tune until `sampler_wait_for_trainer_time ≈ 0` *and*
`trainer_wait_for_sampler_time` is small — the trainer is the marginal-cost
resource and is fully utilized; the sampler always has work queued
just-in-time.  In the synchronous baseline, only the second condition is
achievable; `sampler_wait_for_trainer_time` is bounded below by the train +
sync wall time because the runner intentionally stops overlap.

### Other useful metrics

- `train/ppo_kl` — intra-step KL between the current policy and the
  `old_policy_logprobs` snapshot.  Large with `ppo_n_minibatches > 1` means
  the inner loop is genuinely doing work; a near-zero value means
  `ppo_n_minibatches=1` would have the same effect at lower cost.
- `train/inference_kld`, `train/inference_diff` — drift between rollout-time
  inference logprobs and train-time policy logprobs.  These are the best
  available signal for behavior-policy drift in this recipe because
  `async/version_offset_*` only records submit-version lag.  If they spike even
  when `O` is small, look for slow rollout tails, excessive concurrency, or a
  silently failing `weight_sync_fn`.
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
    ppo_n_minibatches=2,                 # K = 2 inner PPO steps per rollout
    max_completion_tokens=16384,
    deployment=DeployConfig(tokenizer_model="Qwen/Qwen2.5-1.5B-Instruct"),
    trainer=TrainerConfig(training_shape_id="..."),
)

main(cfg, rollout_fn_factory=make_rollout_fn, rows=rows)
```

`synchronous_training=True` forces the loop to drain rollouts before every
train step; useful as a baseline for measuring async overlap savings.  Pair it
with `max_head_offpolicy_versions=0` when you want the most conservative
on-policy baseline.

## Loss path

Async is **client-side only**.  `loss_path` is fixed; the server-side built-in
path forbids `kl_beta>0` and `pipeline_parallelism>1`, both of which the async
loop relies on.  Use `rl_loop.py` if you need the server-side fast path.

## TIS / drift metrics

`utils/rl/common.py::run_loss_loop` computes TIS weights and the
`inference_diff` / `inference_kld` / `ppo_kl` drift metrics over **active
positions only** (`loss_mask>0`).  Including masked bridge/user/tool tokens
biases the geometric-mean TIS toward 1 and the drift metrics toward 0.  This
matches slime's `tis_level="geometric"` `masked_mean` and AReaL's
`masked_mean(log_diff, mask, expand=True)` semantics.

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
- [`gradient-accumulation.md`](gradient-accumulation.md) — PPO minibatch
  gradient normalization
- [`dynamic-filter.md`](dynamic-filter.md) — async runner accepts the same
  `dynamic_filter_fn` signature
