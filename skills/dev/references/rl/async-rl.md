# RL: async recipe (`async_rl_loop.py`)

The async recipe overlaps rollout sampling with training: while the trainer runs
`fwd_bwd + optim_step` on batch *N*, the sampler is producing batch *N+1* against
the previous policy version.  This is distinct from the synchronous
`rl_loop.py`, which trains and samples in lockstep.

| | `rl_loop.py` (sync) | `async_rl_loop.py` (async) |
|---|---|---|
| Sampler/trainer overlap | None — drains rollouts before each step | Overlapped via an off-policy budget |
| Rollout API | `make_rollout_fn(rl_cfg, deploy_mgr) -> rollout_fn` | `rollout_fn_factory(setup) -> rollout_fn` |
| Per-call signature | `rollout_fn(prompt, n)` returns N samples | `rollout_fn(sample_prompt) -> RolloutSample \| None` (one trajectory per call) |
| Concurrency | `ConcurrencyConfig` (adaptive AIMD) | Sample-level cap on the async runner (`max_concurrency_rollout_sample`) |
| Off-policy | Strict on-policy | Configurable budget in weight-sync versions |
| PPO inner steps | One `fwd_bwd + optim_step` per rollout | `ppo_n_minibatches` × inner steps per rollout |

Use the async recipe when (a) you need the rollout/train overlap to fit in your
GPU budget, or (b) you want PPO inner-loop minibatching.  Use the sync recipe
otherwise — it's simpler and avoids the off-policy reasoning below.

## The rollout API

The user supplies one trajectory per call:

```python
async def rollout_fn(sample_prompt: dict) -> RolloutSample | None: ...
```

`sample_prompt` is the dataset row's dict, renamed once it crosses the
dataset/sampling seam (the dataset emits a "row"; the sampler treats it as a
prompt to draw a sample against).  Return `None` to drop the sample (counts as
one lost sample within the row's group).

`RolloutSample` is three parallel lists plus a scalar reward:

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

The factory pattern keeps per-rollout dependencies (sampler, tokenizer, sample
kwargs, custom state) out of the per-call signature:

```python
def make_rollout_fn(setup: RolloutSetup) -> RolloutFn:
    sampler = DeploymentSampler(setup.inference_base_url, setup.model, setup.api_key, setup.tokenizer)
    async def rollout_fn(sample_prompt: dict) -> RolloutSample | None:
        ...
    return rollout_fn
```

`RolloutSetup.extras` carries arbitrary caller state (replaces the older
`RolloutContext.ctx_extras` setattr injection).

## The off-policy gate (sample-level)

The runner gates rollout submission so a sample is admitted iff:

```
samples_in_flight + samples_accepted < (max_head_offpolicy_versions + version + 1) * batch_size_samples
                                  AND
samples_in_flight < max_concurrency_rollout_sample   (when set)
```

All bookkeeping is in samples (LLM calls), the same unit the deployment's
`max_batch_size` gates on.

| Knob | Unit | Meaning |
|---|---|---|
| `prompt_groups_per_step` | prompts | `B_p`: how many prompts make one optimizer step |
| `completions_per_prompt` | samples/prompt | `cpp`: GRPO group size per prompt |
| `max_head_offpolicy_versions` | weight-sync versions | `O`: how many sync boundaries past submit a sample may land at the trainer.  `0` is strict on-policy |
| `max_concurrency_rollout_sample` | samples | `C_s`: hard cap on in-flight LLM calls; map to `deployment.max_batch_size`.  Must be `>= cpp` or the gate deadlocks |
| `ppo_n_minibatches` | minibatches | `K`: inner PPO steps per rollout batch (each with `old_policy_logprobs` snapshot reused) |

Derived: `B_s = B_p × cpp` (samples per outer batch), `R = C_s / B_s` (active set
relative to one batch).

**Version semantics.** `version` increments once per `weight_sync_fn` call
(once per outer rollout batch when `weight_sync_interval=1`, which the recipe
pins).  It is **not** an optimizer-step counter — with `K>1` the same version
spans `K` optim steps.

**Sizing rule of thumb.**  For sustained overlap you need `O >= R - 1`.  AReaL's
GSM8K example uses `R=1` with `O=2`; we typically run `R=4` with `O=4` (one
margin step over the minimum).  See WandB `perf/wait_time_ratio` and
`perf/overlap_ratio` to confirm the rollout side never starves.

**Diagnosing waits.**

- `perf/trainer_wait_for_sampler_time > 0` → the trainer is waiting on rollouts;
  rollouts are the bottleneck (raise replicas or `C_s`, or accept the wait as
  Amdahl-bound).
- `perf/sampler_wait_for_trainer_time > 0` → the rollout side is throttled by
  the staleness budget; raise `O` or accept that the trainer is the bottleneck.
- Healthy async has the first metric > 0 and the second `~0`: concurrency cap
  binds before staleness.

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
    infra=InfraConfig(training_shape_id="..."),
)

main(cfg, rollout_fn_factory=make_rollout_fn, rows=rows)
```

`synchronous_training=True` forces the loop to drain rollouts before every
train step; useful as a baseline for measuring the async overlap savings (it
makes `perf/sampler_wait_for_trainer_time` ≈ train+sync wall time).

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

- `single_turn_token_in/` — pre-tokenized rows; `rollout_fn` calls the
  `/v1/completions` token-in/token-out path once per row.
- `multi_turn_message_in/` — OpenAI-style messages; `rollout_fn` runs a retry
  loop using `MessageTrajectoryAssembler` to keep prior assistant tokens
  exact across turns.  Ports AReaL's `examples/multi_turn_math/` to this recipe.

Each example exposes a `rollout_fn_factory(setup)` and a `train.py` that wires
the dataset and factory into `recipes.async_rl_loop.main`.

## Related

- [`recipes.md`](../recipes.md) — sync `rl_loop.py` overview
- [`hotload.md`](hotload.md) — weight sync internals (the version counter that
  `max_head_offpolicy_versions` budgets against)
- [`gradient-accumulation.md`](gradient-accumulation.md) — PPO minibatch
  gradient normalization
- [`dynamic-filter.md`](dynamic-filter.md) — async runner accepts the same
  `dynamic_filter_fn` signature
