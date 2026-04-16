# RL: losses, training loop, TIS, R3

RL code lives in `training/utils/rl/`. All losses accept `(policy_logprobs, inference_logprobs, advantages, ratios)` and return a scalar; they're registered in `utils/rl/losses.py` and selected via `Config.policy_loss`.

## Loss selection (`Config.policy_loss`)

| Value | File | Ratio form | Clip | TIS |
|-------|------|-----------|------|-----|
| `"grpo"` (default) | `grpo.py` | per-token `exp(π - π_inf)` | symmetric `1±eps` | yes |
| `"dapo"` | `dapo.py` | per-token | asymmetric (upper 0.28, lower 0.2) | yes |
| `"gspo"` | `gspo.py` | **sequence-level** `exp(mean(π - π_inf))` | symmetric | no (sequence-level) |
| `"dro"` | `dro.py` | per-token | none (quadratic penalty on `r-1`) | no |
| `"cispo"` | `cispo.py` | per-token, **clip is detached** | symmetric | no |
| `"importance_sampling"` | `is_loss.py` | per-token, unclipped, capped by `ratio_log_cap` | none | no |
| `"reinforce"` | `reinforce.py` | per-token | none | no |

Call out the key defaults: `eps=0.2`, `dapo_upper=0.28`, `ratio_log_cap=None`. Override via `Config.is_config` when exposed (`ISConfig.ratio_log_cap`, `ISConfig.clip_high`, `ISConfig.clip_low`).

## TIS -- `utils/rl/tis.py`

Per-token weight that corrects the numerical gap between training and inference logprobs (bf16 rounding, different kernels). Used inside `grpo.py` and `dapo.py`.

- Input: `policy_logprobs`, `inference_logprobs` (from `DeploymentSampler` with `echo=True`)
- Output: per-token weights multiplied into the policy loss
- Disable by setting `TISConfig.enabled=False` (not recommended; GRPO diverges across different serving stacks without it)

## R3 -- `utils/rl/router_replay.py`

MoE routing replay: inference deployments use different routing from the trainer; R3 replays the routing matrix from inference back into training to align expert selection.

Requirements:

1. MoE base model (not dense)
2. Deployment shape includes `--enable-moe-stats` in `extra_args`
3. Sampler request: `include_routing_matrix=True`
4. Routing matrices come back as base64-encoded int16 arrays in `SampledCompletion.routing_matrices`

Without R3 on MoE: policy sees different experts than inference → training signal corrupted.

## Training loop -- `utils/rl/train.py`

### `run_rl_loop(policy, ref, sampler, syncer, ...)`

Pipelined on-policy orchestration. Overlaps inference sampling, policy forward/backward, and weight sync.

Parameters (passed via `RunConfig`):

| Param | Default | Effect |
|-------|---------|--------|
| `weight_sync_interval` | 1 | Hotload every N steps. Increase to 4-8 for >30B models. |
| `group_size` | 8 | Completions per prompt (GRPO groups). |
| `max_rows` | dataset size | Number of prompts per epoch. |
| `epochs` | 1 | Full passes over dataset. |
| `dcp_save_interval` | 0 (off) | Save a DCP checkpoint every N steps for resume. |

Return value: final metrics dict.

## Advantages

`utils/rl/losses.py::compute_group_advantages(group)` returns `(rewards - mean) / std` per group; no-ops when all rewards are identical. `filter_zero_advantage_groups(groups)` drops degenerate groups before the gradient step.

If the training loop exits with `steps: 0`: every group had uniform rewards. Increase `max_rows`, relax the reward function's threshold, or check that `group_size >= 2`.

## IGPO -- `utils/rl/igpo.py`

Used by `recipes/igpo_loop.py`. For each multi-turn trajectory with T turns:

1. Run 1 baseline scoring pass (whole trajectory concatenated)
2. Run T ablation passes (drop turn t, re-score)
3. IG reward for turn t: `baseline_logprob - without_turn_t_logprob`
4. Each token in turn t gets advantage `A_{i,t}` (per-turn, centered within the prompt group)

Scoring passes run concurrently via `ThreadPoolExecutor`; controlled by `Config.ig_scoring_workers` (default 4).

## Pipeline parallel -- `utils/rl/pp.py`

`recommend_pp_batch(config, profile)` returns a `PPConfig` that aligns batch sizes across PP stages. Call once at startup; recipes apply the result automatically. Not needed for non-PP training.

## LossSpec -- `utils/rl/spec.py`

Protocol for custom losses:

```python
class LossSpec(Protocol):
    def make_loss_fn(self, *, is_config: ISConfig, ...) -> Callable:
        ...
```

Register a custom spec in `utils/rl/losses.py::LOSS_REGISTRY` if you want it accessible via the `policy_loss` string. Otherwise pass the factory directly to `run_rl_loop`.

## Metrics logged automatically

`utils/rl/metrics.py` adds these to `metrics.jsonl` every step:

| Metric | Meaning |
|--------|---------|
| `response_len/mean`, `response_len/p50`, `response_len/p99` | Completion length distribution |
| `response_len/frac_truncated` | Fraction hitting `max_new_tokens` |
| `reward/mean`, `reward/std` | Batch reward statistics |
| `advantage/mean`, `advantage/std`, `advantage/abs_mean` | After group centering |
| `train/tokens_per_second`, `train/samples_per_second` | Throughput |
| `hotload/save_time_s`, `hotload/hotload_time_s`, `hotload/warmup_time_s` | From `syncer.last_timing` |
| `igpo/turns_mean`, `igpo/ig_reward_mean` (IGPO only) | Per-turn statistics |
