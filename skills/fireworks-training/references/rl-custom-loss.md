# RL: customize the objective

The RL recipes are reference loops with one fixed objective and no loss,
anchor, KL, TIS or sampling-support options. To change the objective, fork the
loop and edit its train step using the loss sketches below.

## What the reference loops train

`rl_loop`, `async_rl_loop` and `async_rl_loop_serverless` train server-side
GRPO: the trainer's `"ppo"` loss clipped to `[1 - eps_clip, 1 + eps_clip_high]`
against the rollout logprobs, with no reference model, KL or importance-sampling
correction. MoE models get completion-only router replay. This sequential
whole-window sketch shows the loss math:

```python
from training.utils.rl.algorithm import grpo as objective
from training.utils.rl.common import inference_drift_metrics
from training.utils.rl.losses import combine_prompt_groups

batch = combine_prompt_groups(prompt_groups)          # PolicyBatch; anchor = rollout logprobs
datums = objective.build_datums(batch, policy_loss)   # advantages, anchor, response mask
result = policy.forward_backward(datums, objective.LOSS_FN, loss_fn_config=objective.loss_fn_config(policy_loss))
result.metrics.update(objective.metrics(batch, result, policy_loss))
result.metrics.update(inference_drift_metrics(batch, result))
policy.optim_step(adam, grad_accumulation_normalization=objective.NORMALIZATION, emit_grad_norm_metrics=True)
```

The sync recipe builds and trains the complete batch; async recipes submit
currently ready groups together. Both keep the loss call visible for forks.
In async recipes, `track_group` records the native
future; `retain_submission` retains it even if cancellation races submission.
`submission_window` drains retained requests on failure, and `drain_groups` joins
every result before one optimizer update. `add_batch_diagnostics` computes full-window diagnostics once
after that join. A fork of `train_group` must preserve that return contract and
all-success optimizer boundary; do not optimize separately for each group.

Every util returns a flat metrics dict; merge it into `result.metrics` and the
step logger records it as `train/<name>` (see
[`step-metrics.md`](step-metrics.md#policy-objectives)).

## Objective modules

`training/utils/rl/algorithm/<name>.py`, no registry. Every module has the same
surface, so swapping objectives is one import plus its config:

| Name | What it is |
|---|---|
| `<Name>Config` | Frozen settings, validated on construction |
| `LOSS_FN`, `loss_fn_config(cfg)` | Trainer built-in loss (absent for `reinforce`, `score_centering`) |
| `build_datums(batch, cfg)` | Datums for that loss |
| `metrics(batch, result, cfg)` | `{PREFIX}_ratio_mean`, `{PREFIX}_clip_frac`, `{PREFIX}_kl` |
| `make_loss_fn(batch, cfg)` | Same objective as a client loss on `batch.data`, same metric names |
| `NORMALIZATION` | `grad_accumulation_normalization` for `optim_step` (GSPO: `num_sequences`) |

Modules: `grpo` (prefix `ppo`), `gspo`, `dapo`, `cispo`, `dro`, `dppo`,
`importance_sampling` (prefix `is`), `reinforce`, `score_centering`. `igpo` is
an advantage estimator, not a loss; `sao` holds the experimental value-head
recipe's DIS loss.

```python
from training.utils.rl.algorithm import cispo as objective
policy_loss = objective.CISPOConfig(eps_low=0.2, eps_high=0.28)
# server side: the train step above, unchanged
# client side: the batch's own datums carry the loss mask
result = policy.forward_backward_custom(batch.data, objective.make_loss_fn(batch, policy_loss))
```

A new client loss is `loss_fn(data, logprobs_list) -> (scalar_loss, metrics)`;
build it on `common.run_loss_loop(batch, ...)` and name its metrics with
`common.client_loss_metrics`.

## Recompute the anchor

Clip against a pre-update trainer snapshot instead of the rollout logprobs, for
one extra forward (`perf/old_policy_forward_time`). Everything downstream reads
`batch.anchor_logprobs`:

```python
from training.utils.rl.anchor import anchor_drift_metrics, recompute_anchor

batch, forward = recompute_anchor(policy, batch)
datums = objective.build_datums(batch, policy_loss)
...
result.metrics.update(anchor_drift_metrics(batch))   # anchor_k1, anchor_k3
```

Off-policy async (`max_head_offpolicy_versions > 0`) clips against the stale
sampling policy by default; recomputing gives the proximal snapshot that
decoupled PPO uses.

## Truncated importance sampling (TIS)

With a recomputed anchor, weight trained tokens by `min(pi_old / mu, cap)`
between the snapshot and the sampler. The weight scales the datum advantages,
so it works with any server-side loss:

```python
from training.utils.rl.tis import apply_tis

datums, tis = apply_tis(datums, batch, cap=2.0)   # level="sequence": one weight per response
result.metrics.update(tis)                         # tis_weight_mean, tis_clip_frac
```

With the rollout logprobs as the anchor every weight is 1.

## Add reference KL

Client-side GRPO and REINFORCE add a reference penalty when the batch carries
reference logprobs. In the fork, drop the loop's reference guard, pass
`reference_required=True` to `build_service_client`, and:

```python
from dataclasses import replace

reference = service.create_reference_client(policy_client=training_client)
ref = reference.forward(batch.data, "cross_entropy").result()
batch = replace(batch, ref_logprobs=[list(o["logprobs"].data) for o in ref.loss_fn_outputs])
policy_loss = objective.GRPOConfig(eps_clip=0.2, kl_beta=0.001)
result = policy.forward_backward_custom(batch.data, objective.make_loss_fn(batch, policy_loss))
```

Logs `train/ref_kl` and `train/mean_kl_penalty`. The built-in loss has no
reference input, so `loss_fn_config` rejects `kl_beta > 0`.

## Replay the sampling support (top-p)

With `top_p < 1`, train over the recorded sampling support instead of the full
vocabulary:

```python
from training.utils.rl.sampling_support import (
    attach_top_sampling_references, sampling_support_kwargs, sampling_support_loss_fn_config,
)

sample_kwargs.update(sampling_support_kwargs(top_p=0.95, support_size=4096))
...
datums = attach_top_sampling_references(objective.build_datums(batch, policy_loss), batch.top_sampling_references)
result = policy.forward_backward(
    datums, objective.LOSS_FN, loss_fn_config=sampling_support_loss_fn_config(objective.loss_fn_config(policy_loss))
)
result.metrics.update(objective.metrics(batch, result, policy_loss))   # skip inference_drift_metrics
```

- `support_size` is a storage width, not a top-k filter; it must cover every
  trained token's support (trainer limit 20,000). Incomplete support at a
  trained position is rejected.
- Needs a sampler bound to the trainer's shared storage, a serving image that
  accepts `top_sampling_format="parquet_v1"`, and
  `training_client.supports_target_logprob_support`.
- Only server-side losses read the support. Drift against full-vocabulary
  serving logprobs is meaningless here, so the fork skips the drift call.
- Do not combine with a recomputed anchor; the snapshot is full-vocabulary.

## Turn-level advantages (IGPO)

`rl_loop.main(cfg, rollout_fn_factory=..., prepare_batch_fn=igpo.prepare_batch(gamma=1.0))`.
Each `RolloutRun.metadata` sets `TURN_SPANS` and `ANSWER_TOKENS`; one trainer
forward scores the answer after every turn and sets `batch.token_advantages`,
which `build_datums` and every client loss read. See
`training/examples/rl/multihop_qa/`.

## Invariants

- Keep target tokens, prompt boundaries and logprob rows aligned.
- Change the anchor only through `PolicyBatch.anchor_logprobs`; never overwrite
  the rollout logprobs. Only `apply_tis` reweights advantages.
- Use raw inference logprobs only for drift metrics.
- Use the objective's `NORMALIZATION`
  ([`rl-gradient-accumulation.md`](rl-gradient-accumulation.md)).
- Vision datums keep Tinker's expanded coordinates: `model_input.length == N - 1`,
  image positions are zero placeholders with zero weight and advantage.
