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
| `LOSS_FN`, `loss_fn_config(cfg)` | Module's trainer built-in loss (absent for the client `reinforce`, `score_centering` modules) |
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

## Server TIS/MIS

All built-in RL policy losses support token TIS/MIS on compatible trainers.
Anchored losses such as PPO opt in with:

```python
loss_fn_config = {
    "anchor_logprob_source": "current_detached",
    "weighting": "mis", "weight_low": 0.2, "weight_high": 5.0,
    "target_logprob_support": "sampling_support",
}
```

Require `supports_policy_importance_sampling`, raw advantages, sampler logprobs,
and an explicit response mask. The trainer reuses its forward for a detached
anchor and weighting; no separate anchor forward is needed. Use exactly one
optimizer update per weight sync, finish publication before the next training
window, and do not reuse a batch. Rollouts can remain async. PPO's ratio is 1
with a nonzero gradient, so clipping is inactive and its gradient matches
REINFORCE with the same weights. Keep the objective's existing normalization;
MIS rejection must not shrink its denominator. Separate auxiliaries are unweighted.

## Score-centering paths

Both client-side and server-side SC are supported; neither is selected by default.

| Path | API | Objective |
|---|---|---|
| Client SC | `score_centering.make_loss_fn` + `forward_backward_custom` | Explicit vanilla policy-gradient correction; partial top-k head/tail approximation. |
| Built-in REINFORCE + SC | `loss_fn="reinforce"`, `score_centering=True` | Explicit correction after detached none/TIS/MIS weighting; full vocabulary or complete support replay. |

With identical heads, masks and normalization, client SC matches unweighted
full-vocabulary server SC. The old STE option `train_inference_correction` has
been removed; server-side `score_centering=True` is supported only on REINFORCE.

The new path requires a compatible SDK release and trainer capabilities
`supports_reinforce` / `supports_score_centering`. Reference loops retain GRPO.
Send ordinary single-target datums with raw advantages, sampler logprobs and
an explicit response mask; do not use client SC's expanded datums or preapply TIS.

```python
from training.utils.rl.sampling_support import attach_sampling_mask_references, attach_top_sampling_references

datums = attach_sampling_mask_references(datums, batch.sampling_mask_references)
datums = attach_top_sampling_references(datums, batch.top_sampling_references)
result = policy.forward_backward(datums, "reinforce", loss_fn_config={
    "score_centering": True, "weighting": "tis", "weight_high": 2.0,
    "target_logprob_support": "sampling_support",
})
policy.optim_step(adam, grad_accumulation_normalization="num_loss_tokens")
```

SC needs inline sampler IDs/logprobs or Parquet references (see replay below).
The example requires complete support captured during rollout. `apply_tis`
rewrites returned datums, not client SC's `batch.advantages`; use the server
path for SC combined with TIS/MIS.
SC accepts a partial probability head in either normalization mode.
`sampling_support` requires a complete ID-only replay mask.
SC, none/TIS/MIS weighting and mask replay are independent options and can be
combined. v1 uses temperature 1; SC/replay are text-only. Do not preweight
advantages or add PPO/GSPO clipping. Response-mask tokens count even when their
advantage is zero.

For replay, capture `sampling_mask="parquet_v1"`. For SC, additionally request
`top_sampling_logprobs=128` and `top_sampling_format="parquet_v1"`. The SDK carries
each through its own field: `sampling_mask_references` for replay and
`top_sampling_references` for SC; replay-only needs no head.
The complete mask defines normalization; the bounded head uses approximate
head/tail SC. Replay requires `sampling_mask`; top-k-only files are rejected.
Keep loss selection in the caller; the async loop retains its default objective.

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

With the rollout logprobs as the anchor every weight is 1. For REINFORCE + SC,
use built-in `weighting="tis"` instead: it weights each candidate inside the
SC correction as well as the sampled token. Keep advantages raw.

## Add reference KL

Client-side GRPO and REINFORCE add a reference penalty when the batch carries
reference logprobs. Keep reference setup outside the stock sync/async loops;
use `training.utils.rl.reference` in your custom train step. Reference scores
come from the frozen baseline, while rollout logprobs remain the clipping
anchor. They are different distributions with different purposes.

For **LoRA**, create the SDK base-only client once on the policy service:

```python
from training.utils.rl.reference import LoraReference, with_reference_logprobs

# No explicit reference shape/job: SDK shares the LoRA trainer and disables
# adapters on this client. Only forward on it; never optimize the reference.
reference = LoraReference(service.create_reference_client(policy_client=training_client))
```

For **full-parameter** tuning, provision a separate base-model inference
deployment through the SDK. Use the same validated deployment shape as the
policy rollout and enough context for the full input plus final target (and
one generated token for `compute_logprobs`). Do not bind this deployment to
policy hot-loads or publish policy checkpoints to it. No reference training
shape or second trainer is needed:

```python
from fireworks.training.sdk.deployment import DeploymentConfig
from training.utils.rl.reference import SamplerReference, with_reference_logprobs

base_sampler = service.create_inference_deployment_sampler(
    DeploymentConfig(
        deployment_id="frozen-base-reference",
        base_model=cfg.base_model,
        deployment_shape=cfg.deployment.deployment_shape,
        min_replica_count=1,
        enable_hot_load=False,
        hot_load_bucket_type=None,
    ),
    cleanup_on_close="delete",
    tokenizer=tokenizer,
)
reference_sampler = service.create_sampling_client(deployment_sampler=base_sampler)
reference = SamplerReference(reference_sampler)
```

Close the sampling client and service in `finally`/an exit stack. Service
cleanup deletes only deployments it created, leaving reattached resources to
their owner. Reference inference is separately billed. Validate model/tokenizer
identity and context coverage when choosing the reference deployment; a shorter
LoRA training shape does not constrain full-parameter inference references.
An independent reference can run in another region. To place it separately,
create and wait for a `DeploymentConfig` with an explicit region through
`DeploymentManager`, then attach it with
`service.create_deployment_sampler_for_model(deployment.name, tokenizer=tokenizer)`
and `service.create_sampling_client(deployment_sampler=base_sampler)`. The caller
owns deletion of that independently created deployment. Use a region with
available inference quota; the policy trainer's training quota is separate.

For either source, score the exact token IDs without decoding/re-encoding:

```python
batch = with_reference_logprobs(batch, reference)
policy_loss = objective.GRPOConfig(eps_clip=0.2, kl_beta=0.001)
result = policy.forward_backward_custom(batch.data, objective.make_loss_fn(batch, policy_loss))
```

The helper returns a new batch, preserves its rollout anchor, validates score
lengths/finiteness and times `reference_forward`. In an async fork, use
`batch = await asyncio.to_thread(with_reference_logprobs, batch, reference)`
under a bounded semaphore. The sampler adapter supports shifted text datums,
appends the final target token and removes the unscored first prompt position.
It requests raw base logprobs at temperature 1 without top-p/top-k truncation.
Both adapters discard policy routing replay and sampling-support metadata so
the frozen reference chooses its own experts and scores its full vocabulary.
Trainer and inference scores can differ numerically because their execution
and precision differ. Multimodal scoring needs a compatible reference trainer;
the sampler adapter cannot score image-containing inputs.

Logs `train/ref_kl` and `train/mean_kl_penalty`. The built-in loss has no
reference input, so `loss_fn_config` rejects `kl_beta > 0`. The stock RL loops
keep their fixed GRPO objective and do not provision or score references.

## Replay the sampling support (top-p)

Independent mask/head capture requires `fireworks-ai>=1.2.28` and a compatible trainer.

With `top_p < 1`, train over the recorded sampling support instead of the full
vocabulary:

```python
from training.utils.rl.sampling_support import (
    attach_top_sampling_references, attach_sampling_mask_references,
    replay_mask_kwargs, sc_head_kwargs, sampling_support_loss_fn_config,
)

sample_kwargs.update(replay_mask_kwargs(top_p=0.95) | sc_head_kwargs(128))
...
datums = attach_sampling_mask_references(objective.build_datums(batch, policy_loss), batch.sampling_mask_references)
datums = attach_top_sampling_references(datums, batch.top_sampling_references)
result = policy.forward_backward(
    datums, objective.LOSS_FN, loss_fn_config=sampling_support_loss_fn_config(objective.loss_fn_config(policy_loss))
)
result.metrics.update(objective.metrics(batch, result, policy_loss))   # skip inference_drift_metrics
```

- `sampling_mask="parquet_v1"` records the complete support (trainer limit
  20,000); incomplete masks are rejected. `sc_head_kwargs(128)` requests a separate
  probability head for SC. Omit `sc_head_kwargs` for replay without SC.
- Needs a sampler bound to the trainer's shared storage, a serving image that
  accepts `sampling_mask="parquet_v1"`, and
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
  the rollout logprobs. Client `apply_tis` reweights advantages; built-in
  REINFORCE weighting receives raw advantages.
- Use raw inference logprobs only for drift metrics.
- Use the objective's `NORMALIZATION`
  ([`rl-gradient-accumulation.md`](rl-gradient-accumulation.md)).
- Vision datums keep Tinker's expanded coordinates: `model_input.length == N - 1`,
  image positions are zero placeholders with zero weight and advantage.
