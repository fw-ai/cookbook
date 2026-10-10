# RL: customize a loss

Keep group construction, advantages, reference/anchor forwards, backward, and
optimizer updates visible in the recipe. Isolate loss formulas and their typed
options under `training/utils/rl/algorithm/`; these modules never call a trainer.

## Recipe contract

`training/recipes/rl_loop.py` runs client-side GRPO. The async recipe resolves
`policy_loss` and `loss_execution` once at startup, then executes the selected
objective in its training loop:

1. Compute group-normalized advantages from rollout rewards.
2. If the objective uses reference KL, collect the reference logprobs.
3. For clipped objectives, prepare the fixed anchor selected by `anchor_logp`.
4. Run the client closure with `forward_backward_custom`, reusing its anchor
   forward, or assemble the built-in inputs and call `forward_backward`.
5. Apply the optimizer update and publish sampler weights in the recipe.

For example, the async client GRPO path uses:

```python
Config(
    policy_loss="grpo",
    loss_execution="client",
    anchor_logp="old_policy",
    kl_beta=0.001,
    eps_clip=0.2,
    eps_clip_high=None,
)
```

Use `kl_beta=0` to disable reference KL and provisioning. Unused reference
settings and built-in objectives that cannot consume reference KL are rejected
at startup. Client GRPO applies a differentiable k3 reference-KL penalty;
`ref_kl` is not only a logging estimator.

External TIS advantage preweighting is removed. The default clipping anchor is
still the pre-update trainer snapshot (`anchor_logp="old_policy"`);
`anchor_logp="rollout"` instead clips against recorded sampling logprobs and
skips the anchor forward. Sampling probabilities remain a separate input for
importance-sampling objectives and drift metrics. Removing external TIS changes
the estimator when trainer and sampler policies differ.

## Choose client or built-in execution

The async recipe uses `loss_execution="client" | "builtin"` for registered
objectives. Built-in GRPO maps to the trainer's `"ppo"` loss and requires
`kl_beta=0`. Its datum logprobs must contain the selected clipping anchor:

```python
from training.utils.rl.anchor import prepare_policy_anchor
from training.utils.rl.losses import build_grpo_datums

anchor, _ = prepare_policy_anchor(
    policy, data, rollout_logprobs, cfg.anchor_logp,
)
eps_high = cfg.eps_clip if cfg.eps_clip_high is None else cfg.eps_clip_high
grpo_datums = build_grpo_datums(
    data,
    advantages,
    anchor,
    prompt_lens,
    include_response_mask=True,
)
result = policy.forward_backward(
    grpo_datums,
    "ppo",
    loss_fn_config={
        "clip_low_threshold": 1.0 - cfg.eps_clip,
        "clip_high_threshold": 1.0 + eps_high,
    },
)
```

Objectives without a built-in implementation, or whose loss name the installed
SDK does not accept, fail at startup. Custom research losses remain ordinary
closures; there is no automatic execution fallback.

## Sampling-support metadata

With an SDK release containing the sampling-support API, both deployment
sampling and the Tinker-style `FiretitanSamplingClient.sample` path can record
support. For the latter, opt in with:

```python
from fireworks.training.sdk import FiretitanSamplingParams

params = FiretitanSamplingParams(top_p=0.95, top_sampling_logprobs=K)
response = sampler.sample(prompt, 1, params).result()
support = response.sequences[0].top_sampling_references
```

`top_sampling_logprobs` is a storage-width limit, not a top-k sampling filter.
The wrapper requests `parquet_v1` automatically. Leave it unset to avoid
recording support; the default target distribution remains `full_vocabulary`.
References contain generated-token rows only, including when
`include_prompt_logprobs=True`. When assembling a datum, preserve the
next-token alignment: reference row j belongs to generated target token j;
mask prompt positions and serialize the aligned references using
`top_sampling_model_input_kwargs` from `fireworks.training.sdk.routing`.

Using the recorded support requires a shared storage binding, a compatible
serving image, and `training_client.supports_target_logprob_support`. Explicitly
set `loss_fn_config={"target_logprob_support": "sampling_support"}` on the
built-in backward request. This renormalizes the target distribution over the
recorded support; it does not change sampler settings. K must cover the complete
support at every trained token: truncated support is rejected by the trainer.
`r3_ttl_seconds` controls retention when support or routing recording is enabled.
No deployment-start replay switch is required. Client custom losses and
unsupported trainers cannot silently substitute for this built-in path.

## Add a research algorithm

Create or update `training/utils/rl/algorithm/<algorithm>.py`. Keep its formula,
typed options, and validation together. To expose it through the async recipe,
export a `POLICY_LOSS` definition and register it in
`training/utils/rl/algorithm/__init__.py`. The definition maps options into a
client closure or built-in `loss_fn_config`; it must not perform trainer I/O.

Validate settings when constructing a closure and resolving recipe options.
Each option must affect the objective or documented observability. For a
one-off custom loss, fork the closest recipe and replace its loss call while
keeping rollout, checkpoint, optimizer, and weight-sync operations explicit.

IGPO owns per-token turn advantages and uses `make_igpo_loss_fn` with
`forward_backward_custom` in its dedicated recipe.

## Custom-loss interface

```python
def my_loss(data, logprobs_list):
    # data: aligned tensors (advantages, inference logprobs, prompt lens, etc.)
    # logprobs_list: per-datum training logprobs from the trainer forward pass
    loss = compute_loss(data, logprobs_list)   # scalar torch.Tensor, requires_grad
    return loss, {"loss": float(loss.item())}  # (loss_tensor, metrics_dict)
```

The recipe passes the closure into
`training_client.forward_backward_custom(datums, my_loss).result()`.

`training/utils/rl/algorithm/grpo.py` is the reference implementation:
advantages, logprobs, and optional KL return a scalar plus metrics. Other direct
builders live beside it (`dapo.py`, `dro.py`, `gspo.py`, `cispo.py`, and so on).
The previous `training/utils/rl/<algorithm>.py` import paths remain aliases.

## Preserve invariants

- Import loss builders at module scope.
- Require exact datum, prompt-boundary, and logprob alignment.
- Keep sampling probabilities separate from the fixed clipping anchor. Use
  `old_policy_logprobs` for snapshot clipping, and rollout probabilities for IS.
- Keep advantages unweighted; the chosen objective owns any weighting.
- Keep raw inference logprobs observability-only; report drift metrics without
  feeding them into the policy ratio.
- Provision and run a reference only when the selected loss consumes it.
- Raise on incompatible configuration; never ignore or downgrade it.
- Test the direct builder and the recipe boundary. Delete tests that only
  exercise removed dispatch machinery.

## RL-only `Config` fields commonly changed

All live on `rl_loop.Config`:

| Field | Default | Meaning |
|---|---|---|
| `completions_per_prompt` | `4` | GRPO group size: responses sampled per prompt. |
| `prompt_groups_per_step` | `1` | Prompt groups per `forward_backward + optim_step` pair. |
| `kl_beta` | `0.001` | Reference-KL coefficient; `0` skips the reference. |
| `eps_clip`, `eps_clip_high` | `0.2`, `None` | PPO clip for GRPO. |
| `router_replay`, `router_replay_completion_only` | `True`, `True` | Replay MoE routes for generated tokens by default. Set completion-only to `False` only when full-sequence replay is worth the serving cost of `echo=True`. |
| `grad_accumulation_normalization` | `None` | No server-side normalization by default. Use `NUM_LOSS_TOKENS` for raw-sum losses. See [`rl-gradient-accumulation.md`](rl-gradient-accumulation.md). |

Trainer accelerator type and count are not cookbook config fields; select them
indirectly with `training_shape_id`. Other shape-owned fields such as
`node_count` and `custom_image_tag` come from the training profile; never
hand-set them.

## Do not

- Reimplement `forward_backward_custom`; replace the documented direct builder
  in a fork of the closest recipe.
- Add an automatic fallback. A fork calls either `forward_backward` or
  `forward_backward_custom` explicitly.
- Silently reuse the GRPO reference. Keep or remove reference provisioning
  based on what the replacement closure consumes.
- Forget `grad_accumulation_normalization`. Match it to whether the loss returns
  a raw sum or a pre-normalized mean; double-normalization is the common bug.

Run the focused RL and provisioning tests under `training/` with the `dev`
extra, then lint every changed Python file.

## See also

- Built-in GRPO datum preparation: `training/utils/rl/losses.py`.
- Policy objectives and their registry: `training/utils/rl/algorithm/`.
- `forward_backward_custom` signature and behavior:
  `fireworks.training.sdk.client.FiretitanTrainingClient.forward_backward_custom`.
