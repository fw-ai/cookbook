# RL: customize a loss

Keep the recipe's algorithm visible at one call site. Do not add a registry,
string selector, runtime import, automatic fallback, or generic bundle of every
algorithm's configuration.

## Generic recipe contract

Treat `training/recipes/rl_loop.py` and `async_rl_loop.py` as client-side GRPO
recipes:

1. Compute group-normalized advantages from rollout rewards.
2. When `kl_beta > 0`, provision a reference and collect reference logprobs.
3. Snapshot old-policy logprobs on the trainer.
4. Call `make_grpo_loss_fn(...)` directly with advantages, reference and
   rollout logprobs, prompt boundaries, `kl_beta`, PPO clipping, and `TISConfig`.
5. Pass that closure to `policy.forward_backward_custom(...)`.

Keep the public algorithm knobs shallow:

```python
Config(
    kl_beta=0.001,
    eps_clip=0.2,
    eps_clip_high=None,
    tis=TISConfig(cap=5.0, level="token"),
)
```

Use `kl_beta=0` to disable reference KL and reference provisioning. Reject
reference-trainer settings when KL is disabled instead of ignoring them. The
client GRPO builder applies a differentiable k3 reference-KL penalty; `ref_kl`
is not only a logging estimator.

Async also exposes the PPO anchor explicitly:

```python
AsyncConfig(anchor_logp="old_policy")  # trainer snapshot + active TIS
AsyncConfig(anchor_logp="rollout")     # skip snapshot; TIS ratio is identity
```

Treat this as an estimator choice, not a performance-only switch. Validate
rollout-anchor rows against `target_tokens` exactly.

## Switch to the trainer built-in

Make the switch explicitly in a recipe fork. The trainer's built-in `"ppo"`
kernel does not consume reference logprobs, so require `kl_beta=0`.

```python
from training.utils.rl.losses import build_grpo_datums

eps_high = cfg.eps_clip if cfg.eps_clip_high is None else cfg.eps_clip_high
grpo_datums = build_grpo_datums(
    data,
    advantages,
    old_policy_logprobs,
    rollout_logprobs,
    prompt_lens,
    cfg.tis,
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

Delete the now-unused reference forward from that fork. Do not keep both paths
behind an `if` or silently fall back to the client loss.

## Add a research algorithm

Create or update one direct builder under `training/utils/rl/<algorithm>.py`.
Keep algorithm-specific configuration and a
`validate_<algorithm>_config(...)` helper beside that builder. Call validation
unconditionally when constructing the loss closure so invalid settings fail
before the first training forward. Every config field must affect the objective
or explicitly documented observability; delete unused knobs. Fork the closest
recipe and replace only its documented loss call; keep rollout, checkpoint,
optimizer, and weight-sync plumbing intact.

Add configuration to the generic recipe only when it remains part of that
recipe's GRPO contract. Do not add every research algorithm's knobs to the
generic `Config`.

IGPO owns per-token turn advantages and currently uses
`make_igpo_loss_fn(...)` with `forward_backward_custom(...)`. Keep it separate
from generic GRPO dispatch.

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

`training/utils/rl/grpo.py` is the reference implementation: advantages,
logprobs, and optional KL return a scalar plus metrics. Other direct builders
live beside it (`dapo.py`, `dro.py`, `gspo.py`, `cispo.py`, and so on).

## Preserve invariants

- Import loss builders at module scope.
- Require exact datum, prompt-boundary, and logprob alignment.
- Apply TIS only on active loss-mask positions.
- Keep rollout logprobs as the TIS behavior-policy denominator.
- Keep raw inference logprobs observability-only; report drift metrics without
  feeding them into PPO or TIS.
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
| `router_replay` | `False` | Record routing at rollout time and replay during training for MoE models. |
| `grad_accumulation_normalization` | `None` | No server-side normalization by default. Use `NUM_LOSS_TOKENS` for raw-sum losses. See [`rl-gradient-accumulation.md`](rl-gradient-accumulation.md). |

Shape-owned fields (`accelerator_type`, `node_count`, and `custom_image_tag`)
come from the training profile; never hand-set them.

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
- Direct client loss builders: `training/utils/rl/{grpo,dapo,dro,gspo,cispo}.py`.
- `forward_backward_custom` signature and behavior:
  `fireworks.training.sdk.client.FiretitanTrainingClient.forward_backward_custom`.
