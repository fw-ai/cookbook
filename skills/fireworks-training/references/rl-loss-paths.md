# RL loss execution

`training/recipes/rl_loop.py` defaults to client-side GRPO.
`training/recipes/async_rl_loop.py` has the same default plus one narrow
`server_side_grpo=True` opt-in for the trainer's built-in PPO kernel. Both
paths still compute group-normalized advantages in the recipe.

There is no generic loss selector, registry, runtime import, or fallback. The
server opt-in is GRPO-only and requires `kl_beta=0`. The public algorithm knobs
remain `kl_beta`, `eps_clip`, `eps_clip_high`, and `tis`; the two recipes also
share `anchor_logp="old_policy" | "rollout"`.

## Default client path

The recipe performs an optional reference forward when `kl_beta > 0`, snapshots
old-policy logprobs, and calls:

```python
policy.forward_backward_custom(
    data,
    make_grpo_loss_fn(
        advantages=advantages,
        ref_logprobs=ref_logprobs,
        prompt_len=prompt_lens,
        inf_logprobs=rollout_logprobs,
        old_policy_logprobs=old_policy_logprobs,
        kl_beta=cfg.kl_beta,
        eps_clip=cfg.eps_clip,
        eps_clip_high=cfg.eps_clip_high,
        tis_config=cfg.tis,
    ),
    precomputed_forward=old_policy_result,
)
```

This one closure owns PPO clipping, behavioral TIS, and optional reference KL.
Set `kl_beta=0` to skip reference provisioning.

Both recipes default to `anchor_logp="old_policy"`: snapshot trainer logprobs for the
PPO anchor and compute TIS against rollout behavior logprobs. Setting
`anchor_logp="rollout"` skips the snapshot, anchors PPO directly on rollout
logprobs, and makes the TIS ratio identity.

The sync, dedicated async, and serverless async client-loss recipes reuse the
old-policy forward result to construct the custom-loss gradients. The trainer
still performs the differentiable forward/backward recomputation; only the
duplicate standalone custom-loss forward is removed. Because the reused
old-policy logprobs are exactly the PPO anchor, this makes
`train/ppo_ratio_mean=1` and `train/ppo_clip_frac=0` for that update instead of
values differing from identity only by a redundant forward's numerical noise.

This is only a compute-path optimization. Train/inference K1 and K3 retain
their historical definition: mean active-token drift within each sequence,
mean across sequences within a trainer chunk, then mean across reported chunks
at the optimizer step.

## Dedicated async built-in path

With `Config(server_side_grpo=True, kl_beta=0)`, the async recipe prepares the
built-in datum contract with `build_grpo_datums(...)` and calls
`forward_backward(..., "ppo")`. `anchor_logp="rollout"` skips the old-policy
forward; `anchor_logp="old_policy"` retains it and folds TIS into per-token
advantages. The exact trainer logprobs returned by the built-in call produce
the same inference K1/K3 and PPO ratio/clip diagnostics without another
forward pass.

This path requires a trainer topology that supports built-in RL losses. It is
not added to the sync or experimental serverless recipes.

## Switching or adding another loss

Fork the recipe at its documented direct `forward_backward_custom` call.
For the exact built-in switch and new-algorithm workflow, read
[`rl-custom-loss.md`](rl-custom-loss.md).

Do not generalize `server_side_grpo` into a loss selector. A different
algorithm still belongs in an explicit recipe fork.

## Multimodal datum contract

Vision RL uses the canonical Tinker expanded sequence coordinates. For an
unshifted sequence of length `N`, including every image slot:

- `datum.model_input.length == N - 1`;
- `target_tokens`, `weights`, forward logprobs, and backward gradients all have
  length `N - 1`;
- image positions in `target_tokens` are zero wire placeholders; and
- image positions have zero weight/advantage and contribute no loss.

Do not strip image positions or compress tensors into text-only coordinates.
