# RL loss execution

`training/recipes/rl_loop.py` and `training/recipes/async_rl_loop.py` are
intentionally opinionated client-side GRPO recipes. They compute
group-normalized advantages and call `make_grpo_loss_fn(...)` directly through
`forward_backward_custom(...)`.

There is no loss selector, registry, runtime import, or fallback. The public
algorithm knobs are `kl_beta`, `eps_clip`, `eps_clip_high`, and `tis`; the
two recipes also share `anchor_logp="old_policy" | "rollout"`.

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

## Switching or adding a loss

Fork the recipe at its documented direct `forward_backward_custom` call.
For the exact built-in switch and new-algorithm workflow, read
[`rl-custom-loss.md`](rl-custom-loss.md).

The server built-in `"ppo"` path cannot apply reference KL. A built-in fork
must require `kl_beta=0`, prepare the kernel's datum contract explicitly, and
call `forward_backward(...)`. Do not keep both paths behind a config selector.

## Multimodal datum contract

Vision RL uses the canonical Tinker expanded sequence coordinates. For an
unshifted sequence of length `N`, including every image slot:

- `datum.model_input.length == N - 1`;
- `target_tokens`, `weights`, forward logprobs, and backward gradients all have
  length `N - 1`;
- image positions in `target_tokens` are zero wire placeholders; and
- image positions have zero weight/advantage and contribute no loss.

Do not strip image positions or compress tensors into text-only coordinates.
