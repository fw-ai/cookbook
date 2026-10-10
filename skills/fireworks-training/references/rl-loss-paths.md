# RL loss execution

Recipes compute group-relative advantages and keep them unweighted. PPO and
GSPO clip against a fixed trainer snapshot by default (`anchor_logp="old_policy"`).
Set `anchor_logp="rollout"` to use recorded sampling probabilities instead.
There is no TIS config or separate advantage correction.

## Client path

`rl_loop` uses `forward_backward_custom` and an ordinary loss closure. The SDK
reuses the snapshot forward for the differentiable callback. Optional reference KL remains controlled by
`kl_beta`; raw inference logprobs are used only for drift diagnostics.

```python
anchor, forward = prepare_policy_anchor(policy, data, rollout_logprobs, cfg.anchor_logp)
policy.forward_backward_custom(
    data,
    make_grpo_loss_fn(
        advantages=advantages,
        ref_logprobs=ref_logprobs,
        prompt_len=prompt_lens,
        inf_logprobs=rollout_logprobs,
        old_policy_logprobs=anchor,
        kl_beta=cfg.kl_beta,
        eps_clip=cfg.eps_clip,
        eps_clip_high=cfg.eps_clip_high,
    ),
    precomputed_forward=forward,
)
```

## Built-in path

The async recipe's `server_side_grpo=True` uses built-in PPO, or GSPO when
selected explicitly. These paths require `kl_beta=0`. Datum preparation sends
the selected anchor, masked raw advantages, and independent response membership.
Returned trainer logprobs supply diagnostics without another forward.

Removing external TIS weights changes the estimator when rollout and trainer
policies differ, even with the snapshot anchor restored. Existing trainer images
still accept the same built-in
and custom-loss protocols. New trainer-only features require their advertised
capabilities.

For research losses, replace the direct closure described in
[`rl-custom-loss.md`](rl-custom-loss.md). Keep optimizer normalization, masks,
checkpointing and weight synchronization explicit.

## Multimodal datum contract

The synchronous `rl_loop` default rollout supports both text and image messages
through `service.create_deployment_sampler(...)`. It sends the renderer's image
payloads with the token prompt and preserves the same image chunks for training;
Shape CI uses this default path too. R3 format negotiation stays in the SDK:
it selects Parquet when the trainer and inference advertise a shared R3 store.
Do not construct an unbound sampler or force an R3 format in the recipe.

Vision RL uses the canonical Tinker expanded sequence coordinates. For an
unshifted sequence of length `N`, including every image slot:

- `datum.model_input.length == N - 1`;
- `target_tokens`, `weights`, forward logprobs, and backward gradients all have
  length `N - 1`;
- image positions in `target_tokens` are zero wire placeholders; and
- image positions have zero weight/advantage and contribute no loss.

Do not strip image positions or compress tensors into text-only coordinates.
