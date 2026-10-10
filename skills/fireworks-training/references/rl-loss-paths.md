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

The async recipe's `loss_execution="builtin"` runs the selected objective's
trainer loss (`grpo` maps to built-in `"ppo"`); `"client"` runs the portable
closure. Built-in execution requires `kl_beta=0`. Datum preparation sends the
selected anchor, masked raw advantages, and response membership. Returned
trainer logprobs supply diagnostics without another forward.

Objectives live in `training/utils/rl/algorithm/`, one module per algorithm.
Each module exports a `POLICY_LOSS` definition: its typed options, the client
closure, and the built-in `loss_fn_config` translation. `resolve_policy_loss`
validates the choice once at startup; the recipe then calls
`forward`/`forward_backward`/`forward_backward_custom` itself.

### Sampling support (top-p mask replay)

With `top_p < 1`, rollouts sample from a truncated, renormalized distribution.
To train against the same support, set:

```python
Config(
    loss_execution="builtin",
    anchor_logp="rollout",
    top_p=0.95,
    top_sampling_logprobs=4096,  # Capacity; must cover every trained token's support
    target_logprob_support="sampling_support",
    kl_beta=0,
)
```

The sampler writes each completion token's top-K support to trainer-shared
storage and returns `SampledCompletion.top_sampling_references`. Rollout
assembly aligns them with model-input positions (prompt and tool tokens are
gaps); the recipe attaches them to built-in datums as
`model_input.top_sampling_references`, and the trainer normalizes target
logprobs over that support. It rejects incomplete support at trained response
positions, including zero-advantage responses. `top_sampling_logprobs` is only
accepted with `sampling_support`; `top_p < 1` with `full_vocabulary` logs a warning.

`full_vocabulary` is the default and requests no sampling-support recording or
transfer. On a compatible deployment with trainer-shared storage, this is a
per-request choice; it requires no separate replay startup flag. The recorded
support must be complete: the current trainer supports at most 20,000 candidates
per token, and some distributions exceed that limit even with `top_p < 1`.
Increasing K changes recording cost; it does not change the sampling threshold.
Support replay preserves references through text/image rollouts and agent/TITO
materialization. Built-in PPO/GSPO ratio and clipping diagnostics remain available;
inference-drift metrics are omitted because replay normalizes over a different
support from raw full-vocabulary inference logprobs.

Use an SDK release that supports Parquet sampling references when enabling it.
The serving image must support `top_sampling_format="parquet_v1"`; an existing
training profile can pin an older serving image even when the trainer is current.
Replay preserves existing trainer limits: GSPO is currently unsupported with
`global_rolling` batching.

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
