# Training API losses and datums

*Source of truth: live [Dedicated Training](https://docs.fireworks.ai/fine-tuning/training-api/dedicated.md), [service client reference](https://docs.fireworks.ai/fine-tuning/training-api/reference/service-client.md), and the maintained cookbook recipes. Defer signatures to the installed SDK and pinned cookbook checkout.*

Use this reference when selecting a loss path, constructing datums, or forking a recipe. Checkpoint behavior belongs in `sdk-checkpoints.md`; hotload recovery belongs in `rl-hotload.md`.

## Choose the loss path

| Need | API pattern | Start from |
|---|---|---|
| Standard SFT | Recipe-owned weighted token loss | `training/recipes/sft_loop.py` |
| DPO or ORPO | Pairwise recipe objective | `dpo_loop.py` or `orpo_loop.py` |
| Standard synchronous RL | Cookbook GRPO path | `rl_loop.py` |
| Standard asynchronous RL | Cookbook async objective and scheduler | `async_rl_loop.py` |
| New research objective | Fork the closest maintained recipe | `rl-custom-loss.md` |
| Direct built-in loss | `forward_backward(...)` | Installed SDK contract |
| Direct client closure | `forward_backward_custom(...)` | Closest recipe implementation |

Prefer a maintained recipe over a blank loop. A direct SDK call is appropriate only when the reviewed task needs a contract the recipes do not expose.

## Datum invariants

- Preserve exact token IDs produced by the selected renderer.
- Apply loss only to intended policy or target tokens.
- Keep prompt, system, user, tool-result, and repaired-context tokens masked.
- Keep per-token weights aligned with tokens and log probabilities.
- Validate every row locally before upload or trainer creation.
- For multimodal datums, follow the public docs and renderer reference rather than inventing an image schema in the skill.

Read `renderer.md` and `renderer-verification.md` before changing tokenization or masks.

## Built-in and custom calls

`forward_backward` uses a trainer-supported objective and the datum fields that objective requires. `forward_backward_custom` returns local differentiable log probabilities to a reviewed closure that produces one scalar loss and metrics.

```python
loss, metrics = loss_fn(data, logprobs_list)
```

The closure must:

1. Return a scalar differentiable loss.
2. Preserve token alignment.
3. Normalize by the reviewed unit, such as loss tokens or sequences.
4. Report enough metrics to detect empty masks, zero variance, NaNs, and clipping saturation.

Do not assume that a weighted SFT datum satisfies a built-in cross-entropy contract. Use the recipe path or inspect the installed SDK's required datum fields.

## Gradient accumulation

Accumulate by calling forward/backward multiple times before one optimizer step. Do not configure removed trainer-launch accumulation fields.

Normalization must be explicit when microbatches have different token or sequence counts. Read `rl-gradient-accumulation.md` for supported normalization modes and failure checks.

## DPO, ORPO, and RL

- DPO and ORPO use the pairwise schema and recipe implementation in `sdk-recipes.md`.
- GRPO and related objectives require grouped samples, aligned behavior log probabilities, and reviewed advantage normalization.
- Custom agent trajectories must satisfy `rl-agentic.md`.
- Async scheduling and off-policy admission belong in `rl-async.md`, not in the loss closure.

## Checkpoints are a separate contract

Do not conflate:

1. Sampler snapshots for weight sync.
2. DCP state for exact resume.
3. Promotable checkpoints for creating a Fireworks model.

Read `sdk-checkpoints.md` for save, resume, and promote behavior and `rl-hotload.md` for deployment synchronization.

## Preflight checklist

- Pin the cookbook commit and compatible SDK version.
- Select the closest maintained recipe.
- Validate datum schema and renderer output on a small local sample.
- Verify non-empty masks and finite loss.
- Record the normalization unit and gradient-accumulation behavior.
- Resolve checkpoint, resume, deployment, and teardown configuration.
- Include all resolved values in the final-plan confirmation.

## Related

- Recipe catalog: `sdk-recipes.md`
- Custom RL objectives: `rl-custom-loss.md`, `rl-loss-paths.md`
- Gradient accumulation: `rl-gradient-accumulation.md`
- Checkpoints: `sdk-checkpoints.md`
- Hotload: `rl-hotload.md`
- Renderer correctness: `renderer-verification.md`
