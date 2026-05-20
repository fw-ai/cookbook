# Async RL examples

> ⚠️ **EXPERIMENTAL — `async_rl_loop` is under active development.**
> Config/API may change without backward-compat shims.  See
> [`/skills/dev/references/rl/async-rl.md`](/skills/dev/references/rl/async-rl.md).

The recipe is intentionally minimal-surface: **the only thing you customize
is the rollout function** (`rollout_fn(sample_prompt) -> RolloutSample`).
Gate, advantage, ref forward, weight sync, KL/TIS, PPO inner loop, and
checkpoints are all handled by `recipes.async_rl_loop.main`.

Two minimal rollouts for the async recipe (`training/recipes/async_rl_loop.py`):

- `single_turn_token_in/` — pre-tokenized rows; one `/v1/completions` call per
  `rollout_fn` invocation (recipe invokes `rollout_fn` `completions_per_prompt`
  times per row).
- `multi_turn_message_in/` — OpenAI-style messages; ports AReaL's
  `examples/multi_turn_math/` retry loop.

Each example exposes `rollout_fn_factory(setup) -> rollout_fn` (signature
`async def rollout_fn(sample_prompt) -> RolloutSample | None`) and a `train.py`
that wires the dataset and factory into `recipes.async_rl_loop.main`.

For the API contract and recipe knobs, see
[`/skills/dev/references/rl/async-rl.md`](/skills/dev/references/rl/async-rl.md).

The `train.py` files use placeholder model and tokenizer names — replace with
your own model and tokenizer identifiers before running.
