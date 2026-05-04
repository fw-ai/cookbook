# RL rollout primitives

> ⚠️ **EXPERIMENTAL — the async RL recipe these primitives serve is under
> active development.** API may change without backward-compat shims.

Token-native rollout API for the async RL recipe (`training/recipes/async_rl_loop.py`).

For the user-facing contract (`rollout_fn(sample_prompt) -> RolloutSample`,
`RolloutSetup`, `rollout_fn_factory`, off-policy gate sizing), see
[`/skills/dev/references/rl/async-rl.md`](/skills/dev/references/rl/async-rl.md).

## Layout

| File | Purpose |
| --- | --- |
| `rollout/types.py` | `Rollout`, `RolloutSample`, `rollout_to_prompt_group` (trainer packing). |
| `rollout/assembler.py` | Token-native multi-turn assembly with prefix checks. |
| `rollout/message.py` | Generic message-in TITO bridge that preserves prior assistant tokens. |
| `rollout/renderer.py` | Optional renderer-backed single-turn helper. |
| `rollout/remote.py` | Optional service payload packer. |

The correctness-critical path is `TrajectoryAssembler`: every next model
request must extend the accumulated token sequence, except for an explicit
generic boundary trim passed by the caller.

`MessageTrajectoryAssembler` wraps it for OpenAI-style message loops:
preserves prior assistant token IDs exactly; tokenizes only appended `tool` /
`user` / `system` messages; appends the next assistant generation prompt;
rejects edits to prior messages; supports bounded rollback to an earlier
assistant checkpoint.  No model-specific TITO subclasses live here — keep
tokenizer-specific boundary policy in user code.

## Tests

Invariants only: `test_rollout_types.py`, `test_rollout_assembler.py`,
`test_rollout_message.py`, `test_rollout_helpers.py`.  No remote-service
mocks or example-specific policy tests unless they guard a real trainer or
token-alignment invariant.
