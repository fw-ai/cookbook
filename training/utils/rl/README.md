# RL rollout primitives

> ⚠️ **EXPERIMENTAL — the async RL recipe these primitives serve is under
> active development.** API may change without backward-compat shims.

Token-native rollout API for the async RL recipe (`training/recipes/async_rl_loop.py`).

For the user-facing contract (`rollout_fn(sample_prompt) -> RolloutRun`,
`RolloutSetup`, `rollout_fn_factory`, off-policy gate sizing), see
[`/skills/fireworks-training/references/rl-async.md`](/skills/fireworks-training/references/rl-async.md).

## Layout

| File | Purpose |
| --- | --- |
| `rollout/types.py` | `Rollout`, `RolloutRun`, `RolloutSample`, `rollout_to_prompt_group` (trainer packing). |
| `rollout/assembler.py` | Token-native multi-turn assembly with prefix checks. |
| `rollout/message.py` | Frozen legacy public-import surface; retained for compatibility, not new integrations. |
| `rollout/renderer.py` | Optional renderer-backed single-turn helper. |
| `rollout/remote.py` | Optional service payload packer. |

The correctness-critical path is `TrajectoryAssembler`: every next model
request must extend the accumulated token sequence, except for an explicit
generic boundary trim passed by the caller.

New message-based multi-turn integrations use an SDK `TITOSidecar` inside each
agent environment through a cookbook harness adapter. The legacy
`MessageTrajectoryAssembler` and agent utilities remain byte-for-byte stable
because they are public imports, but no production example depends on them and
new callers must not extend that island.

## Tests

Invariants only: `test_rollout_types.py`, `test_rollout_assembler.py`,
`test_rollout_message.py`, `test_rollout_helpers.py`.  No remote-service
mocks or example-specific policy tests unless they guard a real trainer or
token-alignment invariant.
