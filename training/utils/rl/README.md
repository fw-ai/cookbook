# RL rollout primitives

> ⚠️ **EXPERIMENTAL — the async RL recipe these primitives serve is under
> active development.** API may change without backward-compat shims.

Token-native rollout API for the RL recipes (`training/recipes/rl_loop.py`, `async_rl_loop.py`).

For the user-facing contract (`rollout_fn(sample_prompt) -> RolloutRun`,
`RolloutSetup`, `rollout_fn_factory`, off-policy gate sizing), see
[`/skills/fireworks-training/references/rl-async.md`](/skills/fireworks-training/references/rl-async.md).

## Layout

| File | Purpose |
| --- | --- |
| `rollout/setup.py` | `RolloutSetup`, `default_sample_kwargs`, the rollout context helpers. |
| `rollout/types.py` | `Rollout`, `RolloutRun`, `RolloutSample`, `rollout_to_prompt_group` (trainer packing). |
| `rollout/assembler.py` | Token-native multi-turn assembly with prefix checks. |
| `rollout/turn_traces.py` | Pack per-turn `{prompt_ids, completion_ids, completion_logprobs}` traces into one sample; turn spans for IGPO. |
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

## Policy training

The RL recipes train server-side GRPO against the rollout logprobs. Forks call
these modules to change the objective; see
[`/skills/fireworks-training/references/rl-custom-loss.md`](/skills/fireworks-training/references/rl-custom-loss.md).

| File | Purpose |
| --- | --- |
| `losses.py` | `PromptGroup`, `PolicyBatch` (one `anchor_logprobs` field, optional `token_advantages`), `combine_prompt_groups`, `build_policy_datums`. |
| `algorithm/` | One module per objective with the same surface (`Config`, `LOSS_FN`, `loss_fn_config`, `build_datums`, `metrics`, `make_loss_fn`, `NORMALIZATION`). No registry. `igpo` sets turn-level advantages. |
| `anchor.py` | `recompute_anchor(policy, batch)` (clip against a trainer snapshot), `anchor_drift_metrics`. |
| `tis.py` | `apply_tis`: truncated importance weights between the recomputed anchor and the rollout. |
| `sampling_support.py` | Top-p support replay: `sampling_support_kwargs`, `attach_top_sampling_references`, `sampling_support_loss_fn_config`. |
| `heldout.py` | `make_heldout_evaluation` (a recipe `evaluation_fn` over fixed held-out rows) and `run_evaluation`. |
| `common.py` | Shared client loss loop, trust-region metrics and `inference_drift_metrics`. |

## Tests

Invariants only: `test_rollout_types.py`, `test_rollout_assembler.py`,
`test_rollout_message.py`, `test_rollout_helpers.py`.  No remote-service
mocks or example-specific policy tests unless they guard a real trainer or
token-alignment invariant.
