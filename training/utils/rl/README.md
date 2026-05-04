# RL Rollout Primitives

The rollout surface is token-native.  A user supplies:

```python
async def rollout_fn(sample_prompt) -> RolloutSample | None: ...
```

returning one trajectory per call as a `RolloutSample` with aligned
`tokens`, `logprobs`, and `loss_mask`.  Prompt, user, tool, and environment
tokens have `loss_mask=0`; assistant-generated tokens have `loss_mask=1`.
The framework fans each dataset row out to `completions_per_prompt`
parallel `rollout_fn` calls and joins them into a `PromptGroup` via
`GroupAssembler`.

Per-rollout context (sampler, tokenizer, sample kwargs, custom state) is
assembled once at startup as a `RolloutSetup` and closed over by the
user-supplied `rollout_fn_factory(setup) -> rollout_fn`; there is no
per-call `ctx` argument.  Custom rollout engines pass extra state via
`RolloutSetup.extras`.

## Layout

| File | Purpose |
| --- | --- |
| `rollout/types.py` | `Rollout`, `RolloutSample`, and trainer packing. |
| `rollout/assembler.py` | Token-native multi-turn assembly with prefix checks. |
| `rollout/message.py` | Generic message-in TITO bridge that preserves prior assistant tokens. |
| `rollout/renderer.py` | Optional renderer-backed single-turn helper. |
| `rollout/remote.py` | Optional service payload packer. |

The correctness-critical path is `TrajectoryAssembler`: every next model
request must extend the accumulated token sequence, except for an explicit
generic boundary trim passed by the caller.

## Generic Rollout Examples

The generic rollout contract is covered by two small examples under
`training/examples/rl/`:

* `single_turn_token_in`
* `multi_turn_message_in`

Existing domain examples remain separate from these generic examples.

## Message-In TITO

`MessageTrajectoryAssembler` wraps `TrajectoryAssembler` for OpenAI-style
message loops.  It:

* preserves prior assistant output token IDs exactly;
* tokenizes only appended `tool`, `user`, or `system` messages;
* appends the next assistant generation prompt;
* rejects edits to prior messages;
* supports bounded rollback to an earlier assistant checkpoint.

There are no model-specific TITO subclasses in the cookbook.  If a deployment
needs tokenizer-specific boundary behavior, keep that policy in user code and
pass the resulting token sequence into the generic assembler.

## Testing

Keep tests focused on invariants:

* `test_rollout_types.py`
* `test_rollout_assembler.py`
* `test_rollout_message.py`
* `test_rollout_helpers.py`

Avoid adding remote-service mocks or example-specific policy tests unless they
guard a real trainer or token-alignment invariant.
