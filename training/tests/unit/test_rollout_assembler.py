"""Unit tests for ``TrajectoryAssembler``.

The assembler carries the AReaL prefix-equality invariant for multi-turn
rollouts.  Each engine call's ``input_tokens`` must extend the
already-accumulated sequence verbatim; if the rollout function
re-tokenized text between turns the assembler raises
:class:`PrefixMismatch` instead of training on misaligned tokens.
"""

from __future__ import annotations

import pytest

from training.utils.rl.rollout import remote as tr
from training.utils.rl.rollout import (
    InferenceCall,
    PrefixMismatch,
    TrajectoryAssembler,
)


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_single_call_builds_payload():
    asm = TrajectoryAssembler(tokenizer_id="test-tok")
    asm.add_call(
        InferenceCall(
            input_tokens=[1, 2, 3],
            output_tokens=[10, 11],
            output_logprobs=[-0.1, -0.2],
            finish_reason="stop",
        ),
    )
    payload = asm.to_payload(total_reward=0.5)

    assert payload.tokenizer_id == "test-tok"
    assert payload.total_reward == 0.5
    assert payload.finish_reason == "stop"
    assert getattr(payload, "_assembled") is True
    assert [t.role for t in payload.turns] == ["user", "assistant"]
    assert payload.turns[0].token_ids == [1, 2, 3]
    assert payload.turns[1].token_ids == [10, 11]
    assert payload.turns[1].logprobs == [-0.1, -0.2]


def test_multi_turn_extends_prefix():
    asm = TrajectoryAssembler()
    asm.add_call(
        InferenceCall(
            input_tokens=[1, 2, 3],
            output_tokens=[10, 11],
            output_logprobs=[-0.1, -0.2],
        ),
    )
    # Next call's input is prior prompt + assistant output + a 2-token suffix.
    asm.add_call(
        InferenceCall(
            input_tokens=[1, 2, 3, 10, 11, 50, 51],
            output_tokens=[20, 21, 22],
            output_logprobs=[-0.3, -0.4, -0.5],
        ),
        role_before="tool",
    )
    payload = asm.to_payload(total_reward=1.0)

    roles = [t.role for t in payload.turns]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert payload.turns[2].token_ids == [50, 51]
    assert payload.turns[3].token_ids == [20, 21, 22]
    assert payload.turns[3].logprobs == [-0.3, -0.4, -0.5]
    assert asm.accumulated_tokens == [1, 2, 3, 10, 11, 50, 51, 20, 21, 22]


def test_to_flat_returns_aligned_arrays():
    asm = TrajectoryAssembler()
    asm.add_call(
        InferenceCall(
            input_tokens=[1, 2],
            output_tokens=[10, 11],
            output_logprobs=[-0.1, -0.2],
        ),
    )
    asm.add_call(
        InferenceCall(
            input_tokens=[1, 2, 10, 11, 99],
            output_tokens=[20],
            output_logprobs=[-0.3],
        ),
    )
    tokens, logprobs, mask = asm.to_flat()

    assert tokens == [1, 2, 10, 11, 99, 20]
    assert mask == [0, 0, 1, 1, 0, 1]
    assert logprobs == [0.0, 0.0, -0.1, -0.2, 0.0, -0.3]


def test_add_call_allows_explicit_boundary_trim():
    asm = TrajectoryAssembler()
    asm.add_call(
        InferenceCall(
            input_tokens=[1, 2],
            output_tokens=[777],
            output_logprobs=[-0.1],
        ),
    )

    asm.add_call(
        InferenceCall(
            input_tokens=[1, 2, 50],
            output_tokens=[20],
            output_logprobs=[-0.2],
        ),
        max_trim_tokens=1,
    )

    tokens, logprobs, mask = asm.to_flat()
    assert tokens == [1, 2, 50, 20]
    assert logprobs == [0.0, 0.0, 0.0, -0.2]
    assert mask == [0, 0, 0, 1]
    assert asm.accumulated_tokens == [1, 2, 50, 20]


def test_add_environment_tokens_records_turn_but_not_engine_visible():
    """``add_environment_tokens`` is for tokens the ENGINE never sees as
    input (e.g. incremental-prompt adapters where the engine carries its
    own state).  The tokens MUST be recorded in the trajectory the trainer
    consumes, but they MUST NOT extend the engine-visible accumulated
    sequence — otherwise ``add_call``'s strict-prefix invariant on the
    very next engine call would reject any input that doesn't start with
    those env-only tokens."""
    asm = TrajectoryAssembler()
    asm.add_call(
        InferenceCall(
            input_tokens=[1],
            output_tokens=[10],
            output_logprobs=[-0.1],
        ),
    )
    asm.add_environment_tokens([42, 43], role="tool")

    # The engine-visible prefix is unchanged by add_environment_tokens.
    assert asm.accumulated_tokens == [1, 10]

    # The next engine call's input_tokens starts at the engine-visible
    # prefix [1, 10] — NOT [1, 10, 42, 43].  This is the documented
    # contract for env-only tokens.
    asm.add_call(
        InferenceCall(
            input_tokens=[1, 10],
            output_tokens=[20],
            output_logprobs=[-0.2],
        ),
    )

    # Trajectory the trainer consumes: includes the tool turn between
    # the two assistant turns.
    roles = [t.role for t in asm.to_payload(total_reward=0.0).turns]
    assert roles == ["user", "assistant", "tool", "assistant"]


def test_add_environment_tokens_does_not_break_next_add_call_prefix():
    """Regression: previously ``add_environment_tokens`` extended the
    engine-visible ``_seq``, so the next ``add_call`` REQUIRED the env
    tokens to appear in its ``input_tokens`` — which contradicts the
    documented purpose of the helper.  An incremental-engine adapter
    that records a tool reply as env-only and then issues the next
    engine call (with a fresh, narrower input prefix) used to hit
    ``PrefixMismatch`` immediately."""
    asm = TrajectoryAssembler()
    asm.add_call(
        InferenceCall(
            input_tokens=[1],
            output_tokens=[10],
            output_logprobs=[-0.1],
        ),
    )
    asm.add_environment_tokens([42, 43], role="tool")
    # The next engine call still passes only the engine-visible prefix
    # plus its own new tokens.  Must not raise PrefixMismatch.
    asm.add_call(
        InferenceCall(
            input_tokens=[1, 10],  # env tokens [42, 43] are NOT here
            output_tokens=[20],
            output_logprobs=[-0.2],
        ),
    )


# ---------------------------------------------------------------------------
# Error paths -- the whole point of the helper
# ---------------------------------------------------------------------------


def test_prefix_mismatch_raises_with_index():
    asm = TrajectoryAssembler()
    asm.add_call(
        InferenceCall(
            input_tokens=[1, 2, 3],
            output_tokens=[10, 11],
            output_logprobs=[-0.1, -0.2],
        ),
    )
    # Caller "re-tokenized" and produced a different token at position 2.
    with pytest.raises(PrefixMismatch) as exc_info:
        asm.add_call(
            InferenceCall(
                input_tokens=[1, 2, 999, 10, 11, 50],
                output_tokens=[20],
                output_logprobs=[-0.3],
            ),
        )
    msg = str(exc_info.value)
    assert "index 2" in msg
    assert "prior_token=3" in msg
    assert "input_token=999" in msg
    assert "re-tokenized" in msg


def test_prefix_too_short_raises():
    asm = TrajectoryAssembler()
    asm.add_call(
        InferenceCall(
            input_tokens=[1, 2, 3, 4, 5],
            output_tokens=[10],
            output_logprobs=[-0.1],
        ),
    )
    # Next input is shorter than what the engine has already seen.
    with pytest.raises(PrefixMismatch):
        asm.add_call(
            InferenceCall(
                input_tokens=[1, 2, 3],
                output_tokens=[20],
                output_logprobs=[-0.2],
            ),
        )


def test_logprob_length_mismatch_raises():
    asm = TrajectoryAssembler()
    with pytest.raises(ValueError, match="output_logprobs length"):
        asm.add_call(
            InferenceCall(
                input_tokens=[1],
                output_tokens=[10, 11, 12],
                output_logprobs=[-0.1, -0.2],
            ),
        )


def test_empty_to_payload_raises():
    asm = TrajectoryAssembler()
    with pytest.raises(RuntimeError, match="empty"):
        asm.to_payload(total_reward=0.0)


def test_no_assistant_to_payload_raises():
    asm = TrajectoryAssembler()
    asm.add_environment_tokens([1, 2, 3], role="user")
    with pytest.raises(RuntimeError, match="no assistant"):
        asm.to_payload(total_reward=0.0)


# ---------------------------------------------------------------------------
# Round-trip through the existing packer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_assembler_payload_round_trips_through_packer():
    asm = TrajectoryAssembler(tokenizer_id="test-tok")
    asm.add_call(
        InferenceCall(
            input_tokens=[1, 2, 3],
            output_tokens=[10, 11],
            output_logprobs=[-0.1, -0.2],
        ),
    )
    asm.add_call(
        InferenceCall(
            input_tokens=[1, 2, 3, 10, 11, 50],
            output_tokens=[20, 21],
            output_logprobs=[-0.3, -0.4],
        ),
        role_before="user",
    )
    payload = asm.to_payload(total_reward=1.5)

    sample = await tr.pack_payload_to_sample(
        payload,
        tokenizer_id="test-tok",
    )
    # Concatenated tokens / mask / logprobs come straight from the assembler.
    assert sample.tokens == [1, 2, 3, 10, 11, 50, 20, 21]
    assert sample.loss_mask == [0, 0, 0, 1, 1, 0, 1, 1]
    assert sample.logprobs == [0.0, 0.0, 0.0, -0.1, -0.2, 0.0, -0.3, -0.4]
    assert sample.reward == pytest.approx(1.5)
