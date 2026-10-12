"""Which tokens of a FrozenLake episode are trained.

The rollout records one trace per turn and builds each next prompt from tokens:
previous prompt + completion + turn close + tool reply. ``episode_to_sample``
packs the episode into one sequence where only the model's tool calls carry
loss; the close that frames a turn and the tool replies are context.
"""

from __future__ import annotations

import pytest
from eval_protocol.models import EvaluationRow, InputMetadata

from training.examples.rl.frozen_lake.train_frozen_lake import (
    episode_to_sample,
    frozen_lake_reward,
)
from training.utils.rl.rollout import PrefixMismatch
from training.utils.rl.rollout.turn_traces import append_turn_close

CLOSE = [90, 91]  # e.g. "<|im_end|>", "\n"
TOOL_REPLY = [20, 21]


def _episode(traces: list[dict]) -> EvaluationRow:
    row = EvaluationRow(input_metadata=InputMetadata(row_id="seed_1"))
    row.execution_metadata.extra = {
        "token_turn_traces": traces,
        "step_rewards": [0.0] * (len(traces) - 1) + [1.0],
    }
    return frozen_lake_reward(row)


def _two_turns(first_completion: list[int], first_logprobs: list[float]) -> list[dict]:
    prompt = [1, 2, 3]
    return [
        {"prompt_ids": prompt, "completion_ids": first_completion, "completion_logprobs": first_logprobs},
        {
            "prompt_ids": prompt + append_turn_close(first_completion, CLOSE) + TOOL_REPLY,
            "completion_ids": [30],
            "completion_logprobs": [-0.3],
        },
    ]


def test_clean_stop_trains_the_close_the_model_emitted():
    sample = episode_to_sample(_episode(_two_turns([10, 90], [-0.1, -0.2])))

    # The model's own "<|im_end|>" (90) is trained; only the missing "\n" is framing.
    assert sample.tokens == [1, 2, 3, 10, 90, 91, 20, 21, 30]
    assert sample.loss_mask == [0, 0, 0, 1, 1, 0, 0, 0, 1]
    assert sample.logprobs == [0.0, 0.0, 0.0, -0.1, -0.2, 0.0, 0.0, 0.0, -0.3]


def test_truncated_turn_never_trains_a_synthesized_close():
    sample = episode_to_sample(_episode(_two_turns([10, 11], [-0.1, -0.2])))

    assert sample.tokens == [1, 2, 3, 10, 11, 90, 91, 20, 21, 30]
    assert sample.loss_mask == [0, 0, 0, 1, 1, 0, 0, 0, 0, 1]


def test_kimi_tool_call_prefill_is_trained_with_probability_one():
    # The rollout forces the tool-call section marker (99) before sampling.
    sample = episode_to_sample(_episode(_two_turns([99, 10], [0.0, -0.1])))

    assert sample.loss_mask[3:5] == [1, 1]
    assert sample.logprobs[3:5] == [0.0, -0.1]


def test_re_rendered_prompt_is_rejected_instead_of_misaligned():
    traces = _two_turns([10, 11], [-0.1, -0.2])
    traces[1]["prompt_ids"] = [1, 2, 3, 12, 90, 91, 20, 21]  # re-tokenized turn 1

    with pytest.raises(PrefixMismatch):
        episode_to_sample(_episode(traces))


def test_missing_logprobs_are_rejected():
    with pytest.raises(ValueError, match="output_logprobs length"):
        episode_to_sample(_episode(_two_turns([10, 11], [-0.1])))
