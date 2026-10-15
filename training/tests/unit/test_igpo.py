"""IGPO turn-level advantages and the policy datums they produce."""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fireworks.training import sdk
from training.utils.rl.algorithm.igpo import (
    ANSWER_TOKENS,
    TURN_SPANS,
    compute_turn_advantages,
    expand_turn_advantages,
    information_gain,
    prepare_batch,
    score_answer_logprobs,
    token_advantages,
)
from training.utils.rl.losses import (
    PromptGroup,
    build_policy_datums,
    combine_prompt_groups,
)


def _datum(tokens: list[int], weights: list[int]) -> sdk.Datum:
    return sdk.Datum(
        model_input=sdk.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": sdk.TensorData(data=tokens[1:], dtype="int64", shape=[len(tokens) - 1]),
            "weights": sdk.TensorData(data=weights, dtype="int64", shape=[len(weights)]),
        },
    )


class _Scorer:
    """Trainer stand-in: ``log P(answer | context)`` depends on context length."""

    def __init__(self, scores_by_context_len: dict[int, float]):
        self.scores_by_context_len = scores_by_context_len
        self.calls: list[list[sdk.Datum]] = []

    def forward(self, data, loss_fn):
        assert loss_fn == "cross_entropy"
        self.calls.append(list(data))
        outputs = []
        for datum in data:
            # One answer token: the context is the whole model input.
            score = self.scores_by_context_len[datum.model_input.length]
            n = datum.model_input.length
            outputs.append({"logprobs": sdk.TensorData(data=[score] * n, dtype="float32", shape=[n])})
        return SimpleNamespace(loss_fn_outputs=outputs)


def test_information_gain_skips_the_last_turn() -> None:
    assert information_gain([-3.0, -1.0, -1.5]) == [2.0, -0.5, 0.0]
    assert information_gain([-3.0]) == [0.0]
    assert information_gain([]) == []


def test_score_answer_logprobs_averages_answer_positions_in_one_forward() -> None:
    calls = []

    class Client:
        def forward(self, data, loss_fn):
            calls.append(data)
            return SimpleNamespace(
                loss_fn_outputs=[
                    {"logprobs": sdk.TensorData(data=[-9.0, -1.0, -3.0], dtype="float32", shape=[3])},
                    {"logprobs": sdk.TensorData(data=[-9.0, -9.0, -2.0], dtype="float32", shape=[3])},
                ]
            )

    scores = score_answer_logprobs(Client(), [[1, 2], [1, 2, 3]], [[7, 8], [9]])

    assert scores == [-2.0, -2.0]
    assert len(calls) == 1
    assert calls[0][0].model_input.to_ints() == [1, 2, 7]
    assert calls[0][0].loss_fn_inputs["target_tokens"].data == [2, 7, 8]


def test_turn_advantages_normalize_streams_separately_then_discount() -> None:
    a = math.sqrt(2) / 2
    advantages = compute_turn_advantages(
        ig_rewards=[[1.5, 0.0], [0.0]],
        outcome_rewards=[[0.0, 1.0], [0.0]],
        gamma=0.5,
    )

    assert advantages[0] == pytest.approx([1.5 * a, a], rel=1e-5)
    assert advantages[1] == pytest.approx([-2 * a], rel=1e-5)


def test_expand_turn_advantages_maps_token_spans_to_targets() -> None:
    assert expand_turn_advantages([2.0, -1.0], [(2, 4), (4, 6)], 5) == [
        0.0,
        2.0,
        2.0,
        -1.0,
        -1.0,
    ]
    with pytest.raises(ValueError, match="one advantage per turn"):
        expand_turn_advantages([1.0], [(1, 2), (2, 3)], 3)


def _group() -> PromptGroup:
    """Run A has two turns, run B one; prompt [1, 2], answer [9]."""
    return PromptGroup(
        data=[
            _datum([1, 2, 3, 4, 5, 6], [0, 1, 1, 1, 1]),
            _datum([1, 2, 7, 8, 5, 6], [0, 1, 1, 1, 1]),
        ],
        advantages=[0.0, 0.0],
        prompt_len=2,
        rewards=[1.0, 0.0],
        inf_logprobs=[[0.0, -0.1, -0.2, -0.3, -0.4], [0.0, -0.5, -0.5, -0.5, -0.5]],
        run_metadata=[
            {ANSWER_TOKENS: [9], TURN_SPANS: [(2, 4), (4, 6)]},
            {ANSWER_TOKENS: [9], TURN_SPANS: [(2, 6)]},
        ],
    )


def test_igpo_token_advantages_score_every_turn_context_once() -> None:
    scorer = _Scorer({2: -2.0, 4: -0.5})
    batch = combine_prompt_groups([_group()])

    advantaged, metrics = token_advantages(scorer, batch, gamma=0.5)

    assert len(scorer.calls) == 1
    assert [d.model_input.to_ints() for d in scorer.calls[0]] == [
        [1, 2],
        [1, 2, 3, 4],
        [1, 2],
    ]
    a = math.sqrt(2) / 2
    rows = advantaged.token_advantages
    assert rows[0] == pytest.approx([0.0, 1.5 * a, 1.5 * a, a, a], rel=1e-5)
    assert rows[1] == pytest.approx([0.0] + [-2 * a] * 4, rel=1e-5)
    assert metrics == pytest.approx({"igpo/avg_turns": 1.5, "igpo/mean_ig": 1.5, "igpo/baseline_logp": -2.0})
    # The input batch is not mutated; only token_advantages differ.
    assert batch.token_advantages is None
    assert advantaged.data is batch.data and advantaged.advantages is batch.advantages


def test_prepare_batch_is_an_rl_loop_hook() -> None:
    batch = combine_prompt_groups([_group()])
    prepared, metrics = prepare_batch(gamma=0.5)(_Scorer({2: -2.0, 4: -0.5}), batch)
    assert len(prepared.token_advantages) == 2
    assert metrics["igpo/avg_turns"] == 1.5


def test_igpo_token_advantages_require_one_datum_per_run() -> None:
    group = _group()
    group.run_metadata = group.run_metadata[:1]
    with pytest.raises(ValueError, match="one datum and reward per rollout run"):
        token_advantages(_Scorer({}), combine_prompt_groups([group]), gamma=0.5)


@pytest.mark.parametrize(("key", "message"), [(ANSWER_TOKENS, "answer tokens"), (TURN_SPANS, "without turns")])
def test_igpo_token_advantages_require_run_metadata(key, message) -> None:
    group = _group()
    group.run_metadata = [{k: v for k, v in meta.items() if k != key} for meta in group.run_metadata]
    with pytest.raises(ValueError, match=message):
        token_advantages(_Scorer({}), combine_prompt_groups([group]), gamma=0.5)


def test_igpo_datums_carry_masked_token_advantages_and_rollout_anchor() -> None:
    group = PromptGroup(
        data=[_datum([10, 11, 12, 13], [1, 0, 1])],
        advantages=[0.0],
        prompt_len=1,
        rewards=[1.0],
        # The tool token at target 1 is masked; its rollout logprob may be null.
        inf_logprobs=[[-1.0, None, -0.9]],
    )
    batch = combine_prompt_groups([group])

    datums = build_policy_datums(replace(batch, token_advantages=[[2.0, -10.0, -1.0]]))

    reference = build_policy_datums(batch)[0].loss_fn_inputs
    inputs = datums[0].loss_fn_inputs
    assert inputs["advantages"].data == [2.0, 0.0, -1.0]
    assert inputs["logprobs"].data == reference["logprobs"].data
    assert inputs["logprobs"].data == pytest.approx([-1.0, 0.0, -0.9])
    assert inputs["response_mask"].data == reference["response_mask"].data
    assert inputs["target_tokens"].data == [11, 12, 13]


def test_igpo_datums_require_one_advantage_per_target() -> None:
    batch = combine_prompt_groups([_group()])
    with pytest.raises(ValueError, match="4 values for 5 target tokens"):
        build_policy_datums(replace(batch, token_advantages=[[0.0] * 4, [0.0] * 5]))
    with pytest.raises(ValueError, match="token_advantages=1"):
        build_policy_datums(replace(batch, token_advantages=[[0.0] * 5]))


@pytest.mark.parametrize("logprobs", [[-1.0, -0.9], [None, -0.9, -0.8]])
def test_igpo_datums_reject_missing_active_rollout_logprobs(logprobs) -> None:
    group = PromptGroup(
        data=[_datum([10, 11, 12, 13], [1, 1, 1])],
        advantages=[0.0],
        prompt_len=1,
        rewards=[1.0],
        inf_logprobs=[logprobs],
    )
    batch = combine_prompt_groups([group])
    with pytest.raises(ValueError, match="rollout_logprobs|anchor_logprobs"):
        build_policy_datums(replace(batch, token_advantages=[[1.0, 1.0, 1.0]]))


def test_batch_preparation_preserves_group_local_normalization_and_weighted_metrics() -> None:
    first = _group()
    second = replace(
        first,
        data=[_datum([1, 2, 3, 4], [0, 0, 1])],
        advantages=[0.0],
        prompt_len=3,
        rewards=[0.25],
        inf_logprobs=[[0.0, 0.0, -0.3]],
        run_metadata=[{ANSWER_TOKENS: [9], TURN_SPANS: [(3, 4)]}],
    )
    groups = [first, second]
    scorer = _Scorer({2: -2.0, 3: -5.5, 4: -0.5})
    prepared, metrics = prepare_batch(gamma=0.5)(scorer, combine_prompt_groups(groups))
    assert len(scorer.calls) == 1
    expected = [token_advantages(scorer, combine_prompt_groups([group]), gamma=0.5)[0] for group in groups]
    for actual, expected_row in zip(
        prepared.token_advantages, [r for b in expected for r in b.token_advantages], strict=True
    ):
        assert actual == pytest.approx(expected_row)
    assert metrics == pytest.approx({"igpo/avg_turns": 4 / 3, "igpo/mean_ig": 1.5, "igpo/baseline_logp": -9.5 / 3})


def test_batch_preparation_rejects_misaligned_advantage_rows():
    group = _group()
    batch = combine_prompt_groups([group])
    # A malformed preparation input has fewer actual datums than group runs.
    batch.data = batch.data[:-1]
    scorer = _Scorer({2: -2.0, 3: -5.5, 4: -0.5})
    with pytest.raises(ValueError, match="advantage rows"):
        prepare_batch(gamma=0.5)(scorer, batch)
