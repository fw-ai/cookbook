"""The fixed clipping anchor and the sampling distribution have distinct roles."""

import math
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import tinker
import torch

from training.utils.rl.algorithm import grpo, reinforce
from training.utils.rl.anchor import anchor_drift_metrics, recompute_anchor
from training.utils.rl.losses import PolicyBatch, build_policy_datums


def datum():
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(data=[2, 3], dtype="int64", shape=[2]),
            "weights": tinker.TensorData(data=[0, 1], dtype="float32", shape=[2]),
        },
    )


def batch(rollout, anchor=None, advantage=1.0):
    return PolicyBatch(
        data=[datum()],
        advantages=[advantage],
        prompt_lens=[2],
        rollout_logprobs=[rollout],
        anchor_logprobs=None if anchor is None else [anchor],
    )


@pytest.mark.parametrize(
    "anchor,expected_loss,expected_grad", [(0.2, -1.1, -1.1), (0.1, -1.2, 0)]
)
def test_anchor_controls_clipping_without_preweighting_advantages(
    anchor, expected_loss, expected_grad
):
    pi = torch.tensor([0.0, math.log(0.22)], requires_grad=True)
    b = batch([None, math.log(0.1)], [None, math.log(anchor)])
    loss, _ = grpo.make_loss_fn(b, grpo.GRPOConfig())(b.data, [pi])
    loss.backward()
    assert loss.item() == pytest.approx(expected_loss)
    assert pi.grad.tolist() == pytest.approx([0, expected_grad])


def test_reinforce_ratio_uses_rollout_not_anchor():
    pi = torch.tensor([0.0, math.log(0.2)], requires_grad=True)
    b = batch([None, math.log(0.1)], [None, math.log(0.2)])
    loss, metrics = reinforce.make_loss_fn(b, reinforce.ReinforceConfig())(b.data, [pi])
    assert loss.item() == pytest.approx(-2.0)
    assert metrics["reinforce_ratio_mean"] == pytest.approx(2.0)
    # The anchor is still what the k3 diagnostic compares against.
    assert metrics["reinforce_kl"] == pytest.approx(0.0)


def test_builtin_anchor_changes_denominator_but_preserves_mask_and_raw_advantage():
    rows = build_policy_datums(batch([None, math.log(0.1)], [None, math.log(0.2)], advantage=2.0))
    inputs = rows[0].loss_fn_inputs
    assert inputs["logprobs"].data == pytest.approx([0, math.log(0.2)])
    assert inputs["advantages"].data == [0, 2]
    assert inputs["response_mask"].data == [0, 1]


@pytest.mark.parametrize("future", [False, True])
def test_snapshot_is_reusable_for_both_sdk_client_interfaces(future):
    calls = []
    result = SimpleNamespace(
        loss_fn_outputs=[{"logprobs": SimpleNamespace(data=[-1.0, -2.0])}]
    )

    def forward(*args, **kwargs):
        calls.append((args, kwargs))
        if future:
            pending = Future()
            pending.set_result(result)
            return pending
        return result

    policy = SimpleNamespace(forward=forward)
    original = batch([None, -3.0])
    anchored, reusable = recompute_anchor(policy, original)
    assert anchored.anchor_logprobs == [[-1.0, -2.0]]
    assert anchored.rollout_logprobs is original.rollout_logprobs
    assert original.anchor_logprobs is original.rollout_logprobs
    assert reusable is result
    assert len(calls) == 1
    assert calls[0][0][1] == "cross_entropy"


def test_policy_datums_use_the_recomputed_anchor():
    policy = SimpleNamespace(
        forward=lambda *args: SimpleNamespace(
            loss_fn_outputs=[{"logprobs": SimpleNamespace(data=[-5.0, math.log(0.3)])}]
        )
    )
    anchored, _ = recompute_anchor(policy, batch([0.0, math.log(0.1)]))
    (row,) = build_policy_datums(anchored)
    assert row.loss_fn_inputs["logprobs"].data == pytest.approx([0.0, math.log(0.3)])
    assert row.loss_fn_inputs["advantages"].data == [0.0, 1.0]


@pytest.mark.parametrize("outputs", [[], [{"logprobs": SimpleNamespace(data=[-1.0])}]])
def test_misaligned_snapshot_fails_before_training(outputs):
    policy = SimpleNamespace(forward=lambda *args: SimpleNamespace(loss_fn_outputs=outputs))
    with pytest.raises(ValueError, match="align with target"):
        recompute_anchor(policy, batch([0, -1]))


def test_anchor_drift_metrics_compare_the_anchor_with_rollout_logprobs():
    metrics = anchor_drift_metrics(batch([0.0, math.log(0.1)], [0.0, math.log(0.2)]))
    diff = math.log(2.0)
    assert metrics == pytest.approx(
        {"anchor_k1": diff, "anchor_k3": math.exp(diff) - diff - 1.0}
    )
    assert anchor_drift_metrics(batch([0.0, math.log(0.1)])) == {"anchor_k1": 0.0, "anchor_k3": 0.0}
