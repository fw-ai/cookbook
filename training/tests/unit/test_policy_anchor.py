"""The fixed clipping anchor and the sampling distribution have distinct roles."""

import math
from types import SimpleNamespace
from concurrent.futures import Future

import pytest
import tinker
import torch

from training.utils.rl.anchor import prepare_policy_anchor
from training.utils.rl.algorithm.grpo import make_grpo_loss_fn
from training.utils.rl.losses import build_grpo_datums


def datum():
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(data=[2, 3], dtype="int64", shape=[2]),
            "weights": tinker.TensorData(data=[0, 1], dtype="float32", shape=[2]),
        },
    )


@pytest.mark.parametrize(
    "anchor,expected_loss,expected_grad", [(0.2, -1.1, -1.1), (0.1, -1.2, 0)]
)
def test_anchor_controls_clipping_without_preweighting_advantages(
    anchor, expected_loss, expected_grad
):
    pi = torch.tensor([0.0, math.log(0.22)], requires_grad=True)
    loss_fn = make_grpo_loss_fn(
        advantages=[1.0],
        ref_logprobs=[],
        prompt_len=[2],
        inf_logprobs=[[None, math.log(0.1)]],
        kl_beta=0,
        old_policy_logprobs=[[None, math.log(anchor)]],
    )
    loss, _ = loss_fn([datum()], [pi])
    loss.backward()
    assert loss.item() == pytest.approx(expected_loss)
    assert pi.grad.tolist() == pytest.approx([0, expected_grad])


def test_builtin_anchor_changes_denominator_but_preserves_mask_and_raw_advantage():
    rows = build_grpo_datums(
        [datum()],
        [2.0],
        [[None, math.log(0.1)]],
        [2],
        old_policy_logprobs=[[None, math.log(0.2)]],
        include_response_mask=True,
    )
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
    rollout = [[None, -3.0]]
    anchor, reusable = prepare_policy_anchor(policy, [datum()], rollout, "old_policy")
    assert anchor == [[-1.0, -2.0]]
    assert reusable is result
    assert len(calls) == 1
    anchor, reusable = prepare_policy_anchor(policy, [datum()], rollout, "rollout")
    assert anchor is rollout
    assert reusable is None
    assert len(calls) == 1


def test_misaligned_snapshot_fails_before_training():
    policy = SimpleNamespace(forward=lambda *args: SimpleNamespace(loss_fn_outputs=[]))
    with pytest.raises(ValueError, match="align with target"):
        prepare_policy_anchor(policy, [datum()], [[0, -1]], "old_policy")


def test_igpo_accepts_explicit_snapshot_anchor():
    from training.utils.rl.algorithm.igpo import make_igpo_loss_fn

    pi = torch.tensor([0.0, math.log(0.22)], requires_grad=True)
    fn = make_igpo_loss_fn(
        per_token_advantages=[[0., 1.]], ref_logprobs=[], prompt_lens=[2],
        inf_logprobs=[[None, math.log(0.1)]], kl_beta=0,
        old_policy_logprobs=[[None, math.log(0.2)]],
    )
    loss, _ = fn([datum()], [pi])
    loss.backward()
    assert pi.grad.tolist() == pytest.approx([0., -1.1])
