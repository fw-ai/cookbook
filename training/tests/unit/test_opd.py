import pytest
import tinker
import torch

from training.utils.distillation import build_rl_topk_forward_kl_datums
from training.utils.distillation.sampling import TopKDist
from training.utils.rl.grpo import make_grpo_loss_fn
from training.utils.rl.gspo import make_gspo_loss_fn
from training.utils.rl.rollout import Rollout, RolloutRun, RolloutSample
from training.utils.rl.rollout.types import rollout_to_prompt_group


def _datum():
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(data=[2, 3], dtype="int64", shape=[2]),
            "weights": tinker.TensorData(data=[1, 1], dtype="int64", shape=[2]),
        },
    )


def test_rollout_packs_teacher_logprobs():
    sample = RolloutSample(
        tokens=[1, 2, 3],
        logprobs=[0.0, -2.0, -2.0],
        teacher_logprobs=[0.0, -1.0, -1.0],
        loss_mask=[0, 1, 1],
        reward=1.0,
    )
    group = rollout_to_prompt_group(
        Rollout(runs=[RolloutRun(segments=[sample])]),
        advantage_fn=lambda rewards: rewards,
    )
    assert group is not None
    assert group.teacher_logprobs == [[-1.0, -1.0]]


def test_rollout_packs_teacher_topk_and_preserves_mask():
    dist = TopKDist(token_ids=[7, 8], logprobs=[-0.2, -1.8])
    sample = RolloutSample(
        tokens=[1, 2, 3],
        logprobs=[0.0, -2.0, -2.0],
        teacher_topk=[None, dist, dist],
        loss_mask=[0, 1, 1],
        reward=1.0,
    )
    group = rollout_to_prompt_group(
        Rollout(runs=[RolloutRun(segments=[sample])]),
        advantage_fn=lambda rewards: rewards,
    )
    assert group is not None
    assert group.teacher_topk == [[dist, dist]]
    datums, metrics = build_rl_topk_forward_kl_datums(
        group.data, group.teacher_topk, top_k=2
    )
    assert datums[0].loss_fn_inputs["target_tokens"].shape == [2, 3]
    assert datums[0].loss_fn_inputs["weights"].shape == [2, 3]
    assert metrics["sdft_active_positions"] == 2


@pytest.mark.parametrize("loss_builder", [make_grpo_loss_fn, make_gspo_loss_fn])
def test_opd_gradient_raises_mass_when_teacher_is_higher(loss_builder):
    student = torch.tensor([-2.0, -2.0], requires_grad=True)
    kwargs = {
        "advantages": [0.0],
        "ref_logprobs": [],
        "inf_logprobs": [[-2.0, -2.0]],
        "prompt_len": [1],
        "old_policy_logprobs": [[-2.0, -2.0]],
        "teacher_logprobs": [[-1.0, -1.0]],
        "opd_beta": 0.05,
    }
    if loss_builder is make_grpo_loss_fn:
        kwargs["kl_beta"] = 0.0
    loss, metrics = loss_builder(**kwargs)([_datum()], [student])
    loss.backward()
    assert student.grad is not None
    assert torch.all(student.grad < 0)
    assert metrics["opd_kl_mean"] == pytest.approx(-1.0)


@pytest.mark.parametrize("loss_builder", [make_grpo_loss_fn, make_gspo_loss_fn])
def test_topk_opd_gradient_raises_teacher_candidate_mass(loss_builder):
    dist = TopKDist(
        token_ids=[7, 8],
        logprobs=[torch.log(torch.tensor(0.8)).item(), torch.log(torch.tensor(0.2)).item()],
    )
    datums, _ = build_rl_topk_forward_kl_datums(
        [_datum()], [[dist, dist]], top_k=2
    )
    student = torch.tensor(
        [[-2.0, -0.3, -2.0], [-2.0, -0.3, -2.0]],
        requires_grad=True,
    )
    kwargs = {
        "advantages": [0.0],
        "ref_logprobs": [],
        "inf_logprobs": [[-2.0, -2.0]],
        "prompt_len": [1],
        "old_policy_logprobs": [[-2.0, -2.0]],
        "opd_beta": 0.05,
        "opd_top_k": 2,
    }
    if loss_builder is make_grpo_loss_fn:
        kwargs["kl_beta"] = 0.0
    loss, metrics = loss_builder(**kwargs)(datums, [student])
    loss.backward()
    assert student.grad is not None
    assert torch.allclose(student.grad[:, 0], torch.zeros(2))
    assert torch.all(student.grad[:, 1:] < 0)
    assert metrics["opd_forward_kl_mean"] > 0


def test_opd_requires_aligned_teacher_logprobs():
    loss_fn = make_grpo_loss_fn(
        advantages=[0.0],
        ref_logprobs=[],
        inf_logprobs=[[-2.0, -2.0]],
        prompt_len=[1],
        old_policy_logprobs=[[-2.0, -2.0]],
        teacher_logprobs=[[-1.0]],
        opd_beta=0.05,
        kl_beta=0.0,
    )
    with pytest.raises(ValueError, match="must align with target tokens"):
        loss_fn([_datum()], [torch.tensor([-2.0, -2.0], requires_grad=True)])
