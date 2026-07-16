"""Multimodal wire-contract coverage for every cookbook built-in RL loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import tinker
import torch
from tinker._compat import model_dump
from tinker.lib._pydantic_conv import to_pydantic_request
from tinker.types import ImageChunk

from training.utils.rl.builtin_losses import BUILTIN_LOSSES
from training.utils.rl.cispo import CISPOConfig
from training.utils.rl.dapo import DAPOConfig
from training.utils.rl.dro import DROConfig
from training.utils.rl.gspo import GSPOConfig
from training.utils.rl.losses import (
    LossConfig,
    build_builtin_loss_datums,
    build_loss_fn,
    get_builtin_loss_config,
    validate_loss_path,
)
from training.utils.rl.rollout.renderer import _build_multimodal_rollout_sample
from training.utils.rl.rollout.types import Rollout, rollout_to_prompt_group


@dataclass(frozen=True)
class _BuiltinCase:
    name: str
    config: LossConfig
    trainer_loss_fn: str
    trainer_loss_config: dict[str, Any]


_BUILTIN_CASES = (
    _BuiltinCase(
        name="grpo",
        config=LossConfig(
            policy_loss="grpo",
            loss_path="builtin",
            kl_beta=0.0,
            eps_clip=0.15,
            eps_clip_high=0.25,
        ),
        trainer_loss_fn="ppo",
        trainer_loss_config={
            "clip_low_threshold": 0.85,
            "clip_high_threshold": 1.25,
        },
    ),
    _BuiltinCase(
        name="importance_sampling",
        config=LossConfig(
            policy_loss="importance_sampling",
            loss_path="builtin",
            kl_beta=0.0,
            ratio_log_cap=7.5,
        ),
        trainer_loss_fn="importance_sampling",
        trainer_loss_config={"ratio_log_cap": 7.5},
    ),
    _BuiltinCase(
        name="dapo_without_dual_clip",
        config=LossConfig(
            policy_loss="dapo",
            loss_path="builtin",
            kl_beta=0.0,
            dapo=DAPOConfig(
                eps_clip=0.11,
                eps_clip_high=0.22,
                eps_clip_c=None,
                ratio_log_cap=8.0,
            ),
        ),
        trainer_loss_fn="ppo",
        trainer_loss_config={
            "clip_low_threshold": 0.89,
            "clip_high_threshold": 1.22,
            "ratio_log_cap": 8.0,
        },
    ),
    _BuiltinCase(
        name="dapo_with_dual_clip",
        config=LossConfig(
            policy_loss="dapo",
            loss_path="builtin",
            kl_beta=0.0,
            dapo=DAPOConfig(
                eps_clip=0.11,
                eps_clip_high=0.22,
                eps_clip_c=2.5,
                ratio_log_cap=8.0,
            ),
        ),
        trainer_loss_fn="dapo",
        trainer_loss_config={
            "clip_low_threshold": 0.89,
            "clip_high_threshold": 1.22,
            "ratio_log_cap": 8.0,
            "eps_clip_c": 2.5,
        },
    ),
    _BuiltinCase(
        name="dro",
        config=LossConfig(
            policy_loss="dro",
            loss_path="builtin",
            kl_beta=0.0,
            dro=DROConfig(beta=0.07),
        ),
        trainer_loss_fn="dro",
        trainer_loss_config={"beta": 0.07},
    ),
    _BuiltinCase(
        name="gspo",
        config=LossConfig(
            policy_loss="gspo",
            loss_path="builtin",
            kl_beta=0.0,
            gspo=GSPOConfig(
                clip_ratio_low=0.1,
                clip_ratio_high=0.3,
                seq_ratio_log_cap=6.0,
            ),
        ),
        trainer_loss_fn="gspo",
        trainer_loss_config={
            "clip_low_threshold": 0.9,
            "clip_high_threshold": 1.3,
            "seq_ratio_log_cap": 6.0,
        },
    ),
    _BuiltinCase(
        name="cispo",
        config=LossConfig(
            policy_loss="cispo",
            loss_path="builtin",
            kl_beta=0.0,
            cispo=CISPOConfig(
                eps_low=0.12,
                eps_high=0.34,
                ratio_log_cap=9.0,
            ),
        ),
        trainer_loss_fn="cispo",
        trainer_loss_config={
            "clip_low_threshold": 0.88,
            "clip_high_threshold": 1.34,
            "ratio_log_cap": 9.0,
        },
    ),
)


def _multimodal_prompt_group():
    prompt = tinker.ModelInput(
        chunks=[
            tinker.EncodedTextChunk(tokens=[10, 11]),
            ImageChunk(
                data=b"test-image-bytes",
                format="png",
                expected_tokens=3,
            ),
            tinker.EncodedTextChunk(tokens=[12]),
        ]
    )
    run = _build_multimodal_rollout_sample(
        prompt_model_input=prompt,
        completion_tokens=[30, 31],
        completion_logprobs=[-0.1, -0.2],
        raw_completion_logprobs=[-1.1, -1.2],
        reward=1.0,
        finish_reason="stop",
        text="answer",
    )
    prompt_group = rollout_to_prompt_group(
        Rollout(runs=[run]),
        advantage_fn=lambda rewards: list(rewards),
    )
    assert prompt_group is not None
    assert prompt_group.prompt_lens is not None
    return prompt_group


def test_multimodal_builtin_matrix_covers_every_registered_trainer_loss() -> None:
    """Registry drift cannot silently leave a built-in objective untested."""

    assert {case.config.policy_loss for case in _BUILTIN_CASES} == set(BUILTIN_LOSSES)
    assert {case.trainer_loss_fn for case in _BUILTIN_CASES} == {
        "importance_sampling",
        "ppo",
        "cispo",
        "dapo",
        "gspo",
        "dro",
    }


@pytest.mark.parametrize("case", _BUILTIN_CASES, ids=lambda case: case.name)
def test_multimodal_builtin_loss_serializes_expanded_coordinates(
    case: _BuiltinCase,
) -> None:
    """Every built-in path sends the same canonical expanded wire datum."""

    validate_loss_path(case.config)
    trainer_loss_fn, trainer_loss_config = get_builtin_loss_config(case.config)
    assert trainer_loss_fn == case.trainer_loss_fn
    assert trainer_loss_config == pytest.approx(case.trainer_loss_config)

    prompt_group = _multimodal_prompt_group()
    datum = prompt_group.data[0]
    expected_length = datum.model_input.length
    assert expected_length == 7
    assert datum.loss_fn_inputs["target_tokens"].data == [11, 0, 0, 0, 12, 30, 31]
    assert datum.loss_fn_inputs["weights"].data == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    assert prompt_group.inf_logprobs[0] == [0.0, 0.0, 0.0, 0.0, 0.0, -0.1, -0.2]

    builtin_datum = build_builtin_loss_datums(
        data=prompt_group.data,
        advantages=prompt_group.advantages,
        old_policy_logprobs=prompt_group.inf_logprobs,
        inf_logprobs=prompt_group.inf_logprobs,
        prompt_lens=prompt_group.prompt_lens,
        policy_loss=case.config.policy_loss,
    )[0]

    wire_request = tinker.types.ForwardBackwardRequest(
        forward_backward_input=tinker.types.ForwardBackwardInput(
            data=[builtin_datum],
            loss_fn=trainer_loss_fn,
            loss_fn_config=trainer_loss_config,
        ),
        model_id="test-model",
        seq_id=1,
    )
    wire_payload = model_dump(
        to_pydantic_request(wire_request),
        exclude_unset=False,
        exclude_none=True,
        mode="json",
    )
    forward_backward_input = wire_payload["forward_backward_input"]
    assert forward_backward_input["loss_fn"] == case.trainer_loss_fn
    assert forward_backward_input["loss_fn_config"] == pytest.approx(
        case.trainer_loss_config
    )

    serialized = forward_backward_input["data"][0]["loss_fn_inputs"]
    assert set(serialized) == {"target_tokens", "logprobs", "advantages"}
    for field in serialized.values():
        assert field["shape"] == [expected_length]
        assert len(field["data"]) == expected_length

    # Image labels are zero wire placeholders. Prompt/image positions carry
    # neither behavior-policy logprobs nor advantages; only completion labels
    # at positions 5 and 6 contribute to the objective.
    assert serialized["target_tokens"] == {
        "data": [11, 0, 0, 0, 12, 30, 31],
        "dtype": "int64",
        "shape": [expected_length],
    }
    assert serialized["logprobs"] == {
        "data": pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, -0.1, -0.2]),
        "dtype": "float32",
        "shape": [expected_length],
    }
    assert serialized["advantages"] == {
        "data": pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]),
        "dtype": "float32",
        "shape": [expected_length],
    }


def test_multimodal_client_grpo_gradients_share_builtin_coordinates() -> None:
    """Vanilla client-side GRPO keeps the same expanded target index space."""

    prompt_group = _multimodal_prompt_group()
    datum = prompt_group.data[0]
    target_count = datum.model_input.length
    loss_fn = build_loss_fn(
        LossConfig(policy_loss="grpo", loss_path="client", kl_beta=0.0)
    )(
        prompt_group.advantages,
        [[0.0] * target_count],
        prompt_group.prompt_lens,
        prompt_group.inf_logprobs,
        prompt_group.inf_logprobs,
    )
    forward_logprobs = torch.tensor(
        prompt_group.inf_logprobs[0],
        dtype=torch.float32,
        requires_grad=True,
    )

    loss, metrics = loss_fn([datum], [forward_logprobs])
    loss.backward()

    assert metrics["active_tokens"] == 2
    assert forward_logprobs.shape == (target_count,)
    assert forward_logprobs.grad is not None
    assert forward_logprobs.grad[:5].tolist() == pytest.approx([0.0] * 5)
    assert all(abs(value) > 0.0 for value in forward_logprobs.grad[5:].tolist())
