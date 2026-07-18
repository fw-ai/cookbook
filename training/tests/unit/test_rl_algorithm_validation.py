"""Construction-time validation for direct client-side RL algorithms."""

from __future__ import annotations

import pytest

from training.utils.rl.cispo import CISPOConfig, make_cispo_loss_fn
from training.utils.rl.dapo import DAPOConfig, make_dapo_loss_fn
from training.utils.rl.dro import DROConfig, make_dro_loss_fn
from training.utils.rl.grpo import make_grpo_loss_fn, validate_grpo_config
from training.utils.rl.gspo import GSPOConfig, make_gspo_loss_fn
from training.utils.rl.igpo import make_igpo_loss_fn
from training.utils.rl.is_loss import make_is_loss_fn
from training.utils.rl.reinforce import make_reinforce_loss_fn


def _group_loss_inputs() -> dict:
    return {
        "advantages": [],
        "ref_logprobs": [],
        "inf_logprobs": [],
        "prompt_len": [],
        "old_policy_logprobs": [],
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kl_beta": -0.1}, "kl_beta"),
        ({"eps_clip": -0.1}, "eps_clip"),
        ({"eps_clip_high": -0.1}, "eps_clip"),
    ],
)
def test_grpo_builder_validates_config(kwargs, message) -> None:
    inputs = {
        **_group_loss_inputs(),
        "kl_beta": 0.0,
        "eps_clip": 0.2,
        "eps_clip_high": None,
        **kwargs,
    }

    with pytest.raises(ValueError, match=message):
        make_grpo_loss_fn(**inputs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"reference_job_id": "ref"}, "require kl_beta > 0"),
        ({"anchor_logp": "latest"}, "anchor_logp"),
        ({"ppo_n_minibatches": 0}, "ppo_n_minibatches"),
    ],
)
def test_grpo_recipe_validation(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_grpo_config(
            kl_beta=0.0,
            eps_clip=0.2,
            eps_clip_high=None,
            **kwargs,
        )


@pytest.mark.parametrize(
    "config",
    [
        DAPOConfig(eps_clip=-0.1),
        DAPOConfig(eps_clip_high=-0.1),
        DAPOConfig(eps_clip_c=1.0),
        DAPOConfig(ratio_log_cap=-0.1),
    ],
)
def test_dapo_builder_validates_config(config) -> None:
    with pytest.raises(ValueError, match="DAPO"):
        make_dapo_loss_fn(**_group_loss_inputs(), dapo_config=config)


def test_dro_builder_validates_config() -> None:
    with pytest.raises(ValueError, match="DRO beta"):
        make_dro_loss_fn(
            **_group_loss_inputs(),
            dro_config=DROConfig(beta=-0.1),
        )


@pytest.mark.parametrize(
    "config",
    [
        GSPOConfig(clip_ratio_low=-0.1),
        GSPOConfig(clip_ratio_high=-0.1),
        GSPOConfig(seq_ratio_log_cap=-0.1),
    ],
)
def test_gspo_builder_validates_config(config) -> None:
    with pytest.raises(ValueError, match="GSPO"):
        make_gspo_loss_fn(**_group_loss_inputs(), gspo_config=config)


@pytest.mark.parametrize(
    "config",
    [
        CISPOConfig(eps_low=-0.1),
        CISPOConfig(eps_high=-0.1),
        CISPOConfig(ratio_log_cap=-0.1),
    ],
)
def test_cispo_builder_validates_config(config) -> None:
    with pytest.raises(ValueError, match="CISPO"):
        make_cispo_loss_fn(**_group_loss_inputs(), cispo_config=config)


def test_is_builder_validates_config() -> None:
    with pytest.raises(ValueError, match="IS ratio_log_cap"):
        make_is_loss_fn(**_group_loss_inputs(), ratio_log_cap=-0.1)


def test_reinforce_builder_validates_config() -> None:
    inputs = _group_loss_inputs()
    inputs["prompt_lens"] = inputs.pop("prompt_len")

    with pytest.raises(ValueError, match="REINFORCE kl_beta"):
        make_reinforce_loss_fn(**inputs, kl_beta=-0.1)


@pytest.mark.parametrize(
    "kwargs",
    [{"kl_beta": -0.1}, {"eps_clip": -0.1}],
)
def test_igpo_builder_validates_config(kwargs) -> None:
    inputs = {
        "per_token_advantages": [],
        "ref_logprobs": [],
        "prompt_lens": [],
        "inf_logprobs": [],
        "kl_beta": 0.0,
        "eps_clip": 0.2,
        **kwargs,
    }

    with pytest.raises(ValueError, match="IGPO"):
        make_igpo_loss_fn(**inputs)
