"""Construction-time validation of objective configs."""

from __future__ import annotations

import pytest

from training.utils.rl.algorithm.cispo import CISPOConfig
from training.utils.rl.algorithm.dapo import DAPOConfig
from training.utils.rl.algorithm.dppo import DPPOConfig
from training.utils.rl.algorithm.dro import DROConfig
from training.utils.rl.algorithm.grpo import GRPOConfig
from training.utils.rl.algorithm.gspo import GSPOConfig
from training.utils.rl.algorithm.importance_sampling import ISConfig
from training.utils.rl.algorithm.reinforce import ReinforceConfig
from training.utils.rl.algorithm.score_centering import ScoreCenteringConfig


@pytest.mark.parametrize(
    ("config_cls", "kwargs", "message"),
    [
        (GRPOConfig, {"kl_beta": -0.1}, "kl_beta"),
        (GRPOConfig, {"eps_clip": -0.1}, "eps_clip"),
        (GRPOConfig, {"eps_clip_high": -0.1}, "eps_clip"),
        (DAPOConfig, {"eps_clip": -0.1}, "DAPO"),
        (DAPOConfig, {"eps_clip_high": -0.1}, "DAPO"),
        (DAPOConfig, {"eps_clip_c": 1.0}, "DAPO dual-clip"),
        (DAPOConfig, {"ratio_log_cap": -0.1}, "DAPO"),
        (DROConfig, {"beta": -0.1}, "DRO beta"),
        (DPPOConfig, {"divergence": "invalid"}, "DPPO divergence"),
        (DPPOConfig, {"threshold": -0.1}, "DPPO threshold"),
        (DPPOConfig, {"ratio_log_cap": -0.1}, "DPPO"),
        (ScoreCenteringConfig, {"top_k": 0}, "score centering"),
        (ScoreCenteringConfig, {"top_k": 9}, "score centering"),
        (ScoreCenteringConfig, {"tail_mass_epsilon": 0}, "score centering"),
        (GSPOConfig, {"clip_ratio_low": -0.1}, "GSPO"),
        (GSPOConfig, {"clip_ratio_high": -0.1}, "GSPO"),
        (GSPOConfig, {"seq_ratio_log_cap": -0.1}, "GSPO"),
        (GSPOConfig, {"token_reduction": "invalid"}, "GSPO"),
        (CISPOConfig, {"eps_low": -0.1}, "CISPO"),
        (CISPOConfig, {"eps_high": -0.1}, "CISPO"),
        (CISPOConfig, {"ratio_log_cap": -0.1}, "CISPO"),
        (ISConfig, {"ratio_log_cap": -0.1}, "IS ratio_log_cap"),
        (ReinforceConfig, {"kl_beta": -0.1}, "REINFORCE kl_beta"),
    ],
)
def test_config_validates_on_construction(config_cls, kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        config_cls(**kwargs)


def test_dppo_threshold_defaults_per_divergence() -> None:
    assert DPPOConfig(divergence="binary_tv").threshold == 0.15
    assert DPPOConfig(divergence="binary_kl").threshold == 0.05
    assert DPPOConfig(threshold=0.3).threshold == 0.3


def test_grpo_upper_epsilon_defaults_to_lower() -> None:
    assert GRPOConfig(eps_clip=0.1).upper_eps == 0.1
    assert GRPOConfig(eps_clip=0.1, eps_clip_high=0.3).upper_eps == 0.3
