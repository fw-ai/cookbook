"""Contracts for the self-contained objective modules in ``training.utils.rl.algorithm``."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import get_args

import pytest
import tinker
import torch

from training.utils.client import GradAccNormalization
from training.utils.rl.algorithm import (
    cispo,
    dapo,
    dppo,
    dro,
    grpo,
    gspo,
    importance_sampling,
    reinforce,
    score_centering,
)
from training.utils.rl.losses import PolicyBatch
from training.utils.supervised import build_datum_from_token_mask

SERVER_SIDE = [
    (grpo, grpo.GRPOConfig(), "ppo"),
    (gspo, gspo.GSPOConfig(), "gspo"),
    (dapo, dapo.DAPOConfig(), "dapo"),
    (cispo, cispo.CISPOConfig(), "cispo"),
    (dro, dro.DROConfig(), "dro"),
    (dppo, dppo.DPPOConfig(), "dppo"),
    (importance_sampling, importance_sampling.ISConfig(), "is"),
]
CLIENT_ONLY = [
    (reinforce, reinforce.ReinforceConfig()),
    (score_centering, score_centering.ScoreCenteringConfig()),
]
CONFIG_CLASSES = {
    grpo: "GRPOConfig",
    gspo: "GSPOConfig",
    dapo: "DAPOConfig",
    cispo: "CISPOConfig",
    dro: "DROConfig",
    dppo: "DPPOConfig",
    importance_sampling: "ISConfig",
    reinforce: "ReinforceConfig",
    score_centering: "ScoreCenteringConfig",
}
# Client-only diagnostics beyond the shared ``{prefix}_*`` names.
CLIENT_EXTRAS = {"dro_quad_penalty", "dppo_divergence_mean"}


def _batch() -> PolicyBatch:
    datum = build_datum_from_token_mask(token_ids=[10, 11, 12, 13], token_mask=[0, 1, 1, 1]).datum
    return PolicyBatch(
        data=[datum],
        advantages=[1.0],
        prompt_lens=[2],
        rollout_logprobs=[[-0.4, -0.3, -0.1]],
    )


def _result(*rows):
    return SimpleNamespace(
        loss_fn_outputs=[
            {"logprobs": tinker.TensorData(data=row, dtype="float32", shape=[len(row)])} for row in rows
        ]
    )


def _datum(weights: list[float]) -> tinker.Datum:
    n = len(weights)
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(list(range(1, n + 1))),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(data=list(range(2, n + 2)), dtype="int64", shape=[n]),
            "weights": tinker.TensorData(data=weights, dtype="float32", shape=[n]),
        },
    )


def _parity_batch() -> tuple[PolicyBatch, list[list[float]]]:
    """Two datums with masked tool tokens, ratios inside and outside every trust region."""
    data = [_datum([0, 1, 1, 0, 1]), _datum([0, 0, 1, 1, 1, 1])]
    rollout = [[0.0, -1.0, -0.5, 0.0, -2.0], [0.0, 0.0, -0.3, -0.9, -1.4, -0.2]]
    anchor = [[0.0, -1.1, -0.4, 0.0, -1.5], [0.0, 0.0, -0.35, -0.8, -1.6, -0.25]]
    policy = [[-3.0, -0.6, -0.45, -7.0, -1.6], [-2.0, -4.0, -0.2, -1.3, -1.55, -0.24]]
    batch = PolicyBatch(
        data=data,
        advantages=[1.0, -0.5],
        prompt_lens=[2, 3],
        rollout_logprobs=rollout,
        anchor_logprobs=anchor,
    )
    return batch, policy


def test_server_side_losses_are_trainer_loss_names() -> None:
    import fireworks.training.sdk  # noqa: F401  (registers Fireworks loss names)

    supported = set(get_args(tinker.types.LossFnType))
    names = {module.LOSS_FN for module, *_ in SERVER_SIDE}

    # dppo becomes available once the SDK registers it as a request loss name.
    assert names - supported <= {"dppo"}


@pytest.mark.parametrize("module, config, prefix", SERVER_SIDE)
def test_server_objective_surface(module, config, prefix) -> None:
    assert module.PREFIX == prefix
    assert isinstance(module.LOSS_FN, str)
    assert isinstance(module.loss_fn_config(config), dict)
    assert isinstance(module.NORMALIZATION, GradAccNormalization)
    assert callable(module.build_datums) and callable(module.metrics) and callable(module.make_loss_fn)
    assert isinstance(getattr(module, CONFIG_CLASSES[module]), type)


@pytest.mark.parametrize("module, config", CLIENT_ONLY)
def test_client_only_objective_surface(module, config) -> None:
    assert isinstance(config, getattr(module, CONFIG_CLASSES[module]))
    assert isinstance(module.NORMALIZATION, GradAccNormalization)
    assert callable(module.build_datums) and callable(module.make_loss_fn)
    assert not hasattr(module, "LOSS_FN")
    assert not hasattr(module, "loss_fn_config")


@pytest.mark.parametrize("module", list(CONFIG_CLASSES))
def test_configs_are_frozen(module) -> None:
    config = getattr(module, CONFIG_CLASSES[module])()
    field = next(iter(vars(config)))
    with pytest.raises(FrozenInstanceError):
        setattr(config, field, getattr(config, field))


def test_normalization_matches_objective_reduction() -> None:
    assert gspo.NORMALIZATION is GradAccNormalization.NUM_SEQUENCES
    token_mean = [grpo, dapo, cispo, dro, dppo, importance_sampling, reinforce, score_centering]
    assert all(m.NORMALIZATION is GradAccNormalization.NUM_LOSS_TOKENS for m in token_mean)


@pytest.mark.parametrize("module, config, prefix", SERVER_SIDE)
def test_each_objective_owns_its_config_and_metrics(module, config, prefix) -> None:
    values = module.metrics(_batch(), _result([-0.4, -0.2, -0.1]), config)

    assert values[f"{prefix}_ratio_mean"] > 0
    assert f"{prefix}_kl" in values
    assert values["active_tokens"] == 2
    assert "inference_k1" not in values


TIGHT = [
    (grpo, grpo.GRPOConfig(eps_clip=0.05)),
    (gspo, gspo.GSPOConfig()),
    (dapo, dapo.DAPOConfig(eps_clip=0.05, eps_clip_high=0.05)),
    (cispo, cispo.CISPOConfig(eps_low=0.05, eps_high=0.05)),
    (dppo, dppo.DPPOConfig(threshold=0.05)),
]


@pytest.mark.parametrize(
    "module, config, prefix", SERVER_SIDE + [(m, c, m.PREFIX) for m, c in TIGHT]
)
def test_server_and_client_report_the_same_metrics(module, config, prefix) -> None:
    batch, policy = _parity_batch()

    server = module.metrics(batch, _result(*policy), config)
    loss_fn = module.make_loss_fn(batch, config)
    _, client = loss_fn(batch.data, [torch.tensor(row, requires_grad=True) for row in policy])

    server_keys = {k for k in server if k.startswith(f"{prefix}_")}
    client_keys = {k for k in client if k.startswith(f"{prefix}_")}
    assert {f"{prefix}_ratio_mean", f"{prefix}_kl"} <= server_keys
    assert server_keys == client_keys - CLIENT_EXTRAS
    for key in server_keys:
        assert client[key] == pytest.approx(server[key], rel=1e-5, abs=1e-6), key


@pytest.mark.parametrize("module, config", TIGHT)
def test_parity_batch_exercises_clipping(module, config) -> None:
    batch, policy = _parity_batch()
    values = module.metrics(batch, _result(*policy), config)
    assert 0.0 < values[f"{module.PREFIX}_clip_frac"] <= 1.0


def test_unclipped_objectives_report_no_clip_fraction() -> None:
    batch, policy = _parity_batch()
    for module, config in ((dro, dro.DROConfig()), (importance_sampling, importance_sampling.ISConfig())):
        assert f"{module.PREFIX}_clip_frac" not in module.metrics(batch, _result(*policy), config)


@pytest.mark.parametrize("module, config, prefix", SERVER_SIDE)
def test_build_datums_share_the_policy_datum_contract(module, config, prefix) -> None:
    (datum,) = module.build_datums(_batch(), config)
    assert set(datum.loss_fn_inputs) == {"target_tokens", "logprobs", "advantages", "response_mask"}


def test_importance_sampling_metrics_use_rollout_probabilities() -> None:
    values = importance_sampling.metrics(
        _batch(), _result([-0.4, -0.3, -0.1]), importance_sampling.ISConfig()
    )

    assert values["is_ratio_mean"] == pytest.approx(1.0)
    assert "is_clip_frac" not in values


def test_builtin_gspo_rejects_token_sum_reduction() -> None:
    with pytest.raises(ValueError, match="token_reduction='mean'"):
        gspo.loss_fn_config(gspo.GSPOConfig(token_reduction="sum"))


def test_builtin_grpo_rejects_reference_kl() -> None:
    with pytest.raises(ValueError, match="make_loss_fn"):
        grpo.loss_fn_config(grpo.GRPOConfig(kl_beta=0.1))


@pytest.mark.parametrize("module, config", [(grpo, grpo.GRPOConfig(kl_beta=0.1)), (reinforce, reinforce.ReinforceConfig(kl_beta=0.1))])
def test_client_reference_kl_needs_reference_rows(module, config) -> None:
    with pytest.raises(ValueError, match="ref_logprobs"):
        module.make_loss_fn(_batch(), config)


def test_loss_fn_configs_map_clip_epsilons_to_thresholds() -> None:
    assert grpo.loss_fn_config(grpo.GRPOConfig(eps_clip=0.2, eps_clip_high=0.3)) == pytest.approx(
        {"clip_low_threshold": 0.8, "clip_high_threshold": 1.3}
    )
    assert dapo.loss_fn_config(dapo.DAPOConfig(eps_clip_c=3.0))["eps_clip_c"] == 3.0
    assert "eps_clip_c" not in dapo.loss_fn_config(dapo.DAPOConfig())
    assert dppo.loss_fn_config(dppo.DPPOConfig(divergence="binary_kl")) == {
        "divergence": "binary_kl",
        "ratio_log_cap": 20.0,
        "threshold": 0.05,
    }
    assert dro.loss_fn_config(dro.DROConfig(beta=0.1)) == {"beta": 0.1}
    assert importance_sampling.loss_fn_config(importance_sampling.ISConfig()) == {}
