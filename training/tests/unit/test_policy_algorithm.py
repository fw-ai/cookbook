"""Contracts for the policy-objective registry in ``training.utils.rl.algorithm``."""

from __future__ import annotations

import importlib
from typing import get_args, get_type_hints

import pytest

from training.recipes import async_rl_loop
from training.utils.rl.algorithm import (
    POLICY_LOSSES,
    resolve_policy_loss,
    supported_builtin_loss_fns,
)


def test_recipe_policy_loss_names_match_the_registry() -> None:
    hints = get_type_hints(async_rl_loop.Config)

    assert set(get_args(hints["policy_loss"])) == set(POLICY_LOSSES)


@pytest.mark.parametrize(
    "old, new",
    [
        ("grpo", "grpo"),
        ("gspo", "gspo"),
        ("cispo", "cispo"),
        ("dapo", "dapo"),
        ("dro", "dro"),
        ("dppo", "dppo"),
        ("is_loss", "importance_sampling"),
        ("score_centering", "score_centering"),
        ("reinforce", "reinforce"),
        ("igpo", "igpo"),
    ],
)
def test_previous_module_paths_alias_the_algorithm_modules(old, new) -> None:
    previous = importlib.import_module(f"training.utils.rl.{old}")

    assert previous is importlib.import_module(f"training.utils.rl.algorithm.{new}")


def test_every_builtin_objective_names_a_trainer_loss() -> None:
    supported = supported_builtin_loss_fns()
    builtin = {
        loss.builtin_name for loss in POLICY_LOSSES.values() if loss.builtin_name
    }

    # dppo becomes available once the SDK registers it as a request loss name.
    assert builtin - supported <= {"dppo"}


def test_objectives_without_an_anchor_always_use_rollout_probabilities() -> None:
    loss = resolve_policy_loss(
        "importance_sampling",
        options=None,
        execution="client",
        anchor="old_policy",
        kl_beta=0,
    )

    assert loss.anchor == "rollout"


def test_unknown_objective_lists_the_supported_names() -> None:
    with pytest.raises(ValueError, match="expected one of"):
        resolve_policy_loss(
            "ppo2", options=None, execution="client", anchor="rollout", kl_beta=0
        )
