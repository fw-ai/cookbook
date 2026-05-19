from __future__ import annotations

import importlib

import pytest

from training.utils import (
    LRScheduleConfig,
    LR_SCHEDULE_KINDS,
    TwoPointLinearConfig,
    compute_learning_rate,
    validate_lr_schedule_config,
)


def test_compute_learning_rate_preserves_constant_warmup_behavior():
    peak_lr = 1e-4
    schedule = LRScheduleConfig(kind="constant", warmup_steps=4)

    assert compute_learning_rate(1, 10, peak_lr, schedule) == pytest.approx(2.5e-5)
    assert compute_learning_rate(4, 10, peak_lr, schedule) == pytest.approx(peak_lr)
    assert compute_learning_rate(5, 10, peak_lr, schedule) == pytest.approx(peak_lr)


def test_compute_learning_rate_cosine_decays_to_min_lr_after_warmup():
    peak_lr = 1e-4
    schedule = LRScheduleConfig(
        kind="cosine",
        warmup_steps=1,
        min_lr_ratio=0.1,
    )

    assert compute_learning_rate(1, 4, peak_lr, schedule) == pytest.approx(peak_lr)
    assert compute_learning_rate(2, 4, peak_lr, schedule) == pytest.approx(7.75e-5)
    assert compute_learning_rate(3, 4, peak_lr, schedule) == pytest.approx(3.25e-5)
    assert compute_learning_rate(4, 4, peak_lr, schedule) == pytest.approx(1e-5)


def test_compute_learning_rate_generalized_cosine_uses_power():
    peak_lr = 1e-4
    schedule = LRScheduleConfig(kind="generalized_cosine", cosine_power=2.0)

    assert compute_learning_rate(1, 3, peak_lr, schedule) == pytest.approx(peak_lr)
    assert compute_learning_rate(2, 3, peak_lr, schedule) == pytest.approx(2.5e-5)
    assert compute_learning_rate(3, 3, peak_lr, schedule) == pytest.approx(0.0)


def test_compute_learning_rate_linear_decays_to_min_lr():
    peak_lr = 1e-4
    schedule = LRScheduleConfig(kind="linear", min_lr_ratio=0.1)

    assert compute_learning_rate(1, 3, peak_lr, schedule) == pytest.approx(peak_lr)
    assert compute_learning_rate(2, 3, peak_lr, schedule) == pytest.approx(5.5e-5)
    assert compute_learning_rate(3, 3, peak_lr, schedule) == pytest.approx(1e-5)


def test_compute_learning_rate_two_point_linear_interpolates_control_points():
    peak_lr = 1e-4
    schedule = LRScheduleConfig(
        kind="two_point_linear",
        two_point_linear=TwoPointLinearConfig(
            x1=0.25,
            y1=0.8,
            x2=0.75,
            y2=0.2,
        ),
    )

    assert compute_learning_rate(1, 5, peak_lr, schedule) == pytest.approx(1e-4)
    assert compute_learning_rate(2, 5, peak_lr, schedule) == pytest.approx(8e-5)
    assert compute_learning_rate(3, 5, peak_lr, schedule) == pytest.approx(5e-5)
    assert compute_learning_rate(4, 5, peak_lr, schedule) == pytest.approx(2e-5)
    assert compute_learning_rate(5, 5, peak_lr, schedule) == pytest.approx(0.0)


def test_validate_lr_schedule_config_accepts_all_supported_kinds():
    for kind in LR_SCHEDULE_KINDS:
        validate_lr_schedule_config(LRScheduleConfig(kind=kind))


def test_validate_lr_schedule_config_ignores_irrelevant_cosine_power():
    validate_lr_schedule_config(LRScheduleConfig(kind="constant", cosine_power=0.0))
    validate_lr_schedule_config(LRScheduleConfig(kind="linear", cosine_power=-1.0))
    validate_lr_schedule_config(LRScheduleConfig(kind="cosine", cosine_power=0.0))


@pytest.mark.parametrize(
    "recipe_module_name",
    (
        "training.recipes.sft_loop",
        "training.recipes.dpo_loop",
        "training.recipes.orpo_loop",
    ),
)
@pytest.mark.parametrize("kind", sorted(LR_SCHEDULE_KINDS))
def test_recipe_configs_accept_shared_lr_schedule_config(recipe_module_name, kind):
    recipe_module = importlib.import_module(recipe_module_name)

    cfg = recipe_module.Config(
        log_path="/tmp/test_lr_schedule_config",
        lr_schedule=LRScheduleConfig(kind=kind),
    )

    assert isinstance(cfg.lr_schedule, LRScheduleConfig)
    validate_lr_schedule_config(cfg.lr_schedule)


def test_validate_lr_schedule_config_rejects_invalid_values():
    schedule = LRScheduleConfig(kind="exponential")
    with pytest.raises(ValueError, match="lr_schedule"):
        validate_lr_schedule_config(schedule)

    schedule = LRScheduleConfig(kind="cosine", min_lr_ratio=1.5)
    with pytest.raises(ValueError, match="min_lr_ratio"):
        validate_lr_schedule_config(schedule)

    schedule = LRScheduleConfig(kind="cosine", warmup_steps=-1)
    with pytest.raises(ValueError, match="warmup_steps"):
        validate_lr_schedule_config(schedule)

    schedule = LRScheduleConfig(kind="generalized_cosine", cosine_power=0.0)
    with pytest.raises(ValueError, match="cosine_power"):
        validate_lr_schedule_config(schedule)

    schedule = LRScheduleConfig(
        kind="two_point_linear",
        two_point_linear=TwoPointLinearConfig(x1=0.7, x2=0.6),
    )
    with pytest.raises(ValueError, match="two_point_linear\\.x1"):
        validate_lr_schedule_config(schedule)

    schedule = LRScheduleConfig(
        kind="two_point_linear",
        two_point_linear=TwoPointLinearConfig(x1=0.3, y1=0.2, x2=0.6, y2=0.4),
    )
    with pytest.raises(ValueError, match="two_point_linear\\.y1"):
        validate_lr_schedule_config(schedule)
