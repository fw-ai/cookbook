"""Tests for the two training-shape launch paths.

Exercises the cookbook ``create_trainer_job`` through to
``TrainerJobConfig`` construction for each path, verifying that the
correct config fields are set.

Two paths tested:
  1. Shape path  (profile provided, shape owns infra)
  2. Manual path (no profile, all fields from InfraConfig)
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import training.utils.infra as infra_module
from training.utils.config import InfraConfig
from fireworks.training.sdk.trainer import (
    TrainerJobConfig,
    TrainingShapeProfile,
)

BASE_MODEL = "accounts/fireworks/models/qwen3-1p7b"

PROFILE = TrainingShapeProfile(
    training_shape_version="accounts/fw/trainingShapes/ts-qwen3-1p7b/versions/v42",
    trainer_image_tag="0.35.0",
    max_supported_context_length=4096,
    node_count=1,
    deployment_shape_version="accounts/fw/deploymentShapes/ds-qwen3-1p7b/versions/v1",
    deployment_image_tag="4.24.22",
    accelerator_type="NVIDIA_H100_80GB",
    accelerator_count=1,
    base_model_weight_precision="bfloat16",
    pipeline_parallelism=1,
)


class _CapturingMgr:
    """Fake TrainerJobManager that captures the TrainerJobConfig."""

    account_id = "test-account"

    def __init__(self):
        self.captured: TrainerJobConfig | None = None

    def create_and_wait(self, config, **kwargs):
        self.captured = config
        return SimpleNamespace(job_id="job-smoke")


# -----------------------------------------------------------------------
# Path 1: Shape path (profile provided, shape owns infra)
# -----------------------------------------------------------------------


class TestShapePath:
    def test_config_has_minimal_fields(self):
        mgr = _CapturingMgr()
        infra_module.create_trainer_job(
            mgr,
            base_model=BASE_MODEL,
            infra=InfraConfig(region="US_VIRGINIA_1"),
            profile=PROFILE,
            grad_accum=2,
        )
        c = mgr.captured

        assert c.training_shape_ref == PROFILE.training_shape_version
        assert c.base_model == BASE_MODEL
        assert c.gradient_accumulation_steps == 2
        assert c.region == "US_VIRGINIA_1"

        assert c.accelerator_type is None
        assert c.accelerator_count is None
        assert c.custom_image_tag is None
        assert c.max_context_length == PROFILE.max_supported_context_length
        assert c.node_count is None

    def test_extra_args_passed_through(self):
        mgr = _CapturingMgr()
        infra_module.create_trainer_job(
            mgr,
            base_model=BASE_MODEL,
            infra=InfraConfig(extra_args=["--foo"]),
            profile=PROFILE,
        )
        assert mgr.captured.extra_args == ["--foo"]


# -----------------------------------------------------------------------
# Path 2: Manual path (no profile, all fields from InfraConfig)
# -----------------------------------------------------------------------


class TestManualPath:
    def test_infra_fields_sent_directly(self):
        mgr = _CapturingMgr()
        infra_module.create_trainer_job(
            mgr,
            base_model=BASE_MODEL,
            infra=InfraConfig(
                region="US_VIRGINIA_1",
                accelerator_type="NVIDIA_H100_80GB",
                accelerator_count=1,
                custom_image_tag="manual:1",
                node_count=1,
            ),
            max_seq_len=4096,
        )
        c = mgr.captured

        assert c.training_shape_ref is None
        assert c.accelerator_type == "NVIDIA_H100_80GB"
        assert c.accelerator_count == 1
        assert c.custom_image_tag == "manual:1"
        assert c.max_context_length == 4096
        assert c.node_count == 1

    def test_no_shape_no_overrides(self):
        """Bare minimum manual launch with only base_model."""
        mgr = _CapturingMgr()
        infra_module.create_trainer_job(
            mgr,
            base_model=BASE_MODEL,
            infra=InfraConfig(),
        )
        c = mgr.captured

        assert c.training_shape_ref is None
        assert c.accelerator_type is None
        assert c.accelerator_count is None
        assert c.custom_image_tag is None
