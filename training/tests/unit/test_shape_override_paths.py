"""Tests for the two training-shape launch paths.

Exercises the cookbook ``create_trainer_job`` through to
``TrainerJobConfig`` construction for each path, verifying that the
correct config fields are set.

Two paths tested:
  1. Shape path  (profile provided, shape owns infra)
  2. Manual path (no profile, all fields from InfraConfig)
"""

from __future__ import annotations

from typing import Any
from types import SimpleNamespace

import pytest

import training.utils.infra as infra_module
from training.utils.config import InfraConfig, purpose_annotation_value
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

    def create(self, config):
        self.captured = config
        return SimpleNamespace(job_id="job-smoke", job_name="jobs/job-smoke")

    def wait_for_ready(self, job_id, **kwargs):
        return SimpleNamespace(job_id=job_id)


class _FailingMgr:
    account_id = "test-account"

    def __init__(self, message: str):
        self.message = message

    def create(self, config):
        raise RuntimeError(self.message)


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

    def test_reference_failure_is_explicit(self, caplog):
        mgr = _FailingMgr("shape not valid for forward-only")

        with caplog.at_level("ERROR"):
            with pytest.raises(
                RuntimeError,
                match="Failed to create reference trainer job 'grpo-reference'",
            ) as exc_info:
                infra_module.create_trainer_job(
                    mgr,
                    base_model=BASE_MODEL,
                    infra=InfraConfig(region="US_VIRGINIA_1"),
                    profile=PROFILE,
                    display_name="grpo-reference",
                    forward_only=True,
                )

        assert "Failed to create reference trainer job 'grpo-reference'" in caplog.text
        assert str(exc_info.value.__cause__) == "shape not valid for forward-only"

    def test_policy_failure_is_explicit(self, caplog):
        mgr = _FailingMgr("shape not valid for policy")

        with caplog.at_level("ERROR"):
            with pytest.raises(
                RuntimeError,
                match="Failed to create policy trainer job 'grpo-policy'",
            ) as exc_info:
                infra_module.create_trainer_job(
                    mgr,
                    base_model=BASE_MODEL,
                    infra=InfraConfig(region="US_VIRGINIA_1"),
                    profile=PROFILE,
                    display_name="grpo-policy",
                    forward_only=False,
                )

        assert "Failed to create policy trainer job 'grpo-policy'" in caplog.text
        assert str(exc_info.value.__cause__) == "shape not valid for policy"


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


# -----------------------------------------------------------------------
# purpose propagation
# -----------------------------------------------------------------------


class _PurposeCapturingMgr:
    """Fake TrainerJobManager that simulates the SDK create → _post chain.

    ``create()`` mirrors the real SDK: it builds a minimal payload from
    the config (including ``config.purpose`` when the SDK exposes it)
    and routes it through ``_post()``.  This lets us verify that the
    ``purpose`` field reaches the outgoing JSON regardless of whether
    the SDK or the cookbook shim is responsible for injecting it.
    """

    account_id = "test-account"

    def __init__(self):
        self.captured_config: TrainerJobConfig | None = None
        self.captured_post_json: dict | None = None

    class _FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"name": "accounts/test-account/rlorTrainerJobs/job-p"}

    def create(self, config):
        self.captured_config = config
        payload: dict[str, Any] = {"trainingConfig": {}}
        purpose = getattr(config, "purpose", None)
        if purpose:
            payload["purpose"] = purpose
        self._post("/v1/accounts/test-account/rlorTrainerJobs", json=payload)
        return SimpleNamespace(job_id="job-p", job_name="accounts/test-account/rlorTrainerJobs/job-p")

    def _post(self, path, json=None, **kw):
        self.captured_post_json = json
        return self._FakeResp()

    def wait_for_ready(self, job_id, **kwargs):
        return SimpleNamespace(job_id=job_id)


class TestPurpose:
    """Verify purpose reaches the REST payload regardless of SDK version.

    The cookbook uses ``TrainerJobConfig.purpose`` when the SDK exposes
    it, and falls back to the ``_create_with_purpose_shim`` (which
    wraps ``_post``) for older SDK versions.  Either way, the field
    must appear in the outgoing JSON.
    """

    def test_purpose_injected_shape_path(self):
        mgr = _PurposeCapturingMgr()
        infra_module.create_trainer_job(
            mgr,
            base_model=BASE_MODEL,
            infra=InfraConfig(purpose="PURPOSE_PILOT"),
            profile=PROFILE,
        )
        assert mgr.captured_config is not None
        assert mgr.captured_post_json["purpose"] == "PURPOSE_PILOT"

    def test_purpose_injected_manual_path(self):
        mgr = _PurposeCapturingMgr()
        infra_module.create_trainer_job(
            mgr,
            base_model=BASE_MODEL,
            infra=InfraConfig(purpose="PURPOSE_PILOT"),
        )
        assert mgr.captured_config is not None
        assert mgr.captured_post_json["purpose"] == "PURPOSE_PILOT"

    def test_purpose_none_not_in_payload(self):
        mgr = _PurposeCapturingMgr()
        infra_module.create_trainer_job(
            mgr,
            base_model=BASE_MODEL,
            infra=InfraConfig(),
            profile=PROFILE,
        )
        assert mgr.captured_config is not None
        assert "purpose" not in mgr.captured_post_json


# -----------------------------------------------------------------------
# purpose_annotation_value
# -----------------------------------------------------------------------


class TestPurposeAnnotationValue:
    def test_pilot(self):
        assert purpose_annotation_value("PURPOSE_PILOT") == "pilot"

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid purpose"):
            purpose_annotation_value("bad_value")

    def test_unspecified_raises(self):
        with pytest.raises(ValueError, match="Invalid purpose"):
            purpose_annotation_value("PURPOSE_UNSPECIFIED")
