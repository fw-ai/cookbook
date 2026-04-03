"""Shared fixtures for remote smoke tests on validated Qwen3-8B shapes."""

from __future__ import annotations

import logging
import os

import pytest

from fireworks.training.sdk.deployment import DeploymentManager
from fireworks.training.sdk.trainer import TrainerJobManager
from training.utils.config import InfraConfig

logger = logging.getLogger(__name__)

DEFAULT_SMOKE_BASE_MODEL = "accounts/fireworks/models/qwen3-8b"
DEFAULT_SMOKE_TOKENIZER_MODEL = "Qwen/Qwen3-8B"
DEFAULT_SMOKE_TRAINING_SHAPE = (
    "accounts/fireworks/trainingShapes/qwen3-8b-128k"
)
DEFAULT_SMOKE_REF_TRAINING_SHAPE = (
    "accounts/fireworks/trainingShapes/qwen3-8b-128k-forward-only"
)
DEFAULT_SMOKE_DEPLOYMENT_SHAPE = None
DEFAULT_SMOKE_BASE_URL = "https://dev.api.fireworks.ai"


def _get_env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


@pytest.fixture(scope="session")
def smoke_base_model() -> str:
    return _get_env("FIREWORKS_SMOKE_BASE_MODEL", DEFAULT_SMOKE_BASE_MODEL)


@pytest.fixture(scope="session")
def smoke_tokenizer_model() -> str:
    return _get_env("FIREWORKS_SMOKE_TOKENIZER_MODEL", DEFAULT_SMOKE_TOKENIZER_MODEL)


@pytest.fixture(scope="session")
def smoke_training_shape() -> str:
    return _get_env("FIREWORKS_SMOKE_TRAINING_SHAPE", DEFAULT_SMOKE_TRAINING_SHAPE)


@pytest.fixture(scope="session")
def smoke_ref_training_shape() -> str:
    return _get_env(
        "FIREWORKS_SMOKE_REF_TRAINING_SHAPE",
        DEFAULT_SMOKE_REF_TRAINING_SHAPE,
    )


@pytest.fixture(scope="session")
def smoke_deployment_shape() -> str | None:
    return _get_env(
        "FIREWORKS_SMOKE_DEPLOYMENT_SHAPE",
        DEFAULT_SMOKE_DEPLOYMENT_SHAPE,
    )


@pytest.fixture(scope="session")
def smoke_custom_image_tag() -> str | None:
    return _get_env("FIREWORKS_CUSTOM_IMAGE_TAG")


@pytest.fixture(scope="session")
def smoke_infra(smoke_training_shape, smoke_custom_image_tag) -> InfraConfig:
    """Build the right InfraConfig for the smoke environment.

    * CI sets ``FIREWORKS_CUSTOM_IMAGE_TAG`` to test a freshly-built image
      -> manual path (no training shape, direct infra fields).
    * Without a custom image -> shape path (validated training shape owns
      all infra fields, which is what production users actually use).
    """
    if smoke_custom_image_tag:
        logger.info("Smoke: manual path (custom_image_tag=%s)", smoke_custom_image_tag)
        return InfraConfig(custom_image_tag=smoke_custom_image_tag)
    logger.info("Smoke: shape path (training_shape_id=%s)", smoke_training_shape)
    return InfraConfig(training_shape_id=smoke_training_shape)


@pytest.fixture(scope="session")
def smoke_dpo_infra(
    smoke_training_shape,
    smoke_ref_training_shape,
    smoke_custom_image_tag,
) -> InfraConfig:
    """InfraConfig for DPO smoke tests (includes ref_training_shape_id).

    DPO always needs a reference model. Use a dedicated forward-only shape
    so the smoke path mirrors the validated production configuration.
    """
    if smoke_custom_image_tag:
        return InfraConfig(custom_image_tag=smoke_custom_image_tag)
    return InfraConfig(
        training_shape_id=smoke_training_shape,
        ref_training_shape_id=smoke_ref_training_shape,
    )


@pytest.fixture(scope="session")
def smoke_sdk_managers():
    api_key = _get_env("FIREWORKS_API_KEY")
    if not api_key:
        pytest.skip("FIREWORKS_API_KEY not set")

    base_url = _get_env("FIREWORKS_BASE_URL", DEFAULT_SMOKE_BASE_URL)
    inference_url = _get_env("FIREWORKS_INFERENCE_URL", base_url)
    hotload_api_url = _get_env("FIREWORKS_HOTLOAD_API_URL", base_url)

    additional_headers = {}
    gateway_secret = _get_env("FIREWORKS_GATEWAY_SECRET")
    if gateway_secret:
        additional_headers["X-Fireworks-Gateway-Secret"] = gateway_secret

    rlor_mgr = TrainerJobManager(
        api_key=api_key,
        base_url=base_url,
        additional_headers=additional_headers or None,
    )
    deploy_mgr = DeploymentManager(
        api_key=api_key,
        base_url=base_url,
        inference_url=inference_url,
        hotload_api_url=hotload_api_url,
        additional_headers=additional_headers or None,
    )
    return rlor_mgr, deploy_mgr
