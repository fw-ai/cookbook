"""Shared fixtures for remote smoke tests on small Qwen3-4B shapes."""

from __future__ import annotations

import logging
import os

import pytest

from fireworks.training.sdk.deployment import DeploymentManager
from fireworks.training.sdk.trainer import TrainerJobManager
from training.utils.config import InfraConfig

logger = logging.getLogger(__name__)

DEFAULT_SMOKE_BASE_MODEL = "accounts/fireworks/models/qwen3-4b"
DEFAULT_SMOKE_HF_TOKENIZER_NAME = "Qwen/Qwen3-4B"
DEFAULT_SMOKE_TRAINING_SHAPE = "ts-qwen3-4b-smoke-v1"
DEFAULT_SMOKE_BASE_URL = "https://dev.api.fireworks.ai"


def _get_env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


@pytest.fixture(scope="session")
def smoke_base_model() -> str:
    return _get_env("FIREWORKS_SMOKE_BASE_MODEL", DEFAULT_SMOKE_BASE_MODEL)


@pytest.fixture(scope="session")
def smoke_hf_tokenizer_name() -> str:
    return _get_env("FIREWORKS_SMOKE_HF_TOKENIZER_NAME", DEFAULT_SMOKE_HF_TOKENIZER_NAME)


@pytest.fixture(scope="session")
def smoke_training_shape() -> str:
    return _get_env("FIREWORKS_SMOKE_TRAINING_SHAPE", DEFAULT_SMOKE_TRAINING_SHAPE)


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
def smoke_dpo_infra(smoke_training_shape, smoke_custom_image_tag) -> InfraConfig:
    """InfraConfig for DPO smoke tests (includes ref_training_shape_id).

    DPO always needs a reference model. On the shape path, both policy
    and reference use the same training shape -- the control plane's
    ``applyForwardOnlyConfig`` handles the ``--forward-only`` difference.
    """
    if smoke_custom_image_tag:
        return InfraConfig(custom_image_tag=smoke_custom_image_tag)
    return InfraConfig(
        training_shape_id=smoke_training_shape,
        ref_training_shape_id=smoke_training_shape,
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
