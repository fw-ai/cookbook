"""Shared fixtures for remote smoke tests on the SDK-managed Qwen3.5 shape."""

from __future__ import annotations

import logging
import os

import pytest

from fireworks.training.sdk.deployment import DeploymentManager
from fireworks.training.sdk.trainer import TrainerJobManager
from training.utils.config import TrainerConfig

logger = logging.getLogger(__name__)

DEFAULT_SMOKE_BASE_MODEL = "accounts/fireworks/models/qwen3p5-9b"
DEFAULT_SMOKE_TOKENIZER_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_SMOKE_TRAINING_SHAPE = "accounts/fireworks/trainingShapes/qwen3p5-9b-256k"
DEFAULT_SMOKE_REFERENCE_TRAINING_SHAPE = (
    "accounts/fireworks/trainingShapes/qwen3p5-9b-256k-lora"
)
DEFAULT_SMOKE_LORA_TRAINING_SHAPE = (
    "accounts/fireworks/trainingShapes/qwen3p5-9b-256k-lora"
)
DEFAULT_SMOKE_DEPLOYMENT_SHAPE = (
    "accounts/fireworks/deploymentShapes/rft-qwen3p5-9b-v2/versions/n864rzzy"
)
DEFAULT_SMOKE_MINIMAL_TRAINING_SHAPE = "qwen3-4b-minimum"
DEFAULT_SMOKE_MINIMAL_REF_TRAINING_SHAPE = "qwen3-4b-minimum-lora"
DEFAULT_SMOKE_BASE_URL = "https://api.fireworks.ai"


def _get_env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


@pytest.fixture(scope="session")
def smoke_base_model() -> str:
    return _get_env("FIREWORKS_SMOKE_BASE_MODEL", DEFAULT_SMOKE_BASE_MODEL)


@pytest.fixture(scope="session")
def smoke_tokenizer_model() -> str:
    return _get_env("FIREWORKS_SMOKE_TOKENIZER_MODEL", DEFAULT_SMOKE_TOKENIZER_MODEL)


@pytest.fixture(scope="session")
def smoke_training_shape(port_lora_rank) -> str:
    if port_lora_rank:
        return _get_env(
            "FIREWORKS_SMOKE_LORA_TRAINING_SHAPE", DEFAULT_SMOKE_LORA_TRAINING_SHAPE
        )
    return _get_env("FIREWORKS_SMOKE_TRAINING_SHAPE", DEFAULT_SMOKE_TRAINING_SHAPE)


@pytest.fixture(scope="session")
def smoke_reference_training_shape(port_lora_rank) -> str | None:
    if port_lora_rank:
        return None
    return _get_env(
        "FIREWORKS_SMOKE_REFERENCE_TRAINING_SHAPE",
        DEFAULT_SMOKE_REFERENCE_TRAINING_SHAPE,
    )


@pytest.fixture(scope="session")
def smoke_deployment_shape() -> str:
    return _get_env("FIREWORKS_SMOKE_DEPLOYMENT_SHAPE", DEFAULT_SMOKE_DEPLOYMENT_SHAPE)


@pytest.fixture(scope="session")
def smoke_custom_image_tag() -> str | None:
    return _get_env("FIREWORKS_CUSTOM_IMAGE_TAG")


@pytest.fixture(scope="session")
def smoke_training_profile(smoke_sdk_managers, smoke_training_shape, port_lora_rank):
    """Resolve the training shape before provisioning live resources."""
    rlor_mgr, _deploy_mgr = smoke_sdk_managers
    profile = rlor_mgr.resolve_training_profile(smoke_training_shape)
    if port_lora_rank:
        assert profile.supports_lora, (
            f"LoRA track requires a LoRA-capable shape: {smoke_training_shape}"
        )
    return profile


@pytest.fixture(scope="session")
def smoke_reference_training_profile(
    smoke_sdk_managers, smoke_reference_training_shape
):
    """Resolve the full-param reference shape before provisioning."""
    if smoke_reference_training_shape is None:
        return None
    rlor_mgr, _deploy_mgr = smoke_sdk_managers
    return rlor_mgr.resolve_training_profile(smoke_reference_training_shape)


@pytest.fixture(scope="session")
def smoke_trainer_config(
    smoke_training_shape,
    smoke_reference_training_shape,
    smoke_custom_image_tag,
    smoke_training_profile,
    smoke_reference_training_profile,
) -> TrainerConfig:
    """Build the SDK-managed trainer config for the smoke environment."""
    _ = smoke_reference_training_profile
    logger.info(
        "Smoke: SDK-managed trainer shape=%s version=%s",
        smoke_training_shape,
        smoke_training_profile.training_shape_version,
    )
    return TrainerConfig(
        training_shape_id=smoke_training_shape,
        reference_training_shape_id=smoke_reference_training_shape,
        custom_image_tag=smoke_custom_image_tag,
    )


@pytest.fixture(scope="session")
def smoke_minimal_training_shape() -> str:
    return _get_env(
        "FIREWORKS_SMOKE_MINIMAL_TRAINING_SHAPE",
        DEFAULT_SMOKE_MINIMAL_TRAINING_SHAPE,
    )


@pytest.fixture(scope="session")
def smoke_minimal_ref_training_shape() -> str:
    return _get_env(
        "FIREWORKS_SMOKE_MINIMAL_REF_TRAINING_SHAPE",
        DEFAULT_SMOKE_MINIMAL_REF_TRAINING_SHAPE,
    )


@pytest.fixture(scope="session")
def smoke_minimal_grpo_trainer(
    smoke_minimal_training_shape,
    smoke_minimal_ref_training_shape,
    smoke_custom_image_tag,
) -> TrainerConfig:
    """Two minimal 1xGPU training shapes (policy + frozen reference)."""
    if smoke_custom_image_tag:
        return TrainerConfig(custom_image_tag=smoke_custom_image_tag)
    return TrainerConfig(
        training_shape_id=smoke_minimal_training_shape,
        reference_training_shape_id=smoke_minimal_ref_training_shape,
    )


@pytest.fixture(scope="session")
def smoke_sdk_managers():
    api_key = _get_env("FIREWORKS_API_KEY")
    if not api_key:
        # Inside GitHub Actions a missing FIREWORKS_API_KEY almost always
        # means the repo secret was never configured -- previously the
        # smoke matrix silently skipped and was reported green, hiding
        # the fact that no e2e was actually running. Fail loudly there.
        if os.environ.get("GITHUB_ACTIONS") == "true":
            pytest.fail(
                "FIREWORKS_API_KEY is empty in GitHub Actions. The repo "
                "secret is not configured -- add it under Settings -> "
                "Secrets and variables -> Actions."
            )
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
