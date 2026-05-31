"""Shared fixtures for algorithm E2E tests.

These tests create real RLOR jobs and deployments on Fireworks infrastructure.
They default to the SDK-managed qwen3p5-9b single-shape port path.

Requires FIREWORKS_API_KEY to be set.

Override defaults via environment variables:
  FIREWORKS_E2E_MODEL, FIREWORKS_E2E_REGION, FIREWORKS_E2E_TRAINING_SHAPE,
  FIREWORKS_E2E_DEPLOYMENT_SHAPE, FIREWORKS_E2E_TOKENIZER_MODEL,
  FIREWORKS_BASE_URL
"""

from __future__ import annotations

import os
import logging

import pytest

from fireworks.training.sdk.trainer import TrainerJobManager
from fireworks.training.sdk.deployment import DeploymentManager

DEFAULT_MODEL = "accounts/fireworks/models/qwen3p5-9b"
DEFAULT_TOKENIZER_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_REGION = "US_OHIO_1"
DEFAULT_TRAINING_SHAPE = "accounts/fireworks/trainingShapes/qwen3p5-9b-256k"
DEFAULT_REFERENCE_TRAINING_SHAPE = (
    "accounts/fireworks/trainingShapes/qwen3p5-9b-256k-forward-only"
)
DEFAULT_LORA_TRAINING_SHAPE = "accounts/fireworks/trainingShapes/qwen3p5-9b-256k-lora"
DEFAULT_TRAINING_ACCELERATOR = None
DEFAULT_DEPLOYMENT_ACCELERATOR = "NVIDIA_B200_180GB"
DEFAULT_DEPLOYMENT_SHAPE = "accounts/fireworks/deploymentShapes/rft-qwen3p5-9b-v2/versions/n864rzzy"

GSM8K_SAMPLE_URL = "https://raw.githubusercontent.com/eval-protocol/python-sdk/main/development/gsm8k_sample.jsonl"


@pytest.fixture(autouse=True, scope="session")
def _configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )


def _get_env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


@pytest.fixture(scope="module")
def sdk_managers():
    """Create TrainerJobManager + DeploymentManager from env vars.

    Skips the entire module if FIREWORKS_API_KEY is not set.
    """
    api_key = _get_env("FIREWORKS_API_KEY")
    if not api_key:
        pytest.skip("FIREWORKS_API_KEY not set -- skipping E2E tests")

    base_url = _get_env("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
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


@pytest.fixture(scope="module")
def e2e_region() -> str:
    return _get_env("FIREWORKS_E2E_REGION", DEFAULT_REGION)


@pytest.fixture(scope="module")
def e2e_model() -> str:
    return _get_env("FIREWORKS_E2E_MODEL", DEFAULT_MODEL)


@pytest.fixture(scope="module")
def e2e_tokenizer_model() -> str:
    """HuggingFace model name for the tokenizer (client-side tokenization)."""
    return _get_env("FIREWORKS_E2E_TOKENIZER_MODEL", DEFAULT_TOKENIZER_MODEL)


@pytest.fixture(scope="module")
def e2e_training_accelerator() -> str | None:
    """Accelerator for RLOR trainer jobs (None = server auto-configures)."""
    return _get_env("FIREWORKS_E2E_TRAINING_ACCELERATOR", DEFAULT_TRAINING_ACCELERATOR)


@pytest.fixture(scope="module")
def e2e_training_shape(port_lora_rank) -> str | None:
    """Training shape for SDK-managed trainer jobs."""
    if port_lora_rank:
        return _get_env("FIREWORKS_E2E_LORA_TRAINING_SHAPE", DEFAULT_LORA_TRAINING_SHAPE)
    return _get_env("FIREWORKS_E2E_TRAINING_SHAPE", DEFAULT_TRAINING_SHAPE)


@pytest.fixture(scope="module")
def e2e_reference_training_shape(port_lora_rank) -> str | None:
    """Forward-only reference shape for full-param SDK-managed jobs."""
    if port_lora_rank:
        return None
    return _get_env(
        "FIREWORKS_E2E_REFERENCE_TRAINING_SHAPE",
        DEFAULT_REFERENCE_TRAINING_SHAPE,
    )


@pytest.fixture(scope="module")
def e2e_training_profile(sdk_managers, e2e_training_shape, port_lora_rank):
    """Resolve the training shape before provisioning live resources."""
    rlor_mgr, _deploy_mgr = sdk_managers
    profile = rlor_mgr.resolve_training_profile(e2e_training_shape)
    if port_lora_rank:
        assert profile.supports_lora, f"LoRA track requires a LoRA-capable shape: {e2e_training_shape}"
    return profile


@pytest.fixture(scope="module")
def e2e_reference_training_profile(sdk_managers, e2e_reference_training_shape):
    """Resolve the full-param forward-only reference shape before provisioning."""
    if e2e_reference_training_shape is None:
        return None
    rlor_mgr, _deploy_mgr = sdk_managers
    return rlor_mgr.resolve_training_profile(e2e_reference_training_shape)


@pytest.fixture(scope="module")
def e2e_deployment_accelerator() -> str | None:
    """Accelerator for inference deployments (required by the deployment API)."""
    return _get_env("FIREWORKS_E2E_DEPLOYMENT_ACCELERATOR", DEFAULT_DEPLOYMENT_ACCELERATOR)


@pytest.fixture(scope="module")
def e2e_deployment_shape() -> str | None:
    """Deployment shape for the SDK-managed sampler."""
    return _get_env("FIREWORKS_E2E_DEPLOYMENT_SHAPE", DEFAULT_DEPLOYMENT_SHAPE)


@pytest.fixture(scope="module")
def custom_image_tag() -> str | None:
    return _get_env("FIREWORKS_CUSTOM_IMAGE_TAG")
