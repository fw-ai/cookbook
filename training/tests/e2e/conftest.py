"""Shared fixtures for algorithm E2E tests.

These tests create real RLOR jobs and deployments on Fireworks infrastructure.
All algorithms run on qwen3-30b-a3b (MoE) by default.

Requires FIREWORKS_API_KEY and FIREWORKS_ACCOUNT_ID to be set.

Override defaults via environment variables:
  FIREWORKS_E2E_MODEL, FIREWORKS_E2E_REGION, FIREWORKS_E2E_DEPLOYMENT_SHAPE,
  FIREWORKS_E2E_TOKENIZER_MODEL, FIREWORKS_ACCOUNT_ID, FIREWORKS_BASE_URL
"""

from __future__ import annotations

import os
import logging

import pytest

from fireworks.training.sdk.trainer import TrainerJobManager
from fireworks.training.sdk.deployment import DeploymentManager

DEFAULT_MODEL = "accounts/fireworks/models/qwen3-30b-a3b"
DEFAULT_TOKENIZER_MODEL = "Qwen/Qwen3-30B-A3B"
DEFAULT_REGION = "US_OHIO_1"
DEFAULT_TRAINING_ACCELERATOR = None
DEFAULT_DEPLOYMENT_ACCELERATOR = "NVIDIA_B200_180GB"
# For GRPO MoE tests, set FIREWORKS_E2E_DEPLOYMENT_SHAPE to a shape that
# enables router stats (e.g. --enable-moe-stats).
DEFAULT_DEPLOYMENT_SHAPE = None

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

    Skips the entire module if FIREWORKS_API_KEY or FIREWORKS_ACCOUNT_ID is not set.
    """
    api_key = _get_env("FIREWORKS_API_KEY")
    if not api_key:
        pytest.skip("FIREWORKS_API_KEY not set -- skipping E2E tests")

    account_id = _get_env("FIREWORKS_ACCOUNT_ID")
    if not account_id:
        pytest.skip("FIREWORKS_ACCOUNT_ID not set -- skipping E2E tests")

    base_url = _get_env("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    inference_url = _get_env("FIREWORKS_INFERENCE_URL", base_url)
    hotload_api_url = _get_env("FIREWORKS_HOTLOAD_API_URL", base_url)

    additional_headers = {}
    gateway_secret = _get_env("FIREWORKS_GATEWAY_SECRET")
    if gateway_secret:
        additional_headers["X-Fireworks-Gateway-Secret"] = gateway_secret

    rlor_mgr = TrainerJobManager(
        api_key=api_key,
        account_id=account_id,
        base_url=base_url,
        additional_headers=additional_headers or None,
    )
    deploy_mgr = DeploymentManager(
        api_key=api_key,
        account_id=account_id,
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
def e2e_deployment_accelerator() -> str | None:
    """Accelerator for inference deployments (required by the deployment API)."""
    return _get_env("FIREWORKS_E2E_DEPLOYMENT_ACCELERATOR", DEFAULT_DEPLOYMENT_ACCELERATOR)


@pytest.fixture(scope="module")
def e2e_deployment_shape() -> str | None:
    """Deployment shape for MoE models (includes --enable-moe-stats for R3)."""
    return _get_env("FIREWORKS_E2E_DEPLOYMENT_SHAPE", DEFAULT_DEPLOYMENT_SHAPE)


@pytest.fixture(scope="module")
def custom_image_tag() -> str | None:
    return _get_env("FIREWORKS_CUSTOM_IMAGE_TAG")
