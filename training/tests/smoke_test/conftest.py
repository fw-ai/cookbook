"""Shared fixtures for remote smoke tests on small Qwen3-4B shapes."""

from __future__ import annotations

import os

import pytest

from fireworks.training.sdk.deployment import DeploymentManager
from fireworks.training.sdk.trainer import TrainerJobManager

DEFAULT_SMOKE_BASE_MODEL = "accounts/fireworks/models/qwen3-4b"
DEFAULT_SMOKE_TOKENIZER_MODEL = "Qwen/Qwen3-4B"
DEFAULT_SMOKE_TRAINING_SHAPE = "ts-qwen3-4b-smoke-v1"
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
def smoke_custom_image_tag() -> str | None:
    return _get_env("FIREWORKS_CUSTOM_IMAGE_TAG")


@pytest.fixture(scope="session")
def smoke_sdk_managers():
    api_key = _get_env("FIREWORKS_API_KEY")
    if not api_key:
        pytest.skip("FIREWORKS_API_KEY not set")

    account_id = _get_env("FIREWORKS_ACCOUNT_ID")
    if not account_id:
        pytest.skip("FIREWORKS_ACCOUNT_ID not set")

    base_url = _get_env("FIREWORKS_BASE_URL", DEFAULT_SMOKE_BASE_URL)
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
