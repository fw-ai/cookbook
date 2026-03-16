"""Pre-flight validation for cookbook configs.

Application-level checks only (credentials, base_model format, dataset).
Infra-field validation lives in the SDK's ``TrainerJobConfig.validate()``.
"""

from __future__ import annotations

import logging
import re

from fireworks.training.sdk.errors import format_sdk_error, DOCS_SDK
from training.utils.config import DeployConfig, WeightSyncConfig

logger = logging.getLogger(__name__)
_RESOURCE_ID_LEGACY_RE = re.compile(r"^[a-z0-9-]+$")


def _validate_output_model_id(output_model_id: str | None) -> list[str]:
    """Return a list of error strings for invalid output model IDs (empty if valid)."""
    if output_model_id in (None, ""):
        return []

    problems: list[str] = []
    if len(output_model_id) > 63:
        problems.append("must be at most 63 characters")
    if output_model_id.startswith("-"):
        problems.append("must not start with '-'")
    if output_model_id.endswith("-"):
        problems.append("must not end with '-'")
    if not _RESOURCE_ID_LEGACY_RE.fullmatch(output_model_id):
        problems.append("must contain only lowercase a-z, 0-9, and hyphen (-)")

    if problems:
        return [
            format_sdk_error(
                "Invalid output_model_id",
                f"'{output_model_id}' is not a valid Fireworks model ID.",
                "Use 1-63 characters of lowercase a-z, 0-9, or hyphen (-).\n"
                "  Underscores, spaces, slashes, and uppercase letters are not allowed.\n"
                "  The ID must not start or end with '-'.\n"
                "  Example: deepmath-qwen3-8b-dev",
            )
        ]
    return []


def validate_config(
    base_model: str,
    dataset: str,
    hotload: WeightSyncConfig | None = None,
    deploy: DeployConfig | None = None,
    output_model_id: str | None = None,
) -> None:
    """Pre-flight validation. Catches misconfiguration before provisioning GPUs."""
    errors: list[str] = []

    if not base_model:
        errors.append(
            format_sdk_error(
                "Missing base_model",
                "No base model specified.",
                "Set base_model (e.g. 'accounts/fireworks/models/qwen3-8b').",
            )
        )
    elif not base_model.startswith("accounts/"):
        errors.append(
            format_sdk_error(
                "Invalid base_model format",
                f"'{base_model}' doesn't match expected format.",
                "Use format: accounts/ACCOUNT/models/MODEL_NAME\n"
                "  Example: accounts/fireworks/models/qwen3-8b",
            )
        )

    if not dataset:
        errors.append(
            format_sdk_error(
                "Missing dataset",
                "No dataset path or URL specified.",
                "Set dataset to a local path or URL to a JSONL file.",
            )
        )

    errors.extend(_validate_output_model_id(output_model_id))

    if errors:
        raise RuntimeError("\n\n".join(errors))


def validate_preflight(
    args,
    fw_api_key: str | None,
    fw_account_id: str | None,
    *,
    skip_credential_check: bool = False,
) -> None:
    """Catch common CLI configuration issues before creating expensive resources."""
    errors: list[str] = []

    if not skip_credential_check:
        if not fw_api_key:
            errors.append(
                format_sdk_error(
                    "Missing FIREWORKS_API_KEY",
                    "No API key found in --fireworks-api-key or FIREWORKS_API_KEY env var.",
                    "export FIREWORKS_API_KEY='your-key-here'\n"
                    "  Get your API key at: https://fireworks.ai/account/api-keys",
                    docs_url=DOCS_SDK,
                )
            )
        if not fw_account_id:
            errors.append(
                format_sdk_error(
                    "Missing FIREWORKS_ACCOUNT_ID",
                    "No account ID found in --fireworks-account-id or FIREWORKS_ACCOUNT_ID env var.",
                    "export FIREWORKS_ACCOUNT_ID='your-account-id'\n"
                    "  Find your account ID at: https://fireworks.ai/account",
                    docs_url=DOCS_SDK,
                )
            )

    if errors:
        raise RuntimeError("\n\n".join(errors))

    validate_config(
        base_model=getattr(args, "base_model", "") or "",
        dataset=getattr(args, "dataset", "") or "",
        hotload=WeightSyncConfig(
            weight_sync_interval=getattr(args, "weight_sync_interval", 0),
            weight_sync_before_training=getattr(args, "weight_sync_before_training", False),
        ),
        deploy=DeployConfig(
            deployment_id=getattr(args, "hot_load_deployment_id", None),
        ),
        output_model_id=getattr(args, "output_model_id", None),
    )
