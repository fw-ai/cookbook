"""Pre-flight validation for cookbook configs."""

from __future__ import annotations

import logging

from fireworks.training.sdk.errors import DOCS_HOTLOAD, DOCS_API_KEYS, DOCS_DEPLOYMENTS, format_sdk_error
from training.utils.config import InfraConfig, DeployConfig, ResumeConfig, HotloadConfig

logger = logging.getLogger(__name__)


def validate_config(
    base_model: str,
    dataset: str,
    hotload: HotloadConfig,
    deploy: DeployConfig,
    infra: InfraConfig,
    resume: ResumeConfig | None = None,
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
                "Use format: accounts/ACCOUNT/models/MODEL_NAME\n" "  Example: accounts/fireworks/models/qwen3-8b",
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

    if resume and resume.resume_from and not resume.resume_from.startswith(("gs://", "/")):
        if resume.resume_job_id is None:
            logger.warning(
                "resume_from='%s' looks like a checkpoint name, not a full path. "
                "If resuming from a different job, set resume_job_id.",
                resume.resume_from,
            )

    if infra.node_count < 1:
        errors.append(
            format_sdk_error(
                "Invalid node_count",
                f"node_count={infra.node_count} must be >= 1.",
                "Set InfraConfig(node_count=1) or higher.",
            )
        )

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
                    docs_url=DOCS_API_KEYS,
                )
            )
        if not fw_account_id:
            errors.append(
                format_sdk_error(
                    "Missing FIREWORKS_ACCOUNT_ID",
                    "No account ID found in --fireworks-account-id or FIREWORKS_ACCOUNT_ID env var.",
                    "export FIREWORKS_ACCOUNT_ID='your-account-id'\n"
                    "  Find your account ID at: https://fireworks.ai/account",
                    docs_url=DOCS_API_KEYS,
                )
            )

    if errors:
        raise RuntimeError("\n\n".join(errors))

    validate_config(
        base_model=getattr(args, "base_model", "") or "",
        dataset=getattr(args, "dataset", "") or "",
        hotload=HotloadConfig(
            hot_load_interval=getattr(args, "hot_load_interval", 0),
            hot_load_before_training=getattr(args, "hot_load_before_training", False),
        ),
        deploy=DeployConfig(
            deployment_id=getattr(args, "hot_load_deployment_id", None),
        ),
        infra=InfraConfig(),
    )
