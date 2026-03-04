"""Pre-flight validation for cookbook configs."""

from __future__ import annotations

import logging

from fireworks.training.sdk.errors import DOCS_HOTLOAD, DOCS_API_KEYS, DOCS_DEPLOYMENTS, format_sdk_error
from training_cookbook.utils.config import InfraConfig, DeployConfig, ResumeConfig, HotloadConfig

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

    if hotload.hot_load_interval > 0 and not deploy.deployment_id:
        errors.append(
            format_sdk_error(
                "Hotload requires a deployment",
                f"hot_load_interval={hotload.hot_load_interval} but no deployment_id is configured.",
                "Set deployment_id in DeployConfig when using hotload.\n"
                "  Example: DeployConfig(deployment_id='my-deployment', create_deployment=True)",
                docs_url=DOCS_HOTLOAD,
            )
        )

    if hotload.hot_load_before_training and not deploy.deployment_id:
        errors.append(
            format_sdk_error(
                "hot_load_before_training requires a deployment",
                "Cannot hotload before training without a deployment_id.",
                "Set deployment_id in DeployConfig.",
                docs_url=DOCS_HOTLOAD,
            )
        )

    if deploy.create_deployment and not deploy.deployment_id:
        errors.append(
            format_sdk_error(
                "create_deployment requires a deployment_id",
                "Cannot create a deployment without specifying an ID.",
                "Set deployment_id in DeployConfig.",
                docs_url=DOCS_DEPLOYMENTS,
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


def validate_streaming_config(
    prompt_groups_per_step: int,
    completions_per_prompt: int,
    max_samples_per_fwd_bwd: int,
    min_samples_per_fwd_bwd: int | None = None,
) -> None:
    """Validate streaming / batching parameters before creating resources.

    Catches config mismatches that would otherwise surface as silent
    training bugs or OOMs deep into a run.
    """
    errors: list[str] = []

    effective_min_samples = (
        min_samples_per_fwd_bwd
        if min_samples_per_fwd_bwd is not None
        else max_samples_per_fwd_bwd
    )

    if completions_per_prompt < 1:
        errors.append(
            f"completions_per_prompt must be >= 1, got {completions_per_prompt}"
        )
    if prompt_groups_per_step < 1:
        errors.append(
            f"prompt_groups_per_step must be >= 1, got {prompt_groups_per_step}"
        )
    if max_samples_per_fwd_bwd < completions_per_prompt:
        errors.append(
            f"max_samples_per_fwd_bwd ({max_samples_per_fwd_bwd}) must be >= completions_per_prompt ({completions_per_prompt}) "
            f"so at least one full prompt group fits per fwd_bwd call"
        )
    if (
        completions_per_prompt > 0
        and max_samples_per_fwd_bwd % completions_per_prompt != 0
    ):
        errors.append(
            f"max_samples_per_fwd_bwd ({max_samples_per_fwd_bwd}) should be divisible by completions_per_prompt ({completions_per_prompt}) "
            f"to avoid partial groups in a fwd_bwd call"
        )

    if effective_min_samples < completions_per_prompt:
        errors.append(
            f"min_samples_per_fwd_bwd ({effective_min_samples}) must be >= completions_per_prompt ({completions_per_prompt}) "
            f"so at least one full prompt group fits per fwd_bwd call"
        )
    if effective_min_samples > max_samples_per_fwd_bwd:
        errors.append(
            f"min_samples_per_fwd_bwd ({effective_min_samples}) must be <= max_samples_per_fwd_bwd ({max_samples_per_fwd_bwd})"
        )
    if (
        completions_per_prompt > 0
        and effective_min_samples % completions_per_prompt != 0
    ):
        errors.append(
            f"min_samples_per_fwd_bwd ({effective_min_samples}) should be divisible by completions_per_prompt ({completions_per_prompt}) "
            f"to avoid partial groups in a fwd_bwd call"
        )

    if errors:
        raise ValueError("\n".join(errors))

    min_prompt_groups = effective_min_samples // completions_per_prompt
    max_prompt_groups = max_samples_per_fwd_bwd // completions_per_prompt
    fwd_bwd_calls_per_step = -(-prompt_groups_per_step // min_prompt_groups)
    logger.info(
        "Streaming config: min_samples_per_fwd_bwd=%d, max_samples_per_fwd_bwd=%d, "
        "completions_per_prompt=%d -> fire fwd_bwd at %d prompt_groups (%d samples), "
        "cap at %d prompt_groups (%d samples), ~%d fwd_bwd calls per step (%d prompt_groups/step)",
        effective_min_samples,
        max_samples_per_fwd_bwd,
        completions_per_prompt,
        min_prompt_groups,
        effective_min_samples,
        max_prompt_groups,
        max_samples_per_fwd_bwd,
        fwd_bwd_calls_per_step,
        prompt_groups_per_step,
    )


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
            create_deployment=getattr(args, "create_deployment", False),
        ),
        infra=InfraConfig(),
    )
