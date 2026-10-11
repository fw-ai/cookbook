"""Training-time plumbing shared by the Harbor task recipes.

Every recipe in this package is task rows + one harness ``make_rollout_fn`` +
a recipe ``Config`` + ``main``. This module holds what they would otherwise
repeat: CLI groups, the typed ``HarborSettings`` each harness factory closes
over, dedicated resource config, sampling-only scoring against an existing
deployment, and W&B/E2B runtime settings. Lightweight row and record helpers
live in ``records.py`` so dataset preparation does not import the training
stack.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fireworks.training.sdk.deployment import DeploymentSampler

from training.examples.harbor_rl.records import write_json
from training.examples.harbor_rl.tito.evaluate import evaluate_rows
from training.examples.harbor_rl.tito.rollout import (
    DEFAULT_MAX_CONCURRENT_TRIALS,
    DEFAULT_ROLLOUT_RETRIES,
    HarborSettings,
)
from training.examples.harbor_rl.tito.sidecar import TITOSidecarSettings
from training.examples.harbor_rl.tito.trial import (
    DEFAULT_HARBOR_RETRYABLE_EXCEPTIONS,
    DEFAULT_HARNESS_TOOL_TIMEOUT_SECONDS,
)
from training.utils import (
    DeployConfig,
    TrainerConfig,
    read_api_extra_headers_env,
    resolve_router_replay_enabled,
)
from training.utils.rl.rollout.lifecycle import close_rollout_fn
from training.utils.rl.rollout.setup import (
    RolloutFnFactory,
    RolloutSetup,
    default_sample_kwargs,
)
from training.utils.tokenizers import load_tokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI groups
# ---------------------------------------------------------------------------


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """Policy model, tokenizer, and the certified TITO renderer contract."""
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument(
        "--renderer-name",
        required=True,
        help=(
            "Production-certified TITO renderer matching the model and "
            "tokenizer; unsupported pairs fail closed"
        ),
    )
    parser.add_argument(
        "--tito-prompt-mode",
        choices=("full_history", "incremental"),
        default="full_history",
        help="Incremental is experimental and needs a model-specific junction",
    )


def add_harness_args(
    parser: argparse.ArgumentParser,
    *,
    max_concurrent_trials: int = DEFAULT_MAX_CONCURRENT_TRIALS,
    tool_timeout_seconds: int = DEFAULT_HARNESS_TOOL_TIMEOUT_SECONDS,
    terminal_failure_reward: float | None = None,
    retry_include_exceptions: Sequence[str] = (),
    tito_debug: bool = False,
) -> None:
    """Harbor trial and harness controls with recipe-specific defaults."""
    parser.add_argument("--harbor-trial-config", default=None, help="TrialConfig YAML")
    parser.add_argument(
        "--max-concurrent-trials",
        type=int,
        default=max_concurrent_trials,
        help="Active Harbor trials; each trial owns one trajectory",
    )
    parser.add_argument(
        "--harness-tool-timeout-seconds",
        type=int,
        default=tool_timeout_seconds,
        help="Per-tool timeout; must stay below the trial's agent timeout",
    )
    parser.add_argument("--rollout-retries", type=int, default=DEFAULT_ROLLOUT_RETRIES)
    parser.add_argument(
        "--retry-include-exception",
        action="append",
        default=list(retry_include_exceptions),
        help="Additional transient Harbor/provider exception type; repeatable",
    )
    parser.add_argument(
        "--terminal-failure-reward",
        type=float,
        default=terminal_failure_reward,
        help=(
            "Reward for terminal agent failures without a verifier reward; "
            "unset retries and then discards them"
        ),
    )
    parser.add_argument(
        "--tito-debug", action=argparse.BooleanOptionalAction, default=tito_debug
    )


def add_resource_args(
    parser: argparse.ArgumentParser,
    *,
    replica_count: int,
    sample_timeout: int,
    weight_sync_timeout: int,
    shapes: bool = True,
) -> None:
    """Dedicated trainer/deployment controls; omitted IDs are SDK-created."""
    parser.add_argument("--trainer-job-id", default=None, help="Reattach a trainer")
    parser.add_argument(
        "--deployment-id", default=None, help="Reattach and hot-load a deployment"
    )
    if shapes:
        parser.add_argument("--training-shape-id", default=None)
        parser.add_argument("--deployment-shape", default=None)
    parser.add_argument("--replica-count", type=int, default=replica_count)
    parser.add_argument("--sample-timeout", type=int, default=sample_timeout)
    parser.add_argument(
        "--weight-sync-timeout",
        type=int,
        default=weight_sync_timeout,
        help="Seconds to wait for each sampler hot-load",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Keep the trainer and deployment after the run (cleanup_on_exit=False)",
    )


def add_wandb_args(parser: argparse.ArgumentParser, *, project: str) -> None:
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument(
        "--wandb-project", default=os.environ.get("WANDB_PROJECT", project)
    )
    parser.add_argument("--wandb-run-name", default=None)


# ---------------------------------------------------------------------------
# Config mapping
# ---------------------------------------------------------------------------


def trainer_config(args: argparse.Namespace) -> TrainerConfig:
    return TrainerConfig(
        job_id=args.trainer_job_id,
        training_shape_id=getattr(args, "training_shape_id", None),
    )


def deploy_config(args: argparse.Namespace, **overrides: Any) -> DeployConfig:
    return DeployConfig(
        deployment_id=args.deployment_id,
        deployment_shape=getattr(args, "deployment_shape", None),
        tokenizer_model=args.tokenizer_model,
        tokenizer_revision=args.tokenizer_revision,
        replica_count=args.replica_count,
        sample_timeout=args.sample_timeout,
        **overrides,
    )


def check_sampling_only_args(args: argparse.Namespace) -> None:
    """Sampling-only scores an existing deployment and never touches a trainer."""
    if not args.deployment_id:
        raise ValueError("--sampling-only requires --deployment-id")
    if args.trainer_job_id:
        raise ValueError("--sampling-only does not accept --trainer-job-id")


def harbor_settings(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    environment: str,
    trials_dir: Path | None,
    debug_redact_text: bool = False,
) -> HarborSettings:
    """Map the shared CLI groups onto the settings a harness factory closes over.

    ``trials_dir=None`` runs each trial in a temporary directory; pass a path
    to retain Harbor results, agent logs, and TITO artifacts.
    """
    return HarborSettings(
        tito=TITOSidecarSettings(
            renderer_name=args.renderer_name,
            bundle_root=run_dir / "sidecar-bundles",
            prompt_mode=args.tito_prompt_mode,
            debug_enabled=args.tito_debug,
            debug_redact_text=debug_redact_text,
        ),
        environment=environment,
        trial_config=args.harbor_trial_config,
        trials_dir=trials_dir,
        rollout_retries=args.rollout_retries,
        retry_include_exceptions=DEFAULT_HARBOR_RETRYABLE_EXCEPTIONS
        | frozenset(args.retry_include_exception),
        max_concurrent_trials=args.max_concurrent_trials,
        terminal_failure_reward=args.terminal_failure_reward,
        tool_timeout_seconds=args.harness_tool_timeout_seconds,
    )


# ---------------------------------------------------------------------------
# Sampling-only scoring
# ---------------------------------------------------------------------------


@contextmanager
def deployment_rollout_setup(
    *,
    base_model: str,
    deployment_id: str,
    tokenizer_model: str,
    tokenizer_revision: str | None,
    max_completion_tokens: int,
    max_seq_len: int,
    temperature: float,
    sample_timeout: int,
    completions_per_prompt: int,
) -> Iterator[RolloutSetup]:
    """Yield the ``RolloutSetup`` ``async_rl_loop`` would build, without a trainer.

    The setup owns a sampler on ``deployment_id`` that is closed on exit.
    Router Replay is requested only for MoE base models (a dense deployment
    rejects the routing request).
    """
    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    headers = read_api_extra_headers_env()
    tokenizer = load_tokenizer(tokenizer_model, tokenizer_revision)
    sample_kwargs = default_sample_kwargs(
        max_tokens=max_completion_tokens,
        temperature=temperature,
        max_seq_len=max_seq_len,
        http_timeout=sample_timeout,
        router_replay=resolve_router_replay_enabled(
            requested=True,
            api_key=api_key,
            base_url=base_url,
            additional_headers=headers,
            base_model=base_model,
        ),
    )
    sampler = DeploymentSampler(
        inference_url=base_url,
        model=deployment_id,
        api_key=api_key,
        tokenizer=tokenizer,
        additional_headers=headers,
    )
    try:
        yield RolloutSetup(
            sampler=sampler,
            tokenizer=tokenizer,
            tokenizer_id=tokenizer_model,
            sample_kwargs=sample_kwargs,
            completions_per_prompt=completions_per_prompt,
            inference_base_url=base_url,
            api_key=api_key,
            model=deployment_id,
        )
    finally:
        sampler.close()


def score_rows(
    rollout_fn_factory: RolloutFnFactory,
    setup: RolloutSetup,
    rows: Sequence[dict[str, Any]],
    *,
    completions_per_prompt: int,
    max_concurrency: int | None,
    metric_prefix: str = "sampling",
) -> dict[str, float | int]:
    """Run fixed rows through the training rollout once and close it.

    The rollout runs in training mode (no ``evaluation=True``), so the metrics
    include the trainable-token and TITO statistics a training step would see.
    """
    rollout_fn = rollout_fn_factory(setup)

    async def run() -> dict[str, float | int]:
        try:
            return await evaluate_rows(
                rollout_fn,
                rows,
                completions_per_prompt=completions_per_prompt,
                metric_prefix=metric_prefix,
                step=0,
                max_concurrency=max_concurrency,
            )
        finally:
            await close_rollout_fn(rollout_fn)

    return asyncio.run(run())


def run_sampling_only(
    args: argparse.Namespace,
    rollout_fn_factory: RolloutFnFactory,
    rows: Sequence[dict[str, Any]],
    *,
    completions_per_prompt: int,
    max_concurrency: int | None,
    result_path: Path,
    **record: Any,
) -> dict[str, float | int]:
    """Score ``rows`` with the training rollout on ``--deployment-id``.

    No trainer is created and no weights move. The metrics and ``record`` are
    written to ``result_path``.
    """
    check_sampling_only_args(args)
    with deployment_rollout_setup(
        base_model=args.base_model,
        deployment_id=args.deployment_id,
        tokenizer_model=args.tokenizer_model,
        tokenizer_revision=args.tokenizer_revision,
        max_completion_tokens=args.max_completion_tokens,
        max_seq_len=args.max_seq_len,
        temperature=args.temperature,
        sample_timeout=args.sample_timeout,
        completions_per_prompt=completions_per_prompt,
    ) as setup:
        metrics = score_rows(
            rollout_fn_factory,
            setup,
            rows,
            completions_per_prompt=completions_per_prompt,
            max_concurrency=max_concurrency,
        )
    write_json(
        result_path,
        {
            "mode": "sampling_only",
            "deployment_id": args.deployment_id,
            "tasks": [row["task_name"] for row in rows],
            "completions_per_prompt": completions_per_prompt,
            "metrics": metrics,
            **record,
        },
    )
    logger.info(
        "Sampling-only reward %.4f over %d tasks x %d completions (%s)",
        float(metrics.get("sampling/reward", 0.0)),
        len(rows),
        completions_per_prompt,
        result_path,
    )
    return metrics


# ---------------------------------------------------------------------------
# Runtime environment
# ---------------------------------------------------------------------------


def pin_wandb_run_id(entity: str, resume_run_id: str | None = None) -> str | None:
    """Fix the W&B run id through the environment so a later stage can resume it."""
    if not entity:
        return None
    run_id = resume_run_id or uuid.uuid4().hex[:8]
    os.environ["WANDB_RUN_ID"] = run_id
    os.environ["WANDB_RESUME"] = "must" if resume_run_id else "never"
    return run_id


def require_env(*names: str, wandb_entity: str = "") -> None:
    """Fail before any resource is created when a credential is missing."""
    required = (*names, "WANDB_API_KEY") if wandb_entity else names
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise ValueError(f"set {', '.join(missing)} before launching")


def set_e2b_request_timeout(seconds: float) -> None:
    """Harbor does not expose E2B's control-plane timeout in ``TrialConfig``.

    Keep provider requests alive while a high-concurrency wave queues.
    """
    if seconds <= 0:
        raise ValueError("E2B request timeout must be positive")
    import e2b.connection_config

    e2b.connection_config.REQUEST_TIMEOUT = seconds


__all__ = [
    "add_harness_args",
    "add_model_args",
    "add_resource_args",
    "add_wandb_args",
    "check_sampling_only_args",
    "deploy_config",
    "deployment_rollout_setup",
    "harbor_settings",
    "pin_wandb_run_id",
    "require_env",
    "run_sampling_only",
    "score_rows",
    "set_e2b_request_timeout",
    "trainer_config",
]
