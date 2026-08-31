"""Run Harbor/Mini-SWE through an environment-local TITO sidecar."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
from typing import TYPE_CHECKING, Any

from training.examples.rl.harbor.mini_swe.constants import (
    MINI_SWE_HARBOR_IMPORT_PATH,
    PINNED_MINI_SWE_VERSION,
)
from training.examples.rl.harbor.tito.sidecar import (
    build_launch_spec,
    build_sidecar_bundle,
    launch_spec_json,
    resolve_max_context_tokens,
)
from training.examples.rl.harbor.tito.rollout import (
    DEFAULT_MAX_CONCURRENT_TRIALS,
    DEFAULT_ROLLOUT_RETRIES,
    materialize_harbor_trajectory,
    resolve_trials_dir,
    run_with_fresh_trajectory_retries,
    trial_workspace,
)
from training.examples.rl.harbor.tito.trial import (
    DEFAULT_HARBOR_RETRYABLE_EXCEPTIONS,
    DEFAULT_HARNESS_TOOL_TIMEOUT_SECONDS,
    load_harbor_trial_config,
    run_harbor_trial,
    task_config_from_row,
    task_initial_instruction,
    task_name_from_row,
    validate_harbor_retry_exceptions,
)
from training.utils.rl.rollout import RolloutRun
from training.utils.rl.rollout.lifecycle import ActiveRolloutTasks

if TYPE_CHECKING:
    from training.recipes.async_rl_loop import RolloutFn, RolloutSetup

logger = logging.getLogger(__name__)


class _MiniSweRolloutRunner:
    """Allocate one independent sidecar trajectory per Mini-SWE attempt."""

    def __init__(self, setup: RolloutSetup) -> None:
        if setup.sample_kwargs.get("echo"):
            raise ValueError("Mini-SWE TITO supports completion-only Router Replay")
        self._setup = setup
        self._tito_debug_enabled = bool(setup.extras.get("tito_debug_enabled", False))
        self._trial_config = load_harbor_trial_config(
            setup.extras.get("harbor_trial_config")
        )
        self._harbor_environment = str(
            setup.extras.get("harbor_environment", "docker")
        ).lower()
        if self._harbor_environment not in {"docker", "e2b"}:
            raise ValueError(
                "rollout_extras['harbor_environment'] must be docker or e2b"
            )
        configured_retry_names = setup.extras.get("retry_include_exceptions")
        self._retry_include_exceptions = validate_harbor_retry_exceptions(
            DEFAULT_HARBOR_RETRYABLE_EXCEPTIONS
            if configured_retry_names is None
            else configured_retry_names
        )
        self._trials_dir = resolve_trials_dir(
            setup.extras.get("harbor_trials_dir"),
            self._trial_config,
        )
        self._rollout_retries = int(
            setup.extras.get("rollout_retries", DEFAULT_ROLLOUT_RETRIES)
        )
        if self._rollout_retries < 0:
            raise ValueError("rollout_extras['rollout_retries'] must be >= 0")
        max_concurrent_trials = int(
            setup.extras.get(
                "max_concurrent_trials",
                DEFAULT_MAX_CONCURRENT_TRIALS,
            )
        )
        if max_concurrent_trials < 1:
            raise ValueError("rollout_extras['max_concurrent_trials'] must be >= 1")
        self._trial_semaphore = asyncio.Semaphore(max_concurrent_trials)
        terminal_failure_reward = setup.extras.get("terminal_failure_reward")
        self._terminal_failure_reward = (
            None if terminal_failure_reward is None else float(terminal_failure_reward)
        )
        if self._terminal_failure_reward is not None and not math.isfinite(
            self._terminal_failure_reward
        ):
            raise ValueError("rollout_extras['terminal_failure_reward'] must be finite")
        self._reward_key = str(setup.extras.get("harbor_reward_key", "reward")).strip()
        if not self._reward_key:
            raise ValueError("rollout_extras['harbor_reward_key'] must not be empty")
        self._tool_timeout_seconds = int(
            setup.extras.get(
                "harness_tool_timeout_seconds",
                DEFAULT_HARNESS_TOOL_TIMEOUT_SECONDS,
            )
        )
        if self._tool_timeout_seconds < 1:
            raise ValueError(
                "rollout_extras['harness_tool_timeout_seconds'] must be positive"
            )
        self._max_context_tokens = resolve_max_context_tokens(setup)
        self._sidecar_bundle = build_sidecar_bundle(setup)
        self._active_rollouts = ActiveRolloutTasks()

    async def __call__(
        self,
        sample_prompt: dict[str, Any],
        *,
        cursor_index: int = 0,
        sample_index: int = 0,
        epoch: int = 0,
        evaluation: bool = False,
        **_: Any,
    ) -> RolloutRun | None:
        async with self._active_rollouts.track():
            return await self._run(
                sample_prompt,
                cursor_index=cursor_index,
                sample_index=sample_index,
                epoch=epoch,
                evaluation=evaluation,
            )

    async def _run(
        self,
        sample_prompt: dict[str, Any],
        *,
        cursor_index: int,
        sample_index: int,
        epoch: int,
        evaluation: bool,
    ) -> RolloutRun | None:
        task_config = task_config_from_row(sample_prompt)
        task_name = task_name_from_row(sample_prompt)
        run_id = f"harbor-mini-swe:{task_name}:{epoch}:{cursor_index}:{sample_index}"
        rollout_group_id = f"harbor-mini-swe:{task_name}:{epoch}:{cursor_index}"
        canonical_initial_prompt_hash = hashlib.sha256(
            task_initial_instruction(task_config).encode("utf-8")
        ).hexdigest()

        async def attempt(retry_index: int) -> RolloutRun | None:
            return await self._run_attempt(
                task_config=task_config,
                task_name=task_name,
                run_id=run_id,
                rollout_group_id=rollout_group_id,
                sample_index=sample_index,
                evaluation=evaluation,
                canonical_initial_prompt_hash=canonical_initial_prompt_hash,
                retry_index=retry_index,
            )

        return await run_with_fresh_trajectory_retries(
            attempt,
            task_name=task_name,
            retries=self._rollout_retries,
        )

    async def _run_attempt(
        self,
        *,
        task_config: Any,
        task_name: str,
        run_id: str,
        rollout_group_id: str,
        sample_index: int,
        evaluation: bool,
        canonical_initial_prompt_hash: str,
        retry_index: int,
    ) -> RolloutRun | None:
        metadata = {
            "harness": "mini_swe",
            "harbor_environment_type": self._harbor_environment,
            "task_name": task_name,
            "run_id": run_id,
            "rollout_group_id": rollout_group_id,
            "rollout_member_index": sample_index,
            "retry_index": retry_index,
            "evaluation": evaluation,
            "canonical_initial_prompt_hash": canonical_initial_prompt_hash,
        }
        launch_spec = launch_spec_json(
            build_launch_spec(
                self._setup,
                self._sidecar_bundle,
                call_classifier="all_policy",
                metadata=metadata,
            )
        )
        with trial_workspace(
            self._trials_dir,
            prefix="harbor-mini-swe-tito-",
        ) as trial_root:
            async with self._trial_semaphore:
                outcome = await run_harbor_trial(
                    task_config=task_config,
                    inference_key=self._setup.api_key,
                    run_id=(f"{run_id}:retry-{retry_index}" if retry_index else run_id),
                    harbor_environment=self._harbor_environment,
                    sidecar_bundle_path=self._sidecar_bundle.path,
                    sidecar_launch_spec=launch_spec,
                    trial_config=self._trial_config,
                    trials_dir=trial_root,
                    context_limit=self._max_context_tokens,
                    output_limit=int(self._setup.sample_kwargs["max_tokens"]),
                    agent_import_path=MINI_SWE_HARBOR_IMPORT_PATH,
                    agent_provider="openai",
                    agent_version=PINNED_MINI_SWE_VERSION,
                    reward_key=self._reward_key,
                    retry_include_exceptions=self._retry_include_exceptions,
                    tool_timeout_seconds=self._tool_timeout_seconds,
                    terminal_failure_reward=self._terminal_failure_reward,
                )
            rollout = materialize_harbor_trajectory(
                outcome,
                max_context_tokens=self._max_context_tokens,
                debug_enabled=self._tito_debug_enabled,
            )
            if rollout is None:
                logger.warning(
                    "Harbor/Mini-SWE retained a rewardless trajectory for %s",
                    outcome.task_name,
                )
                return None
            rollout.run_id = run_id
            rollout.metadata.update(
                {
                    "task_name": outcome.task_name,
                    "trial_name": outcome.trial_name,
                    "harbor_rewards": outcome.rewards,
                    "harbor_reward_key": self._reward_key,
                    "harbor_exception_type": outcome.exception_type,
                    "tito_harness": "mini_swe",
                    "harbor_environment_type": outcome.environment_type,
                    "mini_swe_version": PINNED_MINI_SWE_VERSION,
                    "tito_retry_index": retry_index,
                }
            )
            return rollout

    async def aclose(self) -> None:
        await self._active_rollouts.cancel_and_wait()


def make_rollout_fn(setup: RolloutSetup) -> RolloutFn:
    return _MiniSweRolloutRunner(setup)


__all__ = ["PINNED_MINI_SWE_VERSION", "make_rollout_fn"]
