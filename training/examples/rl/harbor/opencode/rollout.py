"""Run Harbor/OpenCode through an environment-local TITO sidecar."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from training.examples.rl.harbor.opencode.constants import (
    DEFAULT_OPENCODE_VERSION,
    OPENCODE_HARBOR_IMPORT_PATH,
)
from training.examples.rl.harbor.opencode.artifacts import tool_timeout_count
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
    HarborTrialOutcome,
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
_TRIAL_START_INTERVAL_SECONDS = 1.0


class _HarborRolloutRunner:
    """Allocate one independent sidecar trajectory per Harbor attempt."""

    def __init__(self, setup: RolloutSetup) -> None:
        if setup.sample_kwargs.get("echo"):
            raise ValueError(
                "Fireworks Harbor supports completion-only Router Replay; "
                "set router_replay_completion_only=True"
            )
        self._setup = setup
        self._tito_debug_enabled = bool(setup.extras.get("tito_debug_enabled", False))
        self._rollout_retries = int(
            setup.extras.get("rollout_retries", DEFAULT_ROLLOUT_RETRIES)
        )
        if self._rollout_retries < 0:
            raise ValueError("rollout_extras['rollout_retries'] must be >= 0")
        configured_retry_names = setup.extras.get("retry_include_exceptions")
        self._retry_include_exceptions = validate_harbor_retry_exceptions(
            DEFAULT_HARBOR_RETRYABLE_EXCEPTIONS
            if configured_retry_names is None
            else configured_retry_names
        )
        max_concurrent_trials = int(
            setup.extras.get("max_concurrent_trials", DEFAULT_MAX_CONCURRENT_TRIALS)
        )
        if max_concurrent_trials < 1:
            raise ValueError("rollout_extras['max_concurrent_trials'] must be >= 1")
        self._trial_semaphore = asyncio.Semaphore(max_concurrent_trials)
        # Harbor concurrency bounds active physical trials. Pace only their
        # Docker starts so a full wave does not stampede host setup services;
        # this is not rollout-dispatch or sidecar admission control.
        self._trial_start_lock = asyncio.Lock()
        self._next_trial_start = 0.0
        self._trial_start_interval_seconds = _TRIAL_START_INTERVAL_SECONDS

        terminal_failure_reward = setup.extras.get("terminal_failure_reward")
        self._terminal_failure_reward = (
            None if terminal_failure_reward is None else float(terminal_failure_reward)
        )
        if self._terminal_failure_reward is not None and not math.isfinite(
            self._terminal_failure_reward
        ):
            raise ValueError("rollout_extras['terminal_failure_reward'] must be finite")
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
        self._trials_dir = resolve_trials_dir(
            setup.extras.get("harbor_trials_dir"),
            self._trial_config,
        )
        self._context_limit = resolve_max_context_tokens(setup)
        self._max_sample_tokens = int(setup.sample_kwargs.get("max_tokens") or 0)
        self._opencode_version = str(
            setup.extras.get("opencode_version", DEFAULT_OPENCODE_VERSION)
        )
        if self._opencode_version != DEFAULT_OPENCODE_VERSION:
            raise ValueError(
                "TITO/OpenCode supports only the classifier-certified version "
                f"{DEFAULT_OPENCODE_VERSION}; got {self._opencode_version}"
            )
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
        self._task_selector = setup.extras.get("task_selector")
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
        **kwargs: Any,
    ) -> RolloutRun | None:
        async with self._active_rollouts.track():
            return await self._run(
                sample_prompt,
                cursor_index=cursor_index,
                sample_index=sample_index,
                epoch=epoch,
                evaluation=evaluation,
                **kwargs,
            )

    async def _run(
        self,
        sample_prompt: dict[str, Any],
        *,
        cursor_index: int = 0,
        sample_index: int = 0,
        epoch: int = 0,
        evaluation: bool = False,
        **_: Any,
    ) -> RolloutRun | None:
        use_selector = self._task_selector is not None and not evaluation
        if use_selector:
            sample_prompt = await self._task_selector.row_for_group(cursor_index)
        task_name = task_name_from_row(sample_prompt)
        task_config = task_config_from_row(sample_prompt)
        run_id = f"harbor-opencode:{task_name}:{epoch}:{cursor_index}:{sample_index}"
        rollout_group_id = f"harbor-opencode:{task_name}:{epoch}:{cursor_index}"
        canonical_initial_prompt_hash = hashlib.sha256(
            task_initial_instruction(task_config).encode("utf-8")
        ).hexdigest()
        result = await self._run_opencode(
            task_config=task_config,
            task_name=task_name,
            run_id=run_id,
            rollout_group_id=rollout_group_id,
            rollout_member_index=sample_index,
            evaluation=evaluation,
            canonical_initial_prompt_hash=canonical_initial_prompt_hash,
        )
        if use_selector:
            await self._task_selector.record(
                cursor_index,
                sample_index,
                result.segments[0].reward if result and result.segments else None,
            )
        return result

    async def _run_opencode(
        self,
        *,
        task_config: Any,
        task_name: str,
        run_id: str,
        rollout_group_id: str | None = None,
        rollout_member_index: int | None = None,
        evaluation: bool = False,
        canonical_initial_prompt_hash: str | None = None,
    ) -> RolloutRun | None:
        async def attempt(retry_index: int) -> RolloutRun | None:
            return await self._run_opencode_attempt(
                task_config=task_config,
                task_name=task_name,
                run_id=run_id,
                rollout_group_id=rollout_group_id,
                rollout_member_index=rollout_member_index,
                evaluation=evaluation,
                canonical_initial_prompt_hash=canonical_initial_prompt_hash,
                retry_index=retry_index,
            )

        return await run_with_fresh_trajectory_retries(
            attempt,
            task_name=task_name,
            retries=self._rollout_retries,
        )

    async def _run_opencode_attempt(
        self,
        *,
        task_config: Any,
        task_name: str,
        run_id: str,
        rollout_group_id: str | None,
        rollout_member_index: int | None,
        evaluation: bool,
        canonical_initial_prompt_hash: str | None,
        retry_index: int,
    ) -> RolloutRun | None:
        metadata = {
            "harness": "opencode",
            "harbor_environment_type": self._harbor_environment,
            "task_name": task_name,
            "run_id": run_id,
            "rollout_group_id": rollout_group_id,
            "rollout_member_index": rollout_member_index,
            "retry_index": retry_index,
            "evaluation": evaluation,
            "canonical_initial_prompt_hash": canonical_initial_prompt_hash,
        }
        launch_spec = launch_spec_json(
            build_launch_spec(
                self._setup,
                self._sidecar_bundle,
                call_classifier="tools_present",
                metadata=metadata,
            )
        )
        with trial_workspace(
            self._trials_dir,
            prefix="harbor-opencode-tito-",
        ) as trial_root:
            outcome = await self._run_admitted_trial(
                task_config=task_config,
                inference_key=self._setup.api_key,
                run_id=(f"{run_id}:retry-{retry_index}" if retry_index else run_id),
                trials_dir=trial_root,
                harbor_environment=self._harbor_environment,
                sidecar_bundle_path=self._sidecar_bundle.path,
                sidecar_launch_spec=launch_spec,
                context_limit=self._context_limit,
                output_limit=self._max_sample_tokens,
                agent_import_path=OPENCODE_HARBOR_IMPORT_PATH,
                agent_provider="fireworks-rl",
                agent_version=self._opencode_version,
                tool_timeout_seconds=self._tool_timeout_seconds,
            )
            rollout = materialize_harbor_trajectory(
                outcome,
                max_context_tokens=self._context_limit,
                debug_enabled=self._tito_debug_enabled,
            )
            if rollout is None:
                logger.warning(
                    "Harbor/OpenCode retained an untrained rewardless "
                    "trajectory for %s (%s, %s)",
                    outcome.task_name,
                    outcome.exception_type,
                    outcome.trajectory_artifact.terminal_reason,
                )
                return None
            rollout.run_id = run_id
            rollout.metadata.update(
                {
                    "task_name": outcome.task_name,
                    "trial_name": outcome.trial_name,
                    "harbor_rewards": outcome.rewards,
                    "harbor_exception_type": outcome.exception_type,
                    "harness_tool_timeout_count": tool_timeout_count(
                        outcome.trial_path
                    ),
                    "harness_tool_timeout_seconds": self._tool_timeout_seconds,
                    "tito_harness": "opencode",
                    "harbor_environment_type": outcome.environment_type,
                    "tito_retry_index": retry_index,
                }
            )
            return rollout

    async def _run_trial(
        self,
        *,
        task_config: Any,
        inference_key: str,
        run_id: str,
        trials_dir: Path | None = None,
        **agent_kwargs: Any,
    ) -> HarborTrialOutcome:
        return await run_harbor_trial(
            task_config=task_config,
            inference_key=inference_key,
            run_id=run_id,
            trial_config=self._trial_config,
            trials_dir=trials_dir if trials_dir is not None else self._trials_dir,
            terminal_failure_reward=self._terminal_failure_reward,
            retry_include_exceptions=self._retry_include_exceptions,
            **agent_kwargs,
        )

    async def _run_admitted_trial(
        self,
        *,
        task_config: Any,
        inference_key: str,
        run_id: str,
        trials_dir: Path | None = None,
        **agent_kwargs: Any,
    ) -> HarborTrialOutcome:
        async with self._trial_semaphore:
            if self._harbor_environment == "docker":
                await self._pace_trial_start()
            started = time.monotonic()
            try:
                return await self._run_trial(
                    task_config=task_config,
                    inference_key=inference_key,
                    run_id=run_id,
                    trials_dir=trials_dir,
                    **agent_kwargs,
                )
            finally:
                logger.debug(
                    "Harbor/OpenCode trial wall time: %.3fs",
                    time.monotonic() - started,
                )

    async def _pace_trial_start(self) -> None:
        async with self._trial_start_lock:
            now = time.monotonic()
            scheduled = max(now, self._next_trial_start)
            self._next_trial_start = scheduled + self._trial_start_interval_seconds
        delay = scheduled - now
        if delay > 0:
            await asyncio.sleep(delay)

    async def aclose(self) -> None:
        await self._active_rollouts.cancel_and_wait()


def make_rollout_fn(setup: RolloutSetup) -> RolloutFn:
    """Create one logical Harbor rollout with bounded fresh-trajectory retries."""
    return _HarborRolloutRunner(setup)


__all__ = ["make_rollout_fn"]
