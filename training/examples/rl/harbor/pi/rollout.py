"""Run Harbor/Pi through an environment-local TITO sidecar."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

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
from training.utils.rl.async_rl.errors import RecoverableRolloutError
from training.utils.rl.rollout import RolloutRun
from training.utils.rl.rollout.lifecycle import ActiveRolloutTasks

from .artifacts import tool_timeout_count
from .constants import PI_HARBOR_IMPORT_PATH, PINNED_PI_REVISION, PINNED_PI_VERSION

if TYPE_CHECKING:
    from training.recipes.async_rl_loop import RolloutFn, RolloutSetup

logger = logging.getLogger(__name__)
_TITO_RESPONSE_ID_PREFIX = "chatcmpl-"


def _pi_abandoned_turn_ids(event_stream: str) -> set[str]:
    """Extract Pi length attempts discarded by successful overflow recovery."""

    last_length_response_id: str | None = None
    abandoned: set[str] = set()
    for line in event_stream.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping):
            continue
        if event.get("type") == "message_end":
            message = event.get("message")
            if not isinstance(message, Mapping) or message.get("role") != "assistant":
                continue
            if message.get("stopReason") == "length":
                response_id = message.get("responseId")
                last_length_response_id = (
                    response_id if isinstance(response_id, str) else None
                )
            else:
                last_length_response_id = None
            continue
        if (
            event.get("type") == "compaction_end"
            and event.get("reason") == "overflow"
            and event.get("willRetry") is True
            and event.get("aborted") is not True
        ):
            if not last_length_response_id or not last_length_response_id.startswith(
                _TITO_RESPONSE_ID_PREFIX
            ):
                raise ValueError(
                    "Pi retried after overflow compaction without a recognizable "
                    "length-truncated TITO response ID"
                )
            abandoned.add(last_length_response_id[len(_TITO_RESPONSE_ID_PREFIX) :])
            last_length_response_id = None
    return abandoned


class _PiRolloutRunner:
    """Allocate one independent sidecar trajectory per Harbor/Pi attempt."""

    def __init__(self, setup: RolloutSetup) -> None:
        if setup.sample_kwargs.get("echo"):
            raise ValueError("Pi TITO supports completion-only Router Replay")
        self._setup = setup
        self._tito_debug_enabled = bool(setup.extras.get("tito_debug_enabled", False))
        self._max_context_tokens = resolve_max_context_tokens(setup)
        self._pi_revision = str(setup.extras.get("pi_revision", PINNED_PI_REVISION))
        if self._pi_revision != PINNED_PI_REVISION:
            raise ValueError(
                "TITO/Pi supports only the classifier-certified revision "
                f"{PINNED_PI_REVISION}; got {self._pi_revision}"
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
        run_id = f"harbor-pi:{task_name}:{epoch}:{cursor_index}:{sample_index}"
        rollout_group_id = f"harbor-pi:{task_name}:{epoch}:{cursor_index}"
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
            "harness": "pi",
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
                call_classifier="adapter_metadata",
                metadata=metadata,
            )
        )
        with trial_workspace(self._trials_dir, prefix="harbor-pi-tito-") as trial_root:
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
                    agent_import_path=PI_HARBOR_IMPORT_PATH,
                    agent_provider="fireworks-tito",
                    agent_version=PINNED_PI_VERSION,
                    tool_timeout_seconds=self._tool_timeout_seconds,
                    terminal_failure_reward=self._terminal_failure_reward,
                    retry_include_exceptions=self._retry_include_exceptions,
                )
            lifecycle_path = outcome.trial_path / "agent" / "pi.txt"
            if not lifecycle_path.is_file():
                raise RecoverableRolloutError(
                    f"Pi trial produced no lifecycle stream: {lifecycle_path}"
                )
            try:
                abandoned_turn_ids = _pi_abandoned_turn_ids(
                    lifecycle_path.read_text(encoding="utf-8", errors="replace")
                )
            except ValueError as exc:
                raise RecoverableRolloutError(
                    f"Pi lifecycle stream could not be reconciled: {exc}"
                ) from exc
            rollout = materialize_harbor_trajectory(
                outcome,
                max_context_tokens=self._max_context_tokens,
                debug_enabled=self._tito_debug_enabled,
                harness_abandoned_turn_ids=abandoned_turn_ids,
            )
            if rollout is None:
                logger.warning(
                    "Harbor/Pi retained an untrained rewardless trajectory for %s",
                    outcome.task_name,
                )
                return None
            rollout.run_id = run_id
            rollout.metadata.update(
                {
                    "tito_harness": "pi",
                    "harbor_environment_type": outcome.environment_type,
                    "pi_revision": self._pi_revision,
                    "pi_version": PINNED_PI_VERSION,
                    "pi_abandoned_turn_count": len(abandoned_turn_ids),
                    "harness_tool_timeout_count": tool_timeout_count(
                        outcome.trial_path
                    ),
                    "harness_tool_timeout_seconds": self._tool_timeout_seconds,
                    "trial_name": outcome.trial_name,
                    "harbor_rewards": outcome.rewards,
                    "tito_retry_index": retry_index,
                }
            )
            return rollout

    async def aclose(self) -> None:
        await self._active_rollouts.cancel_and_wait()


def make_rollout_fn(setup: RolloutSetup) -> RolloutFn:
    return _PiRolloutRunner(setup)


__all__ = ["PINNED_PI_REVISION", "PINNED_PI_VERSION", "make_rollout_fn"]
