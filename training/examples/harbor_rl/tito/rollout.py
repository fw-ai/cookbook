"""Harness-neutral Harbor rollout over an environment-local TITO sidecar.

An agent adapter describes itself once as a :class:`HarborHarness`; every
adapter then shares :class:`HarborRollout` for the rollout-function contract,
run-level :class:`HarborSettings`, bounded fresh-trajectory retries, trial
admission, and artifact-process materialization.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import tempfile
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping, Set
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from training.examples.harbor_rl.tito._artifact_io import ArtifactProcessPool
from training.examples.harbor_rl.tito.sidecar import (
    TITOSidecarSettings,
    build_launch_spec,
    build_sidecar_bundle,
    launch_spec_json,
    resolve_max_context_tokens,
)
from training.examples.harbor_rl.tito.trial import (
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
from training.utils.phase_tracing import phase_span
from training.utils.rl import trial_events
from training.utils.rl.async_rl.errors import RecoverableRolloutError
from training.utils.rl.rollout import RolloutRun
from training.utils.rl.rollout.lifecycle import ActiveRolloutTasks
from training.utils.rl.rollout.tito import materialize_tito_trajectory

if TYPE_CHECKING:
    from training.utils.rl.rollout.setup import RolloutSetup

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONCURRENT_TRIALS = 24
DEFAULT_ROLLOUT_RETRIES = 3
_RETRY_DELAY_SECONDS = 15
# Harbor concurrency bounds active physical trials. Pace only local Docker
# starts so a full wave does not stampede host setup services.
_DOCKER_START_INTERVAL_SECONDS = 1.0

_T = TypeVar("_T")


def resolve_trials_dir(
    configured: str | Path | None,
    trial_config: Mapping[str, object],
) -> Path | None:
    """Resolve an explicit trial directory before the Harbor YAML fallback."""

    value = configured if configured is not None else trial_config.get("trials_dir")
    return Path(value).expanduser() if value else None


@contextmanager
def trial_workspace(
    configured: Path | None,
    *,
    prefix: str,
) -> Iterator[Path]:
    """Keep temporary Harbor artifacts alive through trajectory materialization."""

    if configured is not None:
        yield configured
        return
    with tempfile.TemporaryDirectory(prefix=prefix) as temporary:
        yield Path(temporary)


async def run_with_fresh_trajectory_retries(
    operation: Callable[[int], Awaitable[_T]],
    *,
    task_name: str,
    retries: int,
) -> _T | None:
    """Retry transient Harbor failures with a fresh sidecar trajectory."""

    if retries < 0:
        raise ValueError("rollout retries must be non-negative")
    for attempt in range(retries + 1):
        trial_events.record_trial_attempt()
        try:
            return await operation(attempt)
        except asyncio.CancelledError:
            raise
        except RecoverableRolloutError as exc:
            reason = getattr(exc, "reason", None) or "recoverable_rollout_error"
            if attempt == retries:
                trial_events.record_trial_discarded(reason)
                logger.warning(
                    "Discarding Harbor task %s after %d attempts: %s",
                    task_name,
                    retries + 1,
                    exc,
                )
                return None
            trial_events.record_trial_retry(reason)
            delay = _RETRY_DELAY_SECONDS * (attempt + 1)
            logger.warning(
                "Harbor task %s failed transiently (attempt %d/%d); "
                "retrying with a fresh trajectory in %ds: %s",
                task_name,
                attempt + 1,
                retries + 1,
                delay,
                exc,
            )
            await asyncio.sleep(delay)
        except Exception as exc:  # noqa: BLE001 - rollout isolation boundary
            trial_events.record_trial_discarded(type(exc).__name__)
            logger.warning(
                "Discarding non-recoverable Harbor task %s: %s: %s",
                task_name,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            return None
    return None


def _attach_trial_phase_metrics(
    rollout: RolloutRun,
    phase_timings: Mapping[str, float],
) -> None:
    """Record Harbor trial phase brackets as one-sample distributions.

    The step-level merge aggregates count/sum/min/max across trials, so each
    trial contributes count=1 with the bracket duration as sum/min/max.
    """
    if not phase_timings:
        return
    tito_metrics = rollout.metadata.get("tito_metrics")
    if not isinstance(tito_metrics, dict):
        raise ValueError("rollout metadata tito_metrics must be a mapping")
    for phase, seconds in phase_timings.items():
        base = f"tito/trial/{phase}"
        value = float(seconds)
        tito_metrics[f"{base}_count"] = 1.0
        tito_metrics[f"{base}_sum"] = value
        tito_metrics[f"{base}_min"] = value
        tito_metrics[f"{base}_max"] = value


def materialize_harbor_trajectory(
    outcome: HarborTrialOutcome,
    *,
    max_context_tokens: int,
    debug_enabled: bool,
    harness_abandoned_turn_ids: Set[str] | None = None,
) -> RolloutRun | None:
    """Convert a validated Harbor artifact without leaking failures to the producer."""

    artifact = outcome.trajectory_artifact
    if artifact is None:
        raise RecoverableRolloutError(
            "Harbor trial returned no validated TITO artifact"
        )
    if outcome.reward is None:
        return None
    try:
        rollout = materialize_tito_trajectory(
            artifact,
            reward=outcome.reward,
            harness_abandoned_turn_ids=harness_abandoned_turn_ids,
            max_context_tokens=max_context_tokens,
            debug_enabled=debug_enabled,
        )
    except RecoverableRolloutError:
        raise
    except Exception as exc:
        raise RecoverableRolloutError(
            f"TITO trajectory materialization failed: {type(exc).__name__}: {exc}"
        ) from exc
    if not rollout.segments:
        raise RecoverableRolloutError(
            "Harbor trial produced no policy-visible trainable segment"
        )
    _attach_trial_phase_metrics(rollout, outcome.phase_timings)
    return rollout


@dataclass(frozen=True)
class HarborHarness:
    """The pinned contract one agent adapter contributes to a Harbor rollout."""

    name: str
    """Harness identity in trajectory metadata and ``harbor-<name>`` run ids."""
    agent_import_path: str
    agent_version: str
    agent_provider: str
    call_classifier: str
    """TITO call classifier mode used by the sidecar for this agent's traffic."""
    trial_kwargs: Mapping[str, Any] = field(default_factory=dict)
    """Extra ``run_harbor_trial`` keywords, such as ``agent_extra_kwargs``."""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    """Constant per-rollout metadata, such as the pinned harness version."""
    materialize: Callable[..., RolloutRun | None] = materialize_harbor_trajectory
    """Picklable ``(outcome, *, max_context_tokens, debug_enabled)`` converter.

    It runs in the artifact process while the trial directory still exists, so
    it may also read harness-owned event logs from ``outcome.trial_path``.
    """
    materialize_evaluation: Callable[[HarborTrialOutcome], RolloutRun | None] | None = (
        None
    )
    """Optional reward-only evaluation path that needs no trainable artifact."""


@dataclass(frozen=True)
class HarborSettings:
    """Run-level Harbor trial settings shared by every harness rollout."""

    tito: TITOSidecarSettings
    environment: str = "docker"
    """``docker`` or ``e2b``."""
    trial_config: str | Path | Mapping[str, Any] | None = None
    """Harbor ``TrialConfig`` YAML path or mapping."""
    trials_dir: str | Path | None = None
    """Retain trials here; ``None`` uses the trial config's ``trials_dir`` or
    a temporary directory per trial."""
    reward_key: str = "reward"
    rollout_retries: int = DEFAULT_ROLLOUT_RETRIES
    retry_include_exceptions: frozenset[str] = DEFAULT_HARBOR_RETRYABLE_EXCEPTIONS
    max_concurrent_trials: int = DEFAULT_MAX_CONCURRENT_TRIALS
    terminal_failure_reward: float | None = None
    """Reward for terminal agent failures without a verifier reward; ``None``
    retries and then discards them."""
    tool_timeout_seconds: int = DEFAULT_HARNESS_TOOL_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if self.environment not in {"docker", "e2b"}:
            raise ValueError("Harbor environment must be docker or e2b")
        if not self.reward_key:
            raise ValueError("Harbor reward_key must not be empty")
        if self.rollout_retries < 0:
            raise ValueError("rollout_retries must be >= 0")
        if self.max_concurrent_trials < 1:
            raise ValueError("max_concurrent_trials must be >= 1")
        if self.tool_timeout_seconds < 1:
            raise ValueError("tool_timeout_seconds must be >= 1")
        if self.terminal_failure_reward is not None and not math.isfinite(
            self.terminal_failure_reward
        ):
            raise ValueError("terminal_failure_reward must be finite")


class HarborRollout:
    """``rollout_fn`` that runs one Harbor trial per attempt for one harness.

    Each call is one logical rollout. Every attempt gets a fresh sidecar
    trajectory, Harbor environment and credential; transient failures retry up
    to ``settings.rollout_retries`` times and then return ``None`` so the loop
    drops the sample instead of training a synthetic reward.
    """

    def __init__(
        self, setup: RolloutSetup, harness: HarborHarness, settings: HarborSettings
    ) -> None:
        self._setup = setup
        self._harness = harness
        self._settings = settings
        self._run_prefix = "harbor-" + harness.name.replace("_", "-")
        self._trial_config = load_harbor_trial_config(settings.trial_config)
        self._retry_include_exceptions = validate_harbor_retry_exceptions(
            settings.retry_include_exceptions
        )
        self._trials_dir = resolve_trials_dir(settings.trials_dir, self._trial_config)
        self._trial_slots = asyncio.Semaphore(settings.max_concurrent_trials)
        self._context_limit = resolve_max_context_tokens(setup)
        self._output_limit = int(setup.sample_kwargs["max_tokens"])
        self._trial_start_interval_seconds = _DOCKER_START_INTERVAL_SECONDS
        self._trial_start_lock = asyncio.Lock()
        self._next_trial_start = 0.0
        self._sidecar_bundle = build_sidecar_bundle(
            setup, settings.tito.bundle_root, renderer_name=settings.tito.renderer_name
        )
        self._active_rollouts = ActiveRolloutTasks()
        self._artifact_processor = ArtifactProcessPool()

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
            task_config = task_config_from_row(sample_prompt)
            task_name = task_name_from_row(sample_prompt)
            group_id = f"{self._run_prefix}:{task_name}:{epoch}:{cursor_index}"
            prompt_hash = hashlib.sha256(
                task_initial_instruction(task_config).encode("utf-8")
            ).hexdigest()
            return await run_with_fresh_trajectory_retries(
                partial(
                    self._run_attempt,
                    task_config,
                    task_name=task_name,
                    group_id=group_id,
                    sample_index=sample_index,
                    evaluation=evaluation,
                    prompt_hash=prompt_hash,
                ),
                task_name=task_name,
                retries=self._settings.rollout_retries,
            )

    async def _run_attempt(
        self,
        task_config: Any,
        retry_index: int,
        *,
        task_name: str,
        group_id: str,
        sample_index: int,
        evaluation: bool,
        prompt_hash: str,
    ) -> RolloutRun | None:
        harness = self._harness
        settings = self._settings
        run_id = f"{group_id}:{sample_index}"
        launch_spec = launch_spec_json(
            build_launch_spec(
                self._setup,
                self._sidecar_bundle,
                settings.tito,
                call_classifier=harness.call_classifier,
                metadata={
                    "harness": harness.name,
                    "harbor_environment_type": settings.environment,
                    "task_name": task_name,
                    "run_id": run_id,
                    "rollout_group_id": group_id,
                    "rollout_member_index": sample_index,
                    "retry_index": retry_index,
                    "evaluation": evaluation,
                    "canonical_initial_prompt_hash": prompt_hash,
                },
            )
        )
        reward_only = evaluation and harness.materialize_evaluation is not None
        materializer = (
            harness.materialize_evaluation
            if reward_only
            else partial(
                harness.materialize,
                max_context_tokens=self._context_limit,
                debug_enabled=settings.tito.debug_enabled,
            )
        )
        with trial_workspace(
            self._trials_dir, prefix=f"{self._run_prefix}-tito-"
        ) as trial_root:
            outcome = await self._run_trial(
                task_name=task_name,
                retry_index=retry_index,
                task_config=task_config,
                inference_key=self._setup.api_key,
                run_id=f"{run_id}:retry-{retry_index}" if retry_index else run_id,
                harbor_environment=settings.environment,
                sidecar_bundle_path=self._sidecar_bundle.path,
                sidecar_launch_spec=launch_spec,
                trial_config=self._trial_config,
                trials_dir=trial_root,
                context_limit=self._context_limit,
                output_limit=self._output_limit,
                agent_import_path=harness.agent_import_path,
                agent_provider=harness.agent_provider,
                agent_version=harness.agent_version,
                reward_key=settings.reward_key,
                terminal_failure_reward=settings.terminal_failure_reward,
                tool_timeout_seconds=settings.tool_timeout_seconds,
                retry_include_exceptions=self._retry_include_exceptions,
                artifact_processor=None if reward_only else self._artifact_processor,
                materializer=materializer,
                require_trajectory_artifact=not reward_only,
                **harness.trial_kwargs,
            )
        rollout = outcome.rollout
        if rollout is None:
            logger.warning(
                "Harbor/%s retained an untrained rewardless trajectory for %s (%s)",
                harness.name,
                outcome.task_name,
                outcome.exception_type,
            )
            return None
        rollout.run_id = run_id
        rollout.metadata.update(
            {
                "task_name": outcome.task_name,
                "trial_name": outcome.trial_name,
                "harbor_rewards": outcome.rewards,
                "harbor_reward_key": settings.reward_key,
                "harbor_exception_type": outcome.exception_type,
                "harbor_environment_type": outcome.environment_type,
                "harness_tool_timeout_seconds": settings.tool_timeout_seconds,
                "tito_harness": harness.name,
                "tito_retry_index": retry_index,
                **harness.metadata,
            }
        )
        return rollout

    async def _run_trial(
        self, *, task_name: str, retry_index: int, **trial_kwargs: Any
    ) -> HarborTrialOutcome:
        async with self._trial_slots:
            if self._settings.environment == "docker":
                await self._pace_trial_start()
            with phase_span(
                "harbor_trial",
                category="rollout",
                attributes={"task_name": task_name, "retry_index": retry_index},
            ):
                return await run_harbor_trial(**trial_kwargs)

    async def _pace_trial_start(self) -> None:
        async with self._trial_start_lock:
            now = time.monotonic()
            scheduled = max(now, self._next_trial_start)
            self._next_trial_start = scheduled + self._trial_start_interval_seconds
        delay = scheduled - now
        if delay > 0:
            await asyncio.sleep(delay)

    async def aclose(self) -> None:
        try:
            await self._active_rollouts.cancel_and_wait()
        finally:
            await self._artifact_processor.aclose()


__all__ = [
    "DEFAULT_MAX_CONCURRENT_TRIALS",
    "DEFAULT_ROLLOUT_RETRIES",
    "HarborHarness",
    "HarborRollout",
    "HarborSettings",
    "materialize_harbor_trajectory",
    "resolve_trials_dir",
    "run_with_fresh_trajectory_retries",
    "trial_workspace",
]
