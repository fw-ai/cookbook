"""Shared Harbor lifecycle mechanics for environment-local TITO rollouts."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from collections.abc import Awaitable, Callable, Iterator, Mapping, Set
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar

from training.examples.rl.harbor.tito.trial import HarborTrialOutcome
from training.utils.rl.async_rl.errors import RecoverableRolloutError
from training.utils.rl.rollout import RolloutRun
from training.utils.rl.rollout.tito import materialize_tito_trajectory

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONCURRENT_TRIALS = 24
DEFAULT_ROLLOUT_RETRIES = 3
_RETRY_DELAY_SECONDS = 15

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
        try:
            return await operation(attempt)
        except asyncio.CancelledError:
            raise
        except RecoverableRolloutError as exc:
            if attempt == retries:
                logger.warning(
                    "Discarding Harbor task %s after %d attempts: %s",
                    task_name,
                    retries + 1,
                    exc,
                )
                return None
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
            logger.warning(
                "Discarding non-recoverable Harbor task %s: %s: %s",
                task_name,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            return None
    return None


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
    return rollout


__all__ = [
    "DEFAULT_MAX_CONCURRENT_TRIALS",
    "DEFAULT_ROLLOUT_RETRIES",
    "materialize_harbor_trajectory",
    "resolve_trials_dir",
    "run_with_fresh_trajectory_retries",
    "trial_workspace",
]
