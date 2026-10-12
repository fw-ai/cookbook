"""Run Harbor/Pi through an environment-local TITO sidecar."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING

from training.examples.harbor_rl.tito.rollout import (
    HarborHarness,
    HarborRollout,
    HarborSettings,
    materialize_harbor_trajectory,
)
from training.examples.harbor_rl.tito.trial import HarborTrialOutcome
from training.utils.rl.async_rl.errors import RecoverableRolloutError
from training.utils.rl.rollout import RolloutRun, RolloutSample

from .artifacts import tool_timeout_count
from .constants import (
    PI_AGENT_PROVIDER,
    PI_HARBOR_IMPORT_PATH,
    PINNED_PI_REVISION,
    PINNED_PI_VERSION,
)

if TYPE_CHECKING:
    from training.utils.rl.rollout.setup import RolloutFn, RolloutSetup

_TITO_RESPONSE_ID_PREFIX = "chatcmpl-"
_TOOL_PROFILES = frozenset({"coding", "textworld"})


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


def _materialize_pi_trajectory(
    outcome: HarborTrialOutcome,
    *,
    max_context_tokens: int,
    debug_enabled: bool,
) -> RolloutRun | None:
    """Reconcile Pi visibility and materialize inside the artifact process."""
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
        max_context_tokens=max_context_tokens,
        debug_enabled=debug_enabled,
        harness_abandoned_turn_ids=abandoned_turn_ids,
    )
    if rollout is not None:
        rollout.metadata.update(
            pi_abandoned_turn_count=len(abandoned_turn_ids),
            harness_tool_timeout_count=tool_timeout_count(outcome.trial_path),
        )
    return rollout


def _materialize_pi_evaluation(
    outcome: HarborTrialOutcome,
) -> RolloutRun | None:
    """Keep verifier reward evaluation independent of trainable artifacts."""
    if outcome.reward is None:
        return None
    return RolloutRun(
        segments=[
            RolloutSample(
                tokens=[],
                logprobs=[],
                loss_mask=[],
                reward=outcome.reward,
            )
        ],
        run_id=outcome.trial_name,
        metadata={"evaluation_reward_only": True},
    )


def make_rollout_fn(
    setup: RolloutSetup, *, harbor: HarborSettings, tool_profile: str = "coding"
) -> RolloutFn:
    """Create one logical Harbor/Pi rollout per call.

    ``tool_profile`` selects ``coding`` or the ``textworld`` action-only tool
    set. Bind the settings with ``functools.partial`` to get a factory.
    """
    if tool_profile not in _TOOL_PROFILES:
        raise ValueError("Pi tool_profile must be coding or textworld")
    return HarborRollout(
        setup,
        HarborHarness(
            name="pi",
            agent_import_path=PI_HARBOR_IMPORT_PATH,
            agent_version=PINNED_PI_VERSION,
            agent_provider=PI_AGENT_PROVIDER,
            call_classifier="adapter_metadata",
            trial_kwargs={"tool_profile": tool_profile},
            metadata={
                "pi_revision": PINNED_PI_REVISION,
                "pi_version": PINNED_PI_VERSION,
            },
            materialize=_materialize_pi_trajectory,
            materialize_evaluation=_materialize_pi_evaluation,
        ),
        harbor,
    )


__all__ = ["PINNED_PI_REVISION", "PINNED_PI_VERSION", "make_rollout_fn"]
