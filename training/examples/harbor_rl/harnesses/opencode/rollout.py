"""Run Harbor/OpenCode through an environment-local TITO sidecar."""

from __future__ import annotations

from typing import TYPE_CHECKING

from training.examples.harbor_rl.harnesses.opencode.artifacts import tool_timeout_count
from training.examples.harbor_rl.harnesses.opencode.constants import (
    DEFAULT_OPENCODE_VERSION,
    OPENCODE_HARBOR_IMPORT_PATH,
)
from training.examples.harbor_rl.tito.rollout import (
    HarborHarness,
    HarborRollout,
    HarborSettings,
    materialize_harbor_trajectory,
)
from training.examples.harbor_rl.tito.trial import HarborTrialOutcome
from training.utils.rl.rollout import RolloutRun

if TYPE_CHECKING:
    from training.utils.rl.rollout.setup import RolloutFn, RolloutSetup


def _materialize_opencode_trajectory(
    outcome: HarborTrialOutcome,
    *,
    max_context_tokens: int,
    debug_enabled: bool,
) -> RolloutRun | None:
    """Materialize the trajectory and count OpenCode bash tool timeouts."""
    rollout = materialize_harbor_trajectory(
        outcome,
        max_context_tokens=max_context_tokens,
        debug_enabled=debug_enabled,
    )
    if rollout is not None:
        rollout.metadata["harness_tool_timeout_count"] = tool_timeout_count(
            outcome.trial_path
        )
    return rollout


def make_rollout_fn(
    setup: RolloutSetup,
    *,
    harbor: HarborSettings,
    version: str = DEFAULT_OPENCODE_VERSION,
) -> RolloutFn:
    """Create one logical Harbor/OpenCode rollout per call.

    Bind the settings with ``functools.partial`` to get a factory.
    """
    if version != DEFAULT_OPENCODE_VERSION:
        raise ValueError(
            "TITO/OpenCode supports only the classifier-certified version "
            f"{DEFAULT_OPENCODE_VERSION}; got {version}"
        )
    return HarborRollout(
        setup,
        HarborHarness(
            name="opencode",
            agent_import_path=OPENCODE_HARBOR_IMPORT_PATH,
            agent_version=version,
            agent_provider="fireworks-rl",
            # Title/summary calls carry no tools and stay auxiliary.
            call_classifier="tools_present",
            materialize=_materialize_opencode_trajectory,
        ),
        harbor,
    )


__all__ = ["make_rollout_fn"]
