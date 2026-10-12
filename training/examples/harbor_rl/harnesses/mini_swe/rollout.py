"""Run Harbor/Mini-SWE through an environment-local TITO sidecar."""

from __future__ import annotations

from typing import TYPE_CHECKING

from training.examples.harbor_rl.harnesses.mini_swe.constants import (
    MINI_SWE_HARBOR_IMPORT_PATH,
    PINNED_MINI_SWE_VERSION,
)
from training.examples.harbor_rl.tito.rollout import (
    HarborHarness,
    HarborRollout,
    HarborSettings,
)

if TYPE_CHECKING:
    from training.utils.rl.rollout.setup import RolloutFn, RolloutSetup


def make_rollout_fn(setup: RolloutSetup, *, harbor: HarborSettings) -> RolloutFn:
    """Create one logical Harbor/Mini-SWE rollout per call.

    Bind the settings with ``functools.partial`` to get a factory.
    """
    return HarborRollout(
        setup,
        HarborHarness(
            name="mini_swe",
            agent_import_path=MINI_SWE_HARBOR_IMPORT_PATH,
            agent_version=PINNED_MINI_SWE_VERSION,
            agent_provider="openai",
            call_classifier="all_policy",
            metadata={"mini_swe_version": PINNED_MINI_SWE_VERSION},
        ),
        harbor,
    )


__all__ = ["PINNED_MINI_SWE_VERSION", "make_rollout_fn"]
