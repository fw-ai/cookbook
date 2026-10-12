"""Run Harbor/mimoagent through an environment-local TITO sidecar.

mimoagent is Xiaomi MiMo's edited fork of mini-swe-agent 1.9.0 (MIT);
see mimoagent/constants.py for the pinned commit and attribution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from training.examples.harbor_rl.harnesses.mimoagent.constants import (
    MIMOAGENT_HARBOR_IMPORT_PATH,
    PINNED_MIMOAGENT_VERSION,
)
from training.examples.harbor_rl.tito.rollout import (
    HarborHarness,
    HarborRollout,
    HarborSettings,
)

if TYPE_CHECKING:
    from training.utils.rl.rollout.setup import RolloutFn, RolloutSetup

_ARMS = frozenset({"bash", "cc"})


def make_rollout_fn(
    setup: RolloutSetup, *, harbor: HarborSettings, arm: str = "bash"
) -> RolloutFn:
    """Create one logical Harbor/mimoagent rollout per call.

    ``arm`` selects the ``bash`` or ``cc`` (mini-claude-code) agent arm. Bind
    the settings with ``functools.partial`` to get a factory.
    """
    if arm not in _ARMS:
        raise ValueError("mimoagent arm must be bash or cc")
    return HarborRollout(
        setup,
        HarborHarness(
            name="mimoagent",
            agent_import_path=MIMOAGENT_HARBOR_IMPORT_PATH,
            agent_version=PINNED_MIMOAGENT_VERSION,
            agent_provider="openai",
            call_classifier="all_policy",
            trial_kwargs={"agent_extra_kwargs": {"arm": arm}},
            metadata={
                "mimoagent_version": PINNED_MIMOAGENT_VERSION,
                "mimoagent_arm": arm,
            },
        ),
        harbor,
    )


__all__ = ["PINNED_MIMOAGENT_VERSION", "make_rollout_fn"]
