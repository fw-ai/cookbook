"""Pinned mimoagent harness contract.

mimoagent (https://github.com/XiaomiMiMo/mimoagent) is Xiaomi MiMo's agentic
rollout framework, an edited fork of mini-swe-agent 1.9.0
(https://github.com/SWE-agent/mini-swe-agent). Both are MIT-licensed:
mimoagent copyright (c) 2026 Xiaomi Corporation; mini-swe-agent copyright
(c) 2025 Kilian A. Lieret and Carlos E. Jimenez. This adapter installs
mimoagent from a pinned upstream commit at image build time and
redistributes no upstream source; the upstream LICENSE.md and NOTICE apply.
"""

# XiaomiMiMo/mimoagent, branch mimo-oss.
PINNED_MIMOAGENT_COMMIT = "467f0a19016f0ac4d63b8d17a1f0da9ba07f232c"
PINNED_MIMOAGENT_VERSION = "0.1.0"
MIMOAGENT_HARBOR_IMPORT_PATH = (
    "training.examples.rl.harbor.mimoagent.agent:ConfigurableMimoAgent"
)

__all__ = [
    "MIMOAGENT_HARBOR_IMPORT_PATH",
    "PINNED_MIMOAGENT_COMMIT",
    "PINNED_MIMOAGENT_VERSION",
]
