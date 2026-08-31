"""Pinned OpenCode harness contract used by preparation and rollout code."""

DEFAULT_OPENCODE_VERSION = "1.18.8"
OPENCODE_HARBOR_IMPORT_PATH = (
    "training.examples.rl.harbor.opencode.agent:ConfigurableOpenCode"
)

__all__ = ["DEFAULT_OPENCODE_VERSION", "OPENCODE_HARBOR_IMPORT_PATH"]
