"""Pinned Pi artifacts certified by the TITO cookbook adapter."""

from types import MappingProxyType

PINNED_PI_REVISION = "5cd93f688aaab89dbb6dfa4aca535f21796ae185"
PINNED_PI_VERSION = "0.84.2"
PI_HARBOR_IMPORT_PATH = "training.examples.harbor_rl.harnesses.pi.agent:ConfigurablePi"
# Harbor model-name provider for the sidecar-backed policy endpoint.
PI_AGENT_PROVIDER = "fireworks-tito"
PI_OPENAI_COMPAT = MappingProxyType(
    {
        "supportsStore": False,
        "supportsDeveloperRole": False,
        "supportsReasoningEffort": False,
        # GLM emits ``reasoning_content: ""`` on tool-call responses. Pi must
        # preserve that empty field when it replays the assistant message.
        "requiresReasoningContentOnAssistantMessages": True,
    }
)

__all__ = [
    "PI_AGENT_PROVIDER",
    "PINNED_PI_REVISION",
    "PINNED_PI_VERSION",
    "PI_HARBOR_IMPORT_PATH",
    "PI_OPENAI_COMPAT",
]
