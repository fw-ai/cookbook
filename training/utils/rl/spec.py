"""Shared types for RL loss registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

BuiltinConfigBuilder = Callable[..., tuple[str, dict[str, Any]]]
ClientLossFactory = Callable[..., Any]


@dataclass(frozen=True)
class LossSpec:
    """How one policy loss is registered for client-side and builtin paths.

    ``builtin_config_builder=None`` means the loss is client-side-only and
    should always execute through ``forward_backward_custom(...)``.
    """

    name: str
    client_loss_factory: ClientLossFactory
    builtin_config_builder: BuiltinConfigBuilder | None = None
