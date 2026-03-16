"""Shared types for RL loss registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

BuiltinConfigBuilder = Callable[..., tuple[str, dict[str, Any]]]
LossFactory = Callable[..., Any]


@dataclass(frozen=True)
class LossSpec:
    """How one policy loss is constructed in custom and builtin paths."""

    name: str
    make_loss_fn: LossFactory
    builtin_config_builder: BuiltinConfigBuilder | None
