"""Structured-message history matching for multi-turn agent rollouts.

A black-box agent (or a driven multi-turn loop) issues one inference request
per turn.  To reconstruct a trainable trajectory we must decide, for each
request, whether it **continues** the chain we have been recording
(``APPEND``), **starts** a fresh one (``NEW``), or **diverges** from it
(``WIPE`` -- e.g. the agent rewrote or replaced its history).  That decision
uses per-message content hashes, which tolerate tokenization drift while
capturing whether the harness appended a turn or replaced history. This
mirrors the structured-message prefix matching used by Slime's coding-agent
adapter.

This decides *routing* only. Token-level stitching of the recorded turns into
a training segment -- which is always token-exact -- lives in
:mod:`training.utils.rl.agent.trajectory`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from enum import Enum
from typing import Any, Hashable


class TurnKind(Enum):
    """How an incoming turn relates to the chain's recorded prefix."""

    NEW = "new"  # the chain has no recorded turn yet
    APPEND = "append"  # the request extends the recorded prefix
    WIPE = "wipe"  # the request diverges before consuming the prefix


@dataclasses.dataclass(frozen=True)
class TurnDecision:
    """Result of classifying one turn against a chain."""

    kind: TurnKind
    matched_prefix_len: int  # leading fingerprint units that matched the chain


@dataclasses.dataclass(frozen=True)
class TurnRequest:
    """Inputs available when classifying an incoming turn.

    The caller is responsible for stripping volatile, non-semantic metadata
    (e.g. Anthropic ``cache_control``) from ``messages`` / ``system`` before
    building this request.
    """

    messages: list[Any] = dataclasses.field(default_factory=list)
    system: Any = None


def common_prefix_len(stored: list[Hashable], incoming: list[Hashable]) -> int:
    """Length of the longest common prefix of two unit sequences."""
    limit = min(len(stored), len(incoming))
    index = 0
    while index < limit and stored[index] == incoming[index]:
        index += 1
    return index


def classify(
    stored_units: list[Hashable] | None, incoming_units: list[Hashable]
) -> TurnDecision:
    """Classify ``incoming_units`` against the chain's ``stored_units``.

    Strategy-independent: ``NEW`` when the chain has no recorded turn, ``APPEND``
    when the incoming fingerprint extends the stored one (the stored sequence is
    a full prefix), ``WIPE`` when it diverges before consuming the stored prefix.
    """
    if not stored_units:
        return TurnDecision(TurnKind.NEW, 0)
    matched = common_prefix_len(stored_units, incoming_units)
    if matched == len(stored_units):
        return TurnDecision(TurnKind.APPEND, matched)
    return TurnDecision(TurnKind.WIPE, matched)


def _stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class MessageHashFingerprinter:
    """Prefix-match by per-message content hash.

    Units are ``[hash(system)] + [hash(message) for message in messages]`` so a
    system change is a divergence at position 0.  The caller must strip volatile
    metadata before building the request (see :class:`TurnRequest`).
    """

    def units(self, request: TurnRequest) -> list[Hashable]:
        units: list[Hashable] = [_stable_hash(request.system)]
        units.extend(_stable_hash(message) for message in request.messages)
        return units
