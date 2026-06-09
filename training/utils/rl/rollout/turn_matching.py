"""Turn-matching strategies for multi-turn / black-box agent rollouts.

A black-box agent (or a driven multi-turn loop) issues one inference request
per turn.  To reconstruct a trainable trajectory we must decide, for each
request, whether it **continues** the chain we have been recording
(``APPEND``), **starts** a fresh one (``NEW``), or **diverges** from it
(``WIPE`` -- e.g. the agent compacted or rewrote its history).  That decision
is a prefix match between the request and the chain's last accepted turn; the
only thing that varies is *which units we prefix-match*:

* :class:`MessageHashFingerprinter` (default) matches per-message content
  hashes -- tolerant of tokenization drift, captures the harness's intent (did
  it append a turn or compact?).  Used by slime and our coding-agent shim.
* :class:`TokenPrefixFingerprinter` matches rendered token ids -- strict: any
  token shift (cache-control, re-tokenization, tool-schema rewrite) is a
  divergence.  Polar-style.

:func:`classify` is identical for both strategies; the strategy *is* the
fingerprinter, switchable via :func:`make_fingerprinter`.  This decides
*routing* only.  Token-level stitching of the recorded turns into a training
segment -- which is always token-exact regardless of strategy -- lives in
:mod:`training.utils.rl.rollout.agent_trajectory`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from enum import Enum
from typing import Any, Hashable, Protocol

DEFAULT_TURN_MATCHING = "message_hash"


class TurnKind(Enum):
    """How an incoming turn relates to the chain's recorded prefix."""

    NEW = "new"        # the chain has no recorded turn yet
    APPEND = "append"  # the request extends the recorded prefix
    WIPE = "wipe"      # the request diverges before consuming the prefix


@dataclasses.dataclass(frozen=True)
class TurnDecision:
    """Result of classifying one turn against a chain."""

    kind: TurnKind
    matched_prefix_len: int  # leading fingerprint units that matched the chain


@dataclasses.dataclass(frozen=True)
class TurnRequest:
    """Inputs available when classifying an incoming turn.

    Carries both the structured ``messages`` (+ ``system``) and the rendered
    ``prompt_ids`` so either strategy reads what it needs; the field the active
    strategy ignores may be left empty.  The caller is responsible for
    stripping volatile, non-semantic metadata (e.g. Anthropic ``cache_control``)
    from ``messages`` / ``system`` before building this request.
    """

    messages: list[Any] = dataclasses.field(default_factory=list)
    system: Any = None
    prompt_ids: list[int] = dataclasses.field(default_factory=list)


class TurnFingerprinter(Protocol):
    """Maps a :class:`TurnRequest` to a sequence of comparable units."""

    name: str

    def units(self, request: TurnRequest) -> list[Hashable]:
        ...


def common_prefix_len(stored: list[Hashable], incoming: list[Hashable]) -> int:
    """Length of the longest common prefix of two unit sequences."""
    limit = min(len(stored), len(incoming))
    index = 0
    while index < limit and stored[index] == incoming[index]:
        index += 1
    return index


def classify(stored_units: list[Hashable] | None, incoming_units: list[Hashable]) -> TurnDecision:
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
    """Prefix-match by per-message content hash (drift-tolerant; default).

    Units are ``[hash(system)] + [hash(message) for message in messages]`` so a
    system change is a divergence at position 0.  The caller must strip volatile
    metadata before building the request (see :class:`TurnRequest`).
    """

    name = "message_hash"

    def units(self, request: TurnRequest) -> list[Hashable]:
        units: list[Hashable] = [_stable_hash(request.system)]
        units.extend(_stable_hash(message) for message in request.messages)
        return units


class TokenPrefixFingerprinter:
    """Prefix-match by rendered token ids (strict; Polar-style).

    ``request.prompt_ids`` must be the prompt rendered for this turn; any token
    shift versus the prior turn (re-tokenization, cache-control, tool-schema
    rewrite) is classified as a divergence.
    """

    name = "token_prefix"

    def units(self, request: TurnRequest) -> list[Hashable]:
        return list(request.prompt_ids)


_FINGERPRINTERS: dict[str, type[TurnFingerprinter]] = {
    MessageHashFingerprinter.name: MessageHashFingerprinter,
    TokenPrefixFingerprinter.name: TokenPrefixFingerprinter,
}


def make_fingerprinter(strategy: str = DEFAULT_TURN_MATCHING) -> TurnFingerprinter:
    """Build the turn-matching fingerprinter for ``strategy``.

    Raises ``ValueError`` on an unknown strategy (a config error at a system
    boundary, not a programmer invariant).
    """
    if strategy not in _FINGERPRINTERS:
        raise ValueError(
            f"unknown turn-matching strategy {strategy!r}; "
            f"expected one of {sorted(_FINGERPRINTERS)}"
        )
    return _FINGERPRINTERS[strategy]()
