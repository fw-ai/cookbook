"""Score one MiMo music completion.

The policy writes ABC notation. The MiMo scorer renders it with ``abc2midi``
and compares MIDI features to a human percentile table. A missing ``abc2midi``
is an infrastructure failure and is raised, not scored as zero.
"""

from __future__ import annotations

from training.examples.rl.harbor.datasets.mimo.music_scorer import (
    Abc2MidiMissing,
    compute_score,
)

__all__ = ["Abc2MidiMissing", "score_music"]


def score_music(completion: str) -> float:
    """Return a reward in ``[0, 1]``. Raises ``Abc2MidiMissing`` when the binary is absent."""
    return float(compute_score("music", completion))
