"""MiMo-V2.6-RL-oss source adapter: everything MiMo lives under this package.

- ``tasks``: materialize one dataset row as a Harbor task directory
  (``write_harbor_task``) for code, terminal_bench, cyber (ARVO), webdev,
  and general_agent rows.
- ``music``: score a music completion (``score_music``); music is not a
  Harbor task.
- ``assets``: scripts copied into task images or ``tests/`` directories.
- ``music_scorer``: the stdlib ABC/MIDI scorer behind ``music``.
"""

from training.examples.rl.harbor.datasets.mimo.music import Abc2MidiMissing, score_music
from training.examples.rl.harbor.datasets.mimo.tasks import (
    unwrap_instance,
    write_harbor_task,
)

__all__ = ["Abc2MidiMissing", "score_music", "unwrap_instance", "write_harbor_task"]
