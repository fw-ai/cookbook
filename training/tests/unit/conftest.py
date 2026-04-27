"""Shared fixtures for the unit-test suite.

Tightens ``TrainingCheckpoints`` polling so loop tests don't burn the
production-friendly 90s/15s/3s defaults under pytest. The focused
``test_checkpoints.py`` suite already constructs ``TrainingCheckpoints``
with explicit tight timeouts; this fixture covers the loop tests that
rely on the recipe constructing ``TrainingCheckpoints`` itself with
default timeouts.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _fast_training_checkpoints(monkeypatch):
    from training.utils.checkpoints import TrainingCheckpoints

    original_init = TrainingCheckpoints.__init__

    def fast_init(self, *args, **kwargs):
        kwargs.setdefault("save_appear_timeout_s", 5.0)
        kwargs.setdefault("save_stabilize_s", 0.0)
        kwargs.setdefault("save_poll_s", 0.01)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(TrainingCheckpoints, "__init__", fast_init)
