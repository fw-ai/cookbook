from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_trainer_cancel_grace_period(monkeypatch):
    monkeypatch.setenv("FW_TRAINER_CANCEL_GRACE_PERIOD_S", "0")
