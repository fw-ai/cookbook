from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_trainer_delete_grace_period(monkeypatch):
    monkeypatch.setenv("FW_TRAINER_DELETE_GRACE_PERIOD_S", "0")
