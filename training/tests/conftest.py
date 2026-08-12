from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_trainer_cancel_grace_period(monkeypatch):
    monkeypatch.setenv("FW_TRAINER_CANCEL_GRACE_PERIOD_S", "0")


@pytest.fixture(params=[0, 16], ids=["full", "lora"], scope="session")
def port_lora_rank(request) -> int:
    """Precision track parameter for live SDK-managed e2e tests."""
    return int(request.param)
