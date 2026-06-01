from __future__ import annotations

import fcntl
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _disable_trainer_cancel_grace_period(monkeypatch):
    monkeypatch.setenv("FW_TRAINER_CANCEL_GRACE_PERIOD_S", "0")


@pytest.fixture(params=[0, 16], ids=["full", "lora"], scope="session")
def port_lora_rank(request) -> int:
    """Precision track parameter for the live SDK-managed port tests."""
    return int(request.param)


@dataclass
class PortTrackState:
    """Persistent state for sequential live port tests.

    The port plan intentionally reuses one trainer/deployment per precision
    track across DPO -> GRPO -> resume. Pytest may be invoked one file at a
    time during manual validation, so this stores only resource IDs and the
    shared log path in a small JSON file between invocations.
    """

    key: str
    path: Path

    @property
    def _lock_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".lock")

    def _load_all(self) -> dict:
        if not self.path.exists():
            return {}
        with self.path.open() as f:
            return json.load(f)

    def _write_all(self, state: dict) -> None:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=self.path.parent,
            prefix=f"{self.path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            json.dump(state, f, indent=2, sort_keys=True)
            temp_path = Path(f.name)
        temp_path.replace(self.path)

    def load(self) -> dict:
        return dict(self._load_all().get(self.key, {}))

    def update(self, **values: str | int | None) -> dict:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            all_state = self._load_all()
            state = dict(all_state.get(self.key, {}))
            for key, value in values.items():
                if value is None:
                    state.pop(key, None)
                else:
                    state[key] = value
            all_state[self.key] = state
            self._write_all(all_state)
        return state

    def clear(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            all_state = self._load_all()
            all_state.pop(self.key, None)
            self._write_all(all_state)


@pytest.fixture
def port_track_state(port_lora_rank) -> PortTrackState:
    state_path = Path(
        os.environ.get(
            "FIREWORKS_PORT_TRACK_STATE",
            os.path.join(tempfile.gettempdir(), "fireworks_cookbook_port_track_state.json"),
        )
    )
    return PortTrackState(key=f"lora_rank_{port_lora_rank}", path=state_path)
