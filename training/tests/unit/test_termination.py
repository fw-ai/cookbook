from pathlib import Path

import pytest

from training.utils.termination import TerminatedBySignal

_RECIPES = Path(__file__).resolve().parents[2] / "recipes"


def test_signal_termination_unwinds_like_system_exit():
    with pytest.raises(SystemExit) as raised:
        raise TerminatedBySignal("SIGTERM")
    assert isinstance(raised.value, TerminatedBySignal)
    assert (raised.value.code, raised.value.signal_name) == (
        "Terminated by SIGTERM",
        "SIGTERM",
    )


def test_recipe_signal_handlers_raise_the_typed_exit():
    handlers = [
        path for path in _RECIPES.rglob("*.py") if "signal.SIGINT" in path.read_text()
    ]
    assert handlers
    for path in handlers:
        source = path.read_text()
        assert 'SystemExit(f"Terminated by' not in source, path
        assert "raise TerminatedBySignal(" in source, path
