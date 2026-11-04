"""The recorded playground stays offline until live inference is configured."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def demo(monkeypatch):
    for name in ("LIPO_ADAPTER", "LIPO_SFT", "LIPO_ALLOW_LIVE", "LIPO_OFFLINE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-only")
    path = Path(__file__).resolve().parents[1] / "walkthrough" / "demo_helpers.py"
    spec = importlib.util.spec_from_file_location("lipogram_demo_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_card", lambda *args: None)
    return module


def test_a_key_alone_does_not_enable_inference(demo, monkeypatch):
    monkeypatch.setattr(demo, "_load_recorded", lambda: [])
    monkeypatch.setattr(demo, "_chat", lambda *args: pytest.fail("unexpected live call"))
    assert demo.ask_lipogram("A new prompt") == []


def test_live_adapter_requires_an_explicit_model(demo, monkeypatch):
    monkeypatch.setenv("LIPO_ALLOW_LIVE", "1")
    monkeypatch.setattr(demo, "_chat", lambda *args: pytest.fail("unexpected live call"))
    with pytest.raises(ValueError, match="LIPO_ADAPTER"):
        demo.ask_lipogram("A new prompt")


def test_live_base_comparison_does_not_require_an_adapter(demo, monkeypatch):
    monkeypatch.setenv("LIPO_ALLOW_LIVE", "1")
    calls = []

    def chat(model, *args):
        calls.append(model)
        return "Bright sun"

    monkeypatch.setattr(demo, "_chat", chat)
    rows = demo.ask_lipogram("A new prompt", arms=["Base, no instruction"])
    assert calls == [demo.BASE]
    assert rows[0]["response"] == "Bright sun"


def test_unconfigured_optional_sft_is_not_called(demo, monkeypatch):
    monkeypatch.setenv("LIPO_ALLOW_LIVE", "1")
    monkeypatch.setattr(demo, "ARMS", {**demo.ARMS, "RL adapter s100, no instruction": ("configured-adapter", None)})
    calls = []
    monkeypatch.setattr(demo, "_chat", lambda model, *args: calls.append(model) or "Bright sun")
    demo.ask_lipogram("A new prompt")
    assert "configured-adapter" in calls
    assert "" not in calls
    assert len(calls) == 3
