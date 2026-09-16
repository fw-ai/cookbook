from __future__ import annotations

import json

import training.examples.tools.client_phase_trace_demo as demo
from training.utils import phase_tracing


def test_demo_runs_bounded_real_recipe_and_writes_trace(monkeypatch, tmp_path):
    configs = []

    class FakeCountdownRun:
        def __init__(self, config):
            configs.append(config)

        def run(self):
            return [
                {
                    "train/trained": True,
                    "train/loss": 0.25,
                    "rollout/raw_samples": 8,
                }
            ]

    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-test")
    monkeypatch.setattr(demo, "ServerlessCountdownRL", FakeCountdownRun)
    phase_tracing._reset_phase_tracing_for_tests()
    try:
        trace_path, metrics = demo.run_demo(
            tmp_path / "trace.json",
            run_dir=tmp_path / "run",
            prompt_groups=2,
            group_size=4,
        )
    finally:
        phase_tracing._reset_phase_tracing_for_tests()

    assert metrics["train/trained"] is True
    assert configs[0].steps == 1
    assert configs[0].prompt_groups_per_step == 2
    assert configs[0].group_size == 4
    assert configs[0].eval_at_start is False
    assert configs[0].eval_at_end is False
    assert configs[0].output_model_id == ""

    payload = json.loads((tmp_path / "trace.json").read_text())
    names = {
        event["name"] for event in payload["traceEvents"] if event.get("ph") == "X"
    }
    assert trace_path == str(tmp_path / "trace.json")
    assert "serverless_countdown_training" in names
