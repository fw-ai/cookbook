from __future__ import annotations

import json

import training.examples.tools.client_phase_trace_demo as demo
from training.utils import phase_tracing


def test_demo_runs_bounded_real_recipe_and_writes_trace(monkeypatch, tmp_path):
    calls = []

    def fake_train(config, **kwargs):
        calls.append((config, kwargs))
        return {"steps": 1, "accepted_groups": 2}

    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-test")
    monkeypatch.setattr(demo, "train_deepmath", fake_train)
    phase_tracing._reset_phase_tracing_for_tests()
    try:
        trace_path, result = demo.run_demo(
            tmp_path / "trace.json",
            run_dir=tmp_path / "run",
            prompt_groups=2,
            group_size=4,
        )
    finally:
        phase_tracing._reset_phase_tracing_for_tests()

    config, kwargs = calls[0]
    assert result["steps"] == 1
    assert config.max_rows == config.prompt_groups_per_step == 2
    assert config.completions_per_prompt == 4
    assert config.dataset == str(demo.SAMPLE_DATASET)
    assert config.metrics_file == str((tmp_path / "run" / "metrics.jsonl").resolve())
    assert kwargs["eval_rows"] == 0

    payload = json.loads((tmp_path / "trace.json").read_text())
    names = {
        event["name"] for event in payload["traceEvents"] if event.get("ph") == "X"
    }
    assert trace_path == str(tmp_path / "trace.json")
    assert "serverless_deepmath_training" in names
