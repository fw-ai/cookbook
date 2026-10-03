from __future__ import annotations

import importlib
import json


def test_prepare_data_writes_expected_jsonl(tmp_path, monkeypatch):
    import training.examples.rl.deepmath.prepare_data as module

    module = importlib.reload(module)
    output_path = tmp_path / "dataset.jsonl"
    fake_rows = [
        {"question": "What is 1+1?", "final_answer": "2"},
        {"question": "What is 2+2?", "final_answer": "4"},
    ]

    monkeypatch.setattr(module, "OUTPUT_PATH", str(output_path))
    monkeypatch.setattr(module, "load_dataset", lambda *args, **kwargs: fake_rows)

    module.main([])

    written = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert written == [
        {
            "messages": [
                {"role": "system", "content": module.SYSTEM_PROMPT},
                {"role": "user", "content": "What is 1+1?"},
            ],
            "ground_truth": "2",
        },
        {
            "messages": [
                {"role": "system", "content": module.SYSTEM_PROMPT},
                {"role": "user", "content": "What is 2+2?"},
            ],
            "ground_truth": "4",
        },
    ]


def test_prepare_data_output_flag_overrides_default_path(tmp_path, monkeypatch):
    import training.examples.rl.deepmath.prepare_data as module

    module = importlib.reload(module)
    output_path = tmp_path / "nested" / "deepmath.jsonl"
    monkeypatch.setattr(module, "OUTPUT_PATH", str(tmp_path / "unused.jsonl"))
    monkeypatch.setattr(
        module,
        "load_dataset",
        lambda *args, **kwargs: [{"question": "What is 1+1?", "final_answer": "2"}],
    )

    module.main(["--output", str(output_path)])

    assert [json.loads(line)["ground_truth"] for line in output_path.read_text().splitlines()] == ["2"]
    assert not (tmp_path / "unused.jsonl").exists()
