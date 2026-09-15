import copy
import json

import pytest

from training.examples.rl.harbor.recipes.terminal_bench.summarize_health import (
    collect_steps,
    read_records,
    render,
    synchronized_step,
)


def record(step=9):
    return {
        "time": "2026-09-15T20:33:00+00:00",
        "metrics": {"train/step": step, "rollout/step": step,
                    "train/inference_k3": .005658, "perf/train_tokens_per_s": 27724},
        "hotload_peers": [
            {"identity": f"peer-{i}", "readiness": True,
             "current_snapshot_identity": f"step-{step}-abc"}
            for i in range(4)
        ],
    }


def test_complete_matching_snapshot_and_deduplication():
    first = record()
    later = copy.deepcopy(first)
    later["time"] = "2026-09-15T20:34:00+00:00"
    later["metrics"]["train/inference_k3"] = .006
    assert collect_steps([first, later], 4) == {9: later}
    table = render({9: later})
    assert "0.006000" in table and "27724" in table
    assert "—" in table  # Missing metrics aren't invented as zero.


@pytest.mark.parametrize("change", [
    lambda r: r["metrics"].update({"rollout/step": 8}),
    lambda r: r["metrics"].update({"train/step": None}),
    lambda r: r["metrics"].update({"train/step": True}),
    lambda r: r["metrics"].update({"train/step": 9.5, "rollout/step": 9.5}),
    lambda r: r["hotload_peers"].pop(),
    lambda r: r["hotload_peers"][0].update({"identity": "peer-1"}),
    lambda r: r["hotload_peers"][0].update({"readiness": False}),
    lambda r: r["hotload_peers"][0].update({"current_snapshot_identity": "step-8-abc"}),
    lambda r: r["hotload_peers"][0].update({"current_snapshot_identity": "step-9-other"}),
    lambda r: [p.update({"current_snapshot_identity": "step-90-abc"}) for p in r["hotload_peers"]],
])
def test_reject_unverified_observation(change):
    value = record()
    change(value)
    assert synchronized_step(value, 4) is None


def test_partial_live_write_is_ignored_but_complete_corrupt_line_is_not(tmp_path):
    path = tmp_path / "health.jsonl"
    value = record()
    path.write_text(json.dumps(value) + '\n{"unfinished":')
    assert list(read_records(path)) == [value]
    path.write_text('{"malformed":\n')
    with pytest.raises(json.JSONDecodeError):
        list(read_records(path))


def test_reject_invalid_expected_peer_count():
    with pytest.raises(ValueError):
        collect_steps([], 0)
