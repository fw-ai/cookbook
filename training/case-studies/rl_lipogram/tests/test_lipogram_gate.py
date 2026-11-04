"""Offline tests for the retention scorer and the pass/fail gate (the review's three P1 reproductions)."""

from pathlib import Path as _TestPath
import sys as _test_sys
_test_sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1]))

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate_retention import build_items, compatible, is_correct  # noqa: E402

HERE = Path(__file__).parent.parent


def test_retention_rejects_suffix_sign_and_negation_matches():
    assert not is_correct("120", "20")
    assert not is_correct("-42", "42")
    assert not is_correct("Not Paris", "Paris")
    assert is_correct("The answer is 1,234.", "1234")
    assert is_correct("paris", "Paris")


def test_retention_items_split_by_constraint_compatibility():
    items = build_items(200, 0)
    incompatible = [i for i in items if not compatible(i["a"])]
    assert 0 < len(incompatible) < len(items)
    assert all("e" in i["a"].lower() for i in incompatible)


def _gate(tmp_path, summary, extra=()):
    p = tmp_path / "summary.json"
    p.write_text(json.dumps(summary))
    r = subprocess.run([sys.executable, str(HERE / "check_bar.py"), str(p), *extra], capture_output=True, text=True)
    return r.returncode, r.stdout


def test_gate_refuses_max_effort_missing_metric_and_missing_retention(tmp_path):
    rc, out = _gate(tmp_path, {"effort": "max", "arms": {"adapter": {"sampled": {"e_free": 0.99}, "greedy": {"e_free": 0.99}}}})
    assert rc == 1 and "BAR MET" not in out and "INVALID" in out


def test_gate_passes_only_a_complete_low_effort_summary(tmp_path):
    good = {"effort": "low", "retention": {"base": 0.95, "adapter": 0.94},
            "arms": {"adapter": {"sampled": {"success": 0.92}, "greedy": {"success": 0.96}}}}
    rc, out = _gate(tmp_path, good)
    assert rc == 0 and "BAR MET" in out
    rc, out = _gate(tmp_path, good, ["--diagnostic"])
    assert rc == 1 and "BAR MET" not in out
    bad = json.loads(json.dumps(good)); bad["retention"]["adapter"] = 0.90
    assert _gate(tmp_path, bad)[0] == 1
