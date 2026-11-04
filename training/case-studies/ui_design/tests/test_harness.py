"""Offline tests for the measurement harness. No API calls; the browser tests use the local scaffold."""

from pathlib import Path as _TestPath
import sys as _test_sys
_test_sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1]))

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import judge  # noqa: E402
import render  # noqa: E402

CSS = "body{margin:0;font-family:sans-serif}.panel{padding:1rem;border:1px solid #333}"
TOGGLE = """import { useState } from "react";
export default function App() {
  const [open, setOpen] = useState(false);
  return (<main><h1>Granite Peak</h1><p>%s</p>
    <button onClick={() => setOpen(!open)}>Hours</button>
    {open && <div className="panel">Open daily 6am to 11pm</div>}
    <p>%s</p></main>);
}"""


def _render(tmp: Path, app: str, interactions, required=("Granite Peak",)):
    out = tmp / "r"
    args = [sys.executable, str(HERE / "render.py"), "--app", str(tmp / "App.jsx"), "--css", str(tmp / "s.css"), "--out", str(out),
            "--required-text", *required]
    (tmp / "App.jsx").write_text(app); (tmp / "s.css").write_text(CSS)
    if interactions is not None:
        (tmp / "i.json").write_text(json.dumps(interactions)); args += ["--interactions", str(tmp / "i.json")]
    subprocess.run(args, check=False, capture_output=True, text=True)
    return json.loads((out / "record.json").read_text())


@pytest.mark.skipif(not (HERE / "scaffold/node_modules").exists(), reason="scaffold not installed")
def test_toggle_must_change_visibility(tmp_path):
    # Real toggle: the panel appears on click and disappears on the second click.
    rec = _render(tmp_path, TOGGLE % ("Welcome to the gym.", "See you soon."), [{"click": "text=Hours", "expect_text": "6am to 11pm"}])
    assert all(i["ok"] for i in rec["checks"]["interactions"]) and len(rec["checks"]["interactions"]) == 2   # 1440 and 390
    # Vacuous toggle: the text is already on the page and the button does nothing. Must FAIL now.
    noop = TOGGLE.replace("{open && <div className=\"panel\">Open daily 6am to 11pm</div>}", "<div className=\"panel\">Open daily 6am to 11pm</div>")
    rec = _render(tmp_path, noop % ("x", "y"), [{"click": "text=Hours", "expect_text": "6am to 11pm"}])
    assert not any(i["ok"] for i in rec["checks"]["interactions"])


@pytest.mark.skipif(not (HERE / "scaffold/node_modules").exists(), reason="scaffold not installed")
def test_rerun_removes_old_screenshots_and_parse_failure_is_not_judgeable(tmp_path):
    rec = _render(tmp_path, TOGGLE % ("a", "b"), [])
    out = tmp_path / "r"
    assert (out / "shots/1440.png").exists()
    # a second run whose response has no contract blocks must leave NO screenshots behind
    (tmp_path / "bad.txt").write_text("Sorry, here is some prose and no code.")
    subprocess.run([sys.executable, str(HERE / "render.py"), "--from-response", str(tmp_path / "bad.txt"), "--out", str(out)], capture_output=True)
    rec = json.loads((out / "record.json").read_text())
    assert rec["model_failure"] and not rec["rendered"] and not (out / "shots").exists()
    with pytest.raises(ValueError):
        judge.render_record(out)


@pytest.mark.skipif(not (HERE / "scaffold/node_modules").exists(), reason="scaffold not installed")
def test_passes_functional_is_none_when_checks_incomplete(tmp_path):
    rec = _render(tmp_path, TOGGLE % ("a", "b"), None, required=())
    assert rec["checks_complete"] is False and rec["passes_functional"] is None and rec["renders_clean"] in (True, False)


def test_judge_rejects_invalid_verdicts():
    assert judge.validate({}) is None
    assert judge.validate({k: "A" for k in judge.CRITERIA}) is None            # overall missing
    assert judge.validate({**{k: "A" for k in judge.CRITERIA}, "overall": "maybe"}) is None
    assert judge.validate({**{k: "tie" for k in judge.CRITERIA}, "overall": "B"})["overall"] == "B"


def test_harness_error_is_distinct_class():
    assert issubclass(render.HarnessError, RuntimeError)


@pytest.mark.skipif(not (HERE / "scaffold/node_modules").exists(), reason="scaffold not installed")
def test_split_markup_in_panel_still_matches(tmp_path):
    app = TOGGLE.replace("Open daily 6am to 11pm", "Open daily 6am <span>to</span> 11pm")
    rec = _render(tmp_path, app % ("a", "b"), [{"click": "text=Hours", "expect_text": "6am to 11pm"}])
    assert all(i["ok"] for i in rec["checks"]["interactions"])


def test_all_viewports_failing_is_candidate_failure(tmp_path, monkeypatch):
    import render as r
    monkeypatch.setattr(r, "build", lambda *a, **k: {"ok": True, "seconds": 0.1, "stderr": "", "stdout": ""})
    monkeypatch.setattr(r, "screenshot_and_check", lambda *a, **k: {"viewports": {}, "console_errors": [], "blocked_requests": [], "interactions": [], "candidate_errors": ["1440: hang"]})
    (tmp_path / "App.jsx").write_text("x"); (tmp_path / "s.css").write_text("")
    out = tmp_path / "o"
    monkeypatch.setattr(sys, "argv", ["render.py", "--app", str(tmp_path / "App.jsx"), "--css", str(tmp_path / "s.css"), "--out", str(out), "--required-text", "x"])
    (tmp_path / "work").mkdir(exist_ok=True)
    r.main()
    rec = json.loads((out / "record.json").read_text())
    assert rec["model_failure"] and not rec["rendered"] and not rec["harness_error"]


def test_overrides_never_count_as_complete(tmp_path):
    if not (HERE / "scaffold/node_modules").exists():
        pytest.skip("scaffold not installed")
    rec = _render(tmp_path, TOGGLE % ("a", "b"), [], required=("Granite Peak",))
    assert rec["checks_complete"] is False and rec["passes_functional"] is None


@pytest.mark.skipif(not (HERE / "scaffold/node_modules").exists(), reason="scaffold not installed")
def test_required_item_split_across_a_row_counts(tmp_path):
    app = """export default function App(){return(<main><h1>Hinode</h1><ul>
      <li><b>Shio</b> <i>clear salt broth, chicken fat</i> <span>$16</span></li>
      <li><b>Shoyu</b> <i>dark with soy</i> <span>$16</span></li></ul>
      <p>Lots of other words here to pass the minimum text length requirement of the harness checks for this page ok.</p></main>)}"""
    rec = _render(tmp_path, app, [], required=("Shio $16", "Shoyu $16", "Miso $19"))
    v = rec["checks"]["viewports"]["1440"]
    assert v["missing_required_text"] == ["Miso $19"]
    assert set(v["missing_exact_phrase"]) == {"Shio $16", "Shoyu $16", "Miso $19"}


def test_invented_numbers():
    from brief_checks import invented_numbers
    brief = {"brief": "Fernery sells six plants. Free shipping over $60.", "required_content": ["Pothos $18", "Arrives in 3 days"], "content_sections": [], "interactions": [], "brand": "", "assets": []}
    assert invented_numbers("Pothos $18. Free shipping over $60, otherwise $8 shipping. Arrives in 3 days.", brief) == ["$8"]
    assert invented_numbers("Pothos $18 arrives in 3 days", brief) == []


@pytest.mark.skipif(not (HERE / "scaffold/node_modules").exists(), reason="scaffold not installed")
def test_tolerant_presence_bends_words_not_numbers(tmp_path):
    app = """export default function App(){return(<main><h1>Willow Creek Seed Library</h1>
      <p>Packet pickup on Tuesday and Thursday, 2pm-6pm, at the annex.</p>
      <p>Workshop fee is $4 for everyone this season, no exceptions at all, ever.</p>
      <p>Lots of other words here to pass the minimum text length requirement of the harness checks for this page ok.</p></main>)}"""
    rec = _render(tmp_path, app, [], required=("Tue and Thu pickup 2pm-6pm", "Workshop fee $3"))
    v = rec["checks"]["viewports"]["1440"]
    assert v["missing_required_text"] == ["Workshop fee $3"]      # reworded days pass; a changed price does not
    assert "Tue and Thu pickup 2pm-6pm" in v["missing_exact_phrase"]


def test_tolerant_in():
    from brief_checks import tolerant_in
    assert tolerant_in("Tue and Thu pickup 2pm-6pm", "Pickup on Tuesday and Thursday, 2pm-6pm, at the annex")
    assert not tolerant_in("Workshop fee $3", "Workshop fee is $4")
    assert not tolerant_in("Workshop fee $3", "Workshop fee is $39")
    assert tolerant_in("Open daily 6am to 11pm", "We are open daily, 6am to 11pm.")
