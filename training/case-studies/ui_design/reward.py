"""The rendered-page reward for the RL pilot. One candidate response -> components, all logged separately:

  rendered     0/1   parsed, built, page loaded with text
  functional   0..1  share of brief checks passed (required content, toggle, overflow, sections, console)
  fidelity     0..1  1 - min(1, unsupported policy claims / 4)
  design       -1..1 design judge vs the brief's REFERENCE page, both orders: the mean over the five criteria
                     (visual_idea, composition, typography, imagery, mobile) of +1/0/-1, averaged over orders, so a page
                     that beats the reference on two criteria and loses three scores -0.2, not -1. The overall verdict is
                     logged but not used: it is nearly binary against a strong reference and gives GRPO no spread.
  reward       scalar: 0 if not rendered; else 0.15*functional + 0.15*fidelity + 0.7*(design+1)/2, with a hard floor:
               a page that fails >50% of functional checks or has >4 policy claims is capped at 0.3 so an attractive
               broken page cannot beat a complete, well-designed one.

No CSS feature is rewarded directly. The reference is fixed per brief so progress has a consistent target.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import factcheck  # noqa: E402
import judge_design  # noqa: E402

PY = sys.executable


def functional_share(rec: dict, brief: dict) -> float:
    s = rec.get("summary", {})
    checks = []
    req = brief.get("required_content", [])
    checks.append(1 - len(s.get("missing_required_text", [])) / max(1, len(req)))
    n_i = s.get("interactions_n") or 0
    checks.append((s.get("interactions_ok") or 0) / n_i if n_i else 1.0)
    checks.append(0.0 if (s.get("max_overflow_px") or 0) > 8 else 1.0)
    checks.append(0.0 if s.get("console_errors") else 1.0)
    if brief.get("content_sections"):
        checks.append(0.0 if s.get("content_section_failures") else 1.0)
    return sum(checks) / len(checks)


def score(response_text: str, brief: dict, brief_path: Path, reference_render: Path | None, workdir: Path | None = None, client: httpx.Client | None = None) -> dict:
    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="reward-"))
    work.mkdir(parents=True, exist_ok=True)
    (work / "response.txt").write_text(response_text)
    subprocess.run([PY, str(HERE / "render.py"), "--from-response", str(work / "response.txt"), "--brief", str(brief_path), "--out", str(work / "render")], capture_output=True)
    rec = json.loads((work / "render/record.json").read_text()) if (work / "render/record.json").exists() else {}
    out = {"rendered": 0, "functional": 0.0, "fidelity": 0.0, "design": -1.0, "design_criteria": {}, "policy_claims": None, "reward": 0.0, "workdir": str(work)}
    text = rec.get("checks", {}).get("viewports", {}).get("1440", {}).get("text") or ""
    if not rec.get("rendered") or len(text.strip()) < 50:   # a blank page is not a rendered page
        out["fail_stage"] = rec.get("stage", "blank" if rec.get("rendered") else ("build" if rec.get("build") else "parse"))
        return out
    out["rendered"] = 1
    out["functional"] = round(functional_share(rec, brief), 3)
    own = client is None
    c = client or httpx.Client()
    try:
        fc = factcheck.check(c, brief, factcheck.page_text(work / "render"))
        claims = len(fc.get("policy_claims", [])) if not fc.get("error") else 0
        out["policy_claims"] = claims; out["fidelity"] = round(max(0.0, 1 - claims / 4), 3)
        if reference_render and (reference_render / "shots/390.png").exists():
            vs = [judge_design.compare(c, brief, work / "render", reference_render, seed=s) for s in (0, 1)]
            ok = [v for v in vs if not v.get("error")]
            if ok:
                crit = [sum({"a": 1, "tie": 0, "b": -1}[v[k]] for k in judge_design.CRITERIA) / len(judge_design.CRITERIA) for v in ok]
                out["design"] = round(sum(crit) / len(crit), 3)
                out["design_overall"] = round(sum({"a": 1, "tie": 0, "b": -1}[v["overall"]] for v in ok) / len(ok), 3)
                out["design_criteria"] = {k: [v[k] for v in ok] for k in judge_design.CRITERIA}
                out["design_reasons"] = [v["reason"][:160] for v in ok]
            else:
                out["design_error"] = True
    finally:
        if own:
            c.close()
    r = 0.15 * out["functional"] + 0.15 * out["fidelity"] + 0.7 * (out["design"] + 1) / 2
    if out["functional"] < 0.5 or (out["policy_claims"] or 0) > 4:
        r = min(r, 0.3)
    out["reward"] = round(r, 4)
    return out
