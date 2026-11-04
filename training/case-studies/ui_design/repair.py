"""Repair a visually strong draft that failed a check, without touching its design.

Given the draft's two files and its exact defects (missing required facts, unsupported claims to remove, overflow
width), ask the base model for a minimal edit, re-render, verify the defects are gone, and verify with the design
judge that the repaired page is not worse than the original (both orders; any 'original wins' on composition or
visual_idea counts as design loss). Writes repaired/ beside the original render.

    python repair.py --brief briefs/main/x.json --candidate results/main-v1/x/skill-1 --out results/main-v1/x/skill-1/repaired
Paid: one edit call (~4k tokens) + render + 2 judge calls.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cost_ledger  # noqa: E402
import factcheck  # noqa: E402
import judge_design  # noqa: E402
from render import parse_response  # noqa: E402

BASE = "accounts/fireworks/models/glm-5p3-flash"
URL = "https://api.fireworks.ai/inference/v1/chat/completions"
COSTS = HERE / "results/costs.jsonl"
CONTRACT = (HERE / "scaffold/OUTPUT_CONTRACT.md").read_text()


def defects(render_dir: Path) -> dict:
    rec = json.loads((render_dir / "record.json").read_text()); s = rec.get("summary", {})
    fc = render_dir / "factcheck.json"
    pol = json.loads(fc.read_text()).get("policy_claims", []) if fc.exists() else []
    return {"missing_text": s.get("missing_required_text", []), "policy_claims": pol, "overflow_px": s.get("max_overflow_px", 0),
            "invented_numbers": s.get("invented_numbers", []), "section_failures": s.get("content_section_failures", 0),
            "interactions": f"{s.get('interactions_ok')}/{s.get('interactions_n')}"}


def edit_request(brief: dict, files: dict, d: dict) -> str:
    asks = []
    if d["missing_text"]:
        asks.append("ADD these facts, verbatim, where they naturally belong (a row, a line, a label): " + "; ".join(d["missing_text"]))
    if d["policy_claims"] or d["invented_numbers"]:
        asks.append("REMOVE or neutralise these statements, which the brief does not support (delete the sentence or replace it with wording that makes no claim): "
                    + "; ".join(d["policy_claims"] + [f"the number {n}" for n in d["invented_numbers"]]))
    if d["overflow_px"] and d["overflow_px"] > 8:
        asks.append(f"FIX horizontal overflow of {d['overflow_px']}px at narrow widths (max-width:100%, overflow-wrap, or a narrower min-width); do not restructure the layout")
    if d["section_failures"]:
        asks.append("Every row of each content_sections collection must be visible inside data-testid=\"content-<id>\" with data-row; add missing rows, change nothing else")
    if d["interactions"] and d["interactions"].split("/")[0] != d["interactions"].split("/")[1]:
        asks.append("The details-toggle must start with the details-panel hidden, show it on click, and hide it again on a second click")
    return (f"Below is a landing page (two files) built from the brief that follows. It has specific defects. Make the SMALLEST edit that fixes them. "
            f"Keep the design exactly as it is: same layout, colours, type, images, section order and copy, except for the lines you must add or remove.\n\n"
            f"DEFECTS TO FIX\n- " + "\n- ".join(asks) + f"\n\nBRIEF\n{brief['brief']}\n\nREQUIRED CONTENT\n" + "\n".join("- " + c for c in brief["required_content"])
            + f"\n\nCURRENT FILES\n```jsx path=src/App.jsx\n{files['src/App.jsx']}\n```\n\n```css path=src/styles.css\n{files['src/styles.css']}\n```\n\n"
            f"Return the complete corrected files in the same two fenced blocks.\n{CONTRACT}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--brief", type=Path, required=True); ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    brief = json.loads(a.brief.read_text())
    files = parse_response((a.candidate / "response.txt").read_text())
    d = defects(a.candidate / "render")
    if a.out.exists():
        import shutil; shutil.rmtree(a.out)
    a.out.mkdir(parents=True)
    t0 = time.time()
    # A "missing" fact is often present in abbreviated form ("Taper Set of 6" in a row with "$24"). Ask first; if every
    # missing item is judged present, there is no text defect to repair.
    present_note = None
    if d["missing_text"]:
        q = (f"Page text:\n{factcheck.page_text(a.candidate / 'render')[:8000]}\n\nFor each of these facts, is it conveyed on the page (abbreviations, "
             f"split across a table row, or reworded all count as conveyed)? Answer ONLY JSON: {{\"present\": [facts that are conveyed], \"absent\": [facts that are not]}}\n"
             + "\n".join("- " + m for m in d["missing_text"]))
        rr = httpx.post(URL, headers={"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"}, timeout=120,
                        json={"model": BASE, "messages": [{"role": "user", "content": q}], "max_tokens": 800, "temperature": 0.0, "response_format": {"type": "json_object"}})
        if rr.status_code == 200:
            cost_ledger.record(COSTS, "repair", BASE, rr.json().get("usage"))
            try:
                pj = json.loads(rr.json()["choices"][0]["message"]["content"]); absent = [str(x) for x in pj.get("absent", [])]
                present_note = {"judged_present": [m for m in d["missing_text"] if m not in absent], "absent": absent}
                d["missing_text"] = absent
            except (KeyError, json.JSONDecodeError, TypeError):
                pass
    nothing_to_edit = not any([d["missing_text"], d["policy_claims"], d["invented_numbers"], d["section_failures"],
                               d["overflow_px"] and d["overflow_px"] > 8, d["interactions"] and d["interactions"].split("/")[0] != d["interactions"].split("/")[1]])
    if nothing_to_edit:
        # accept the original as-is: the only "defect" was the checker's literalness
        result = {"defects_before": d, "present_note": present_note, "no_edit_needed": True, "accepted": True, "seconds": round(time.time() - t0, 1)}
        (a.out / "repair.json").write_text(json.dumps(result, indent=1, ensure_ascii=False)); print(json.dumps(result, indent=1)); return
    body = {"model": BASE, "messages": [{"role": "user", "content": edit_request(brief, files, d)}], "max_tokens": 12000, "temperature": 0.2, "reasoning_effort": "low"}
    r = httpx.post(URL, headers={"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"}, json=body, timeout=httpx.Timeout(connect=30, read=600, write=60, pool=60))
    r.raise_for_status(); resp = r.json()
    cost_ledger.record(COSTS, "repair", BASE, resp.get("usage"))
    (a.out / "response.txt").write_text(resp["choices"][0]["message"].get("content") or "")
    subprocess.run([sys.executable, str(HERE / "render.py"), "--from-response", str(a.out / "response.txt"), "--brief", str(a.brief), "--out", str(a.out / "render")], capture_output=True)
    rec = json.loads((a.out / "render/record.json").read_text()) if (a.out / "render/record.json").exists() else {}
    result = {"defects_before": d, "present_note": present_note, "rendered": rec.get("rendered", False), "functional_after": rec.get("passes_functional"), "seconds": round(time.time() - t0, 1)}
    if rec.get("rendered"):
        with httpx.Client() as c:
            fc = factcheck.check(c, brief, factcheck.page_text(a.out / "render"))
            (a.out / "render/factcheck.json").write_text(json.dumps(fc, indent=1, ensure_ascii=False))
            after = fc.get("policy_claims", [])
            result["policy_claims_after"] = len(after)
            norm = lambda x: x.lower().strip(" .")[:40]
            result["named_claims_remaining"] = [c for c in d["policy_claims"] if any(norm(c) in norm(x) or norm(x) in norm(c) for x in after)]
            # design preserved? original (a) vs repaired (b), both orders; repaired must not lose on visual_idea or composition
            verdicts = [judge_design.compare(c, brief, a.candidate / "render", a.out / "render", seed=s) for s in (0, 1)]
        ok = [v for v in verdicts if not v.get("error")]
        lost = sum(1 for v in ok for k in ("visual_idea", "composition") if v[k] == "a")
        result["design_checks"] = [{k: v.get(k) for k in ("overall", "visual_idea", "composition", "reason")} for v in verdicts]
        result["design_preserved"] = bool(ok) and lost == 0
        # accept when the page is functional, the claims we asked to remove are gone, and the design survived; the absolute
        # count is reported but not gated (the fact checker finds different warm copy on every call)
        result["accepted"] = bool(result["functional_after"]) and not result["named_claims_remaining"] and result["design_preserved"]
    else:
        result["accepted"] = False
    (a.out / "repair.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in result.items() if k != "design_checks"}, indent=1))


if __name__ == "__main__":
    main()
