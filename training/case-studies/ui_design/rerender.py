"""Re-run render.py over every generated candidate in a results folder (after a checker change) and rebuild pilot_rows.json."""
import json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
res = Path(sys.argv[1]); briefs = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "briefs/pilot"
rows = []
for bd in sorted(p for p in res.iterdir() if p.is_dir() and (briefs / f"{p.name}.json").exists()):
    for cd in sorted(p for p in bd.iterdir() if p.is_dir() and (p / "response.txt").exists()):
        arm, d = cd.name.rsplit("-", 1)
        subprocess.run([sys.executable, str(HERE / "render.py"), "--from-response", str(cd / "response.txt"), "--brief", str(briefs / f"{bd.name}.json"), "--out", str(cd / "render")], capture_output=True)
        rec = json.loads((cd / "render/record.json").read_text()); g = json.loads((cd / "generation.json").read_text()); s = rec.get("summary", {})
        rows.append({"brief": bd.name, "arm": arm, "draw": int(d), "generated": True, "finish": g.get("finish_reason"), "tokens": (g.get("usage") or {}).get("completion_tokens"),
                     "seconds_gen": g.get("seconds"), "parsed": rec.get("stage") != "parse", "built": bool(rec.get("build", {}).get("ok")), "rendered": rec.get("rendered", False),
                     "functional": rec.get("passes_functional"), "harness_error": rec.get("harness_error", False), "model_failure": rec.get("model_failure", False),
                     "missing": s.get("missing_required_text"), "missing_exact": [t for v in rec.get("checks", {}).get("viewports", {}).values() for t in v.get("missing_exact_phrase", [])][:20],
                     "section_failures": s.get("content_section_failures"), "interactions": f"{s.get('interactions_ok')}/{s.get('interactions_n')}",
                     "overflow": s.get("max_overflow_px"), "console_errors": s.get("console_errors"), "warnings": s.get("warnings"), "seconds_render": rec.get("seconds_total")})
(res / "pilot_rows.json").write_text(json.dumps(rows, indent=1))
for arm in ("plain", "skill"):
    rs = [r for r in rows if r["arm"] == arm]; n = max(1, len(rs))
    print(f"{arm}: n={len(rs)} functional {sum(bool(r['functional']) for r in rs)/n:.0%}  missing-text failures {sum(bool(r['missing']) for r in rs)}  exact-phrase misses {sum(bool(r['missing_exact']) for r in rs)}")
