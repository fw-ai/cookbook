"""The pilot: every brief x every arm x N draws, generated in parallel, rendered sequentially (one Chromium at a time),
and summarised. Judging is a separate step so pages can be read first.

    python run_pilot.py --arms plain skill --draws 2 --out results/pilot-v2
Paid API calls for generation only; rendering is local.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--briefs", type=Path, default=HERE / "briefs/pilot")
    ap.add_argument("--arms", nargs="+", default=["plain", "skill"])
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None, help="brief ids")
    a = ap.parse_args()
    briefs = sorted(p for p in a.briefs.glob("*.json"))
    if a.only:
        briefs = [p for p in briefs if p.stem in a.only]
    jobs = [(b, arm, d) for b in briefs for arm in a.arms for d in range(a.draws)]
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(briefs)} briefs x {a.arms} x {a.draws} draws = {len(jobs)} generations")

    def gen(job):
        b, arm, d = job
        out = a.out / b.stem / f"{arm}-{d}"
        if (out / "response.txt").exists():
            return job, "cached"
        r = subprocess.run([PY, str(HERE / "generate.py"), "--brief", str(b), "--arm", arm, "--out", str(out)], capture_output=True, text=True)
        return job, (r.stdout.strip().splitlines()[-1] if r.returncode == 0 else "ERROR " + r.stderr[-300:])

    t0 = time.time()
    with ThreadPoolExecutor(a.concurrency) as ex:
        for k, fut in enumerate(as_completed([ex.submit(gen, j) for j in jobs]), 1):
            job, msg = fut.result()
            print(f"  [{k}/{len(jobs)}] {job[0].stem} {job[1]}-{job[2]}: {msg}", flush=True)
    print(f"generation done in {time.time() - t0:.0f}s; rendering")
    rows = []
    for b, arm, d in jobs:
        out = a.out / b.stem / f"{arm}-{d}"
        if not (out / "response.txt").exists():
            rows.append({"brief": b.stem, "arm": arm, "draw": d, "generated": False}); continue
        r = subprocess.run([PY, str(HERE / "render.py"), "--from-response", str(out / "response.txt"), "--brief", str(b), "--out", str(out / "render")], capture_output=True, text=True)
        rec = json.loads((out / "render/record.json").read_text()) if (out / "render/record.json").exists() else {"harness_error": True, "error": r.stderr[-300:]}
        g = json.loads((out / "generation.json").read_text())
        s = rec.get("summary", {})
        rows.append({"brief": b.stem, "arm": arm, "draw": d, "generated": True, "finish": g.get("finish_reason"), "tokens": (g.get("usage") or {}).get("completion_tokens"),
                     "seconds_gen": g.get("seconds"), "parsed": rec.get("stage") != "parse", "built": bool(rec.get("build", {}).get("ok")), "rendered": rec.get("rendered", False),
                     "functional": rec.get("passes_functional"), "harness_error": rec.get("harness_error", False), "model_failure": rec.get("model_failure", False),
                     "missing": s.get("missing_required_text"), "section_failures": s.get("content_section_failures"), "interactions": f"{s.get('interactions_ok')}/{s.get('interactions_n')}",
                     "overflow": s.get("max_overflow_px"), "console_errors": s.get("console_errors"), "warnings": s.get("warnings"), "seconds_render": rec.get("seconds_total")})
        print(f"  {b.stem:26s} {arm}-{d}: parsed={rows[-1]['parsed']} built={rows[-1]['built']} functional={rows[-1]['functional']} missing={rows[-1]['missing']} sections={rows[-1]['section_failures']} inter={rows[-1]['interactions']}", flush=True)
    (a.out / "pilot_rows.json").write_text(json.dumps(rows, indent=1))
    # summary by arm
    lines = ["| arm | n | parsed | built | functional | mean tokens | mean gen s | harness errors |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in a.arms:
        rs = [r for r in rows if r["arm"] == arm and r["generated"]]
        n = max(1, len(rs))
        lines.append(f"| {arm} | {len(rs)} | {sum(r['parsed'] for r in rs)/n:.0%} | {sum(r['built'] for r in rs)/n:.0%} | {sum(bool(r['functional']) for r in rs)/n:.0%} | "
                     f"{sum(r['tokens'] or 0 for r in rs)/n:,.0f} | {sum(r['seconds_gen'] or 0 for r in rs)/n:.0f} | {sum(r['harness_error'] for r in rs)} |")
    (a.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
