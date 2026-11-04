"""Evaluate an adapter on a set of briefs: generate N draws with the plain prompt, render, then judge against the
plain and skill arms' existing draws for the same briefs (both orders, criteria recorded). Writes a summary with
functional rate and preference by criterion.

    python evaluate_adapter.py --adapter accounts/example/models/recorded-model-06 --briefs pack/smoke/dev_briefs.json \
        --reference results/pilot-v2 --draws 4 --out results/eval-smoke
Paid API calls.
"""
from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import judge  # noqa: E402

PY = sys.executable
CHEAP = "accounts/fireworks/models/glm-5p3-flash"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--briefs", type=Path, required=True, help="JSON list of brief ids, or a directory of brief JSON files")
    ap.add_argument("--brief-dir", type=Path, default=HERE / "briefs/pilot")
    ap.add_argument("--reference", type=Path, required=True, help="pilot results with plain/skill draws for the same briefs")
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--ref-draws", type=int, default=2, help="reference draws per arm to compare against")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--judge", default=CHEAP)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.briefs.is_file():
        loaded = json.loads(a.briefs.read_text())
        ids = loaded["dev"] if isinstance(loaded, dict) and "dev" in loaded else loaded
    else:
        ids = sorted(p.stem for p in a.briefs.glob("*.json"))
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(b, d) for b in ids for d in range(a.draws)]

    def gen(job):
        b, d = job
        out = a.out / b / f"adapter-{d}"
        if not (out / "response.txt").exists():
            subprocess.run([PY, str(HERE / "generate.py"), "--brief", str(a.brief_dir / f"{b}.json"), "--arm", a.adapter, "--out", str(out)], capture_output=True)
        subprocess.run([PY, str(HERE / "render.py"), "--from-response", str(out / "response.txt"), "--brief", str(a.brief_dir / f"{b}.json"), "--out", str(out / "render")], capture_output=True)
        rec = json.loads((out / "render/record.json").read_text()) if (out / "render/record.json").exists() else {}
        return b, d, rec

    recs = {}
    with ThreadPoolExecutor(4) as ex:   # renders share one machine
        for b, d, rec in ex.map(gen, jobs):
            recs[(b, d)] = rec
    func = {k: v.get("passes_functional") for k, v in recs.items()}
    print(f"adapter functional: {sum(bool(v) for v in func.values())}/{len(func)}")
    # judge adapter draws against reference arms
    cmp_jobs = []
    for b in ids:
        brief = json.loads((a.brief_dir / f"{b}.json").read_text())
        for d in range(a.draws):
            if not recs[(b, d)].get("rendered"):
                continue
            for arm in ("plain", "skill"):
                for rd in range(a.ref_draws):
                    ref = a.reference / b / f"{arm}-{rd}/render"
                    try:
                        judge.render_record(ref)
                    except ValueError:
                        continue
                    for rep in range(a.repeats):
                        cmp_jobs.append((brief, b, d, arm, rd, rep, ref))
    print(f"{len(cmp_jobs)} comparisons")

    def one(j):
        brief, b, d, arm, rd, rep, ref = j
        with httpx.Client() as c:
            r = judge.compare(c, brief, a.out / b / f"adapter-{d}/render", ref, seed=rep, judge_model=a.judge)
        return {"brief": b, "adapter_draw": d, "vs": arm, "ref_draw": rd, "rep": rep, **{k: v for k, v in r.items() if k != "usage"}}

    out_rows = []
    with ThreadPoolExecutor(a.concurrency) as ex, (a.out / "judgments.jsonl").open("w") as f:
        for fut in as_completed([ex.submit(one, j) for j in cmp_jobs]):
            r = fut.result(); out_rows.append(r); f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lines = [f"# Adapter {a.adapter.split('/')[-1]} on {len(ids)} briefs, {a.draws} draws", "",
             f"functional: {sum(bool(v) for v in func.values())}/{len(func)}", ""]
    for arm in ("plain", "skill"):
        rs = [r for r in out_rows if r["vs"] == arm and not r.get("error")]
        if not rs:
            continue
        c = collections.Counter(r["overall"] for r in rs)
        lines.append(f"## adapter vs {arm} ({len(rs)} comparisons): adapter {c.get('a', 0)}, {arm} {c.get('b', 0)}, tie {c.get('tie', 0)}")
        for crit in judge.CRITERIA:
            cc = collections.Counter(r[crit] for r in rs)
            lines.append(f"  - {crit}: adapter {cc.get('a', 0)} / {arm} {cc.get('b', 0)} / tie {cc.get('tie', 0)}")
        lines.append("")
    (a.out / "summary.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()
