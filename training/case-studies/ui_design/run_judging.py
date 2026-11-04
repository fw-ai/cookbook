"""Judge every pair in a pilot results folder: cross-arm (plain-i vs skill-i) and within-arm (plain-0 vs plain-1,
skill-0 vs skill-1, for judge self-consistency), N repeats each with alternating order, optionally with a second judge
model for agreement. Only pairs where both candidates rendered are judged. Writes judgments.jsonl and a summary.

    python run_judging.py --results results/pilot-v2 --repeats 4 --judges accounts/fireworks/models/kimi-k3 accounts/fireworks/models/glm-5p3-flash
Paid API calls.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import judge  # noqa: E402


def rendered(d: Path) -> bool:
    try:
        judge.render_record(d); return True
    except ValueError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--briefs", type=Path, default=HERE / "briefs/pilot")
    ap.add_argument("--arms", nargs="+", default=["plain", "skill"])
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--judges", nargs="+", default=[judge.DEFAULT_JUDGE])
    ap.add_argument("--concurrency", type=int, default=6)
    a = ap.parse_args()
    pairs = []
    for bp in sorted(a.briefs.glob("*.json")):
        bd = a.results / bp.stem
        cands = {f"{arm}-{d}": bd / f"{arm}-{d}/render" for arm in a.arms for d in range(a.draws)}
        ok = {k: v for k, v in cands.items() if rendered(v)}
        for d in range(a.draws):   # cross-arm
            x, y = f"{a.arms[0]}-{d}", f"{a.arms[1]}-{d}"
            if x in ok and y in ok:
                pairs.append((bp, "cross", x, y))
        for arm in a.arms:         # within-arm
            x, y = f"{arm}-0", f"{arm}-1"
            if x in ok and y in ok:
                pairs.append((bp, f"within-{arm}", x, y))
    jobs = [(bp, kind, x, y, jm, rep) for bp, kind, x, y in pairs for jm in a.judges for rep in range(a.repeats)]
    print(f"{len(pairs)} pairs x {len(a.judges)} judges x {a.repeats} repeats = {len(jobs)} comparisons")
    out = a.results / "judgments.jsonl"
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            r = json.loads(l); done.add((r["brief"], r["kind"], r["x"], r["y"], r["judge_model"], r["rep"]))
    jobs = [j for j in jobs if (j[0].stem, j[1], j[2], j[3], j[4], j[5]) not in done]
    print(f"{len(done)} already done; {len(jobs)} to run")

    def one(job):
        bp, kind, x, y, jm, rep = job
        brief = json.loads(bp.read_text())
        with httpx.Client() as c:
            r = judge.compare(c, brief, a.results / bp.stem / x / "render", a.results / bp.stem / y / "render", seed=rep, judge_model=jm)
        return {"brief": bp.stem, "kind": kind, "x": x, "y": y, "judge_model": jm, "rep": rep, **{k: v for k, v in r.items() if k != "usage"},
                "usage": r.get("usage")}

    with ThreadPoolExecutor(a.concurrency) as ex, out.open("a") as f:
        for k, fut in enumerate(as_completed([ex.submit(one, j) for j in jobs]), 1):
            try:
                row = fut.result()
            except Exception as e:  # noqa: BLE001
                row = {"error": "exception", "detail": str(e)[:300]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            if k % 10 == 0 or k == len(jobs):
                print(f"  [{k}/{len(jobs)}]", flush=True)

    rows = [json.loads(l) for l in out.read_text().splitlines()]
    lines = ["# Judging summary", ""]
    for jm in a.judges:
        rs = [r for r in rows if r.get("judge_model") == jm and not r.get("error")]
        errs = sum(1 for r in rows if r.get("judge_model") == jm and r.get("error"))
        cross = [r for r in rs if r["kind"] == "cross"]
        c = collections.Counter(r["overall"] for r in cross)
        # order sensitivity: for each (brief, pair) with both orders seen, does the overall differ by order?
        byp = collections.defaultdict(lambda: collections.defaultdict(set))
        for r in cross:
            byp[(r["brief"], r["x"], r["y"])][r["swapped"]].add(r["overall"])
        both = [v for v in byp.values() if len(v) == 2]
        flips = sum(1 for v in both if v[False] != v[True])
        within = [r for r in rs if r["kind"].startswith("within")]
        wc = collections.Counter(r["overall"] for r in within)
        lines += [f"## {jm.split('/')[-1]}", "", f"- cross-arm comparisons: {len(cross)} ({errs} errors). overall: plain {c.get('a', 0)}, skill {c.get('b', 0)}, tie {c.get('tie', 0)}",
                  f"- order sensitivity: {flips} of {len(both)} pairs gave a different overall verdict in the two orders",
                  f"- within-arm comparisons: {len(within)}; verdict split a {wc.get('a', 0)} / b {wc.get('b', 0)} / tie {wc.get('tie', 0)} (a fair judge should be near 50/50 or tie-heavy)", ""]
        for crit in judge.CRITERIA:
            cc = collections.Counter(r[crit] for r in cross)
            lines.append(f"  - {crit}: plain {cc.get('a', 0)} / skill {cc.get('b', 0)} / tie {cc.get('tie', 0)}")
        lines.append("")
    if len(a.judges) == 2:
        j1, j2 = a.judges
        key = lambda r: (r["brief"], r["kind"], r["x"], r["y"], r["rep"])
        m1 = {key(r): r["overall"] for r in rows if r.get("judge_model") == j1 and not r.get("error")}
        m2 = {key(r): r["overall"] for r in rows if r.get("judge_model") == j2 and not r.get("error")}
        common = [k for k in m1 if k in m2]
        agree = sum(m1[k] == m2[k] for k in common)
        lines += [f"## Judge agreement", "", f"- {agree} of {len(common)} identical comparisons gave the same overall verdict ({j1.split('/')[-1]} vs {j2.split('/')[-1]})", ""]
    (a.results / "judging_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
