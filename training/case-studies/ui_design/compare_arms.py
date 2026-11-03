"""Fair multi-arm comparison on one brief set: identical briefs, assets, settings, draw counts. For each arm: generate
(or reuse), render, fact-check; then pairwise design-judge every arm against every other on the same draw index, both
orders. Reports, separately: design preference (design judge), usefulness (the original judge's brief_adherence),
functional rate, unsupported policy claims per page, output length, and repeated design patterns (DOM features).

    python compare_arms.py --briefs briefs/main_split.json --brief-dir briefs/main --draws 2 --out results/compare-dev \
        --arm plain=BASE --arm skill=BASE:skill --arm v1=accounts/pyroworks/models/glm53-uidesign-v1 --arm v2a=... --arm v2b=...
Paid.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import factcheck  # noqa: E402
import judge  # noqa: E402
import judge_design  # noqa: E402

PY = sys.executable
BASE = "accounts/fireworks/models/glm-5p3-flash"


def dom_features(dist: Path) -> dict:
    """Cheap structural fingerprint from the built page: hero pattern, section count, dark bands, grid use."""
    html = (dist / "index.html").read_text() if (dist / "index.html").exists() else ""
    js = "".join(p.read_text(errors="ignore") for p in (dist / "assets").glob("*.js")) if (dist / "assets").exists() else ""
    css = "".join(p.read_text(errors="ignore") for p in (dist / "assets").glob("*.css")) if (dist / "assets").exists() else ""
    return {"sections": js.count('"section"') + js.count("<section"), "h1_clamp": "clamp(" in css, "grid": "display:grid" in css.replace(" ", ""),
            "dark_band": css.count("#0") + css.count("#1") > 6, "svg": "<svg" in js or "createElement(\"svg\"" in js, "fonts": css.count("font-family")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--briefs", type=Path, required=True); ap.add_argument("--brief-dir", type=Path, required=True)
    ap.add_argument("--arm", action="append", required=True, help="name=model[:skill]  (BASE = base model)")
    ap.add_argument("--draws", type=int, default=2); ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--reuse", action="append", default=[], help="name=results_dir with existing <brief>/<name>-<d> folders")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    loaded = json.loads(a.briefs.read_text()); ids = loaded["dev"] if isinstance(loaded, dict) else loaded
    arms = {}
    for s in a.arm:
        n, rest = s.split("=", 1); parts = rest.split(":"); arms[n] = (BASE if parts[0] == "BASE" else parts[0], "skill" in parts[1:])
    reuse = dict(r.split("=", 1) for r in a.reuse)
    briefs = {b: json.loads((a.brief_dir / f"{b}.json").read_text()) for b in ids}
    a.out.mkdir(parents=True, exist_ok=True)

    def cand(b, n, d): return a.out / b / f"{n}-{d}"
    def gen(job):
        b, n, d = job; out = cand(b, n, d); model, sk = arms[n]
        if not (out / "response.txt").exists():
            if n in reuse and (Path(reuse[n]) / b / f"{n}-{d}/response.txt").exists():
                import shutil; shutil.copytree(Path(reuse[n]) / b / f"{n}-{d}", out, dirs_exist_ok=True)
            else:
                subprocess.run([PY, str(HERE / "generate.py"), "--brief", str(a.brief_dir / f"{b}.json"), "--arm", "skill" if sk else ("plain" if model == BASE else model), "--out", str(out)], capture_output=True)
        rec = json.loads((out / "render/record.json").read_text()) if (out / "render/record.json").exists() else {}
        stale = rec.get("rendered") and not rec.get("checks", {}).get("viewports", {}).get("1440", {}).get("text")   # reused render from before page text was stored
        if not rec or stale:
            subprocess.run([PY, str(HERE / "render.py"), "--from-response", str(out / "response.txt"), "--brief", str(a.brief_dir / f"{b}.json"), "--out", str(out / "render")], capture_output=True)
            rec = json.loads((out / "render/record.json").read_text()) if (out / "render/record.json").exists() else {}
        if rec.get("rendered") and not (out / "render/factcheck.json").exists():
            with httpx.Client() as c:
                fc = factcheck.check(c, briefs[b], factcheck.page_text(out / "render"))
            (out / "render/factcheck.json").write_text(json.dumps(fc, indent=1, ensure_ascii=False))
        return job

    jobs = [(b, n, d) for b in ids for n in arms for d in range(a.draws)]
    with ThreadPoolExecutor(4) as ex:
        for k, f in enumerate(as_completed([ex.submit(gen, j) for j in jobs]), 1):
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                print("  gen/render error:", str(e)[:160], flush=True)
            if k % 20 == 0: print(f"  gen/render [{k}/{len(jobs)}]", flush=True)
    # pairwise design + usefulness judging, same draw index, both orders
    pairs = [(b, x, y, d) for b in ids for x, y in itertools.combinations(arms, 2) for d in range(a.draws)
             if (cand(b, x, d) / "render/shots/390.png").exists() and (cand(b, y, d) / "render/shots/390.png").exists()]
    print(f"{len(pairs)} pairs x 2 orders x 2 judges")
    def one(p):
        b, x, y, d = p; out = []
        with httpx.Client() as c:
            for s in (0, 1):
                dj = judge_design.compare(c, briefs[b], cand(b, x, d) / "render", cand(b, y, d) / "render", seed=s)
                uj = judge.compare(c, briefs[b], cand(b, x, d) / "render", cand(b, y, d) / "render", seed=s, judge_model=BASE)
                out.append({"brief": b, "x": x, "y": y, "draw": d, "rep": s, "design": {k: dj.get(k) for k in ("overall", "visual_idea", "composition", "typography", "imagery", "mobile", "reason", "error")},
                            "useful": {k: uj.get(k) for k in ("overall", "brief_adherence", "reason", "error")}})
        return out
    J = []
    with ThreadPoolExecutor(a.concurrency) as ex:
        for k, f in enumerate(as_completed([ex.submit(one, p) for p in pairs]), 1):
            J += f.result()
            if k % 25 == 0: print(f"  judge [{k}/{len(pairs)}]", flush=True)
    (a.out / "judgments.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in J))

    # per-arm stats
    stats = {}
    for n in arms:
        recs = [json.loads((cand(b, n, d) / "render/record.json").read_text()) for b in ids for d in range(a.draws) if (cand(b, n, d) / "render/record.json").exists()]
        fcs = [json.loads((cand(b, n, d) / "render/factcheck.json").read_text()) for b in ids for d in range(a.draws) if (cand(b, n, d) / "render/factcheck.json").exists()]
        gens = [json.loads((cand(b, n, d) / "generation.json").read_text()) for b in ids for d in range(a.draws) if (cand(b, n, d) / "generation.json").exists()]
        feats = [dom_features(cand(b, n, d) / "render/dist") for b in ids for d in range(a.draws) if (cand(b, n, d) / "render/dist").exists()]
        stats[n] = {"n": len(recs), "functional": sum(bool(r.get("passes_functional")) for r in recs), "rendered": sum(bool(r.get("rendered")) for r in recs),
                    "policy_claims_mean": round(sum(len(f.get("policy_claims", [])) for f in fcs) / max(1, len(fcs)), 2),
                    "tokens_median": sorted(g["usage"]["completion_tokens"] for g in gens)[len(gens) // 2] if gens else None,
                    "pattern_share": {k: round(sum(bool(f[k]) for f in feats) / max(1, len(feats)), 2) for k in ("h1_clamp", "grid", "dark_band", "svg")} if feats else {}}
    # pairwise tallies
    tally = collections.defaultdict(lambda: collections.Counter()); utally = collections.defaultdict(lambda: collections.Counter())
    for r in J:
        if not r["design"].get("error"): tally[(r["x"], r["y"])][{"a": r["x"], "b": r["y"], "tie": "tie"}[r["design"]["overall"]]] += 1
        if not r["useful"].get("error"): utally[(r["x"], r["y"])][{"a": r["x"], "b": r["y"], "tie": "tie"}[r["useful"]["overall"]]] += 1
    lines = [f"# Arm comparison on {len(ids)} briefs, {a.draws} draws, effort low", "", "| arm | functional | policy claims/page | tokens (median) | patterns (clamp/grid/dark/svg) |", "|---|---:|---:|---:|---|"]
    for n, s in stats.items():
        lines.append(f"| {n} | {s['functional']}/{s['n']} | {s['policy_claims_mean']} | {s['tokens_median']} | {'/'.join(str(s['pattern_share'].get(k, '')) for k in ('h1_clamp', 'grid', 'dark_band', 'svg'))} |")
    lines += ["", "## Design preference (design judge, both orders)", "", "| pair | first | second | tie |", "|---|---:|---:|---:|"]
    for (x, y), c in tally.items(): lines.append(f"| {x} vs {y} | {c[x]} | {c[y]} | {c['tie']} |")
    lines += ["", "## Usefulness preference (original judge, overall)", "", "| pair | first | second | tie |", "|---|---:|---:|---:|"]
    for (x, y), c in utally.items(): lines.append(f"| {x} vs {y} | {c[x]} | {c[y]} | {c['tie']} |")
    (a.out / "summary.md").write_text("\n".join(lines) + "\n"); (a.out / "stats.json").write_text(json.dumps(stats, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
