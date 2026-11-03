"""v2 selection from the existing draws: repair the two classes the audit flagged, then apply an ABSOLUTE design bar to
every surviving draw (selected-in-v1, functional-not-picked, repaired): the draw must beat the brief's best plain-arm
page on the design judge in BOTH orders, and not lose visual_idea in either. No per-brief quota.

    python select_v2.py --results results/main-v1 --briefs briefs/main --split briefs/main_split.json --plain results/main-plain-train --out pack/main-v2
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
import judge_design  # noqa: E402
from generate import task_prompt  # noqa: E402

PY = sys.executable


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True); ap.add_argument("--briefs", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True); ap.add_argument("--plain", type=Path, required=True, help="plain-arm draws for the same briefs (the bar)")
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--stage", choices=["repair", "bar", "pack", "all"], default="all")
    a = ap.parse_args()
    briefs = {p.stem: json.loads(p.read_text()) for p in a.briefs.glob("*.json")}
    train = set(json.loads(a.split.read_text())["train"])
    cls = json.load(open(a.results / "selection_classes.json"))
    a.out.mkdir(parents=True, exist_ok=True)
    repair_classes = [k for k in cls if k.startswith("excluded: policy") or k.startswith("failed functional: missing text") or k == "excluded: invented numbers"]
    to_repair = [(b, c) for k in repair_classes for b, c in cls[k] if b in train and (a.results / b / c / "render/shots/390.png").exists()]
    keep_as_is = [(b, c) for k in ("selected", "candidate, not picked (judge rank)") for b, c in cls[k] if b in train]

    if a.stage in ("repair", "all"):
        print(f"repairing {len(to_repair)} draws")
        def rep(item):
            b, c = item; out = a.results / b / c / "repaired"
            if not (out / "repair.json").exists():
                subprocess.run([PY, str(HERE / "repair.py"), "--brief", str(a.briefs / f"{b}.json"), "--candidate", str(a.results / b / c), "--out", str(out)], capture_output=True)
            return item, json.loads((out / "repair.json").read_text()) if (out / "repair.json").exists() else {"accepted": False, "error": "no result"}
        rep_res = {}
        with ThreadPoolExecutor(a.concurrency) as ex:
            for k, fut in enumerate(as_completed([ex.submit(rep, i) for i in to_repair]), 1):
                item, r = fut.result(); rep_res[item] = r
                if k % 20 == 0: print(f"  [{k}/{len(to_repair)}]", flush=True)
        acc = [i for i, r in rep_res.items() if r.get("accepted")]
        print(f"repair accepted {len(acc)}/{len(to_repair)}; no-edit-needed {sum(1 for r in rep_res.values() if r.get('no_edit_needed'))}")
        (a.out / "repairs.json").write_text(json.dumps({f"{b}/{c}": r for (b, c), r in rep_res.items()}, indent=1, ensure_ascii=False))

    if a.stage in ("bar", "all"):
        rep_res = json.loads((a.out / "repairs.json").read_text())
        pool = [(b, c, a.results / b / c / "render") for b, c in keep_as_is]
        for key, r in rep_res.items():
            if r.get("accepted"):
                b, c = key.split("/"); pool.append((b, c, a.results / b / c / ("render" if r.get("no_edit_needed") else "repaired/render")))
        # the bar: the brief's best functional plain draw (first functional, else first rendered)
        def best_plain(b):
            for d in range(6):
                rd = a.plain / b / f"plain-{d}/render"
                if (rd / "record.json").exists() and json.loads((rd / "record.json").read_text()).get("passes_functional"):
                    return rd
            for d in range(6):
                rd = a.plain / b / f"plain-{d}/render"
                if (rd / "shots/390.png").exists(): return rd
            return None
        print(f"bar: {len(pool)} candidates vs each brief's best plain page, both orders")
        def one(item):
            b, c, rd = item; pb = best_plain(b)
            if pb is None: return item, None
            with httpx.Client() as cl:
                v = [judge_design.compare(cl, briefs[b], rd, pb, seed=s) for s in (0, 1)]
            ok = [x for x in v if not x.get("error")]
            clears = len(ok) == 2 and all(x["overall"] == "a" for x in ok) and not any(x["visual_idea"] == "b" for x in ok)
            return item, {"clears_bar": clears, "verdicts": [{k: x.get(k) for k in ("overall", "visual_idea", "composition", "typography", "imagery", "mobile", "reason")} for x in v]}
        bar = {}
        with ThreadPoolExecutor(a.concurrency) as ex:
            for k, fut in enumerate(as_completed([ex.submit(one, i) for i in pool]), 1):
                item, r = fut.result(); bar[f"{item[0]}/{item[1]}"] = {"render": str(item[2]), **(r or {"clears_bar": None})}
                if k % 25 == 0: print(f"  [{k}/{len(pool)}]", flush=True)
        (a.out / "bar.json").write_text(json.dumps(bar, indent=1, ensure_ascii=False))
        print("clear the bar:", sum(1 for r in bar.values() if r.get("clears_bar")), "of", len(bar))

    if a.stage in ("pack", "all"):
        bar = json.loads((a.out / "bar.json").read_text())
        rows = []; per = collections.Counter()
        for key, r in bar.items():
            if not r.get("clears_bar"): continue
            b, c = key.split("/"); rd = Path(r["render"])
            resp = (rd.parent / "response.txt").read_text()
            rows.append({"messages": [{"role": "user", "content": task_prompt(briefs[b])}, {"role": "assistant", "content": resp}],
                         "meta": {"brief": b, "group": briefs[b]["group"], "family": briefs[b]["family"], "candidate": c, "repaired": "repaired" in str(rd), "source": "v2-bar"}})
            per[b] += 1
        (a.out / "train.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
        import hashlib
        (a.out / "manifest.json").write_text(json.dumps({"rows": len(rows), "briefs": len(per), "max_per_brief": max(per.values()) if per else 0, "repaired_rows": sum(x["meta"]["repaired"] for x in rows),
            "families": dict(collections.Counter(x["meta"]["family"] for x in rows)), "bar": "beats the brief's best plain page on the design judge in both orders, visual_idea not lost",
            "train_sha256": hashlib.sha256((a.out / "train.jsonl").read_bytes()).hexdigest()}, indent=2))
        print(f"pack: {len(rows)} rows from {len(per)} briefs ({sum(x['meta']['repaired'] for x in rows)} repaired) -> {a.out}")


if __name__ == "__main__":
    main()
