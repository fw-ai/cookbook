"""Select SFT rows from the pilot: for each training brief, rank the functional skill-arm draws by a within-arm
round-robin (cheap judge, both orders) and keep the top K. Student input = the PLAIN task prompt (no skill text);
target = the selected skill page's response text exactly as generated (the two contract blocks). Group-disjoint
dev split held out before selection; nothing from dev briefs enters the pack.

    python build_pack.py --results results/pilot-v2 --out pack/smoke --dev-groups transport retail publishing nonprofit --top 2
Paid judge calls (GLM 5.3 Flash) for the round-robin.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import factcheck  # noqa: E402
import judge  # noqa: E402
from generate import task_prompt  # noqa: E402

CHEAP = "accounts/fireworks/models/glm-5p3-flash"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--briefs", type=Path, default=HERE / "briefs/pilot")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dev-groups", nargs="+", default=None)
    ap.add_argument("--split", type=Path, default=None, help="JSON with train/dev id lists (overrides --dev-groups)")
    ap.add_argument("--arms", nargs="+", default=["skill", "plain"], help="candidate arms; the first supplies most rows, others may win a brief outright")
    ap.add_argument("--max-candidates", type=int, default=6, help="per brief, to bound the round-robin")
    ap.add_argument("--max-policy-claims", type=int, default=4, help="exclude candidates whose fact check finds more unsupported policy claims than this")
    ap.add_argument("--top", type=int, default=2)
    ap.add_argument("--judge", default=CHEAP)
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args()
    briefs = {p.stem: json.loads(p.read_text()) for p in a.briefs.glob("*.json")}
    rows = json.load(open(a.results / "pilot_rows.json"))
    if a.split:
        sp = json.loads(a.split.read_text()); dev = sorted(sp["dev"]); train = sorted(sp["train"])
    else:
        dev = sorted(b for b, v in briefs.items() if v["group"] in a.dev_groups)
        train = sorted(b for b in briefs if b not in dev)
    assert not set(dev) & set(train)
    cands = collections.defaultdict(list)
    excluded = collections.Counter()
    for r in rows:
        if r["arm"] not in a.arms or not r["functional"] or r["brief"] not in train:
            continue
        rd = a.results / r["brief"] / f"{r['arm']}-{r['draw']}" / "render"
        rec = json.loads((rd / "record.json").read_text())
        if rec.get("summary", {}).get("invented_numbers"):   # a beautiful page that invents a price is not a demonstration
            excluded["invented_numbers"] += 1; continue
        cands[r["brief"]].append(f"{r['arm']}-{r['draw']}")
    # text-level fact check on the remaining candidates (cheap; results cached beside the render)
    policy = {}
    def fc(item):
        b, c = item; rd = a.results / b / c / "render"; fp = rd / "factcheck.json"
        if fp.exists():
            return b, c, json.loads(fp.read_text())
        with httpx.Client() as cl:
            r = factcheck.check(cl, briefs[b], factcheck.page_text(rd))
        fp.write_text(json.dumps(r, indent=1, ensure_ascii=False)); return b, c, r
    with ThreadPoolExecutor(a.concurrency) as ex:
        for b, c, r in ex.map(fc, [(b, c) for b, cs in cands.items() for c in cs]):
            policy[(b, c)] = len(r.get("policy_claims", [])) if not r.get("error") else 0
    for b in list(cands):
        keep = [c for c in cands[b] if policy[(b, c)] <= a.max_policy_claims]
        excluded["policy_claims"] += len(cands[b]) - len(keep); cands[b] = keep
    # bound the round-robin: keep up to max_candidates per brief, skill arm first (it supplies the visual direction)
    for b in cands:   # fewest unsupported policy claims first, then skill arm first
        cands[b] = sorted(cands[b], key=lambda c: (policy[(b, c)], 0 if c.startswith(a.arms[0]) else 1, c))[: a.max_candidates]
    print("excluded:", dict(excluded))
    jobs = [(b, x, y, rep) for b, cs in cands.items() for x, y in itertools.combinations(sorted(cs), 2) for rep in (0, 1)]
    print(f"train {len(train)} briefs, dev {len(dev)} ({a.dev_groups}); candidates {sum(len(v) for v in cands.values())}; {len(jobs)} round-robin comparisons")

    def one(job):
        b, x, y, rep = job
        with httpx.Client() as c:
            r = judge.compare(c, briefs[b], a.results / b / x / "render", a.results / b / y / "render", seed=rep, judge_model=a.judge)
        return b, x, y, r

    wins = collections.defaultdict(lambda: collections.Counter())
    audit = []
    with ThreadPoolExecutor(a.concurrency) as ex:
        for k, fut in enumerate(as_completed([ex.submit(one, j) for j in jobs]), 1):
            try:
                b, x, y, r = fut.result()
            except Exception as e:  # noqa: BLE001
                audit.append({"error": "exception", "detail": str(e)[:200]}); continue
            audit.append({"brief": b, "x": x, "y": y, **{k: v for k, v in r.items() if k != "usage"}})
            if k % 50 == 0:
                print(f"  [{k}/{len(jobs)}]", flush=True)
            if r.get("error"):
                continue
            score = sum({"a": 1, "b": -1, "tie": 0}[r[c]] for c in ("hierarchy", "composition", "specificity"))   # criteria, not the forced overall
            if score > 0: wins[b][x] += 1
            elif score < 0: wins[b][y] += 1
            else: wins[b][x] += 0.5; wins[b][y] += 0.5
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "roundrobin.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in audit))
    pack, selection = [], {}
    for b in train:
        ranked = sorted(cands[b], key=lambda c: (-wins[b][c], policy[(b, c)])) if len(cands[b]) > 1 else cands[b]
        selection[b] = {"candidates": {c: wins[b][c] for c in cands[b]}, "selected": ranked[: a.top]}
        for c in ranked[: a.top]:
            resp = (a.results / b / c / "response.txt").read_text()
            g = json.loads((a.results / b / c / "generation.json").read_text())
            pack.append({"messages": [{"role": "user", "content": task_prompt(briefs[b])}, {"role": "assistant", "content": resp}],
                         "meta": {"brief": b, "group": briefs[b]["group"], "family": briefs[b]["family"], "candidate": c, "arm": c.rsplit("-", 1)[0], "request_id": g["request_id"], "wins": wins[b][c], "policy_claims": policy[(b, c)]}})
    (a.out / "train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pack))
    (a.out / "dev_briefs.json").write_text(json.dumps(dev, indent=1))
    man = {"source_results": str(a.results), "arms": a.arms, "max_candidates": a.max_candidates, "max_policy_claims": a.max_policy_claims, "excluded": dict(excluded), "top": a.top, "judge": a.judge, "dev_groups": a.dev_groups, "split": str(a.split) if a.split else None, "train_briefs": train, "dev_briefs": dev,
           "rows": len(pack), "selection": selection, "train_sha256": hashlib.sha256((a.out / "train.jsonl").read_bytes()).hexdigest(),
           "student_prompt": "plain task prompt (no skill text); same as the plain arm", "target": "selected response verbatim (skill arm unless a plain draw won the brief)"}
    (a.out / "manifest.json").write_text(json.dumps(man, indent=2))
    print(f"pack: {len(pack)} rows from {sum(1 for b in train if selection[b]['selected'])} briefs -> {a.out}")


if __name__ == "__main__":
    main()
