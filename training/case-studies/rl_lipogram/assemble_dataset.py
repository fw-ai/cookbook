"""Assemble a training file from builder outputs, with provenance. Asserts every row's prompts are disjoint from
the validation split and from the allergy eval set, and that multi-turn rows keep loss masks on earlier turns.

    python assemble_dataset.py --out data/sft-snapshots/trace-allergy-v2 \
        --source data/sft-v2/sft.jsonl:trace --source data/allergy-v2/allergy.jsonl:allergy
No API calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lipogram_data import ground_truth, split_validation, user_messages  # noqa: E402
from lipogram_rewards import e_free  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", action="append", required=True, help="path:kind, e.g. data/sft-v2/sft.jsonl:trace")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, default=HERE / "data/v2-large/lipogram_train.jsonl")
    ap.add_argument("--eval-set", type=Path, default=HERE / "data/allergy-eval-v1/allergy_eval.jsonl")
    ap.add_argument("--validation-groups", type=int, default=16)
    ap.add_argument("--validation-seed", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--require-trace", action="store_true", default=True)
    a = ap.parse_args()
    all_tasks = [json.loads(l) for l in a.tasks.read_text().splitlines() if l.strip()]
    _, validation, _ = split_validation(all_tasks, a.validation_groups, a.validation_seed)
    forbidden = {user_messages(t)[0]["content"].strip().casefold() for t in validation}
    if a.eval_set.exists():
        for l in a.eval_set.read_text().splitlines():
            it = json.loads(l); forbidden.add(it["new_question"].strip().casefold())
    rows, prov, dropped = [], {}, {}
    for spec in a.source:
        path, kind = spec.rsplit(":", 1); p = Path(path)
        src = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        kept = []
        for r in src:
            last = r["messages"][-1]
            assert last["role"] == "assistant" and last.get("trainable", True)
            assert not any(m["role"] == "system" for m in r["messages"]), "training rows carry no system prompt"
            assert e_free(last["content"]), "trained answer must be e-free"
            if a.require_trace:
                assert (last.get("reasoning_content") or "").strip(), f"{p.name}: trained turn has no reasoning trace"
            for m in r["messages"][:-1]:
                if m["role"] == "assistant":
                    assert m.get("trainable") is False, f"{p.name}: earlier assistant turn must be loss-masked"
            if any(m["role"] == "user" and m["content"].strip().casefold() in forbidden for m in r["messages"]):
                dropped[kind] = dropped.get(kind, 0) + 1; continue
            kept.append(r)
        prov[kind] = {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "rows": len(src), "kept": len(kept), "dropped_overlap": len(src) - len(kept)}
        rows += kept
    random.Random(a.seed).shuffle(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    out = a.out / "sft.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    prov_all = {"sources": prov, "total_rows": len(rows), "seed": a.seed, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                "excluded": {"validation_prompts": a.validation_groups and len(validation), "eval_set_questions": a.eval_set.exists()}}
    (a.out / "sources.json").write_text(json.dumps(prov_all, indent=2))
    print(json.dumps(prov_all, indent=2))


if __name__ == "__main__":
    main()
