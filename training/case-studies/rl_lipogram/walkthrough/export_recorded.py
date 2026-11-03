"""Freeze the evidence the lipogram notebook shows into walkthrough/recorded.json.

Reads the result folders (gitignored jsonl) and writes one small JSON the notebook can load from a fresh
checkout: per-arm rates with counts, a few full cases (prompt, base answer, adapter answer, trace excerpt),
sneeze-test draws, the slip-word profile, and the answer-only vs trace comparison. Rerun after any new eval.

    python walkthrough/export_recorded.py
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from lipogram_data import ground_truth  # noqa: E402
from lipogram_rewards import _canonical  # noqa: E402

RUNS = {  # label: (served folder, retention folder, kind)
    "trace-allergy-v1": ("results/trace-allergy-v1-served-max", "results/trace-allergy-v1-retention-max", "trace"),
    "v2-300": ("results/v2-300-served-max", "results/v2-300-retention-max", "trace"),
    "ao-v4": ("results/ao-v4-served-max", "results/ao-v4-retention-max", "answer-only"),
    "ao-v5": ("results/ao-v5-served-max", "results/ao-v5-retention-max", "answer-only"),
}
LOW_RUNS = {"trace adapter v2-300": "results/v2-300-served-low", "answer-only ao-v4": "results/ao-v4-served-low"}


def rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def arm_rates(samples: list[dict]) -> dict:
    out = {}
    for arm in sorted({r["arm"] for r in samples}):
        rs = [r for r in samples if r["arm"] == arm]
        greedy = [r for r in rs if r.get("greedy")]
        sampled = [r for r in rs if not r.get("greedy")] or greedy
        gc = [r for r in greedy if not r["truncated"]]; sc = [r for r in sampled if not r["truncated"]]
        out[arm] = {
            "n_greedy": len(greedy), "greedy_success": sum(r["success"] for r in greedy), "greedy_e_free": sum(r["e_free"] for r in greedy),
            "n_sampled": len(sampled), "sampled_success": sum(r["success"] for r in sampled), "sampled_e_free": sum(r["e_free"] for r in sampled),
            # the token budget cuts off long reasoning; these count only draws that finished
            "greedy_completed": len(gc), "greedy_completed_success": sum(r["success"] for r in gc), "greedy_completed_e_free": sum(r["e_free"] for r in gc),
            "sampled_completed": len(sc), "sampled_completed_success": sum(r["success"] for r in sc), "sampled_completed_e_free": sum(r["e_free"] for r in sc),
            "mean_reasoning_tokens": round(sum(r.get("reasoning_tokens") or 0 for r in rs) / max(1, len(rs))),
            "mean_words": round(sum(r["words"] for r in rs) / max(1, len(rs)), 1),
        }
    return out


def slip_profile(samples: list[dict], arm: str = "adapter") -> dict:
    rs = [r for r in samples if r["arm"] == arm]
    per = collections.Counter(); words = collections.Counter()
    for r in rs:
        ws = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]*", _canonical(r["response"])) if "e" in w.lower()]
        per[min(len(ws), 6)] += 1; words.update(ws)
    return {"n": len(rs), "e_words_per_answer": {str(k): v for k, v in sorted(per.items())}, "top_words": words.most_common(20)}


def pick_cases(samples: list[dict], n: int = 3) -> list[dict]:
    """A few prompts where the adapter was clean and the prompted base was not, for the first card."""
    by = collections.defaultdict(dict)
    for r in samples:
        if r.get("greedy"):
            by[r["source_id"]][r["arm"]] = r
    cases = []
    for sid, arms in by.items():
        a, b = arms.get("adapter"), arms.get("base")
        if a and b and a["success"] and not b["e_free"] and 40 <= a["words"] <= 160:
            cases.append({"source_id": sid, "family": a["family"], "adapter": a["response"], "base": b["response"],
                          "adapter_reasoning_tokens": a.get("reasoning_tokens"), "base_reasoning_tokens": b.get("reasoning_tokens")})
    return cases[:n]


def main() -> None:
    out = {"recorded_date": "2026-10-18", "leaked_validation_ids": ["gen-00279", "gen-00272", "gen-04056"], "base_model": "accounts/fireworks/models/glm-5p3-flash",
           "validation_prompts": 100, "runs": {}, "low_effort": {}, "sneeze": {}, "cases": {}, "slips": {}}
    for label, (served, ret, kind) in RUNS.items():
        s = rows(HERE / served / "samples.jsonl")
        if not s:
            print("skip", label, "(no served samples)"); continue
        man = json.loads((HERE / served / "manifest.json").read_text()) if (HERE / served / "manifest.json").exists() else {}
        rsum = json.loads((HERE / ret / "summary.json").read_text()) if (HERE / ret / "summary.json").exists() else {}
        out["runs"][label] = {"kind": kind, "effort": man.get("effort"), "draws": man.get("draws"), "max_tokens": man.get("max_tokens"), "n_prompts": len({r["source_id"] for r in s}),
                              "arms": arm_rates(s), "retention": rsum.get("retention"), "n_retention": rsum.get("n_compatible")}
        out["slips"][label] = slip_profile(s)
        if kind == "trace" and not out["cases"]:
            out["cases"] = {c["source_id"]: c for c in pick_cases(s)}
            out["cases_from"] = label
    for label, folder in LOW_RUNS.items():
        s = rows(HERE / folder / "samples.jsonl")
        if s:
            out["low_effort"][label] = {"arms": arm_rates(s), "slips": slip_profile(s)}
    # prompts for the cases
    tasks = {}
    for p in [HERE / "data/sft-v2/validation.jsonl"]:
        for t in rows(p):
            gt = ground_truth(t)
            tasks[gt["source_id"]] = gt["question"]
    for sid, c in out["cases"].items():
        c["question"] = tasks.get(sid, "")
    sn = HERE / "results/trace-allergy-v1-sneeze.jsonl"
    if sn.exists():
        draws = rows(sn)
        out["sneeze"] = {"n": len(draws), "answer_reacts": sum(d["answer_reacts"] for d in draws), "trace_notices": sum(d["trace_notices"] for d in draws),
                         "e_free": sum(d["e_count"] == 0 for d in draws), "examples": draws[:12]}
    (HERE / "walkthrough/recorded.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("runs:", list(out["runs"]), "cases:", len(out["cases"]), "sneeze draws:", out["sneeze"].get("n"))


if __name__ == "__main__":
    main()
