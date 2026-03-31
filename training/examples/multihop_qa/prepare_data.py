#!/usr/bin/env python3
"""Download multi-hop QA datasets and convert to JSONL for IGPO training.

Supports three datasets used in the IGPO paper (arXiv:2510.14967):
  - HotpotQA (distractor setting) — bridge and comparison questions
  - MuSiQue — compositional multi-hop (harder, 2–4 hops)
  - 2WikiMultiHopQA — cross-document reasoning

By default, only **hard** difficulty questions are kept (for HotpotQA).
MuSiQue and 2WikiMultiHopQA do not have difficulty labels and are
included as-is (they are inherently harder).

Output format per row:
  {"messages": [{"role": "system", ...}, {"role": "user", ...}],
   "ground_truth": "<answer>",
   "context": {"titles": [...], "paragraphs": [["sent1", ...], ...]},
   "source": "hotpotqa|musique|2wikimultihopqa",
   "question_type": "bridge|comparison|..."}

Usage:
    pip install datasets
    python prepare_data.py --max-rows 2000 --difficulty hard
    python prepare_data.py --dataset musique --max-rows 1000
    python prepare_data.py --dataset all --max-rows 3000
"""

import argparse
import json
import os
import random

from datasets import load_dataset

SYSTEM_PROMPT = (
    "You are a research assistant. Answer the question by searching for "
    "relevant information. You have access to two tools:\n"
    "- search(query): Search for information about a topic. Returns "
    "relevant paragraphs.\n"
    "- submit_answer(answer): Submit your final answer. Use this once you "
    "are confident.\n\n"
    "Search as many times as needed, then submit your answer."
)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "dataset.jsonl")


def format_hotpotqa(row: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["question"]},
        ],
        "ground_truth": row["answer"],
        "context": {
            "titles": row["context"]["title"],
            "paragraphs": row["context"]["sentences"],
        },
        "source": "hotpotqa",
        "question_type": row.get("type", "unknown"),
    }


def format_musique(row: dict) -> dict:
    paragraphs = row.get("paragraphs", [])
    titles = [p.get("title", "") for p in paragraphs]
    sents = [p.get("paragraph_text", "").split(". ") for p in paragraphs]
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["question"]},
        ],
        "ground_truth": row.get("answer", ""),
        "context": {"titles": titles, "paragraphs": sents},
        "source": "musique",
        "question_type": str(row.get("question_decomposition", [{}])[0].get("question", ""))[:50] if row.get("question_decomposition") else "multi-hop",
    }


def format_2wiki(row: dict) -> dict:
    context = row.get("context", {})
    if isinstance(context, dict):
        titles = context.get("title", [])
        paragraphs = context.get("sentences", [])
    elif isinstance(context, str):
        titles, paragraphs = [], [[context]]
    else:
        titles, paragraphs = [], []
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["question"]},
        ],
        "ground_truth": row.get("answer", ""),
        "context": {"titles": titles, "paragraphs": paragraphs},
        "source": "2wikimultihopqa",
        "question_type": row.get("type", "unknown"),
    }


def load_hotpotqa(split: str, difficulty: str, max_rows: int) -> list:
    ds = load_dataset("hotpot_qa", "distractor", split=split)
    print(f"Loaded {len(ds)} rows from HotpotQA ({split})")
    rows = []
    for row in ds:
        if difficulty != "all" and row.get("level", "") != difficulty:
            continue
        rows.append(format_hotpotqa(row))
        if len(rows) >= max_rows:
            break
    print(f"  Kept {len(rows)} rows (difficulty={difficulty})")
    return rows


def load_musique(split: str, max_rows: int) -> list:
    ds = load_dataset("dgslibiern/MuSiQue", split=split)
    print(f"Loaded {len(ds)} rows from MuSiQue ({split})")
    rows = []
    for row in ds:
        if not row.get("answerable", True):
            continue
        rows.append(format_musique(row))
        if len(rows) >= max_rows:
            break
    print(f"  Kept {len(rows)} rows")
    return rows


def load_2wiki(split: str, max_rows: int) -> list:
    ds = load_dataset("ohjoonhee/2WikiMultihopQA", split=split)
    print(f"Loaded {len(ds)} rows from 2WikiMultiHopQA ({split})")
    rows = []
    for row in ds:
        rows.append(format_2wiki(row))
        if len(rows) >= max_rows:
            break
    print(f"  Kept {len(rows)} rows")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Prepare multi-hop QA dataset for IGPO training")
    parser.add_argument("--max-rows", type=int, default=2000,
                        help="Maximum rows to output (total across all datasets)")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--dataset", default="hotpotqa",
                        choices=["hotpotqa", "musique", "2wiki", "all"],
                        help="Which dataset(s) to use")
    parser.add_argument("--difficulty", default="hard",
                        choices=["hard", "medium", "easy", "all"],
                        help="HotpotQA difficulty filter (ignored for other datasets)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    all_rows = []

    if args.dataset in ("hotpotqa", "all"):
        budget = args.max_rows if args.dataset == "hotpotqa" else args.max_rows // 2
        all_rows.extend(load_hotpotqa(args.split, args.difficulty, budget))

    if args.dataset in ("musique", "all"):
        budget = args.max_rows if args.dataset == "musique" else args.max_rows // 4
        try:
            all_rows.extend(load_musique("train" if args.split == "train" else "validation", budget))
        except Exception as e:
            print(f"  Skipping MuSiQue: {e}")

    if args.dataset in ("2wiki", "all"):
        budget = args.max_rows if args.dataset == "2wiki" else args.max_rows // 4
        try:
            all_rows.extend(load_2wiki("train" if args.split == "train" else "validation", budget))
        except Exception as e:
            print(f"  Skipping 2WikiMultiHopQA: {e}")

    if args.dataset == "all":
        random.shuffle(all_rows)
        all_rows = all_rows[:args.max_rows]

    with open(args.output, "w") as f:
        for entry in all_rows:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    sources = {}
    for r in all_rows:
        s = r.get("source", "unknown")
        sources[s] = sources.get(s, 0) + 1
    print(f"\nWrote {len(all_rows)} rows to {args.output}")
    for s, c in sorted(sources.items()):
        print(f"  {s}: {c}")


if __name__ == "__main__":
    main()
