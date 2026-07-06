#!/usr/bin/env python3
"""Download multi-hop QA datasets and convert to JSONL for IGPO training.

Supports four datasets:
  - HotpotQA (distractor setting) — bridge and comparison questions
  - MuSiQue — compositional multi-hop (harder, 2–4 hops)
  - 2WikiMultiHopQA — cross-document reasoning
  - FRAMES (google/frames-benchmark) — recent (2024) end-to-end RAG benchmark,
    824 realistic multi-hop questions needing 2–15 Wikipedia articles. FRAMES
    ships only gold-article links (no passages), so this builder fetches the
    article plaintext from Wikipedia (cached) and assembles per-question
    paragraph pools (gold + sampled distractors) so the task stays
    environment-free, exactly like the HotpotQA/MuSiQue snapshots.

By default, only **hard** difficulty questions are kept (for HotpotQA).
MuSiQue and 2WikiMultiHopQA do not have difficulty labels and are
included as-is (they are inherently harder).

Output format per row:
  {"messages": [{"role": "system", ...}, {"role": "user", ...}],
   "ground_truth": "<answer>",
   "context": {"titles": [...], "paragraphs": [["sent1", ...], ...]},
   "source": "hotpotqa|musique|2wikimultihopqa|frames",
   "question_type": "bridge|comparison|..."}

Usage:
    Follow the setup instructions in ../../README.md.
    python prepare_data.py --max-rows 2000 --difficulty hard
    python prepare_data.py --dataset musique --max-rows 1000
    python prepare_data.py --dataset all --max-rows 3000

    # FRAMES (eval-only upstream -> deterministic train/eval partition):
    python prepare_data.py --dataset frames --split eval --max-rows 200 \
        --output .../multihop_qa_eval.jsonl
    python prepare_data.py --dataset frames --split train --max-rows 624 \
        --output .../multihop_qa_train.jsonl
"""

import argparse
import ast
import json
import os
import random
import re
import time
import urllib.parse
import urllib.request

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
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
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
    ds = load_dataset("bdsaglam/musique", split=split)
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


# ---------------------------------------------------------------------------
# FRAMES (google/frames-benchmark)
# ---------------------------------------------------------------------------
#
# FRAMES is eval-only (824 questions) and ships only gold Wikipedia article
# links — no passages. To keep the agentic-search task environment-free we:
#   1. parse the gold article titles from each question's wiki_links,
#   2. fetch each article's plaintext from the Wikipedia API once (on-disk
#      cache so re-runs are offline/instant),
#   3. chunk articles into paragraphs and assemble a per-question pool of
#      gold paragraphs + sampled distractor paragraphs (~20, like MuSiQue),
#   4. deterministically partition the 824 questions into a held-out eval
#      split and a disjoint train split.

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_UA = "fireworks-rl-e2e/1.0 (training guardrail dataset build; contact: eng@fireworks.ai)"

# Pool shape (mirrors the ~20-paragraph MuSiQue pools).
FRAMES_POOL_SIZE = 22
FRAMES_PARAS_PER_ARTICLE = 6
FRAMES_GOLD_PER_ARTICLE = 4
FRAMES_GOLD_CAP = 14
FRAMES_DEFAULT_EVAL_SIZE = 200


def _frames_gold_titles(row: dict) -> list:
    """Extract de-duplicated Wikipedia article titles from a FRAMES row."""
    raw = row.get("wiki_links")
    urls: list = []
    if isinstance(raw, str):
        try:
            urls = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            urls = []
    elif isinstance(raw, list):
        urls = raw
    titles: list = []
    seen: set = set()
    for url in urls:
        if not url:
            continue
        path = urllib.parse.urlparse(str(url)).path
        if "/wiki/" not in path:
            continue
        title = urllib.parse.unquote(path.split("/wiki/", 1)[1]).replace("_", " ").strip()
        if title and title not in seen:
            seen.add(title)
            titles.append(title)
    return titles


def _fetch_wiki_extract(title: str) -> str:
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": "1",
        "exsectionformat": "plain",
        "redirects": "1",
        "titles": title,
    }
    url = WIKI_API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": WIKI_UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        return page.get("extract", "") or ""
    return ""


def _fetch_all_wiki_extracts(titles: list, cache_path: str) -> dict:
    cache: dict = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
    todo = [t for t in titles if t not in cache]
    print(f"FRAMES: {len(titles)} unique articles, {len(todo)} to fetch (cache: {cache_path})")
    for i, title in enumerate(todo):
        try:
            cache[title] = _fetch_wiki_extract(title)
        except Exception as exc:  # noqa: BLE001 — one bad article must not abort the build
            print(f"  warn: fetch failed for {title!r}: {exc}")
            cache[title] = ""
        if (i + 1) % 50 == 0:
            print(f"  fetched {i + 1}/{len(todo)}")
            with open(cache_path, "w") as f:
                json.dump(cache, f)
        time.sleep(0.05)
    with open(cache_path, "w") as f:
        json.dump(cache, f)
    return cache


def _chunk_article(text: str, max_paras: int, min_chars: int = 120, max_chars: int = 1200) -> list:
    """Split an article's plaintext into a few clean paragraphs (no headers)."""
    paras: list = []
    for block in text.split("\n"):
        block = block.strip()
        if block.startswith("=") or len(block) < min_chars:
            continue
        paras.append(block[:max_chars])
        if len(paras) >= max_paras:
            break
    return paras


def _para_to_sentences(paragraph: str) -> list:
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()]
    return sents or [paragraph]


def load_frames(
    split: str,
    max_rows: int,
    *,
    seed: int,
    wiki_cache: str,
    eval_size: int = FRAMES_DEFAULT_EVAL_SIZE,
) -> list:
    ds = load_dataset("google/frames-benchmark", split="test")
    raw_rows = list(ds)
    print(f"Loaded {len(raw_rows)} rows from FRAMES (test)")

    per_row_titles = [_frames_gold_titles(r) for r in raw_rows]
    all_titles = sorted({t for titles in per_row_titles for t in titles})
    cache = _fetch_all_wiki_extracts(all_titles, wiki_cache)

    # Chunk every article once; reuse for gold pools and the distractor bank.
    article_chunks: dict = {
        t: _chunk_article(cache.get(t, ""), FRAMES_PARAS_PER_ARTICLE) for t in all_titles
    }
    distractor_bank = [
        (t, para) for t in all_titles for para in article_chunks[t]
    ]

    rng = random.Random(seed)
    built: list = []
    skipped = 0
    for row, titles in zip(raw_rows, per_row_titles):
        gold_pairs: list = []
        for title in titles:
            for para in article_chunks.get(title, [])[:FRAMES_GOLD_PER_ARTICLE]:
                gold_pairs.append((title, para))
        gold_pairs = gold_pairs[:FRAMES_GOLD_CAP]
        if not gold_pairs:
            skipped += 1  # no fetchable gold evidence -> unanswerable, drop
            continue

        gold_titles = set(titles)
        need = max(FRAMES_POOL_SIZE - len(gold_pairs), 0)
        pool_pairs = list(gold_pairs)
        if need > 0:
            candidates = [p for p in distractor_bank if p[0] not in gold_titles]
            rng.shuffle(candidates)
            pool_pairs.extend(candidates[:need])
        rng.shuffle(pool_pairs)

        built.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row["Prompt"]},
                ],
                "ground_truth": str(row.get("Answer", "")),
                "context": {
                    "titles": [t for t, _ in pool_pairs],
                    "paragraphs": [_para_to_sentences(p) for _, p in pool_pairs],
                },
                "source": "frames",
                "question_type": str(row.get("reasoning_types", "")),
            }
        )

    # Deterministic disjoint partition: held-out eval is a fixed prefix.
    order = list(range(len(built)))
    random.Random(seed).shuffle(order)
    eval_idx = set(order[:eval_size])
    if split == "eval":
        rows = [built[i] for i in order[:eval_size]]
    else:
        rows = [built[i] for i in order[eval_size:]]
    print(
        f"  Built {len(built)} answerable rows (skipped {skipped}); "
        f"split={split} -> {len(rows)} rows (eval_size={eval_size})"
    )
    if max_rows and max_rows > 0:
        rows = rows[:max_rows]
    return rows


def main():
    parser = argparse.ArgumentParser(description="Prepare multi-hop QA dataset for IGPO training")
    parser.add_argument("--max-rows", type=int, default=2000,
                        help="Maximum rows to output (total across all datasets)")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--dataset", default="hotpotqa",
                        choices=["hotpotqa", "musique", "2wiki", "frames", "all"],
                        help="Which dataset(s) to use")
    parser.add_argument("--difficulty", default="hard",
                        choices=["hard", "medium", "easy", "all"],
                        help="HotpotQA difficulty filter (ignored for other datasets)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wiki-cache",
                        default=os.path.join(os.path.dirname(__file__), "frames_wiki_cache.json"),
                        help="FRAMES only: on-disk cache of fetched Wikipedia article plaintext")
    parser.add_argument("--eval-size", type=int, default=FRAMES_DEFAULT_EVAL_SIZE,
                        help="FRAMES only: held-out eval split size (rest is train)")
    args = parser.parse_args()

    random.seed(args.seed)
    all_rows = []

    if args.dataset == "frames":
        # FRAMES is upstream eval-only; we deterministically partition it into a
        # disjoint held-out eval (fixed eval_size prefix) and train (remainder).
        # Require the split to be named explicitly so an unexpected value -- incl.
        # the ``validation`` default other datasets use -- can't silently fall
        # through to the train partition.
        if args.split not in ("train", "eval"):
            parser.error(
                "--dataset frames requires --split train or --split eval "
                f"(got --split {args.split!r}); the two partitions are "
                "deterministic and disjoint."
            )
        frames_split = args.split
        all_rows = load_frames(
            frames_split,
            args.max_rows,
            seed=args.seed,
            wiki_cache=args.wiki_cache,
            eval_size=args.eval_size,
        )
        with open(args.output, "w") as f:
            for entry in all_rows:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(all_rows)} rows to {args.output} (frames, split={frames_split})")
        return

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
