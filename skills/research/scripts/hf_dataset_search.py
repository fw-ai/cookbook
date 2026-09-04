#!/usr/bin/env python3
"""Search public Hugging Face datasets for discover (read-only, no downloads).

Usage:
  python hf_dataset_search.py --query "receipt ocr json" --method sft --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

HUB_API = "https://huggingface.co/api/datasets"

METHOD_TAGS: dict[str, tuple[str, ...]] = {
    "sft": ("text-generation", "text2text-generation", "summarization"),
    "dpo": ("text-generation",),
    "rft": ("question-answering", "text-generation"),
    "embedding": ("sentence-similarity", "feature-extraction"),
}


def _get(url: str) -> dict | list:
    req = urllib.request.Request(url, headers={"User-Agent": "fireworks-discover/0.1"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode())


def search_datasets(query: str, limit: int = 8) -> list[dict]:
    params = urllib.parse.urlencode({"search": query, "limit": str(limit)})
    data = _get(f"{HUB_API}?{params}")
    return data if isinstance(data, list) else []


def fetch_dataset_detail(repo_id: str) -> dict:
    try:
        detail = _get(f"{HUB_API}/{urllib.parse.quote(repo_id, safe='')}")
        return detail if isinstance(detail, dict) else {}
    except urllib.error.HTTPError:
        return {}


def schema_hint(features: dict | None, method: str | None) -> str:
    if not features:
        return "unknown — inspect dataset card"
    keys = {k.lower() for k in features}
    if method == "dpo" and {"chosen", "rejected"} <= keys:
        return "preference pairs (chosen/rejected)"
    if method == "embedding" and (
        {"query", "passage"} <= keys or {"anchor", "positive"} <= keys
    ):
        return "query/passage or anchor/positive"
    if "messages" in keys:
        return "chat messages"
    if {"instruction", "output"} <= keys or {"input", "output"} <= keys:
        return "instruction/output pairs"
    if "text" in keys:
        return "text field — may need column mapping"
    return f"columns: {', '.join(sorted(keys)[:6])}"


def fit_notes(item: dict, detail: dict, method: str | None) -> str:
    notes: list[str] = []
    tags = set(item.get("tags") or []) | set(detail.get("tags") or [])
    if item.get("gated"):
        notes.append("gated — needs HF_TOKEN")
    license_ = detail.get("license") or item.get("license")
    if license_:
        notes.append(f"license: {license_}")
    if method and method in METHOD_TAGS:
        if not tags.intersection(METHOD_TAGS[method]):
            notes.append(f"weak tag overlap for {method}")
    downloads = item.get("downloads") or 0
    if downloads < 50:
        notes.append("low download count — verify quality")
    return "; ".join(notes) if notes else "ok"


def rank_score(item: dict, method: str | None) -> float:
    downloads = float(item.get("downloads") or 0)
    score = min(downloads / 1000.0, 10.0)
    tags = set(item.get("tags") or [])
    if method and method in METHOD_TAGS and tags.intersection(METHOD_TAGS[method]):
        score += 3.0
    if item.get("gated"):
        score -= 2.0
    return score


def main() -> int:
    parser = argparse.ArgumentParser(description="HF dataset search for discover")
    parser.add_argument("--query", required=True, help="Search keywords (no PII)")
    parser.add_argument("--method", choices=["sft", "dpo", "rft", "embedding"])
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--fetch-detail", action="store_true", help="Fetch card per hit")
    args = parser.parse_args()

    hits = search_datasets(args.query, limit=max(args.limit * 2, 8))
    results: list[dict] = []

    for item in hits:
        repo_id = item.get("id")
        if not repo_id:
            continue
        detail: dict = {}
        features = None
        if args.fetch_detail:
            detail = fetch_dataset_detail(repo_id)
            card = detail.get("cardData") or {}
            features = card.get("dataset_info", [{}])[0].get("features") if card else None
            if not features and isinstance(detail.get("siblings"), list):
                features = None

        entry = {
            "id": repo_id,
            "downloads": item.get("downloads"),
            "tags": item.get("tags") or [],
            "gated": bool(item.get("gated")),
            "hub_url": f"https://huggingface.co/datasets/{repo_id}",
            "card_summary": (detail.get("description") or item.get("description") or "")[:200],
            "schema_hint": schema_hint(features, args.method),
            "fit_notes": fit_notes(item, detail, args.method),
            "_score": rank_score(item, args.method),
        }
        results.append(entry)

    results.sort(key=lambda x: x.pop("_score"), reverse=True)
    results = results[: args.limit]

    json.dump(
        {
            "query": args.query,
            "method": args.method,
            "count": len(results),
            "datasets": results,
        },
        sys.stdout,
        indent=2,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
