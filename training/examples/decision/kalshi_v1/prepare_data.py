#!/usr/bin/env python3
"""Download the pinned kalshi-decision-v1 public Fireworks datasets.

Splits (row counts and sha256 are pinned in SPLITS below):

  train          6,461  — category-balanced training rows
  dev            2,000  — calibration / fit-T evaluation split
  test           8,765  — full held-out eval (original test + 6k expand + unseen)
  unseen_family    765  — same rows as the unseen slice inside test, also
                          published alone for generalization reporting

Usage (from ``training/``)::

    export FIREWORKS_API_KEY=fw_...
    python -m examples.decision.kalshi_v1.prepare_data
    python -m examples.decision.kalshi_v1.prepare_data --splits train,dev
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import requests
from fireworks import Fireworks

ACCOUNT = "fireworks"
DATA_DIR = Path(__file__).resolve().parent / "data"

# Pins match ~/Desktop/Docs/kalshi-decision-v1-publish/publish_manifest.json
# (built from Decision Model Tuning store bc-7cc7d5bc, including test_extra).
SPLITS: dict[str, dict[str, object]] = {
    "train": {
        "dataset_id": "kalshi-decision-v1-train",
        "rows": 6461,
        "sha256": "158e2453238a08b696fb2138e8c5c9639ddc90ce865161ee1cccaebece4da68c",
    },
    "dev": {
        "dataset_id": "kalshi-decision-v1-dev",
        "rows": 2000,
        "sha256": "e62512a80b5449f321fe48973cfbac9596d525410d3a9ee376b9c4b7b6d5b4f7",
    },
    "test": {
        "dataset_id": "kalshi-decision-v1-test",
        "rows": 8765,
        "sha256": "686dba76ca3a07aa56bf9693b963846b27973f4328e9eb7957f1a8a614caafb7",
        "composition": "test(2000) + test_extra(6000) + unseen_family(765)",
    },
    "unseen_family": {
        "dataset_id": "kalshi-decision-v1-unseen-family",
        "rows": 765,
        "sha256": "0c2166295d38f1d1b889b4cd4c65865e58dc4b5c98ea8e4d6299d9c72df1b2b2",
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pick_signed_url(filename_to_urls: dict[str, str], split: str) -> tuple[str, str]:
    """Return (remote_filename, signed_url) for this split's JSONL."""
    preferred = f"{split}.jsonl"
    if preferred in filename_to_urls:
        return preferred, filename_to_urls[preferred]
    jsonl_names = [name for name in filename_to_urls if name.endswith(".jsonl")]
    if len(jsonl_names) == 1:
        return jsonl_names[0], filename_to_urls[jsonl_names[0]]
    raise RuntimeError(
        f"dataset for {split} has unexpected files: {sorted(filename_to_urls)}; "
        f"expected a single .jsonl (preferably {preferred})"
    )


def download_split(client: Fireworks, split: str, force: bool) -> Path:
    meta = SPLITS[split]
    dataset_id = str(meta["dataset_id"])
    expect_sha = str(meta["sha256"])
    expect_rows = int(meta["rows"])
    out_path = DATA_DIR / f"{split}.jsonl"

    if out_path.exists() and not force:
        got_sha = _sha256_file(out_path)
        if got_sha == expect_sha:
            print(f"{split}: already present ({expect_rows} rows, sha256 ok)")
            return out_path
        print(f"{split}: local file hash mismatch; re-downloading")

    endpoint = client.datasets.get_download_endpoint(
        dataset_id=dataset_id,
        account_id=ACCOUNT,
    )
    urls = endpoint.filename_to_signed_urls or {}
    if not urls:
        raise RuntimeError(
            f"no signed URLs for accounts/{ACCOUNT}/datasets/{dataset_id}; "
            "confirm the dataset is READY and PUBLIC"
        )
    _remote_name, url = _pick_signed_url(urls, split)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".jsonl.partial")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)
    tmp_path.replace(out_path)

    got_sha = _sha256_file(out_path)
    got_rows = sum(1 for line in out_path.open() if line.strip())
    if got_sha != expect_sha:
        raise RuntimeError(f"{split}: sha256 {got_sha} != pinned {expect_sha}")
    if got_rows != expect_rows:
        raise RuntimeError(f"{split}: {got_rows} rows != pinned {expect_rows}")
    print(f"{split}: downloaded {got_rows} rows -> {out_path}")
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        default="train,dev,test,unseen_family",
        help="Comma-separated subset of: " + ",".join(SPLITS),
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if hashes match.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    wanted = [part.strip() for part in args.splits.split(",") if part.strip()]
    unknown = [name for name in wanted if name not in SPLITS]
    if unknown:
        print(f"unknown splits: {unknown}; choose from {list(SPLITS)}", file=sys.stderr)
        return 2

    client = Fireworks()
    for split in wanted:
        download_split(client, split, force=args.force)

    print("pinned resources:")
    for split in wanted:
        dataset_id = SPLITS[split]["dataset_id"]
        print(f"  {split}: accounts/{ACCOUNT}/datasets/{dataset_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
