#!/usr/bin/env python3
"""Download the pinned kalshi-decision-v1 datasets from Hugging Face.

Splits (row counts and sha256 are pinned in SPLITS below):

  train          6,461  — category-balanced training rows
  dev            2,000  — calibration / fit-T evaluation split
  test           8,765  — full held-out eval (original test + 6k expand + unseen)
  unseen_family    765  — same rows as the unseen slice inside test, also
                          published alone for generalization reporting

Usage (from ``training/``)::

    python -m training.examples.decision.kalshi_v1.prepare_data
    python -m training.examples.decision.kalshi_v1.prepare_data --splits train,dev
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import requests

HF_REPO_ID = "fireworks-ai/kalshi-decision-v1"
HF_REVISION = "f5cfca40e6b2187ac62ddb3a6bf18a1c224c26de"
DATA_DIR = Path(__file__).resolve().parent / "data"

# Pins for the public accounts/fireworks/datasets/kalshi-decision-v1-* uploads.
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


def download_split(split: str, force: bool) -> Path:
    meta = SPLITS[split]
    expect_sha = str(meta["sha256"])
    expect_rows = int(meta["rows"])
    out_path = DATA_DIR / f"{split}.jsonl"

    if out_path.exists() and not force:
        got_sha = _sha256_file(out_path)
        if got_sha == expect_sha:
            print(f"{split}: already present ({expect_rows} rows, sha256 ok)")
            return out_path
        print(f"{split}: local file hash mismatch; re-downloading")

    url = (
        f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/"
        f"{HF_REVISION}/{split}.jsonl"
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".jsonl.partial")
    try:
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with tmp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        handle.write(chunk)

        # Verify the partial file before replacing any existing good copy.
        got_sha = _sha256_file(tmp_path)
        with tmp_path.open(encoding="utf-8") as handle:
            got_rows = sum(1 for line in handle if line.strip())
        if got_sha != expect_sha:
            raise RuntimeError(f"{split}: sha256 {got_sha} != pinned {expect_sha}")
        if got_rows != expect_rows:
            raise RuntimeError(f"{split}: {got_rows} rows != pinned {expect_rows}")
        tmp_path.replace(out_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

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

    for split in wanted:
        download_split(split, force=args.force)

    print(f"source: https://huggingface.co/datasets/{HF_REPO_ID}/tree/{HF_REVISION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
