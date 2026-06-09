#!/usr/bin/env python3
"""Generate a SWE-Gym JSONL dataset for the coding-agent example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.examples.rl.coding_agent.swegym_data import (
    fetch_dataset_instances,
    row_for_instance,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--refresh-dataset-cache", action="store_true")
    args = parser.parse_args()
    if args.max_rows is not None and args.max_rows < 1:
        raise ValueError("--max-rows must be >= 1")

    instances = fetch_dataset_instances(args.split, refresh=args.refresh_dataset_cache)
    if args.max_rows is not None:
        instances = instances[: args.max_rows]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for instance in instances:
            f.write(json.dumps(row_for_instance(instance, args.split), ensure_ascii=True) + "\n")
    print(f"Wrote {len(instances)} SWE-Gym rows to {output}")


if __name__ == "__main__":
    main()
