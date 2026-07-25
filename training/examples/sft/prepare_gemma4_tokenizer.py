#!/usr/bin/env python3
"""Create a Gemma 4 tokenizer that maps aliases onto reserved vocabulary rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.utils.tokenizers import load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map customer token strings onto Gemma 4 <unusedN> IDs without "
            "changing the tokenizer or embedding-matrix size."
        )
    )
    parser.add_argument(
        "--source",
        default="google/gemma-4-E2B-it",
        help="Hugging Face Gemma 4 tokenizer name or local directory",
    )
    parser.add_argument("--output", required=True, help="Output tokenizer directory")
    parser.add_argument(
        "--token-map",
        type=json.loads,
        required=True,
        metavar="JSON",
        help=(
            "Token alias -> reserved slot, for example "
            '\'{"<start_search>":"<unused123>"}\''
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(
        args.source,
        gemma4_reserved_token_map=args.token_map,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output)

    resolved = {
        alias: tokenizer.encode(alias, add_special_tokens=False)[0]
        for alias in args.token_map
    }
    print(f"Saved fixed-size tokenizer ({len(tokenizer)} tokens) to {output}")
    print(json.dumps(resolved, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
