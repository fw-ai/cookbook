#!/usr/bin/env python3
"""Repair notebook outputs that are missing nbformat-required fields.

Some editors write minimal output objects -- a `stream` without `name`, a
`display_data` without `metadata` -- which are valid enough to render but fail
`nbformat.validate`, and therefore fail CI and the Colab badge.

    python common/fix_nbformat.py path/to/notebook.ipynb [more.ipynb ...]
    python common/fix_nbformat.py --check path/to/notebook.ipynb   # exit 1 if invalid
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nbformat


def repair(nb: dict) -> int:
    """Fill in required-but-missing output fields. Returns the number of fixes."""
    fixed = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            # Only code cells carry outputs; strip stray ones elsewhere.
            if cell.pop("outputs", None) is not None:
                fixed += 1
            if cell.pop("execution_count", None) is not None:
                fixed += 1
            continue
        cell.setdefault("outputs", [])
        cell.setdefault("execution_count", None)
        for out in cell["outputs"]:
            kind = out.get("output_type")
            if kind == "stream" and "name" not in out:
                out["name"] = "stdout"
                fixed += 1
            if kind in ("display_data", "execute_result") and "metadata" not in out:
                out["metadata"] = {}
                fixed += 1
            if kind == "execute_result" and "execution_count" not in out:
                out["execution_count"] = cell.get("execution_count")
                fixed += 1
    return fixed


def main(argv: list[str]) -> int:
    check_only = "--check" in argv
    paths = [Path(a) for a in argv if not a.startswith("-")]
    if not paths:
        print(__doc__)
        return 2

    bad = 0
    for p in paths:
        original = json.loads(p.read_text())

        if check_only:
            # Validate what is actually on disk, not the repaired copy -- otherwise
            # --check reports VALID for a file that would fail in CI.
            n = repair(json.loads(json.dumps(original)))
            try:
                nbformat.validate(nbformat.reads(p.read_text(), as_version=4))
                print(f"{p}: VALID")
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"{p}: INVALID ({n} field(s) missing) -- {str(e).splitlines()[0]}")
            continue

        n = repair(original)
        if n:
            p.write_text(json.dumps(original, indent=1, ensure_ascii=False))
        try:
            nbformat.validate(nbformat.reads(json.dumps(original), as_version=4))
            status = "VALID"
        except Exception as e:  # noqa: BLE001
            status = f"INVALID: {str(e).splitlines()[0]}"
            bad += 1
        print(f"{p}: fixed {n} output field(s) -> {status}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
