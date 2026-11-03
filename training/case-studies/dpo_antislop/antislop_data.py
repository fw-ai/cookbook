"""Preference-row validation for the Anti-slop DPO notebook (``dpo_glm53_flash.ipynb``).

:func:`validate_preference_rows` checks every row of a preference file against
the three schemas ``training/utils/data.py`` accepts and the two mistakes that
produce no error and a meaningless run: prompt prefixes that differ between the
chosen and rejected side, and identical responses. The notebook imports it in
section 2 and keeps the hold-out split visible in the cell.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if (_ROOT / "training" / "utils").is_dir() and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.utils.data import normalize_preference_row  # noqa: E402


def validate_preference_rows(rows, show=3):
    """Validate rows against the three supported DPO schemas.

    Returns (good_rows, report). Checks, ordered by how quietly each breaks a run:
      1. recognized schema (normalize_preference_row accepts it)
      2. both sides carry a non-empty `messages` list
      3. every message has a string `role` and a non-empty string `content`
      4. each side's LAST message is the assistant response being compared
      5. prompt prefixes are IDENTICAL across chosen/rejected  <- the silent killer
      6. the two responses actually differ (identical pair = zero gradient)
    """
    good, problems = [], []
    fmt = {"chosen/rejected": 0, "input/preferred_output": 0, "samples": 0}

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            problems.append((i, f"not a JSON object (got {type(row).__name__})"))
            continue

        if "chosen" in row and "rejected" in row:
            fmt["chosen/rejected"] += 1
        elif "preferred_output" in row and "non_preferred_output" in row:
            fmt["input/preferred_output"] += 1
        elif "samples" in row:
            fmt["samples"] += 1

        norm = normalize_preference_row(row)
        if norm is None:
            problems.append((i, "unrecognized schema -- needs chosen/rejected, "
                                "input+preferred_output+non_preferred_output, or scored samples"))
            continue

        sides, failed = {}, None
        for side in ("chosen", "rejected"):
            msgs = (norm.get(side) or {}).get("messages")
            if not isinstance(msgs, list) or not msgs:
                failed = f"{side}: missing or empty 'messages' list"
                break
            for j, m in enumerate(msgs):
                if not isinstance(m, dict):
                    failed = f"{side}[{j}]: message is not an object"
                elif not isinstance(m.get("role"), str) or not m["role"]:
                    failed = f"{side}[{j}]: missing or invalid 'role'"
                elif not isinstance(m.get("content"), str) or not m["content"].strip():
                    failed = f"{side}[{j}]: 'content' must be a non-empty string"
                if failed:
                    break
            if failed:
                break
            if msgs[-1].get("role") != "assistant":
                failed = (f"{side}: last message must be the assistant response "
                          f"(got role={msgs[-1].get('role')!r})")
                break
            sides[side] = msgs
        if failed:
            problems.append((i, failed))
            continue

        prompt_c = [(m["role"], m["content"]) for m in sides["chosen"][:-1]]
        prompt_r = [(m["role"], m["content"]) for m in sides["rejected"][:-1]]
        if prompt_c != prompt_r:
            problems.append((i, "chosen/rejected prompt prefixes DIFFER -- DPO would measure "
                                "the prompt difference, not response quality"))
        elif not prompt_c:
            problems.append((i, "no prompt before the assistant response"))
        elif sides["chosen"][-1]["content"].strip() == sides["rejected"][-1]["content"].strip():
            problems.append((i, "chosen and rejected responses are identical (zero gradient)"))
        else:
            good.append(row)

    report = {"total": len(rows), "valid": len(good), "rejected": len(problems),
              "formats_seen": {k: v for k, v in fmt.items() if v}, "problems": problems}

    print(f"{report['valid']}/{report['total']} rows valid")
    print("formats seen:", report["formats_seen"] or "NONE RECOGNIZED")
    if problems and show:
        print(f"\n{len(problems)} rejected. First {min(show, len(problems))}:")
        for i, msg in problems[:show]:
            print(f"  row {i}: {msg}")
    return good, report
