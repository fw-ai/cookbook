"""Load and evaluate inspection rules.

Rules describe attribute combinations on ``audit_table`` rows that are
"worth a closer look". They live in ``inspect_rules.yaml`` so the
viewer and CLI scans share one source of truth.

A row that matches at least one rule is "flagged"; the GUI tints it
amber and the bug-check script lists the matching reason(s).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml

DEFAULT_RULES_PATH = Path(__file__).parent.parent / "rules" / "inspect_rules.yaml"


def load_rules(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Parse the YAML rule file. Raises on malformed entries."""
    p = Path(path) if path else DEFAULT_RULES_PATH
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    rules = data.get("rules") or []
    for r in rules:
        for required in ("id", "when", "reason"):
            if required not in r:
                raise ValueError(f"rule missing {required!r}: {r!r}")
        if not isinstance(r["when"], dict):
            raise ValueError(f"rule {r['id']!r}: `when` must be a mapping")
    return rules


def _row_value(row: dict[str, Any], field: str, *, is_special: bool) -> Any:
    if field == "trainable":
        return (row.get("renderer_claim_weight") or 0) > 0.5
    if field == "special":
        return is_special
    return row.get(field)


def _matches(row: dict[str, Any], when: dict[str, Any], *, is_special: bool) -> bool:
    for field, expected in when.items():
        actual = _row_value(row, field, is_special=is_special)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def evaluate(
    row: dict[str, Any],
    rules: Iterable[dict[str, Any]],
    *,
    is_special: bool = False,
) -> list[str]:
    """Return the reason text for every rule that matches the row."""
    return [
        rule["reason"]
        for rule in rules
        if _matches(row, rule["when"], is_special=is_special)
    ]
