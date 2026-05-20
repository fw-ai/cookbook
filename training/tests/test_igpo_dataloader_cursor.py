"""Source-string regression guards for IGPO's cursor wiring.

The cursor mechanics themselves are tested in ``test_dataloader_cursor.py``.
These tests catch a drive-by edit that drops one of the cursor calls or
revives the legacy step-derived formula.
"""

from __future__ import annotations

from pathlib import Path

_IGPO_LOOP = Path(__file__).parent / "../recipes/igpo_loop.py"


def test_igpo_loop_imports_cursor_and_stats_helper():
    src = _IGPO_LOOP.read_text()
    assert "RawRowCursor" in src, "missing RawRowCursor import"
    assert "raw_rows_from_stats" in src, "missing raw_rows_from_stats import"


def test_igpo_loop_calls_cursor_lifecycle():
    """``cursor.resume → cursor.record → ckpt.save(data_consumed=cursor.value)``."""
    src = _IGPO_LOOP.read_text()
    assert "cursor.resume(" in src, "missing cursor.resume(...)"
    assert "cursor.record(" in src, "missing cursor.record(...)"
    assert "data_consumed=cursor.value" in src, (
        "ckpt.save must persist cursor.value; old step-derived formula "
        "re-introduces drift on filter/sample drops"
    )


def test_igpo_loop_source_no_legacy_data_consumed_formula():
    """The legacy step-derived formula must not return."""
    src = _IGPO_LOOP.read_text()
    assert "(step - step_offset) * prompt_groups_per_step" not in src
    assert "(global_step - step_offset) * prompt_groups_per_step" not in src
