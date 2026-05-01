"""Unit tests for ``training.utils.data`` helper utilities."""

from __future__ import annotations

from training.utils import replicate_rows_for_epochs


def test_replicate_rows_for_epochs_each_row_is_independent():
    """The naive ``rows * epochs`` only multiplies list references --
    every epoch shares the same dict instances.  Any rollout function
    that mutates its input row in place (attaching scratch fields,
    normalizing prompt data, caching renders) leaks that mutation
    into every later epoch and subsequent passes train on the
    already-mutated row instead of the original dataset.
    ``replicate_rows_for_epochs`` returns ``epochs * len(rows)``
    INDEPENDENT dict instances so per-epoch mutations cannot leak.
    """
    rows = [{"prompt": "a", "scratch": []}, {"prompt": "b", "scratch": []}]
    out = replicate_rows_for_epochs(rows, epochs=3)
    assert len(out) == 6  # 2 rows x 3 epochs
    # Every output dict is a distinct object -- id() differs even for
    # the same logical row across epochs.
    assert len({id(r) for r in out}) == 6
    # The nested mutable container (``scratch``) is also independent
    # -- that's what makes ``deepcopy`` (vs shallow ``dict(r)``) the
    # right primitive here.
    assert len({id(r["scratch"]) for r in out}) == 6
    # Mutating one copy does not affect any other copy or the originals.
    out[0]["scratch"].append("epoch0-mutation")
    assert all(r["scratch"] == [] for r in rows), (
        "Mutating a replicated copy leaked back into the originals -- "
        "deepcopy expected."
    )
    assert all(out[i]["scratch"] == [] for i in range(1, 6)), (
        "Mutating one copy leaked into other copies -- sibling rows "
        "must be independent across all epoch slots."
    )


def test_replicate_rows_for_epochs_zero_epochs_returns_empty():
    """``epochs=0`` is a degenerate but valid input; the helper
    returns an empty list rather than raising."""
    rows = [{"x": 1}]
    assert replicate_rows_for_epochs(rows, epochs=0) == []


def test_replicate_rows_for_epochs_preserves_per_epoch_order():
    """Rows are emitted epoch-by-epoch so the resume slicing logic
    (which advances a raw-row cursor through ``all_rows``) sees the
    same order it did when ``rows * epochs`` was used."""
    rows = [{"i": 0}, {"i": 1}, {"i": 2}]
    out = replicate_rows_for_epochs(rows, epochs=2)
    assert [r["i"] for r in out] == [0, 1, 2, 0, 1, 2]
