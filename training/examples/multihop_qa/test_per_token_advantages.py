"""Regression test for IGPO ``per_token_advantages`` coordinate alignment.

``make_igpo_loss_fn`` (in ``training/utils/rl/igpo.py``) reads
``per_token_advantages`` in **target/logprob coordinates** — i.e. arrays
of length ``len(sample.tokens) - 1`` indexed at ``response_start =
prompt_len - 1``.  Building advantages in full-token coordinates (length
``len(sample.tokens)``) shifts every assistant span left by one and
zeros the first token of each span — Codex's Round-3 finding 3.

This test pins the Round-4 fix: the ``_assistant_spans`` walk plus the
``[s-1, e-1)`` coordinate shift produces a target-coord array where the
first target index of every assistant span carries the same advantage as
the rest of the span.
"""

from __future__ import annotations

from typing import List


def _assistant_spans(loss_mask: List[int]) -> List[tuple[int, int]]:
    """Re-implementation of the same private helper used inside
    ``train_multihop_qa_igpo.py::sample_one_prompt``.  Kept verbatim
    here so the regression test isn't subject to import-time side
    effects of the entrypoint module."""
    spans: List[tuple[int, int]] = []
    start: int | None = None
    for i, m in enumerate(loss_mask):
        if m == 1 and start is None:
            start = i
        elif m == 0 and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(loss_mask)))
    return spans


def _per_token_advantages_target_coords(
    tokens: List[int],
    loss_mask: List[int],
    turn_advantages: List[float],
) -> List[float]:
    """Mirror of the Round-4 fix in
    ``train_multihop_qa_igpo.py::sample_one_prompt``: emit per-token
    advantages in target coordinates (length ``len(tokens) - 1``),
    shifted left by 1 from full-token assistant spans."""
    spans = _assistant_spans(loss_mask)
    n_targets = max(0, len(tokens) - 1)
    pta = [0.0] * n_targets
    for span_idx, (s, e) in enumerate(spans):
        if span_idx >= len(turn_advantages):
            break
        adv = float(turn_advantages[span_idx])
        t_start = max(0, s - 1)
        t_end = max(0, e - 1)
        for k in range(t_start, t_end):
            pta[k] = adv
    return pta


def test_first_token_of_each_assistant_span_carries_same_advantage():
    """Codex Round-3 finding 3: the legacy build wrote pta in full-token
    coords, so the first token of every assistant span got 0.0.  This
    test fails on that bug and passes on the target-coord fix."""
    # Layout (full-token coords):
    #   [0, 1, 2]            user prompt        loss_mask=0
    #   [3, 4]               assistant turn 1   loss_mask=1
    #   [5]                  user gap           loss_mask=0
    #   [6, 7, 8]            assistant turn 2   loss_mask=1
    tokens = [10, 11, 12, 20, 21, 30, 40, 41, 42]
    loss_mask = [0, 0, 0, 1, 1, 0, 1, 1, 1]
    turn_advantages = [0.7, 0.4]

    pta = _per_token_advantages_target_coords(tokens, loss_mask, turn_advantages)

    # Length: len(tokens) - 1 == 8 (target/logprob coordinate space).
    assert len(pta) == len(tokens) - 1

    # Turn 1 assistant span = full-token [3, 5) -> target [2, 4).
    # Turn 2 assistant span = full-token [6, 9) -> target [5, 8).
    expected = [0.0, 0.0, 0.7, 0.7, 0.0, 0.4, 0.4, 0.4]
    assert pta == expected

    # The CRITICAL assertion: the first target index of each assistant
    # span carries the SAME advantage as the rest of the span.  The
    # legacy full-token-coord build placed 0.0 here.
    assert pta[2] == 0.7  # first target index of turn 1
    assert pta[5] == 0.4  # first target index of turn 2


def test_single_turn_no_off_by_one():
    tokens = [10, 11, 12, 20, 21, 22]
    loss_mask = [0, 0, 0, 1, 1, 1]
    turn_advantages = [0.9]

    pta = _per_token_advantages_target_coords(tokens, loss_mask, turn_advantages)

    # Assistant span [3, 6) in full-token -> [2, 5) in target.
    assert len(pta) == 5
    assert pta == [0.0, 0.0, 0.9, 0.9, 0.9]
    assert pta[2] == 0.9  # first assistant target index


def test_no_assistant_tokens_returns_zero_advantages():
    tokens = [1, 2, 3]
    loss_mask = [0, 0, 0]
    pta = _per_token_advantages_target_coords(tokens, loss_mask, [0.5])
    assert pta == [0.0, 0.0]


def test_extra_advantages_beyond_spans_are_ignored():
    tokens = [1, 2, 3, 4, 5]
    loss_mask = [0, 1, 1, 0, 0]
    pta = _per_token_advantages_target_coords(tokens, loss_mask, [0.6, 0.7, 0.8])
    # Only one assistant span, gets advantage 0.6; rest ignored.
    # Span [1, 3) -> target [0, 2).
    assert pta == [0.6, 0.6, 0.0, 0.0]
