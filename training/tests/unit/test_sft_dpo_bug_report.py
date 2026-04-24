"""Failing tests that document SFT/DPO bugs identified in review.

Every test in this file is expected to **fail on current main**. The accompanying
bug report explains the root cause; once each fix lands, the corresponding test
should flip to green.

Run with::

    pytest training/tests/unit/test_sft_dpo_bug_report.py -v
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import random as _random
from types import SimpleNamespace

import pytest
import torch

import training.recipes.dpo_loop as dpo_module
import training.recipes.orpo_loop as orpo_module
import training.recipes.sft_loop as sft_module
from training.utils.data import load_preference_dataset
from training.utils.losses import (
    make_batch_dpo_loss_fn,
    make_batch_orpo_loss_fn,
    make_sft_loss_fn,
)


# ---------------------------------------------------------------------------
# Bug 1 (P0): DPO loss drops the first response token's logprob.
#
# `make_batch_dpo_loss_fn` slices `logprobs[response_start:]`. Because logprobs
# are shifted by one (logprob[i] predicts token[i+1]), this silently drops
# the prediction of the *first* response token — typically the most
# discriminating token between chosen and rejected.
#
# ORPO's implementation correctly uses `lp_start = max(0, response_start - 1)`.
# ---------------------------------------------------------------------------


def test_dpo_loss_includes_first_response_token() -> None:
    """Construct a pair whose divergence is the *final* token.

    chosen=[A, B, C], rejected=[A, B, D], response_start=2.

    For each side, logprobs length = 2:
        logprobs[0] = logp(B | A)        -- same token on both sides
        logprobs[1] = logp(C | A, B)     -- chosen
        logprobs[1] = logp(D | A, B)     -- rejected

    The signal lives entirely in index 1. With the buggy slice `lp[2:]` it is
    empty, so `pi_chosen = pi_rejected = 0` and the margin collapses to 0.
    """
    chosen_logprobs = torch.tensor([-0.1, -0.1], dtype=torch.float32)
    rejected_logprobs = torch.tensor([-0.1, -2.0], dtype=torch.float32)
    ref_chosen = [-0.5, -1.0]
    ref_rejected = [-0.5, -1.0]
    response_start = 2

    loss_fn = make_batch_dpo_loss_fn(
        [ref_chosen], [ref_rejected], [response_start], beta=0.1,
    )
    _, metrics = loss_fn([None, None], [chosen_logprobs, rejected_logprobs])

    assert metrics["margin"] != pytest.approx(0.0, abs=1e-6), (
        "DPO margin is 0 because the first response token's logprob was sliced away. "
        "Fix: use `lp_start = max(0, response_start - 1)` (mirroring make_batch_orpo_loss_fn)."
    )
    assert metrics["margin"] > 0.0, (
        f"Policy strongly favors chosen over rejected on the only differing token, "
        f"so DPO margin should be positive. Got margin={metrics['margin']:.4f}."
    )


def test_dpo_and_orpo_agree_on_preferred_direction() -> None:
    """Given identical inputs, DPO and ORPO should both identify chosen as preferred."""
    chosen_logprobs = torch.tensor([-0.1, -0.1], dtype=torch.float32)
    rejected_logprobs = torch.tensor([-0.1, -2.0], dtype=torch.float32)
    ref_chosen = [-0.5, -1.0]
    ref_rejected = [-0.5, -1.0]
    response_start = 2

    dpo_loss = make_batch_dpo_loss_fn(
        [ref_chosen], [ref_rejected], [response_start], beta=0.1,
    )
    _, dpo_metrics = dpo_loss([None, None], [chosen_logprobs.clone(), rejected_logprobs.clone()])

    orpo_loss = make_batch_orpo_loss_fn([response_start], orpo_lambda=1.0)
    _, orpo_metrics = orpo_loss([None, None], [chosen_logprobs.clone(), rejected_logprobs.clone()])

    assert orpo_metrics["log_odds_ratio"] > 0.0, (
        "Sanity check: ORPO should prefer chosen; if this fails, the test setup is wrong."
    )
    assert dpo_metrics["accuracy"] == orpo_metrics["accuracy"], (
        f"DPO and ORPO disagree on preferred direction "
        f"(dpo_acc={dpo_metrics['accuracy']}, orpo_acc={orpo_metrics['accuracy']}). "
        "Root cause: DPO's logprob slice is off by one relative to ORPO's."
    )


def test_single_sample_sft_loss_includes_first_response_token() -> None:
    """Same off-by-one in `make_sft_loss_fn` (single-sample helper).

    Full sequence [A, B, C] with response_start=2 (response is just "C").
    target_tokens=[B, C], logprobs=[logp(B|A), logp(C|A,B)].

    Expected loss: -logp(C|A,B) = 1.0.
    Buggy code slices `lp[2:]` which is empty and returns 0.0.
    """
    target_tokens = [10, 20]  # B, C
    response_start = 2
    logprobs = torch.tensor([0.0, -1.0], dtype=torch.float32)

    loss_fn = make_sft_loss_fn(response_start, target_tokens)
    loss, metrics = loss_fn([None], [logprobs])

    assert metrics["response_tokens"] == 1, (
        f"response_start=2 covers exactly 1 response token in a length-3 sequence; "
        f"got {metrics['response_tokens']}"
    )
    assert loss.item() == pytest.approx(1.0, abs=1e-5), (
        f"SFT single-sample loss dropped the first response token's logprob. "
        f"Expected CE=1.0 (from logp(C|A,B)=-1.0), got {loss.item():.4f}. "
        "Fix: use `lp_start = max(0, response_start - 1)` (same fix as DPO)."
    )


# ---------------------------------------------------------------------------
# Bug 2 (P0): DPO does not shuffle data between epochs.
#
# `_train_loop` iterates `tokenized_pairs` in file order during epoch 0, then
# rebuilds `ordered_pairs = [ref_cache[idx] for idx, _ in tokenized_pairs]`
# using the *same* index order for every subsequent epoch.
# ---------------------------------------------------------------------------


def _make_tokenized_pair(idx: int) -> tuple[int, dict]:
    return idx, {
        "chosen_tokens": [1, 2, 3],
        "rejected_tokens": [1, 2, 4],
        "response_start": 2,
        "chosen_datum": {"id": f"c{idx}"},
        "rejected_datum": {"id": f"r{idx}"},
    }


def test_dpo_train_loop_shuffles_data_between_epochs(monkeypatch) -> None:
    """Run `_train_loop` for 2 epochs and capture the per-step pair ordering.

    On current main, epoch 1 visits pairs in the exact same order as epoch 0,
    producing a perfectly correlated data schedule across epochs.
    """
    tokenized_pairs = [_make_tokenized_pair(i) for i in range(8)]
    batches_by_epoch: list[list[list[str]]] = []
    current_epoch_batches: list[list[str]] = []
    step_counter = {"n": 0}
    steps_per_epoch = 4  # 8 pairs / batch_size 2

    def fake_fwd_bwd(batch, policy, beta):
        ids = [p["chosen_datum"]["id"] for p in batch]
        current_epoch_batches.append(ids)
        step_counter["n"] += 1
        if step_counter["n"] % steps_per_epoch == 0:
            batches_by_epoch.append(list(current_epoch_batches))
            current_epoch_batches.clear()
        return SimpleNamespace(metrics={"dpo_loss": 0.0, "margin": 0.0, "accuracy": 0.5})

    monkeypatch.setattr(dpo_module, "_forward_backward_pairs", fake_fwd_bwd)
    monkeypatch.setattr(dpo_module, "flush_timing", lambda: {})
    monkeypatch.setattr(dpo_module, "log_metrics_json", lambda *a, **kw: None)
    monkeypatch.setattr(dpo_module, "wandb_log", lambda *a, **kw: None)

    class FakeReference:
        def forward(self, datums, loss_fn):
            return SimpleNamespace(
                loss_fn_outputs=[
                    {"logprobs": SimpleNamespace(data=[-0.1, -0.2])} for _ in datums
                ]
            )

    class FakePolicy:
        def optim_step(self, _params, **kwargs):
            return SimpleNamespace(metrics={})

    cfg = dpo_module.Config(
        log_path="/tmp/dpo_bug_report",
        beta=0.1,
        epochs=2,
        batch_size=2,
        ref_cache_concurrency=4,
    )

    asyncio.run(
        dpo_module._train_loop(
            tokenized_pairs,
            FakeReference(),
            FakePolicy(),
            adam_params={"lr": 1e-4},
            cfg=cfg,
            step_offset=0,
        )
    )

    assert len(batches_by_epoch) == 2, (
        f"Expected 2 complete epochs, captured {len(batches_by_epoch)}"
    )
    assert batches_by_epoch[0] != batches_by_epoch[1], (
        f"Epoch 1 data order is identical to epoch 0 — no shuffle between epochs.\n"
        f"epoch0={batches_by_epoch[0]}\n"
        f"epoch1={batches_by_epoch[1]}\n"
        f"Fix: shuffle `ordered_pairs` before each epoch ≥ 1 "
        f"(e.g. `random.Random(seed + epoch).shuffle(ordered_pairs)`)."
    )


# ---------------------------------------------------------------------------
# Bug 3 (P0): load_preference_dataset silently drops invalid "samples" rows.
#
# The current code only accepts scores exactly equal to 1.0 or 0.0 and
# silently drops everything else. Per our preference dataset contract, any
# other score should be a hard error (the dataset is malformed).
# ---------------------------------------------------------------------------


def _write_jsonl(tmp_path, rows):
    path = tmp_path / "pref.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(path)


def test_load_preference_dataset_rejects_non_binary_scores(tmp_path) -> None:
    """A samples-format row with a score outside {0.0, 1.0} should raise."""
    rows = [
        {
            "samples": [
                {"messages": [{"role": "assistant", "content": "a"}], "score": 0.5},
                {"messages": [{"role": "assistant", "content": "b"}], "score": 0.8},
            ],
        },
    ]
    path = _write_jsonl(tmp_path, rows)

    with pytest.raises((ValueError, AssertionError)) as excinfo:
        load_preference_dataset(path)

    assert "score" in str(excinfo.value).lower(), (
        f"Expected the error to name the offending field; got: {excinfo.value!r}"
    )


def test_load_preference_dataset_rejects_rows_without_both_chosen_and_rejected(tmp_path) -> None:
    """A samples-format row that yields no chosen or no rejected must raise."""
    rows = [
        {
            "samples": [
                {"messages": [{"role": "assistant", "content": "a"}], "score": 1.0},
            ],
        },
    ]
    path = _write_jsonl(tmp_path, rows)

    with pytest.raises((ValueError, AssertionError)) as excinfo:
        load_preference_dataset(path)

    assert "rejected" in str(excinfo.value).lower()


def test_load_preference_dataset_rejects_duplicate_chosen_samples(tmp_path) -> None:
    """Two samples with score=1.0 in a single row is ambiguous and must raise."""
    rows = [
        {
            "samples": [
                {"messages": [{"role": "assistant", "content": "a"}], "score": 1.0},
                {"messages": [{"role": "assistant", "content": "b"}], "score": 1.0},
                {"messages": [{"role": "assistant", "content": "c"}], "score": 0.0},
            ],
        },
    ]
    path = _write_jsonl(tmp_path, rows)

    with pytest.raises(ValueError) as excinfo:
        load_preference_dataset(path)

    msg = str(excinfo.value).lower()
    assert "ambiguous" in msg or "multiple" in msg, (
        f"Expected the error to flag the duplicate chosen as ambiguous; got: {excinfo.value!r}"
    )


def test_load_preference_dataset_rejects_unknown_row_format(tmp_path) -> None:
    """A row that matches none of the supported preference formats must raise,
    not silently return 0 rows.
    """
    rows = [
        {"foo": "bar"},
        {"chosen": {"messages": [{"role": "assistant", "content": "a"}]}},  # missing rejected
    ]
    path = _write_jsonl(tmp_path, rows)

    with pytest.raises(ValueError) as excinfo:
        load_preference_dataset(path)

    assert "supported preference format" in str(excinfo.value).lower()


def test_load_preference_dataset_rejects_non_list_samples(tmp_path) -> None:
    """`samples` must be a list — a dict / string / int should raise, not crash
    with a cryptic AttributeError mid-iteration.
    """
    rows = [{"samples": {"this": "is a dict, not a list"}}]
    path = _write_jsonl(tmp_path, rows)

    with pytest.raises(ValueError) as excinfo:
        load_preference_dataset(path)

    assert "list" in str(excinfo.value).lower()


def test_load_preference_dataset_rejects_non_dict_sample_entry(tmp_path) -> None:
    """Individual samples entries must be dicts; a string must raise a
    fail-fast ValueError with file:line context, not AttributeError.
    """
    rows = [
        {
            "samples": [
                "not a dict",
                {"messages": [{"role": "assistant", "content": "b"}], "score": 0.0},
            ],
        },
    ]
    path = _write_jsonl(tmp_path, rows)

    with pytest.raises(ValueError) as excinfo:
        load_preference_dataset(path)

    assert "dict" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# Follow-up coverage: DPO must also shuffle epoch 0 so single-epoch runs on
# file-ordered customer data aren't trained in file order.
# ---------------------------------------------------------------------------


def test_dpo_train_loop_shuffles_epoch_zero(monkeypatch) -> None:
    """When epochs=1, epoch 0 must still be shuffled (before _ref_producer).

    Without this, a customer file sorted by source/date/difficulty is trained
    in exactly that order, which biases a single-epoch DPO run — the very
    problem the cross-epoch shuffle was introduced to solve.
    """
    tokenized_pairs = [_make_tokenized_pair(i) for i in range(8)]

    observed_order: list[str] = []

    def fake_fwd_bwd(batch, policy, beta):
        observed_order.extend(p["chosen_datum"]["id"] for p in batch)
        return SimpleNamespace(metrics={"dpo_loss": 0.0, "margin": 0.0, "accuracy": 0.5})

    monkeypatch.setattr(dpo_module, "_forward_backward_pairs", fake_fwd_bwd)
    monkeypatch.setattr(dpo_module, "flush_timing", lambda: {})
    monkeypatch.setattr(dpo_module, "log_metrics_json", lambda *a, **kw: None)
    monkeypatch.setattr(dpo_module, "wandb_log", lambda *a, **kw: None)

    class FakeReference:
        def forward(self, datums, loss_fn):
            return SimpleNamespace(
                loss_fn_outputs=[
                    {"logprobs": SimpleNamespace(data=[-0.1, -0.2])} for _ in datums
                ]
            )

    class FakePolicy:
        def optim_step(self, _params, **kwargs):
            return SimpleNamespace(metrics={})

    cfg = dpo_module.Config(
        log_path="/tmp/dpo_epoch0_shuffle",
        beta=0.1,
        epochs=1,
        batch_size=2,
        ref_cache_concurrency=4,
    )

    asyncio.run(
        dpo_module._train_loop(
            tokenized_pairs,
            FakeReference(),
            FakePolicy(),
            adam_params={"lr": 1e-4},
            cfg=cfg,
            step_offset=0,
        )
    )

    file_order = [f"c{i}" for i in range(8)]
    assert observed_order != file_order, (
        "Epoch 0 trained in exact file order; DPO must shuffle before epoch 0 "
        "so single-epoch runs on sorted customer data aren't biased. "
        "Fix: shuffle tokenized_pairs with random.Random(cfg.seed) at the top "
        "of _train_loop."
    )
    assert sorted(observed_order) == sorted(file_order), (
        f"Every pair must still be visited exactly once per epoch; got {observed_order}"
    )


# ---------------------------------------------------------------------------
# Bug 4 (P1): SFT silently drops the last partial batch each epoch.
#
# `SupervisedDatasetFromHFDataset.__len__` uses integer division and the SFT
# main loop iterates `range(len(dataset))`. With 10 rows and batch_size=4 the
# last 2 rows are never trained on.
# ---------------------------------------------------------------------------


def test_sft_data_pipeline_visits_every_training_row() -> None:
    """Reproduce the SFT main() data pipeline and show that rows are dropped.

    Ref: sft_loop.main() dataset construction and per-epoch iteration. With
    10 rows and batch_size=4, the buggy pipeline trains on only 8 rows
    (``SupervisedDatasetFromHFDataset.__len__`` uses integer division).

    Fix (user preference): ``sft_loop.pad_training_data_to_batch_size`` pads
    ``training_data`` up to a multiple of ``effective_batch_size`` before
    constructing the SupervisedDatasetFromHFDataset.
    """
    import datasets as hf_datasets
    from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset

    raw_training_data = list(range(10))
    effective_batch_size = 4

    # Apply the documented fix from sft_loop.main() before building the dataset.
    training_data = sft_module.pad_training_data_to_batch_size(
        raw_training_data, effective_batch_size,
    )

    sft_dataset = SupervisedDatasetFromHFDataset(
        hf_datasets.Dataset.from_dict(
            {"datum_idx": list(range(len(training_data)))}
        ),
        batch_size=effective_batch_size,
        map_fn=lambda row: training_data[row["datum_idx"]],
    )

    total_batches_per_epoch = len(sft_dataset)
    expected_batches = math.ceil(len(raw_training_data) / effective_batch_size)
    assert total_batches_per_epoch == expected_batches, (
        f"Dataset reports {total_batches_per_epoch} batches/epoch "
        f"but should cover all {len(raw_training_data)} rows "
        f"({expected_batches} batches). "
        f"Fix: pad training_data to a multiple of batch_size before constructing the dataset."
    )

    seen: list[int] = []
    for i_batch in range(total_batches_per_epoch):
        seen.extend(sft_dataset.get_batch(i_batch))

    missing = set(raw_training_data) - set(seen)
    assert not missing, (
        f"Rows silently dropped per epoch: {sorted(missing)}. "
        f"Fix: pad training_data to a multiple of batch_size."
    )


# ---------------------------------------------------------------------------
# Bug 5 (P1): SFT eval carve-out takes the first-N rendered rows without
# shuffling, producing a biased eval set whenever the input is ordered.
# ---------------------------------------------------------------------------


def test_sft_eval_carveout_shuffles_training_data_before_slicing() -> None:
    """Structural check: the eval carveout code path must shuffle before slicing.

    Currently the source performs ``eval_data = training_data[:carveout_count]``
    with no prior shuffle, so any dataset ordered by source/difficulty/date gives
    a biased eval set.
    """
    src = inspect.getsource(sft_module.main)
    carveout_marker = "eval_data = training_data[:carveout_count]"
    if carveout_marker not in src:
        pytest.fail(
            "Could not locate the carveout line in sft_loop.main(); "
            "test needs updating to match the new implementation."
        )
    preamble = src.split(carveout_marker)[0]
    has_shuffle = ("shuffle" in preamble.lower()) or ("random.sample" in preamble)
    if not has_shuffle:
        pytest.fail(
            "SFT eval carveout slices training_data[:N] without shuffling first. "
            "Any dataset ordered by source/difficulty/date yields a biased eval set. "
            "Fix: shuffle training_data with a configurable seed before carving out "
            "the eval set."
        )


# ---------------------------------------------------------------------------
# Bug 6 (P2): ORPO's per-epoch shuffle uses the global `random` state with
# no seed, so identical configs produce different orderings across runs.
# ---------------------------------------------------------------------------


def test_orpo_epoch_shuffle_uses_seeded_rng() -> None:
    """Structural check: ORPO must seed its per-epoch shuffle for reproducibility."""
    src = inspect.getsource(orpo_module)
    # Ensure the global-state shuffle is gone (or, at minimum, that a seeded
    # RNG is in use in the module).
    uses_unseeded_global_shuffle = (
        "random.shuffle(pair_cache)" in src
        and "random.seed" not in src
        and "random.Random" not in src
    )
    assert not uses_unseeded_global_shuffle, (
        "orpo_loop.py calls `random.shuffle(pair_cache)` on the global random state "
        "with no `random.seed` / `random.Random` anywhere in the module, so reruns "
        "with identical configs produce different data orderings. "
        "Fix: expose a `seed` in Config and use `random.Random(seed + epoch).shuffle(...)`."
    )


# ---------------------------------------------------------------------------
# Bug 7 (P2): DPO total_steps underestimates when the last batch is partial.
#
# The producer emits ceil(N/batch) batches per epoch but total_steps uses `//`.
# The UI shows "Step 7/6" for small datasets with a partial last batch.
# ---------------------------------------------------------------------------


def test_dpo_total_steps_matches_actual_step_count(monkeypatch) -> None:
    """Run one epoch with a partial last batch and confirm reported total_steps == executed steps."""
    tokenized_pairs = [_make_tokenized_pair(i) for i in range(5)]  # 5 pairs, bs=2 -> 3 batches

    executed = {"count": 0}
    reported_total_steps = {"value": None}

    def fake_fwd_bwd(batch, policy, beta):
        executed["count"] += 1
        return SimpleNamespace(metrics={"dpo_loss": 0.0, "margin": 0.0, "accuracy": 0.5})

    class RecordingRunner:
        def __init__(self):
            pass

        def start_training(self):
            pass

        def write_status(self, *args, **kwargs):
            total = kwargs.get("total_steps")
            if total is not None:
                reported_total_steps["value"] = total

        def write_metadata(self, *args, **kwargs):
            pass

        def append_metrics(self, *args, **kwargs):
            pass

        def report_rendering_progress(self, *args, **kwargs):
            pass

        def set_accelerator_info(self, *args, **kwargs):
            pass

    monkeypatch.setattr(dpo_module, "_forward_backward_pairs", fake_fwd_bwd)
    monkeypatch.setattr(dpo_module, "flush_timing", lambda: {})
    monkeypatch.setattr(dpo_module, "log_metrics_json", lambda *a, **kw: None)
    monkeypatch.setattr(dpo_module, "wandb_log", lambda *a, **kw: None)

    class FakeReference:
        def forward(self, datums, loss_fn):
            return SimpleNamespace(
                loss_fn_outputs=[
                    {"logprobs": SimpleNamespace(data=[-0.1, -0.2])} for _ in datums
                ]
            )

    class FakePolicy:
        def optim_step(self, _params, **kwargs):
            return SimpleNamespace(metrics={})

    cfg = dpo_module.Config(
        log_path="/tmp/dpo_total_steps",
        beta=0.1,
        epochs=1,
        batch_size=2,
        ref_cache_concurrency=4,
    )

    asyncio.run(
        dpo_module._train_loop(
            tokenized_pairs,
            FakeReference(),
            FakePolicy(),
            adam_params={"lr": 1e-4},
            cfg=cfg,
            step_offset=0,
            runner=RecordingRunner(),
        )
    )

    assert executed["count"] == 3, (
        f"Expected 3 training steps (ceil(5/2)), got {executed['count']}"
    )
    assert reported_total_steps["value"] == 3, (
        f"_train_loop reported total_steps={reported_total_steps['value']} but actually "
        f"executed {executed['count']} steps. "
        "Fix: total_steps = ((N + batch_size - 1) // batch_size) * epochs."
    )
