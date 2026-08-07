"""Tests for the serverless DPO example's pure helpers."""

from __future__ import annotations

import pytest

from training.examples.serverless_dpo.prepare_data import row_to_preference
from training.examples.serverless_dpo.ultrafeedback_dpo import (
    MAX_CHECKPOINT_NAME_LEN,
    Config,
    _account_from_session,
    _align_reference_logprobs,
    _comparison_prompt_messages,
    _control_plane_base_url,
    _find_promotable,
    _full_sequence_for_ref,
    _interleave_datums,
    _serverless_base_url,
    _split_train_eval,
    _step_from_checkpoint_name,
    _unique_model_id,
    _validate_config,
    _validate_datum_length,
    _validate_resume_reference,
)
from training.utils.supervised import build_datum_from_tokens_and_weights


def test_serverless_dpo_defaults_use_supported_model_and_matching_tokenizer():
    cfg = Config()
    assert cfg.base_model == "accounts/fireworks/models/kimi-k3"
    assert cfg.tokenizer_model == "moonshotai/Kimi-K3"
    assert cfg.max_seq_len > 0
    assert cfg.lora_rank > 0
    assert 0 < cfg.dpo_beta < 0.5


@pytest.mark.parametrize("max_seq_len", [0, -1])
def test_serverless_dpo_rejects_invalid_sequence_bound(max_seq_len):
    with pytest.raises(ValueError, match="max_seq_len > 0"):
        _validate_config(Config(max_seq_len=max_seq_len))


def test_serverless_dpo_rejects_non_lora_config():
    with pytest.raises(ValueError, match="lora_rank > 0"):
        _validate_config(Config(lora_rank=0))


@pytest.mark.parametrize("beta", [0.0, -0.1, 0.5, 1.0])
def test_serverless_dpo_rejects_out_of_range_beta(beta):
    with pytest.raises(ValueError, match="dpo_beta"):
        _validate_config(Config(dpo_beta=beta))


@pytest.mark.parametrize("beta", [0.01, 0.1, 0.3, 0.49])
def test_serverless_dpo_accepts_in_range_beta(beta):
    _validate_config(Config(dpo_beta=beta))


def test_serverless_dpo_rejects_negative_save_interval():
    with pytest.raises(ValueError, match="dcp_save_interval"):
        _validate_config(Config(dcp_save_interval=-1))


def test_serverless_dpo_allows_disabled_periodic_checkpoints():
    _validate_config(Config(dcp_save_interval=0))


def test_serverless_dpo_rejects_nonpositive_ref_concurrency():
    with pytest.raises(ValueError, match="ref_concurrency"):
        _validate_config(Config(ref_concurrency=0))


def test_serverless_dpo_rejects_negative_eval_pairs():
    with pytest.raises(ValueError, match="eval_pairs"):
        _validate_config(Config(eval_pairs=-1))


def test_split_train_eval_reserves_tail_rows():
    rows = [{"i": i} for i in range(10)]
    train, held_out = _split_train_eval(rows, 3)
    assert [r["i"] for r in train] == list(range(7))
    assert [r["i"] for r in held_out] == [7, 8, 9]


def test_split_train_eval_zero_keeps_everything_for_training():
    rows = [{"i": i} for i in range(5)]
    train, held_out = _split_train_eval(rows, 0)
    assert train == rows
    assert held_out == []


def test_split_train_eval_rejects_eval_covering_whole_dataset():
    with pytest.raises(ValueError, match="smaller than the dataset"):
        _split_train_eval([{"i": 0}, {"i": 1}], 2)


def test_serverless_dpo_rejects_overlong_checkpoint_name():
    # The promotable id is "{run_id}-{name}-{suffix}" against a 63-char limit,
    # so an overlong name must fail before any training happens.
    with pytest.raises(ValueError, match="capped at"):
        _validate_config(Config(checkpoint_name="x" * (MAX_CHECKPOINT_NAME_LEN + 1)))


def test_serverless_dpo_rejects_overlong_reference_name():
    with pytest.raises(ValueError, match="capped at"):
        _validate_config(Config(reference_checkpoint_name="x" * (MAX_CHECKPOINT_NAME_LEN + 1)))


def test_serverless_dpo_datum_length_accepts_exact_bound():
    _validate_datum_length(1024, 1024)


def test_serverless_dpo_datum_length_rejects_overflow():
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        _validate_datum_length(1025, 1024)


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.fireworks.ai",
        "https://api.fireworks.ai/",
        "https://api.fireworks.ai/training/v1",
        "https://api.fireworks.ai/training/v1/serverless",
    ],
)
def test_serverless_base_url_is_idempotent(base_url):
    assert _serverless_base_url(base_url) == "https://api.fireworks.ai/training/v1/serverless"


@pytest.mark.parametrize(
    "base_url",
    [
        "https://gateway.example.com",
        "https://gateway.example.com/training/v1",
        "https://gateway.example.com/training/v1/serverless",
    ],
)
def test_control_plane_base_url_strips_training_suffixes(base_url):
    assert _control_plane_base_url(base_url) == "https://gateway.example.com"


def test_full_sequence_for_ref_appends_final_target_token():
    # The datum is the shifted next-token layout; appending the last target
    # token must reconstruct the original token sequence exactly.
    tokens = [10, 11, 12, 13]
    datum = build_datum_from_tokens_and_weights(tokens, [0.0, 0.0, 1.0, 1.0]).datum
    full = _full_sequence_for_ref(datum)
    assert list(full.to_ints()) == tokens


def test_full_sequence_for_ref_handles_empty_targets():
    import tinker

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints([10, 11]),
        loss_fn_inputs={"target_tokens": tinker.TensorData(data=[], dtype="int64", shape=[0])},
    )
    full = _full_sequence_for_ref(datum)
    assert full is datum.model_input


def test_align_reference_logprobs_drops_leading_none():
    # compute_logprobs puts None at position 0; the aligned array must line up
    # with the trainer-forward layout (entry j = logprob of predicted token j).
    aligned = _align_reference_logprobs([None, -1.0, -2.0, -3.0, -4.0], datum_length=4)
    assert aligned == [-1.0, -2.0, -3.0, -4.0]


def test_align_reference_logprobs_rejects_length_mismatch():
    with pytest.raises(ValueError, match="expected"):
        _align_reference_logprobs([None, -1.0, -2.0], datum_length=4)


def test_interleave_datums_pairs_chosen_then_rejected():
    pairs = []
    for i in range(3):
        chosen = build_datum_from_tokens_and_weights([100 + i, 1], [1.0, 1.0]).datum
        rejected = build_datum_from_tokens_and_weights([200 + i, 1], [1.0, 1.0]).datum
        pairs.append(type("P", (), {"chosen_datum": chosen, "rejected_datum": rejected})())
    datums = _interleave_datums(pairs)
    assert len(datums) == 6
    first_tokens = [d.model_input.to_ints()[0] for d in datums]
    assert first_tokens == [100, 200, 101, 201, 102, 202]


def test_account_from_session_parses_account_segment():
    assert _account_from_session("accounts/fireworks/trainingSessions/ts-1") == "fireworks"
    assert _account_from_session(None) is None
    assert _account_from_session("trainingSessions/ts-1") is None


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("fireworks/run-abc123/triage-0006", 6),
        ("triage-0000", 0),
        ("triage-final", 0),
        ("", 0),
    ],
)
def test_step_from_checkpoint_name(reference, expected):
    assert _step_from_checkpoint_name(reference) == expected


@pytest.mark.parametrize(
    "reference",
    ["accounts/a/trainingRuns/run-abc/dpo-0008", "dpo-0000", "acct/run-x/dpo-0123"],
)
def test_validate_resume_reference_accepts_numbered_checkpoints(reference):
    _validate_resume_reference(reference)


@pytest.mark.parametrize("reference", ["dpo-final", "triage-final", "dpo", "", "run-x/"])
def test_validate_resume_reference_rejects_unnumbered_checkpoints(reference):
    # A ref without a numeric step suffix would silently restart the step
    # counter and dataset cursor at 0 -- refuse it instead.
    with pytest.raises(ValueError, match="numbered checkpoint"):
        _validate_resume_reference(reference)


def test_find_promotable_matches_bare_and_run_prefixed_names():
    checkpoints = [
        {"name": "ckpts/run-aaa/dpo-0004", "promotable": True},
        {"name": "ckpts/run-aaa/run-aaa-dpo-final-1a2b3c4d", "promotable": True},
    ]
    match = _find_promotable(checkpoints, "dpo-final", "run-aaa")
    assert match is not None
    assert "dpo-final" in match["name"]


def test_find_promotable_skips_non_promotable():
    checkpoints = [{"name": "ckpts/run-aaa/dpo-final", "promotable": False}]
    assert _find_promotable(checkpoints, "dpo-final", "run-aaa") is None


def test_unique_model_id_appends_short_run_suffix():
    run_id = "run-" + "a" * 32
    model_id = _unique_model_id("serverless-dpo-ultrafeedback", run_id)
    assert model_id == "serverless-dpo-ultrafeedback-aaaaaaaa"
    assert len(model_id) <= 63


def test_unique_model_id_falls_back_without_run_id():
    assert _unique_model_id("serverless-dpo-ultrafeedback", None) == "serverless-dpo-ultrafeedback"


def test_row_to_preference_builds_shared_prompt_pair():
    pref = row_to_preference({
        "instruction": "Explain caching in one sentence.",
        "chosen_response": "Caching stores results so later requests are faster.",
        "rejected_response": "Caching is a thing computers do.",
    })
    assert pref is not None
    assert pref["chosen"]["messages"][:1] == pref["rejected"]["messages"][:1]
    assert pref["chosen"]["messages"][-1]["role"] == "assistant"
    assert pref["chosen"]["messages"][-1]["content"] != pref["rejected"]["messages"][-1]["content"]


@pytest.mark.parametrize(
    "example",
    [
        {"instruction": "", "chosen_response": "a", "rejected_response": "b"},
        {"instruction": "q", "chosen_response": "", "rejected_response": "b"},
        {"instruction": "q", "chosen_response": "same", "rejected_response": "same"},
    ],
)
def test_row_to_preference_drops_invalid_rows(example):
    assert row_to_preference(example) is None


def test_comparison_prompt_messages_strips_final_assistant_turn():
    row = {
        "chosen": {"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "good"},
        ]},
        "rejected": {"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "bad"},
        ]},
    }
    assert _comparison_prompt_messages(row) == [{"role": "user", "content": "q"}]


def test_comparison_prompt_messages_rejects_non_preference_row():
    assert _comparison_prompt_messages({"foo": "bar"}) is None
