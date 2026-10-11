from __future__ import annotations

import pytest
import torch

import training.recipes.embedding_loop as module
from training.utils.embedding import bidirectional_info_nce_loss, info_nce_loss_fn
from training.utils.runner import UserConfigError


class _StopAfterProvisioning(RuntimeError):
    pass


def _make_trailing_special_tokenizer():
    """A tiny self-contained fast tokenizer that appends one trailing special.

    Reproduces the Qwen3-Embedding property the recipe relies on -- with
    ``add_special_tokens=True`` HF fast tokenizers reserve room for the
    post-processor's special tokens and truncate *content* first, so the
    trailing sentinel survives truncation -- without needing to download a
    real model.
    """
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")

    eos = "[EOS]"
    vocab = {"[UNK]": 0, eos: 1, "tok": 2}
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok.post_processor = tokenizers.processors.TemplateProcessing(
        single=f"$A {eos}",
        special_tokens=[(eos, vocab[eos])],
    )
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        eos_token=eos,
        pad_token=eos,
    )


def _datum_token_ids(datum) -> list[int]:
    """Extract the token ids from a tinker Datum (chunk-based ModelInput)."""
    model_input = datum.model_input
    to_ints = getattr(model_input, "to_ints", None)
    if callable(to_ints):
        ids = list(to_ints())
        if ids:
            return ids
    ids: list[int] = []
    for chunk in getattr(model_input, "chunks", []) or []:
        chunk_tokens = getattr(chunk, "tokens", None)
        if chunk_tokens:
            ids.extend(int(t) for t in chunk_tokens)
    return ids


def test_trailing_special_token_preserved_after_truncation():
    """`pooling="last"` reads a stable sentinel even for over-length inputs.

    Over-length queries/docs are truncated to exactly max_length, but the
    trailing special token (what last-token pooling reads) must be preserved,
    otherwise the pooled vector would come from an arbitrary mid-sequence
    content token.
    """
    tokenizer = _make_trailing_special_tokenizer()
    # Derive the appended sentinel id instead of assuming eos_token_id (the real
    # Qwen tokenizer appends <|endoftext|>, which differs from tok.eos_token).
    trailing_id = tokenizer("tok", add_special_tokens=True)["input_ids"][-1]

    max_query_len, max_doc_len = 8, 8
    long_query = "tok " * 100
    long_doc = "tok " * 100
    datums = module._build_batch_datums(
        [(long_query, long_doc)],
        tokenizer=tokenizer,
        query_instruction="find relevant passages",
        max_query_len=max_query_len,
        max_doc_len=max_doc_len,
    )
    # Layout is [query, doc] for a single pair.
    query_ids = _datum_token_ids(datums[0])
    doc_ids = _datum_token_ids(datums[1])

    # Truncation actually happened (content trimmed to exactly max_length)...
    assert len(query_ids) == max_query_len
    assert len(doc_ids) == max_doc_len
    # ...and the trailing sentinel survived on both sides.
    assert query_ids[-1] == trailing_id
    assert doc_ids[-1] == trailing_id


def test_trailing_special_token_present_without_truncation():
    """Short inputs are not truncated but still end in the sentinel."""
    tokenizer = _make_trailing_special_tokenizer()
    trailing_id = tokenizer("tok", add_special_tokens=True)["input_ids"][-1]

    datums = module._build_batch_datums(
        [("tok tok", "tok tok tok")],
        tokenizer=tokenizer,
        query_instruction="find",
        max_query_len=64,
        max_doc_len=64,
    )
    query_ids = _datum_token_ids(datums[0])
    doc_ids = _datum_token_ids(datums[1])

    assert len(query_ids) < 64 and len(doc_ids) < 64  # not truncated
    assert query_ids[-1] == trailing_id
    assert doc_ids[-1] == trailing_id


def test_main_rejects_invalid_base_model(monkeypatch):
    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    cfg = module.Config(
        log_path="/tmp/emb_test_logs",
        base_model="qwen3-embedding-8b",
        dataset="/tmp/pairs.jsonl",
        tokenizer_model="Qwen/Qwen3-Embedding-8B",
    )
    with pytest.raises(UserConfigError, match="Invalid base_model"):
        module.main(cfg)


def test_main_rejects_small_batch_size(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    cfg = module.Config(
        log_path=str(tmp_path),
        base_model="accounts/test/models/qwen3-embedding-8b",
        dataset="/tmp/pairs.jsonl",
        tokenizer_model="Qwen/Qwen3-Embedding-8B",
        batch_size=1,
    )
    with pytest.raises(ValueError, match="batch_size must be >= 2"):
        module.main(cfg)


def test_main_requests_cleanup_for_sdk_created_trainer(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "validate_config", lambda *args, **kwargs: None)

    def fake_build_service_client(**kwargs):
        calls.append(kwargs)
        raise _StopAfterProvisioning

    monkeypatch.setattr(module, "build_service_client", fake_build_service_client)
    cfg = module.Config(
        log_path=str(tmp_path),
        dataset="/tmp/pairs.jsonl",
        tokenizer_model="Qwen/Qwen3-Embedding-8B",
    )

    with pytest.raises(_StopAfterProvisioning):
        module.main(cfg)

    assert len(calls) == 1
    assert calls[0]["cleanup_trainer_on_close"] is True


def test_infonce_recall_is_perfect_for_aligned_pairs():
    q = torch.eye(4)
    d = torch.eye(4)
    loss, metrics = bidirectional_info_nce_loss(q, d, temperature=0.02)
    assert metrics["in_batch_recall_at_1"] == pytest.approx(1.0)
    assert loss.item() >= 0.0


def test_info_nce_loss_fn_splits_queries_then_documents():
    torch.manual_seed(0)
    batch = 3
    vectors = torch.randn(2 * batch, 8)

    loss, metrics = info_nce_loss_fn(batch, temperature=0.05)(None, list(vectors))
    expected, _ = bidirectional_info_nce_loss(
        vectors[:batch], vectors[batch:], temperature=0.05
    )

    assert loss.item() == pytest.approx(expected.item())
    assert set(metrics) == {"loss", "loss_q2d", "loss_d2q", "in_batch_recall_at_1"}


def test_main_trains_full_batches_with_client_side_info_nce(monkeypatch, tmp_path):
    events: list[str] = []
    logged: list[dict] = []

    def tokenizer(texts, **_kwargs):
        return {"input_ids": [[len(text) % 7 + 1, 2] for text in texts]}

    class _Policy:
        def forward_backward_custom(self, datums, loss_fn, *, output, pooling):
            events.append(f"fwd-bwd:{len(datums)}:{output}:{pooling}")
            # Each query embeds exactly like its paired document.
            pairs = torch.eye(len(datums) // 2)
            embeddings = list(torch.cat([pairs, pairs]))
            _loss, metrics = loss_fn(datums, embeddings)
            return type("Result", (), {"metrics": metrics})()

        def optim_step(self, _params, **_kwargs):
            events.append("optim")
            return None

    class _Service:
        trainer_job_id = "trainer"

        def create_training_client(self, *_args, **_kwargs):
            return _Policy()

        def close(self):
            events.append("close")

    class _Checkpoints:
        def __init__(self, *_args, **_kwargs):
            pass

        def save(self, name, **kwargs):
            events.append(f"checkpoint:{name}:{kwargs['data_consumed']}")

    pairs = [{"query": f"q{i}", "positive": f"doc {i}"} for i in range(5)]
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    for name in ("setup_wandb", "validate_config", "log_metrics_json", "wandb_finish", "log_run_resources"):
        monkeypatch.setattr(module, name, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "build_service_client", lambda **_kwargs: _Service())
    monkeypatch.setattr(
        module,
        "ReconnectableClient",
        type("RC", (), {"from_training_client": staticmethod(lambda client, **_kwargs: client)}),
    )
    monkeypatch.setattr(module, "TrainingCheckpoints", _Checkpoints)
    monkeypatch.setattr(module, "load_tokenizer", lambda *_args: tokenizer)
    monkeypatch.setattr(module, "load_jsonl_dataset", lambda *_args: list(pairs))
    monkeypatch.setattr(module, "log_metrics", lambda metrics, *, step: logged.append(dict(metrics)))

    result = module.main(
        module.Config(
            log_path=str(tmp_path),
            dataset="/tmp/pairs.jsonl",
            tokenizer_model="Qwen/Qwen3-Embedding-8B",
            batch_size=2,
            epochs=2,
        )
    )

    # 5 pairs -> 2 full batches per epoch; the odd pair is dropped each epoch.
    assert result == {"steps": 4, "job_id": "trainer"}
    assert events.count("fwd-bwd:4:embedding:last") == 4
    assert events.count("optim") == 4
    assert "checkpoint:step-4:8" in events
    assert [record["train/step"] for record in logged] == [1, 2, 3, 4]
    assert [record["train/epoch"] for record in logged] == [1, 1, 2, 2]
    assert logged[0]["train/in_batch_recall_at_1"] == pytest.approx(1.0)
