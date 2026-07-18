from __future__ import annotations

import pytest
import torch

import training.recipes.embedding_loop as module


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
    with pytest.raises(RuntimeError, match="Invalid base_model"):
        module.main(cfg)


def test_main_rejects_invalid_output_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    cfg = module.Config(
        log_path=str(tmp_path),
        base_model="accounts/test/models/qwen3-embedding-8b",
        dataset="/tmp/pairs.jsonl",
        tokenizer_model="Qwen/Qwen3-Embedding-8B",
        output_mode="not_a_mode",
    )
    with pytest.raises(ValueError, match="Unsupported output_mode"):
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


def test_infonce_recall_is_perfect_for_aligned_pairs():
    q = torch.eye(4)
    d = torch.eye(4)
    loss, metrics = module.bidirectional_info_nce_loss(q, d, temperature=0.02)
    assert metrics["in_batch_recall_at_1"] == pytest.approx(1.0)
    assert loss.item() >= 0.0


def test_cos_similarity_matrix_closure_matches_embedding_closure():
    torch.manual_seed(0)
    batch = 3
    vectors = torch.randn(2 * batch, 8)

    emb_loss_fn = module._build_embedding_loss_fn(batch, temperature=0.05)
    emb_loss, _ = emb_loss_fn(None, list(vectors))

    normalized = torch.nn.functional.normalize(vectors, p=2, dim=-1)
    sim_rows = list(normalized @ normalized.t())
    cos_loss_fn = module._build_cosine_similarity_loss_fn(batch, temperature=0.05)
    cos_loss, _ = cos_loss_fn(None, sim_rows)

    assert cos_loss.item() == pytest.approx(emb_loss.item(), rel=1e-5, abs=1e-6)
