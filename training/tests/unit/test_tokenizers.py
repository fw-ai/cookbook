from __future__ import annotations

from types import SimpleNamespace

import training.utils.tokenizers as tokenizers


def test_load_tokenizer_forwards_optional_revision(monkeypatch):
    captured: dict = {}
    fake_tokenizer = object()

    def fake_from_pretrained(model, **kwargs):
        captured.update(model=model, kwargs=kwargs)
        return fake_tokenizer

    monkeypatch.setattr(
        tokenizers.transformers.AutoTokenizer, "from_pretrained", fake_from_pretrained
    )

    result = tokenizers.load_tokenizer("moonshotai/Kimi-K2.6", "2755962")

    assert result is fake_tokenizer
    assert captured["model"] == "moonshotai/Kimi-K2.6"
    assert captured["kwargs"] == {
        "revision": "2755962",
        "trust_remote_code": True,
    }


def test_load_tokenizer_treats_empty_revision_as_unset(monkeypatch):
    captured: dict = {}

    def fake_from_pretrained(model, **kwargs):
        captured.update(model=model, kwargs=kwargs)
        return object()

    monkeypatch.setattr(
        tokenizers.transformers.AutoTokenizer, "from_pretrained", fake_from_pretrained
    )

    tokenizers.load_tokenizer("Qwen/Qwen3-8B", "")

    assert captured["kwargs"]["revision"] is None


def test_load_deployment_tokenizer_uses_generic_deploy_config_fields(monkeypatch):
    captured: dict = {}

    def fake_load_tokenizer(model, revision=None):
        captured.update(model=model, revision=revision)
        return object()

    monkeypatch.setattr(tokenizers, "load_tokenizer", fake_load_tokenizer)

    tokenizers.load_deployment_tokenizer(
        SimpleNamespace(tokenizer_model="model/name", tokenizer_revision="abc123")
    )

    assert captured == {"model": "model/name", "revision": "abc123"}
