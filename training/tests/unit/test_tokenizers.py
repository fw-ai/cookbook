from __future__ import annotations

from types import SimpleNamespace

import pytest

import training.utils.tokenizers as tokenizers
import training.utils.runner as runner
from training.utils.runner import RunnerConfig, RunnerIO


class FakeHuggingFaceHTTPError(Exception):
    def __init__(self, status_code: int, model: str = "Qwen/Qwen3-8B"):
        super().__init__(
            f"Server error '{status_code}' for url 'https://huggingface.co/{model}'"
        )
        self.response = SimpleNamespace(status_code=status_code)


def wrapped_tokenizer_error(status_code: int, model: str = "Qwen/Qwen3-8B") -> OSError:
    try:
        raise FakeHuggingFaceHTTPError(status_code, model)
    except FakeHuggingFaceHTTPError as exc:
        try:
            raise OSError(
                "Unable to load vocabulary from file. Please check that the "
                "provided vocabulary is accessible and not corrupted."
            ) from exc
        except OSError as wrapped:
            return wrapped


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


def test_load_tokenizer_retries_transient_huggingface_failure(monkeypatch):
    calls = 0
    sleeps: list[float] = []
    fake_tokenizer = object()

    def fake_from_pretrained(model, **kwargs):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise wrapped_tokenizer_error(504)
        return fake_tokenizer

    monkeypatch.setattr(
        tokenizers.transformers.AutoTokenizer, "from_pretrained", fake_from_pretrained
    )
    monkeypatch.setattr(tokenizers.time, "sleep", sleeps.append)

    result = tokenizers.load_tokenizer("Qwen/Qwen3-8B")

    assert result is fake_tokenizer
    assert calls == 3
    assert sleeps == [2.0, 4.0]


def test_load_tokenizer_reports_huggingface_unavailability_after_retries(monkeypatch):
    sleeps: list[float] = []

    def fake_from_pretrained(model, **kwargs):
        raise wrapped_tokenizer_error(504)

    monkeypatch.setattr(
        tokenizers.transformers.AutoTokenizer, "from_pretrained", fake_from_pretrained
    )
    monkeypatch.setattr(tokenizers.time, "sleep", sleeps.append)

    with pytest.raises(RuntimeError) as exc_info:
        tokenizers.load_tokenizer("Qwen/Qwen3-8B")

    assert "Hugging Face Hub is unavailable" in str(exc_info.value)
    assert "HTTP 504" in str(exc_info.value)
    assert "exhausted 3 attempts" in str(exc_info.value)
    assert sleeps == [2.0, 4.0]
    assert isinstance(exc_info.value.__cause__, OSError)


def test_load_tokenizer_does_not_retry_non_transient_failure(monkeypatch):
    calls = 0
    sleeps: list[float] = []

    def fake_from_pretrained(model, **kwargs):
        nonlocal calls
        calls += 1
        raise wrapped_tokenizer_error(404)

    monkeypatch.setattr(
        tokenizers.transformers.AutoTokenizer, "from_pretrained", fake_from_pretrained
    )
    monkeypatch.setattr(tokenizers.time, "sleep", sleeps.append)

    with pytest.raises(OSError, match="Unable to load vocabulary"):
        tokenizers.load_tokenizer("missing/model")

    assert calls == 1
    assert sleeps == []


def test_load_tokenizer_trusts_explicit_non_transient_status(monkeypatch):
    calls = 0

    def fake_from_pretrained(model, **kwargs):
        nonlocal calls
        calls += 1
        raise wrapped_tokenizer_error(404, model="org/model-504")

    monkeypatch.setattr(
        tokenizers.transformers.AutoTokenizer, "from_pretrained", fake_from_pretrained
    )

    with pytest.raises(OSError, match="Unable to load vocabulary"):
        tokenizers.load_tokenizer("org/model-504")

    assert calls == 1


def test_huggingface_unavailability_propagates_to_runner_status(monkeypatch):
    status_writes: list[tuple[str, dict]] = []

    def fake_from_pretrained(model, **kwargs):
        raise wrapped_tokenizer_error(503)

    monkeypatch.setattr(
        tokenizers.transformers.AutoTokenizer, "from_pretrained", fake_from_pretrained
    )
    monkeypatch.setattr(tokenizers.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        runner.fileio,
        "write_json",
        lambda path, payload: status_writes.append((path, payload)),
    )

    with pytest.raises(RuntimeError):
        with RunnerIO(RunnerConfig(status_file="status.json")):
            tokenizers.load_tokenizer("Qwen/Qwen3-8B")

    assert status_writes[-1][0] == "status.json"
    assert status_writes[-1][1]["code"] == 9
    assert "Hugging Face Hub is unavailable" in status_writes[-1][1]["message"]
    assert "HTTP 503" in status_writes[-1][1]["message"]
