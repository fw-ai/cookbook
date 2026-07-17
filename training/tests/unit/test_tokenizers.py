from __future__ import annotations

import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils import hf_raise_for_status

import training.utils.tokenizers as tokenizers
import training.utils.runner as runner
from training.utils.runner import RunnerConfig, RunnerIO


@contextmanager
def local_status_server(status_code: int):
    class StatusHandler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.send_response(status_code)
            self.end_headers()

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), StatusHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/model"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def real_wrapped_tokenizer_http_error(url: str) -> OSError:
    response = httpx.head(url)
    try:
        hf_raise_for_status(response)
    except HfHubHTTPError as exc:
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


@pytest.mark.parametrize("status_code", [404, 504])
def test_load_tokenizer_propagates_real_huggingface_http_status(
    monkeypatch, status_code
):
    with local_status_server(status_code) as url:

        def from_pretrained(model, **kwargs):
            raise real_wrapped_tokenizer_http_error(url)

        monkeypatch.setattr(
            tokenizers.transformers.AutoTokenizer, "from_pretrained", from_pretrained
        )

        with pytest.raises(RuntimeError) as exc_info:
            tokenizers.load_tokenizer("org/model-504")

    assert str(exc_info.value) == (
        "Hugging Face Hub request failed while loading tokenizer "
        f"'org/model-504' (HTTP {status_code})."
    )
    tokenizer_error = exc_info.value.__cause__
    assert isinstance(tokenizer_error, OSError)
    hub_error = tokenizer_error.__cause__
    assert isinstance(hub_error, HfHubHTTPError)
    assert hub_error.response.status_code == status_code
    assert isinstance(hub_error.__cause__, httpx.HTTPStatusError)


def test_load_tokenizer_does_not_misclassify_connection_refused_as_http(monkeypatch):
    socket_handle = socket.socket()
    socket_handle.bind(("127.0.0.1", 0))
    port = socket_handle.getsockname()[1]
    socket_handle.close()
    url = f"http://127.0.0.1:{port}/model"

    def from_pretrained(model, **kwargs):
        try:
            httpx.head(url)
        except httpx.ConnectError as exc:
            raise OSError("Unable to reach tokenizer endpoint") from exc
        raise AssertionError("expected connection refusal")

    monkeypatch.setattr(
        tokenizers.transformers.AutoTokenizer, "from_pretrained", from_pretrained
    )

    with pytest.raises(OSError, match="Unable to reach tokenizer endpoint") as exc_info:
        tokenizers.load_tokenizer("offline/model")

    assert isinstance(exc_info.value.__cause__, httpx.ConnectError)
    assert tokenizers._huggingface_http_status_code(exc_info.value) is None


def test_huggingface_unavailability_propagates_to_runner_status(monkeypatch):
    status_writes: list[tuple[str, dict]] = []

    with local_status_server(503) as url:

        def from_pretrained(model, **kwargs):
            raise real_wrapped_tokenizer_http_error(url)

        monkeypatch.setattr(
            tokenizers.transformers.AutoTokenizer, "from_pretrained", from_pretrained
        )
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
    assert "Hugging Face Hub request failed" in status_writes[-1][1]["message"]
    assert "HTTP 503" in status_writes[-1][1]["message"]
