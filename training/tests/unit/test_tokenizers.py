from __future__ import annotations

import json
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest
import tokenizers as tokenizers_lib
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils import hf_raise_for_status
from transformers.tokenization_utils_tokenizers import TokenizersBackend

import training.utils.tokenizers as tokenizers
import training.utils.runner as runner
from training.renderer.verifier.utils import hf_parity
from training.renderer.verifier.utils import tokenizer as verifier_tokenizers
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


@pytest.mark.parametrize(
    ("policy", "expected"),
    [(None, True), (True, True), (False, False)],
)
def test_load_tokenizer_forwards_revision_and_remote_code_policy(
    monkeypatch, policy, expected
):
    captured: dict = {}
    fake_tokenizer = object()

    def fake_from_pretrained(model, **kwargs):
        captured.update(model=model, kwargs=kwargs)
        return fake_tokenizer

    monkeypatch.setattr(
        tokenizers.transformers.AutoTokenizer, "from_pretrained", fake_from_pretrained
    )

    result = tokenizers.load_tokenizer("moonshotai/Kimi-K2.6", "2755962", policy)

    assert result is fake_tokenizer
    assert captured["model"] == "moonshotai/Kimi-K2.6"
    assert captured["kwargs"] == {
        "revision": "2755962",
        "trust_remote_code": expected,
    }


def test_load_mistral_tokenizer_uses_upstream_regex_fix(monkeypatch):
    captured: dict = {}

    def fake_from_pretrained(model, **kwargs):
        captured.update(model=model, kwargs=kwargs)
        return object()

    monkeypatch.setattr(
        tokenizers.transformers.AutoTokenizer, "from_pretrained", fake_from_pretrained
    )

    tokenizers.load_tokenizer(
        "accounts/fireworks/models/mistral-small-24b-instruct-2501"
    )

    assert captured["kwargs"]["fix_mistral_regex"] is True


def test_verifier_tokenizer_paths_repair_mistral_model(monkeypatch):
    captured: dict = {}
    monkeypatch.setenv("HF_TOKEN", "test-token")

    def fake_from_pretrained(model, **kwargs):
        captured.update(model=model, kwargs=kwargs)
        return object()

    monkeypatch.setattr(
        tokenizers.transformers.AutoTokenizer,
        "from_pretrained",
        fake_from_pretrained,
    )

    verifier_tokenizers.load_tokenizer(
        "mistralai/Mistral-Small-24B-Instruct-2501",
    )

    assert captured["kwargs"]["fix_mistral_regex"] is True

    captured.clear()
    hf_parity._load_tokenizer.cache_clear()
    try:
        hf_parity._load_tokenizer(
            "mistralai/Mistral-Small-24B-Instruct-2501",
            "abc123",
            False,
        )
    finally:
        hf_parity._load_tokenizer.cache_clear()

    assert captured["kwargs"] == {
        "revision": "abc123",
        "token": "test-token",
        "trust_remote_code": False,
        "fix_mistral_regex": True,
    }


def test_upstream_mistral_regex_fix_accepts_raw_tokenizer_backend(tmp_path):
    model_dir = tmp_path / "legacy-mistral"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "mistral",
                "transformers_version": "4.57.2",
            }
        )
    )
    legacy_regex = (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|"
        r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    )
    backend = tokenizers_lib.Tokenizer(
        tokenizers_lib.models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = tokenizers_lib.pre_tokenizers.Sequence(
        [
            tokenizers_lib.pre_tokenizers.Split(
                tokenizers_lib.Regex(legacy_regex),
                behavior="isolated",
            ),
            tokenizers_lib.pre_tokenizers.ByteLevel(
                add_prefix_space=False,
                use_regex=False,
            ),
        ]
    )

    patched_backend = TokenizersBackend._patch_mistral_regex(
        backend,
        str(model_dir),
        is_local=True,
        init_kwargs={},
        fix_mistral_regex=True,
    )

    assert patched_backend.pre_tokenizer.pre_tokenize_str("'The'") == [
        ("'The", (0, 4)),
        ("'", (4, 5)),
    ]


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


def test_load_unknown_model_tokenizer_when_generic_config_rope_validation_fails(
    tmp_path,
):
    model_dir = tmp_path / "future-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "future_model_type",
                "rope_theta": 10_000.0,
                "rope_scaling": {
                    "factor": 16.0,
                    "original_max_position_embeddings": 65536,
                    "type": "yarn",
                },
            }
        )
    )
    backend = tokenizers_lib.Tokenizer(
        tokenizers_lib.models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")
    )
    backend.save(str(model_dir / "tokenizer.json"))

    loaded = tokenizers.load_tokenizer(
        str(model_dir),
        trust_remote_code=False,
        local_files_only=True,
    )

    assert loaded.get_vocab() == {"[UNK]": 0}


def test_load_unknown_model_tokenizer_with_deepseek_v4_model_type(tmp_path):
    """Bindwell GSM8K SFT used DeepSeek-V4; its HF config type is unregistered."""
    model_dir = tmp_path / "deepseek-v4-flash"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "rope_theta": 10_000.0,
                "rope_scaling": {
                    "factor": 16.0,
                    "original_max_position_embeddings": 65536,
                    "type": "yarn",
                },
            }
        )
    )
    backend = tokenizers_lib.Tokenizer(
        tokenizers_lib.models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")
    )
    backend.save(str(model_dir / "tokenizer.json"))

    loaded = tokenizers.load_tokenizer(
        str(model_dir),
        trust_remote_code=False,
        local_files_only=True,
    )

    assert loaded.get_vocab() == {"[UNK]": 0}


@pytest.mark.parametrize(
    "error_kwargs",
    [
        pytest.param(
            {
                "name": "max_position_embeddings",
                "use_obj": True,
            },
            id="structured-attribute-error",
        ),
        pytest.param({}, id="message-only-attribute-error"),
    ],
)
def test_load_tokenizer_retries_generic_config_max_position_error(
    monkeypatch, error_kwargs
):
    calls: list[dict] = []
    fake_tokenizer = object()
    config = tokenizers.transformers.PreTrainedConfig()

    def from_pretrained(model, **kwargs):
        calls.append(kwargs)
        if "config" in kwargs:
            return fake_tokenizer
        raise AttributeError(
            "'PreTrainedConfig' object has no attribute 'max_position_embeddings'",
            **(
                {
                    "name": error_kwargs["name"],
                    "obj": config,
                }
                if error_kwargs.get("use_obj")
                else {}
            ),
        )

    monkeypatch.setattr(
        tokenizers.transformers.AutoTokenizer,
        "from_pretrained",
        from_pretrained,
    )

    result = tokenizers.load_tokenizer("deepseek-ai/DeepSeek-V4-Flash")

    assert result is fake_tokenizer
    assert len(calls) == 2
    assert "config" not in calls[0]
    assert calls[1]["config"].max_position_embeddings == 1


def test_verifier_load_tokenizer_retries_generic_config_max_position_error(
    monkeypatch,
):
    calls: list[dict] = []
    fake_tokenizer = object()

    def from_pretrained(model, **kwargs):
        calls.append(kwargs)
        if "config" in kwargs:
            return fake_tokenizer
        raise AttributeError(
            "'PreTrainedConfig' object has no attribute 'max_position_embeddings'"
        )

    monkeypatch.setattr(
        tokenizers.transformers.AutoTokenizer,
        "from_pretrained",
        from_pretrained,
    )

    result = verifier_tokenizers.load_tokenizer("deepseek-ai/DeepSeek-V4-Flash")

    assert result is fake_tokenizer
    assert len(calls) == 2
    assert calls[1]["config"].max_position_embeddings == 1


def test_load_tokenizer_does_not_hide_unrelated_attribute_errors(monkeypatch):
    def from_pretrained(model, **kwargs):
        config = tokenizers.transformers.PreTrainedConfig()
        raise AttributeError(
            "'PreTrainedConfig' object has no attribute 'unrelated_attribute'",
            name="unrelated_attribute",
            obj=config,
        )

    monkeypatch.setattr(
        tokenizers.transformers.AutoTokenizer,
        "from_pretrained",
        from_pretrained,
    )

    with pytest.raises(AttributeError, match="unrelated_attribute"):
        tokenizers.load_tokenizer("org/model")


def test_load_deployment_tokenizer_uses_generic_deploy_config_fields(monkeypatch):
    captured: dict = {}

    def fake_load_tokenizer(model, revision=None, trust_remote_code=None):
        captured.update(
            model=model,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        return object()

    monkeypatch.setattr(tokenizers, "load_tokenizer", fake_load_tokenizer)

    tokenizers.load_deployment_tokenizer(
        SimpleNamespace(
            tokenizer_model="model/name",
            tokenizer_revision="abc123",
            tokenizer_trust_remote_code=False,
        )
    )

    assert captured == {
        "model": "model/name",
        "revision": "abc123",
        "trust_remote_code": False,
    }


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
