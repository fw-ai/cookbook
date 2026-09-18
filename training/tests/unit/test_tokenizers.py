from __future__ import annotations

import importlib
import json
import logging
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import tokenizers as tokenizers_lib
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError
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


def test_kimi_bytes_to_unicode_legacy_gpt2_import_is_restored():
    """Kimi tokenization_kimi.py imports bytes_to_unicode from the gpt2 module."""
    gpt2_module = importlib.import_module("transformers.models.gpt2.tokenization_gpt2")
    original = getattr(gpt2_module, "bytes_to_unicode", None)
    if original is not None:
        delattr(gpt2_module, "bytes_to_unicode")
    try:
        tokenizers.patch_kimi_tokenizer_bytes_to_unicode()
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

        mapping = bytes_to_unicode()
        assert len(mapping) == 256
        assert mapping[ord("a")] == "a"
    finally:
        if original is not None:
            gpt2_module.bytes_to_unicode = original
        elif hasattr(gpt2_module, "bytes_to_unicode"):
            delattr(gpt2_module, "bytes_to_unicode")


def test_kimi_bytes_to_unicode_patch_is_idempotent_and_preserves_existing_attribute():
    gpt2_module = importlib.import_module("transformers.models.gpt2.tokenization_gpt2")
    sentinel = object()
    original = getattr(gpt2_module, "bytes_to_unicode", None)
    gpt2_module.bytes_to_unicode = sentinel
    try:
        tokenizers.patch_kimi_tokenizer_bytes_to_unicode()
        assert gpt2_module.bytes_to_unicode is sentinel
    finally:
        if original is not None:
            gpt2_module.bytes_to_unicode = original
        else:
            delattr(gpt2_module, "bytes_to_unicode")


def test_load_tokenizer_applies_kimi_bytes_to_unicode_patch(monkeypatch):
    calls: list[int] = []
    real_patch = tokenizers.patch_kimi_tokenizer_bytes_to_unicode

    def counting_patch() -> None:
        calls.append(1)
        real_patch()

    monkeypatch.setattr(
        tokenizers, "patch_kimi_tokenizer_bytes_to_unicode", counting_patch
    )
    monkeypatch.setattr(
        tokenizers.transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )

    tokenizers.load_tokenizer("moonshotai/Kimi-K2.5")

    assert calls, "load_tokenizer must apply the Kimi tokenizer compat patch"


def test_huggingface_unavailability_is_preserved_for_managed_adapter(monkeypatch):
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

        with pytest.raises(RuntimeError) as exc_info:
            with RunnerIO(RunnerConfig(status_file="status.json")):
                tokenizers.load_tokenizer("Qwen/Qwen3-8B")

    assert status_writes[-1][0] == "status.json"
    assert status_writes[-1][1]["code"] == 9
    assert status_writes[-1][1]["message"] == str(exc_info.value)


def _write_tokenizer_dir(
    root: Path,
    *,
    tokenizer_class: str,
    model_type: str | None,
    add_bos_token: bool | None = None,
    bos_in_post_processor: bool = False,
    write_tokenizer_json: bool = True,
) -> None:
    """Minimal local tokenizer dir: verbatim tokenizer.json + declarations."""
    root.mkdir(parents=True, exist_ok=True)
    if write_tokenizer_json:
        backend = tokenizers_lib.Tokenizer(
            tokenizers_lib.models.WordLevel(
                {"<s>": 0, "hello": 1, " world": 2},
                unk_token="<s>",
            )
        )
        if bos_in_post_processor:
            backend.post_processor = tokenizers_lib.processors.TemplateProcessing(
                single="<s>:0 $A:0",
                pair="<s>:0 $A:0 <s>:1 $B:1",
                special_tokens=[("<s>", 0)],
            )
        backend.save(str(root / "tokenizer.json"))
    config: dict[str, object] = {
        "tokenizer_class": tokenizer_class,
        "chat_template": "{{ messages }}",
        "bos_token": "<s>",
    }
    if add_bos_token is not None:
        config["add_bos_token"] = add_bos_token
    (root / "tokenizer_config.json").write_text(json.dumps(config))
    if model_type is not None:
        (root / "config.json").write_text(json.dumps({"model_type": model_type}))


def _mismatched_dir(tmp_path: Path, name: str = "qwen38-like", **kwargs) -> Path:
    """Qwen3.8's shape: declares Qwen2Tokenizer, model_type maps elsewhere."""
    model_dir = tmp_path / name
    _write_tokenizer_dir(
        model_dir, tokenizer_class="Qwen2Tokenizer", model_type="qwen3_5", **kwargs
    )
    return model_dir


def _ambiguous(model_dir: Path) -> bool:
    config = json.loads((model_dir / "tokenizer_config.json").read_text())
    return tokenizers._tokenizer_selection_is_ambiguous(
        str(model_dir), config, revision=None, local_files_only=True
    )


def test_mismatched_declared_class_loads_tokenizer_json_verbatim(tmp_path):
    model_dir = _mismatched_dir(tmp_path)

    assert _ambiguous(model_dir)
    loaded = tokenizers.load_tokenizer(str(model_dir), local_files_only=True)

    assert isinstance(loaded, TokenizersBackend)
    assert loaded.encode("hello") == [1]


def test_matching_declared_class_stays_on_auto_tokenizer(tmp_path, monkeypatch):
    model_dir = tmp_path / "qwen2-like"
    _write_tokenizer_dir(
        model_dir, tokenizer_class="Qwen2TokenizerFast", model_type="qwen2"
    )

    assert not _ambiguous(model_dir)

    seen: dict[str, object] = {}

    class FakeAuto:
        @staticmethod
        def from_pretrained(model, **kwargs):
            seen["model"] = model
            return SimpleNamespace(marker="auto")

    monkeypatch.setattr(tokenizers.transformers, "AutoTokenizer", FakeAuto)
    loaded = tokenizers.load_tokenizer(str(model_dir), local_files_only=True)

    assert loaded.marker == "auto"
    assert seen["model"] == str(model_dir)


def test_remote_code_tokenizer_declaration_is_not_ambiguous(tmp_path):
    model_dir = _mismatched_dir(tmp_path)
    config = json.loads((model_dir / "tokenizer_config.json").read_text())
    config["auto_map"] = {"AutoTokenizer": ["tokenization_x.MyTokenizer", None]}
    (model_dir / "tokenizer_config.json").write_text(json.dumps(config))

    assert not _ambiguous(model_dir)


def test_missing_tokenizer_config_keeps_the_default_path(tmp_path, monkeypatch):
    model_dir = tmp_path / "empty"
    model_dir.mkdir()

    monkeypatch.setattr(
        tokenizers.transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(marker="auto"),
    )

    assert (
        tokenizers._read_repo_json(
            str(model_dir),
            "tokenizer_config.json",
            revision=None,
            local_files_only=True,
        )
        is None
    )
    assert tokenizers.load_tokenizer(str(model_dir), local_files_only=True).marker == (
        "auto"
    )


def test_tokenizer_only_dir_follows_serving_fast_class_rule(tmp_path):
    """Serving's fallback when no model config pins a model_type."""
    fast_dir = tmp_path / "fast-only"
    _write_tokenizer_dir(
        fast_dir, tokenizer_class="PreTrainedTokenizerFast", model_type=None
    )
    slow_dir = tmp_path / "slow-only"
    _write_tokenizer_dir(slow_dir, tokenizer_class="Qwen2Tokenizer", model_type=None)

    assert _ambiguous(fast_dir)
    assert not _ambiguous(slow_dir)


def test_declared_add_bos_token_is_forwarded_like_serving(tmp_path):
    """Serving passes add_bos_token/add_eos_token into the verbatim load.

    Transformers only rebuilds the post-processor when they arrive as explicit
    kwargs, so dropping them diverges from inference in both directions.
    """
    adds_bos = _mismatched_dir(tmp_path, name="bos-true", add_bos_token=True)
    strips_bos = _mismatched_dir(
        tmp_path,
        name="bos-false",
        add_bos_token=False,
        bos_in_post_processor=True,
    )

    assert tokenizers.load_tokenizer(str(adds_bos), local_files_only=True).encode(
        "hello"
    ) == [0, 1]
    # tokenizer.json prepends BOS, but the declaration says not to.
    assert tokenizers.load_tokenizer(str(strips_bos), local_files_only=True).encode(
        "hello"
    ) == [1]


def test_undeclared_post_processor_is_left_verbatim(tmp_path):
    """No declaration means no post-processor rebuild: the artifact wins."""
    model_dir = _mismatched_dir(tmp_path, bos_in_post_processor=True)

    loaded = tokenizers.load_tokenizer(str(model_dir), local_files_only=True)

    assert loaded.encode("hello") == [0, 1]


def test_declared_post_processor_kwargs_mirror_serving_presence_rule():
    assert tokenizers._declared_post_processor_kwargs({}) == {}
    assert tokenizers._declared_post_processor_kwargs({"add_bos_token": False}) == {
        "add_bos_token": False
    }
    # Presence, not truthiness: serving forwards an explicit null too.
    assert tokenizers._declared_post_processor_kwargs({"add_eos_token": None}) == {
        "add_eos_token": None
    }


def test_verbatim_load_failure_falls_back_like_serving(tmp_path, monkeypatch, caplog):
    """Sentencepiece-only ambiguous models have no tokenizer.json to load.

    ``mistralai/Ministral-8B-Instruct-2410`` is the live example: declares
    ``LlamaTokenizer`` against model_type ``mistral``, ships no
    ``tokenizer.json``. Serving falls back to AutoTokenizer there, so the
    cookbook must too instead of hard-failing a loadable model.
    """
    model_dir = _mismatched_dir(tmp_path, write_tokenizer_json=False)

    assert _ambiguous(model_dir)

    monkeypatch.setattr(
        tokenizers.transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(marker="auto"),
    )
    with caplog.at_level(logging.WARNING, logger=tokenizers.logger.name):
        loaded = tokenizers.load_tokenizer(str(model_dir), local_files_only=True)

    assert loaded.marker == "auto"
    assert "falling back to AutoTokenizer" in caplog.text


def test_verbatim_hub_failure_does_not_fall_back(tmp_path, monkeypatch):
    """A rate-limited download is not an artifact failure.

    Serving only loads local dirs and cannot hit this, so falling back would
    let a transient Hub error silently choose a different tokenizer than
    inference uses.
    """
    model_dir = _mismatched_dir(tmp_path)

    with local_status_server(429) as url:

        def failing_backend_load(*args, **kwargs):
            hf_raise_for_status(httpx.head(url))
            raise AssertionError("hf_raise_for_status must raise on 429")

        monkeypatch.setattr(
            TokenizersBackend, "from_pretrained", failing_backend_load
        )
        monkeypatch.setattr(
            tokenizers.transformers.AutoTokenizer,
            "from_pretrained",
            lambda *args, **kwargs: pytest.fail("must not reach AutoTokenizer"),
        )

        with pytest.raises(RuntimeError, match=r"\(HTTP 429\)"):
            tokenizers.load_tokenizer(str(model_dir), local_files_only=True)


def test_backend_selection_does_not_fail_open_on_hub_errors(monkeypatch):
    """An unreadable declaration must not silently restore the default path."""
    calls: list[str] = []

    with local_status_server(500) as url:

        def failing_download(model, filename, **kwargs):
            calls.append(filename)
            hf_raise_for_status(httpx.head(url))
            raise AssertionError("hf_raise_for_status must raise on 500")

        monkeypatch.setattr("huggingface_hub.hf_hub_download", failing_download)
        monkeypatch.setattr(
            tokenizers.transformers.AutoTokenizer,
            "from_pretrained",
            lambda *args, **kwargs: pytest.fail("must not reach AutoTokenizer"),
        )

        # Surfaced through the existing Hub-status boundary, so the managed
        # adapter still classifies it.
        with pytest.raises(RuntimeError, match=r"\(HTTP 500\)") as exc_info:
            tokenizers.load_tokenizer("Qwen/Qwen3.8-27B")

    assert isinstance(exc_info.value.__cause__, HfHubHTTPError)
    assert calls == ["tokenizer_config.json"]


def test_absent_hub_declaration_keeps_the_default_path(monkeypatch):
    def missing_download(model, filename, **kwargs):
        raise EntryNotFoundError(f"{filename} not found")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", missing_download)
    monkeypatch.setattr(
        tokenizers.transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(marker="auto"),
    )

    assert tokenizers.load_tokenizer("Qwen/Qwen3.8-27B").marker == "auto"


def test_corrupt_declaration_is_not_silently_ignored(tmp_path, monkeypatch):
    model_dir = _mismatched_dir(tmp_path)
    (model_dir / "tokenizer_config.json").write_text("{ not json")

    monkeypatch.setattr(
        tokenizers.transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("must not reach AutoTokenizer"),
    )

    with pytest.raises(RuntimeError, match="could not parse tokenizer_config.json"):
        tokenizers.load_tokenizer(str(model_dir), local_files_only=True)


def test_sidecar_tokenizer_loads_bundle_verbatim(tmp_path):
    """The sandbox loader reads the bundled tokenizer.json verbatim."""
    from training.renderer.tito.shared import load_sidecar_tokenizer

    bundle = tmp_path / "tokenizer"
    _write_tokenizer_dir(bundle, tokenizer_class="Qwen2Tokenizer", model_type=None)

    loaded = load_sidecar_tokenizer(bundle)

    assert isinstance(loaded, TokenizersBackend)
    assert loaded.encode("hello") == [1]
    assert loaded.chat_template == "{{ messages }}"


def test_sidecar_bundle_round_trips_a_rebuilt_post_processor(tmp_path):
    """``save_pretrained`` bakes the live post-processor into the bundle.

    That is why the sidecar does not re-forward add_bos_token/add_eos_token:
    the certified artifact already encodes them.
    """
    from training.renderer.tito.shared import load_sidecar_tokenizer

    source = _mismatched_dir(tmp_path, add_bos_token=True)
    certified = tokenizers.load_tokenizer(str(source), local_files_only=True)
    bundle = tmp_path / "tokenizer"
    certified.save_pretrained(bundle)

    assert load_sidecar_tokenizer(bundle).encode("hello") == certified.encode("hello")


def test_sidecar_tokenizer_requires_chat_template(tmp_path):
    from training.renderer.tito.shared import load_sidecar_tokenizer

    bundle = tmp_path / "tokenizer"
    _write_tokenizer_dir(bundle, tokenizer_class="Qwen2Tokenizer", model_type=None)
    config = json.loads((bundle / "tokenizer_config.json").read_text())
    del config["chat_template"]
    (bundle / "tokenizer_config.json").write_text(json.dumps(config))

    with pytest.raises(ValueError, match="no bundled chat template"):
        load_sidecar_tokenizer(bundle)


def test_sidecar_tokenizer_missing_tokenizer_json_fails_closed(tmp_path):
    """No fallback here: a sidecar has no other certified token contract."""
    from training.renderer.tito.shared import load_sidecar_tokenizer

    bundle = tmp_path / "tokenizer"
    _write_tokenizer_dir(
        bundle,
        tokenizer_class="Qwen2Tokenizer",
        model_type=None,
        write_tokenizer_json=False,
    )

    with pytest.raises(ValueError, match="missing tokenizer.json"):
        load_sidecar_tokenizer(bundle)
