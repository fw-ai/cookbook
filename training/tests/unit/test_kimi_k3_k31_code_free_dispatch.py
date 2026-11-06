"""Kimi K3/K3.1 code-free dispatch: recognition, canaries, fail-closed pins, parity.

Like the DeepSeek V4 dispatch suite, these tests download the pinned official
Kimi K3/K3.1 tokenizer artifacts from HuggingFace (small JSON/data files only,
never weights). The canary modules are test-local fixtures that record any
execution; they demonstrate that artifact Python cannot run on migrated Kimi
load paths -- including under an explicit ``trust_remote_code=True`` -- and
that tampered data fails closed instead of falling back to checkpoint code.

The checkpoint tokenizer's own Python is loaded only in the deliberately
remote-code oracle fixtures, to certify that the reviewed native bundle is
token-identical.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
import tomllib
from typing import Any

import pytest

from training.renderer._kimi_native_encoding import KimiEncodingAdapter
from training.renderer.kimi_k3 import (
    KIMI_K3_ENCODER_SHA256,
    KIMI_K3_REPOSITORY,
    KIMI_K3_REVISION,
)
from training.renderer.kimi_k3 import _vendored_encoder as _k3_vendored_encoder
from training.renderer.tito import kimi_k31 as _tito_kimi_k31
from training.renderer.verifier.utils import hf_parity
from training.tokenizer_identity import (
    is_kimi_k3_k31_model_name,
    is_kimi_k3_k31_tokenizer_source,
    requires_code_free_tokenizer_loading,
)
from training.utils import tokenizers as cookbook_tokenizers
from training.utils.kimi_native_tokenizer import (
    KIMI_K3_PROFILE,
    KIMI_K31_PROFILE,
    KIMI_K3_REPOSITORY as NATIVE_K3_REPOSITORY,
    KIMI_K3_REVISION as NATIVE_K3_REVISION,
    KIMI_K31_REPOSITORY as NATIVE_K31_REPOSITORY,
    KIMI_K31_REVISION as NATIVE_K31_REVISION,
)
from training.renderer import _tokenizer_utils

K31_REPOSITORY = "moonshotai/Kimi-K3.1-Pro"
K31_REVISION = "d9e99b0792a5e93b78ea9bea62a68167e36ddd9e"

# ``encoding_k3.py`` is needed only by the deliberately remote-code oracle
# (``tokenization_kimi.py`` imports it); the native path never reads it.
_SOURCE_FILES = (
    "config.json",
    "encoding_k3.py",
    "tiktoken.model",
    "tokenization_kimi.py",
    "tokenizer_config.json",
)

_ENCODE_BATTERY = [
    "Hello world",
    "Kimi K3.1 tokenizer parity check",
    "unicode battery: café 漢字 ☃",
    "<|open|>message<|sep|>structural<|close|>message<|sep|>",
    "literal <|sep|> inside user text",
    " ".join(f"tok{i}" for i in range(512)),
    "x" * 26_000,  # past the checkpoint tokenizer's no-whitespace chunk limit
    "ca\u00e9 ".join(["pair"] * 200),
]


@pytest.fixture(scope="module", params=[KIMI_K3_PROFILE, KIMI_K31_PROFILE])
def official_dir(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The pinned reviewed source files of one profile, staged like a trainer dir."""
    hub = pytest.importorskip("huggingface_hub")
    repo, revision = (
        (KIMI_K3_REPOSITORY, KIMI_K3_REVISION) if request.param == KIMI_K3_PROFILE else (K31_REPOSITORY, K31_REVISION)
    )
    directory = tmp_path_factory.mktemp(f"kimi-official-{request.param}")
    for filename in _SOURCE_FILES:
        local = hub.hf_hub_download(repo, filename, revision=revision)
        shutil.copy2(local, directory / filename)
    return directory


def _write_config_canary(directory: Path, marker: Path) -> Path:
    """Poison ``config.json`` (not digest-pinned) with an executable auto_map."""
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text())
    config["auto_map"] = {"AutoConfig": "canary_config.CanaryConfig"}
    config_path.write_text(json.dumps(config))
    (directory / "canary_config.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "from transformers import PreTrainedConfig\n"
        "class CanaryConfig(PreTrainedConfig):\n    model_type = 'canary'\n"
    )
    return marker


def test_canary_never_executes_on_migrated_load_paths(official_dir: Path, tmp_path: Path) -> None:
    assert is_kimi_k3_k31_tokenizer_source(str(official_dir)) is True
    assert requires_code_free_tokenizer_loading(str(official_dir)) is True
    poisoned = tmp_path / "poisoned-config"
    shutil.copytree(official_dir, poisoned)
    marker = _write_config_canary(poisoned, tmp_path / "canary_executed")
    for loaded in (
        cookbook_tokenizers.load_tokenizer(str(poisoned), trust_remote_code=True),
        _tokenizer_utils.get_tokenizer(str(poisoned)),
        _tokenizer_utils.get_tokenizer(str(poisoned), renderer_name="kimi_k31"),
        hf_parity._load_tokenizer(str(poisoned), None, True),
    ):
        assert type(loaded).__name__ == "TokenizersBackend"
        assert not type(loaded).__module__.startswith("transformers_modules.")
    assert not marker.exists()


@pytest.mark.parametrize("hub_source", [False, True], ids=["local", "pinned-hub"])
def test_qa_reference_uses_installed_kimi_encoder(
    official_dir: Path, hub_source: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = json.loads((official_dir / "config.json").read_text())["model_type"]
    repository, revision = (
        (KIMI_K3_REPOSITORY, KIMI_K3_REVISION) if profile == KIMI_K3_PROFILE else (K31_REPOSITORY, K31_REVISION)
    )

    def checkpoint_code_forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Kimi QA must not call the checkpoint AutoTokenizer")

    monkeypatch.setattr(hf_parity.AutoTokenizer, "from_pretrained", checkpoint_code_forbidden)
    source = repository if hub_source else str(official_dir)
    loaded = hf_parity._load_tokenizer(source, revision if hub_source else None, True)
    assert not type(loaded).__module__.startswith("transformers_modules.")
    assert loaded.apply_chat_template.profile == profile
    assert loaded.apply_chat_template([{"role": "user", "content": "Hello <|open|>"}], tokenize=True)


def test_tampered_vocabulary_fails_closed(official_dir: Path, tmp_path: Path) -> None:
    """A recognized Kimi dir with altered bytes raises; nothing falls back to code."""
    tampered = tmp_path / "tampered"
    shutil.copytree(official_dir, tampered)
    with (tampered / "tiktoken.model").open("ab") as stream:
        stream.write(b" \n")
    assert is_kimi_k3_k31_tokenizer_source(str(tampered)) is True  # recognized via model_type
    with pytest.raises(ValueError, match="reviewed pins"):
        cookbook_tokenizers.load_tokenizer(str(tampered), trust_remote_code=True)
    with pytest.raises(ValueError, match="reviewed pins"):
        hf_parity._load_tokenizer(str(tampered), None, True)


def test_explicit_other_family_is_not_commandeered(official_dir: Path, tmp_path: Path) -> None:
    """Shared vocabulary bytes must not re-label an explicitly foreign family."""
    foreign = tmp_path / "foreign"
    shutil.copytree(official_dir, foreign)
    (foreign / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    assert is_kimi_k3_k31_tokenizer_source(str(foreign)) is False
    # Restrict-only signal: checkpoint code stays inert, caller policy decides.
    assert requires_code_free_tokenizer_loading(str(foreign)) is True


def test_same_artifact_renamed_keeps_data_earned_decision(official_dir: Path, tmp_path: Path) -> None:
    renamed = tmp_path / "totally-unrelated-name"
    shutil.copytree(official_dir, renamed)
    assert is_kimi_k3_k31_tokenizer_source(str(renamed)) is True
    loaded = cookbook_tokenizers.load_tokenizer(str(renamed), trust_remote_code=True)
    assert type(loaded).__name__ == "TokenizersBackend"


def test_saved_native_bundle_reloads_through_cookbook_and_renderer(official_dir: Path, tmp_path: Path) -> None:
    native = cookbook_tokenizers.load_tokenizer(str(official_dir))
    native.save_pretrained(tmp_path)
    assert not (tmp_path / "tiktoken.model").exists()
    messages = [{"role": "user", "content": "Hello <|open|>"}]
    expected = native.apply_chat_template(messages, tokenize=True)
    for loaded in (
        cookbook_tokenizers.load_tokenizer(
            str(tmp_path), base_model="moonshotai/Kimi-K3.1-Pro", trust_remote_code=True
        ),
        cookbook_tokenizers.load_tokenizer(str(tmp_path)),
        _tokenizer_utils.get_tokenizer(str(tmp_path), base_model="moonshotai/Kimi-K3"),
        _tokenizer_utils.get_tokenizer(str(tmp_path)),
    ):
        assert loaded.apply_chat_template(messages, tokenize=True) == expected


def test_native_bundle_matches_checkpoint_tokenizer(official_dir: Path) -> None:
    """The reviewed native bundle is token-identical to the checkpoint tokenizer.

    The oracle load deliberately executes the pinned checkpoint Python -- the
    only place it runs -- to certify the code-free path byte for byte.
    """
    transformers = pytest.importorskip("transformers")
    native = cookbook_tokenizers.load_tokenizer(str(official_dir), trust_remote_code=True)
    oracle = transformers.AutoTokenizer.from_pretrained(str(official_dir), trust_remote_code=True)
    assert type(oracle).__name__ == "TikTokenTokenizer"
    native_encoding = KimiEncodingAdapter(native)
    assert native_encoding.is_native
    for text in _ENCODE_BATTERY:
        for allow_special in (True, False):
            expected = list(oracle._encode_text_piece(text, allow_special_tokens=allow_special))
            assert native_encoding.encode_piece(text, allow_special_tokens=allow_special) == expected, (
                text[:40],
                allow_special,
            )
        assert native.encode(text) == oracle.encode(text), text[:40]
        sample = native.encode(text)
        assert native.decode(sample) == oracle.decode(sample), text[:40]
    for name in (
        "<|kimi_image_placeholder|>",
        "<|media_pad|>",
        "<|open|>",
        "<|close|>",
        "<|sep|>",
    ):
        assert native.convert_tokens_to_ids(name) == oracle.convert_tokens_to_ids(name), name


def test_vendored_k3_encoder_matches_certification_sha256() -> None:
    import hashlib

    vendored = Path(_k3_vendored_encoder.__file__)
    assert hashlib.sha256(vendored.read_bytes()).hexdigest() == KIMI_K3_ENCODER_SHA256


def test_renderer_and_tito_pins_match_sdk_source_of_truth() -> None:
    """Cookbook renderer pins and the SDK native loader pins cannot drift apart."""
    assert KIMI_K3_REPOSITORY == NATIVE_K3_REPOSITORY
    assert KIMI_K3_REVISION == NATIVE_K3_REVISION
    assert K31_REPOSITORY == NATIVE_K31_REPOSITORY
    assert K31_REVISION == NATIVE_K31_REVISION
    assert _tito_kimi_k31.KIMI_K31_REPOSITORY == NATIVE_K31_REPOSITORY
    assert _tito_kimi_k31.KIMI_K31_REVISION == NATIVE_K31_REVISION


def test_hub_sources_skip_advisory_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    from training import tokenizer_identity as identity

    def unavailable(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("reviewed Kimi Hub sources must not need advisory reads")

    monkeypatch.setattr(identity._identity, "_hub_download", unavailable)
    assert is_kimi_k3_k31_tokenizer_source(KIMI_K3_REPOSITORY) is True
    assert is_kimi_k3_k31_tokenizer_source(K31_REPOSITORY) is True
    assert is_kimi_k3_k31_model_name("accounts/fireworks/models/kimi-k3p1-pro") is True


def test_vendored_encoder_is_in_cookbook_artifacts() -> None:
    """Explicit setuptools packages must include the runtime native bundle."""
    project = Path(__file__).resolve().parents[2] / "pyproject.toml"
    metadata = tomllib.loads(project.read_text())
    setuptools = metadata["tool"]["setuptools"]
    assert "training.renderer._vendored" in setuptools["packages"]
    assert "training.renderer._vendored.kimi_k3" in setuptools["packages"]
    assert set(setuptools["package-data"]["training.renderer._vendored.kimi_k3"]) >= {
        "LICENSE",
        "PROVENANCE.md",
    }
