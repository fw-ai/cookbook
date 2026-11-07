"""Real V4.1 data and positive-control tests for code-free loading."""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import pytest
import transformers
from huggingface_hub import hf_hub_download
from transformers.models.auto.image_processing_auto import AutoImageProcessor

from training import tokenizer_identity as identity
from training.renderer import _image_processing_utils, _tokenizer_utils, get_renderer
from training.renderer.verifier.utils import hf_parity, tokenizer as verifier
from training.utils import supervised, tokenizers

REPOSITORY = "deepseek-ai/DeepSeek-V4.1-Flash"
REVISION = "2cba9e42aa026125f3ed06c6d98c1db82f7ca027"
BATTERY = ("Hello world", "café 漢字 ☃", "<think>reason</think>answer", "<｜System｜>Reasoning Effort: 75")


@pytest.fixture
def staged(tmp_path: Path) -> Path:
    root = tmp_path / "opaque"
    root.mkdir()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copyfile(hf_hub_download(REPOSITORY, name, revision=REVISION), root / name)
    return root


def arm(root: Path, marker: Path) -> None:
    config = json.loads((root / "tokenizer_config.json").read_text())
    config["auto_map"] = {"AutoTokenizer": [None, "canary_tokenizer.CanaryTokenizer"]}
    (root / "tokenizer_config.json").write_text(json.dumps(config))
    (root / "canary_tokenizer.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "from transformers import PreTrainedTokenizerFast\n"
        "class CanaryTokenizer(PreTrainedTokenizerFast):\n    pass\n"
    )


def assert_stock(loaded: Any, root: Path) -> None:
    expected = transformers.AutoTokenizer.from_pretrained(root, trust_remote_code=False, local_files_only=True)
    assert type(loaded) is type(expected)
    for text in BATTERY:
        assert loaded.encode(text) == expected.encode(text)
        assert loaded.decode(loaded.encode(text)) == expected.decode(expected.encode(text))


@pytest.mark.parametrize("loader", ["cookbook", "renderer", "verifier", "parity", "worker", "pickle"])
@pytest.mark.parametrize("context_only", [False, True])
def test_artifact_canary_denied(
    staged: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, loader: str, context_only: bool
) -> None:
    if context_only:
        config = json.loads((staged / "config.json").read_text())
        config["model_type"] = "qwen3"
        (staged / "config.json").write_text(json.dumps(config))
        with (staged / "tokenizer.json").open("ab") as handle:
            handle.write(b"\n")
    marker = tmp_path / "executed"
    arm(staged, marker)
    # Positive control: same valid artifact executes when code is explicitly allowed.
    permissive = transformers.AutoTokenizer.from_pretrained(staged, trust_remote_code=True, local_files_only=True)
    assert marker.exists() and type(permissive).__module__.startswith("transformers_modules.")
    marker.unlink()
    monkeypatch.setenv("HF_TRUST_REMOTE_CODE", "1")
    source = str(staged)
    if loader == "cookbook":
        loaded = tokenizers.load_tokenizer(source, trust_remote_code=True, base_model="deepseek_v41")
    elif loader == "renderer":
        loaded = _tokenizer_utils.get_tokenizer(source, renderer_name="deepseek_v41")
    elif loader == "verifier":
        loaded = verifier.load_tokenizer(source, renderer_name="deepseek_v41")
    elif loader == "parity":
        loaded = hf_parity._load_tokenizer(source, "local", True, renderer_name="deepseek_v41")
    elif loader == "worker":
        state: dict[str, Any] = {}
        supervised.populate_render_worker_state(
            state,
            tokenizer_model=source,
            renderer_name="deepseek_v41",
            tokenizer_trust_remote_code=True,
            max_seq_len=256,
        )
        loaded = state["tokenizer"]
    else:
        original = transformers.AutoTokenizer.from_pretrained(staged, trust_remote_code=False)
        renderer = get_renderer("deepseek_v41", original, model_name=source)
        loaded = pickle.loads(pickle.dumps(renderer)).tokenizer
    assert not marker.exists()
    assert_stock(loaded, staged)


@pytest.mark.parametrize("form", ["upstream", "auto", "cookbook"])
def test_real_data_recognized_without_claiming_v4(staged: Path, tmp_path: Path, form: str) -> None:
    source = staged
    if form != "upstream":
        tokenizer = (
            tokenizers.load_tokenizer(str(staged), trust_remote_code=False)
            if form == "cookbook"
            else transformers.AutoTokenizer.from_pretrained(staged, trust_remote_code=False)
        )
        source = tmp_path / "saved"
        tokenizer.save_pretrained(source)
    assert (
        hashlib.sha256((source / "tokenizer.json").read_bytes()).hexdigest()
        in identity.DEEPSEEK_V41_TOKENIZER_JSON_SHA256
    )
    assert identity.requires_code_free_tokenizer_loading(str(source))
    assert not identity.is_deepseek_v4_tokenizer_source(str(source))
    assert supervised.resolve_renderer_name(REPOSITORY) == "deepseek_v41"
    for model_type in ("qwen3", "llama"):
        (source / "config.json").write_text(json.dumps({"model_type": model_type}))
        assert identity.requires_code_free_tokenizer_loading(str(source))
        assert not identity.is_deepseek_v4_tokenizer_source(str(source))


@pytest.mark.parametrize("loader", ["renderer", "verifier"])
def test_real_processor_canary_denied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, loader: str) -> None:
    root = tmp_path / "processor"
    root.mkdir()
    marker = tmp_path / "processor-executed"
    (root / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "image_processor_type": "CLIPImageProcessor",
                "auto_map": {"AutoImageProcessor": ["canary_processor.CanaryProcessor", None]},
            }
        )
    )
    (root / "canary_processor.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "from transformers import CLIPImageProcessor\n"
        "class CanaryProcessor(CLIPImageProcessor):\n    pass\n"
    )
    permissive = AutoImageProcessor.from_pretrained(root, trust_remote_code=True)
    assert marker.exists() and type(permissive).__module__.startswith("transformers_modules.")
    marker.unlink()
    monkeypatch.setenv("HF_TRUST_REMOTE_CODE", "1")
    load = _image_processing_utils.get_image_processor if loader == "renderer" else verifier.load_image_processor
    loaded = load(str(root), renderer_name="deepseek_v41")
    assert not marker.exists()
    assert type(loaded).__module__.startswith("transformers.")
    expected = AutoImageProcessor.from_pretrained(root, trust_remote_code=False)
    assert loaded.to_dict() == expected.to_dict()


@pytest.mark.parametrize(
    "base_type", ["deepseek_v4", "deepseek_v41", "deepseek_v41_text", "deepseek_v4.1", "deepseek_v4.1_text"]
)
def test_typed_base_context_does_not_change_resource_name(base_type: str) -> None:
    assert tokenizers.resolve_tokenizer_base_model("accounts/customer/models/opaque", base_type) == base_type


@pytest.mark.parametrize("model_type", ["deepseek_v41_text", "deepseek_v4.1", "deepseek_v4.1_text"])
def test_text_config_context_restricts_unknown_bytes(staged: Path, tmp_path: Path, model_type: str) -> None:
    config = json.loads((staged / "config.json").read_text())
    config["model_type"] = model_type
    (staged / "config.json").write_text(json.dumps(config))
    with (staged / "tokenizer.json").open("ab") as handle:
        handle.write(b"\n")
    marker = tmp_path / "text-canary"
    arm(staged, marker)
    loaded = tokenizers.load_tokenizer(str(staged), trust_remote_code=True)
    assert not marker.exists()
    assert_stock(loaded, staged)


@pytest.mark.parametrize(
    "etag",
    [
        "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b",
        "23136b659802e7e16fc849c551ac4a05a85e708732adeed010610b11dce85290",
        "f8c19e2bc45ba524379b42c29ea07e83c9f856c15c2f7ebb69e36ddcfa1cf650",
    ],
)
def test_hub_restriction_and_semantic_identity_are_separate(etag: str) -> None:
    assert identity._identity._metadata_decision(None, etag, include_v41=True) is True
    assert identity._identity._metadata_decision(None, etag) is False
