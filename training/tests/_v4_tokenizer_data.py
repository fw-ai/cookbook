"""Pinned data and executable canaries shared by cookbook integration tests."""

from pathlib import Path
import json
import shutil
from typing import Any

import pytest

V4_REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
V4_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
V4_ALIAS = "accounts/fireworks/models/deepseek-v4-flash-0731"
V41_ALIAS = "accounts/fireworks/models/deepseek-v4.1-flash"
NON_V4_MODEL = "accounts/acme/models/custom-x"
RENAMED_ALIAS = "accounts/acme/models/production-model"
TOKEN_BATTERY = ["Hello world", "unicode: café 漢字 ☃", "<think>reasoning</think>answer", " ".join(["token"] * 512)]


@pytest.fixture(scope="module")
def v4_files() -> dict[str, Path]:
    # lazy: HF is optional outside tokenizer integration tests.
    from huggingface_hub import hf_hub_download
    from huggingface_hub import constants
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return {
            name: Path(hf_hub_download(V4_REPO, name, revision=V4_REVISION))
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json")
        }
    except LocalEntryNotFoundError:
        if not constants.HF_HUB_OFFLINE:
            raise
        pytest.skip("Pinned V4 tokenizer data is absent from the offline Hub cache; run online once to populate it.")


@pytest.fixture(scope="module")
def official_dir(v4_files: dict[str, Path], tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("v4-official")
    for name, source in v4_files.items():
        shutil.copy2(source, directory / name)
    return directory


def write_canary(directory: Path, marker: Path) -> Path:
    config_path = directory / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["auto_map"] = {"AutoTokenizer": [None, "canary_tokenizer.CanaryTokenizer"]}
    config_path.write_text(json.dumps(config))
    (directory / "canary_tokenizer.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "from transformers import PreTrainedTokenizerFast\n"
        "class CanaryTokenizer(PreTrainedTokenizerFast):\n    pass\n"
    )
    return marker


def tampered_v4_dir(
    v4_files: dict[str, Path], tmp_path: Path, *, with_config: bool = False, name: str = "default"
) -> tuple[Path, Path]:
    directory = tmp_path / name
    directory.mkdir(parents=True)
    for filename in ("tokenizer.json", "tokenizer_config.json") + (("config.json",) if with_config else ()):
        shutil.copy2(v4_files[filename], directory / filename)
    # Preserve token semantics while invalidating the reviewed byte digest.
    with (directory / "tokenizer.json").open("ab") as stream:
        stream.write(b" \n")
    marker = write_canary(directory, tmp_path / f"executed-{name}")
    return directory, marker


def assert_tokenizers_equal(actual: Any, expected: Any) -> None:
    assert type(actual) is type(expected)
    assert actual.special_tokens_map == expected.special_tokens_map
    assert getattr(actual, "chat_template", None) == getattr(expected, "chat_template", None)
    for text in TOKEN_BATTERY:
        tokens = actual.encode(text)
        assert tokens == expected.encode(text), text[:40]
        assert actual.decode(tokens) == expected.decode(tokens)


def assert_stock_tokenizer(tokenizer: Any, directory: Path) -> None:
    # lazy: Transformers is optional outside tokenizer integration tests.
    from transformers import AutoTokenizer

    stock = AutoTokenizer.from_pretrained(str(directory), trust_remote_code=False, local_files_only=True)
    assert_tokenizers_equal(tokenizer, stock)
