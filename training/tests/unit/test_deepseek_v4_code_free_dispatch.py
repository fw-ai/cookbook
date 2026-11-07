"""DeepSeek V4 code-free dispatch: recognition, canaries, and tokenizer parity.

Like the existing TITO renderer tests, these download the pinned official
DeepSeek V4 tokenizer artifacts from HuggingFace (small JSON/data files only,
never weights). The canary modules are test-local fixtures that record any
execution; they demonstrate that artifact Python cannot run on migrated V4
load paths -- including under an explicit ``trust_remote_code=True``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import transformers

from training.tokenizer_identity import (
    DEEPSEEK_V4_TOKENIZER_JSON_SHA256,
    is_deepseek_v4_tokenizer_source,
    requires_code_free_tokenizer_loading,
)
from training.renderer import _tokenizer_utils
from training.tests._v4_tokenizer_data import (
    V4_ALIAS,
    V4_REPO as V4_REPOSITORY,
    V4_REVISION,
    assert_tokenizers_equal as _assert_battery_equal,
    official_dir as official_dir,
    v4_files as v4_files,
    write_canary as _write_canary,
)
from training.utils import tokenizers as cookbook_tokenizers


@pytest.mark.parametrize("base_model", [None, "deepseek_v4"])
@pytest.mark.parametrize("model_type", ["deepseek_v41", "qwen3", "llama", "new_model_family"])
def test_reviewed_tokenizer_bytes_block_foreign_source_metadata_code(
    official_dir: Path, tmp_path: Path, base_model: str | None, model_type: str
) -> None:
    directory = tmp_path / "shared-tokenizer"
    directory.mkdir()
    shutil.copy2(official_dir / "tokenizer.json", directory / "tokenizer.json")
    shutil.copy2(official_dir / "tokenizer_config.json", directory / "tokenizer_config.json")
    (directory / "config.json").write_text(json.dumps({"model_type": model_type}))
    marker = _write_canary(directory, tmp_path / "foreign_source_canary")
    tokenizer = cookbook_tokenizers.load_tokenizer(
        str(directory), trust_remote_code=True, local_files_only=True, base_model=base_model
    )
    assert not marker.exists()
    stock = transformers.AutoTokenizer.from_pretrained(str(directory), trust_remote_code=False, local_files_only=True)
    _assert_battery_equal(tokenizer, stock)


def test_official_config_and_tokenizer_config_declare_no_auto_map(official_dir: Path) -> None:
    """Pinned official V4 metadata declares no remote code on either file.

    Weight artifacts stay on the native path (FireTitan Dsv4Model +
    Dsv4StateDictAdapter) and tokenizer loads stay on stock classes; neither
    pinned config may ever acquire an auto_map declaration.
    """
    config = json.loads((official_dir / "config.json").read_text())
    assert config.get("model_type") == "deepseek_v4"
    assert "auto_map" not in config
    tokenizer_config = json.loads((official_dir / "tokenizer_config.json").read_text())
    assert tokenizer_config.get("tokenizer_class") == "PreTrainedTokenizerFast"
    assert "auto_map" not in tokenizer_config


@pytest.fixture(scope="module")
def official_tokenizer_forms(official_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, bytes]:
    """Generate each supported serialization once for cookbook integration tests."""
    root = tmp_path_factory.mktemp("dsv4-tokenizer-forms")
    forms = {"hub": (official_dir / "tokenizer.json").read_bytes()}
    tokenizer = cookbook_tokenizers.load_tokenizer(str(official_dir))
    saved = root / "saved"
    tokenizer.save_pretrained(saved)
    forms["backend"] = (saved / "tokenizer.json").read_bytes()
    # The plain AutoTokenizer save produces the other pinned saved form (the
    # backend rebuilt from the declared class), as in the SDK save flow.
    fast = transformers.AutoTokenizer.from_pretrained(str(official_dir), trust_remote_code=False)
    fast_saved = root / "saved-fast"
    fast.save_pretrained(fast_saved)
    forms["auto"] = (fast_saved / "tokenizer.json").read_bytes()
    return forms


def test_pinned_digests_cover_hub_and_saved_forms(official_tokenizer_forms: dict[str, bytes]) -> None:
    digests = {hashlib.sha256(data).hexdigest() for data in official_tokenizer_forms.values()}
    assert digests == DEEPSEEK_V4_TOKENIZER_JSON_SHA256


@pytest.mark.parametrize("form", ["hub", "auto", "backend"])
@pytest.mark.parametrize(
    ("model_type", "source", "renderer"),
    [
        ("deepseek_v41", "deepseek-v4.1-flash", "deepseek_v41"),
        ("qwen3", "Qwen/Qwen3-8B", "qwen3"),
        ("llama", "meta-llama/Llama-3.1-8B-Instruct", "llama3"),
    ],
)
def test_other_family_shared_tokenizer_does_not_select_v4_renderer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    official_tokenizer_forms: dict[str, bytes],
    form: str,
    model_type: str,
    source: str,
    renderer: str,
) -> None:
    # lazy: renderer selection needs the optional training runtime.
    from training.utils.supervised import resolve_renderer_name

    monkeypatch.chdir(tmp_path)
    directory = Path(source)
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps({"model_type": model_type}))
    (directory / "tokenizer.json").write_bytes(official_tokenizer_forms[form])
    assert resolve_renderer_name(source) == renderer
    assert not is_deepseek_v4_tokenizer_source(str(directory))
    assert requires_code_free_tokenizer_loading(str(directory))


@pytest.mark.parametrize("config", [{}, {"model_type": "model"}, {"model_type": ""}])
def test_shared_digest_still_identifies_generic_saved_tokenizer(
    tmp_path: Path, official_tokenizer_forms: dict[str, bytes], config: dict[str, str]
) -> None:
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "tokenizer.json").write_bytes(official_tokenizer_forms["auto"])
    assert is_deepseek_v4_tokenizer_source(str(tmp_path))


# ---------------------------------------------------------------------------
# Canaries: artifact Python cannot execute on migrated V4 load paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trust", [None, True])
def test_auto_map_canary_cannot_execute_with_config_present(official_dir: Path, tmp_path: Path, trust) -> None:
    directory = tmp_path / "staged"
    shutil.copytree(official_dir, directory)
    marker = _write_canary(directory, tmp_path / "CANARY_EXECUTED_config")
    tokenizer = cookbook_tokenizers.load_tokenizer(str(directory), trust_remote_code=trust)
    assert not marker.exists()
    assert tokenizer.encode("canary check")  # loaded and functional


def test_auto_map_canary_cannot_execute_in_configless_saved_dir(official_dir: Path, tmp_path: Path) -> None:
    original = cookbook_tokenizers.load_tokenizer(str(official_dir))
    directory = tmp_path / "saved"
    original.save_pretrained(directory)
    assert not (directory / "config.json").exists()
    marker = _write_canary(directory, tmp_path / "CANARY_EXECUTED_saved")
    reloaded = cookbook_tokenizers.load_tokenizer(str(directory))
    assert not marker.exists()
    assert type(reloaded) is type(original)
    _assert_battery_equal(reloaded, original)


def test_renderer_get_tokenizer_local_canary(official_dir: Path, tmp_path: Path) -> None:
    directory = tmp_path / "staged-local"
    shutil.copytree(official_dir, directory)
    marker = _write_canary(directory, tmp_path / "CANARY_EXECUTED_local")
    tokenizer = _tokenizer_utils.get_tokenizer(str(directory))
    assert not marker.exists()
    assert tokenizer.encode("renderer local canary")


def test_env_opt_in_cannot_reopen_recognized_hub_id(monkeypatch: pytest.MonkeyPatch, official_dir: Path) -> None:
    monkeypatch.setenv("HF_TRUST_REMOTE_CODE", "1")
    captured: dict[str, Any] = {}
    real_from_pretrained = transformers.AutoTokenizer.from_pretrained

    def spy(model: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return real_from_pretrained(model, **kwargs)

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", spy)
    _tokenizer_utils._get_hf_tokenizer.cache_clear()
    _tokenizer_utils.get_tokenizer(V4_REPOSITORY)
    assert captured.get("trust_remote_code") is False


@pytest.mark.parametrize(
    ("model_name", "revision"),
    [
        ("moonshotai/Kimi-K2", None),
        ("moonshotai/Kimi-K2-Thinking", "a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55"),
        ("moonshotai/Kimi-K2.5", "2426b45b6af0da48d0dcce71bbce6225e5c73adc"),
        ("moonshotai/Kimi-K2.6", "b5aabbfb20227ed42becbf5541dbffd213942c58"),
    ],
)
@pytest.mark.parametrize("base_model", [V4_REPOSITORY, "Qwen/Qwen3-8B"])
def test_base_model_policy_precedes_kimi_dynamic_import(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    revision: str | None,
    base_model: str,
) -> None:
    monkeypatch.setenv("HF_TRUST_REMOTE_CODE", "1")
    sentinel = object()
    auto_calls: list[tuple[str, dict[str, Any]]] = []
    dynamic_calls: list[tuple[str, str, dict[str, Any]]] = []
    class_loads: list[tuple[str, dict[str, Any]]] = []

    def auto_load(model: str, **kwargs: Any) -> Any:
        auto_calls.append((model, kwargs))
        return sentinel

    def class_load(model: str, **kwargs: Any) -> Any:
        class_loads.append((model, kwargs))
        return sentinel

    def dynamic_load(class_reference: str, model: str, **kwargs: Any) -> Any:
        assert base_model != V4_REPOSITORY, "V4 base context must block the direct dynamic import"
        dynamic_calls.append((class_reference, model, kwargs))
        return SimpleNamespace(from_pretrained=class_load)

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", auto_load)
    monkeypatch.setattr("transformers.dynamic_module_utils.get_class_from_dynamic_module", dynamic_load)
    _tokenizer_utils._get_hf_tokenizer.cache_clear()
    try:
        assert _tokenizer_utils.get_tokenizer(model_name, base_model=base_model) is sentinel
    finally:
        _tokenizer_utils._get_hf_tokenizer.cache_clear()
    if base_model == V4_REPOSITORY:
        assert auto_calls == [(model_name, {"trust_remote_code": False})]
        assert not dynamic_calls
        assert not class_loads
    else:
        assert not auto_calls
        assert dynamic_calls == [("tokenization_kimi.TikTokenTokenizer", model_name, {"revision": revision})]
        assert class_loads == [(model_name, {"revision": revision})]


# ---------------------------------------------------------------------------
# Tokenizer parity: class, encode/decode, special tokens, template, save/reload
# ---------------------------------------------------------------------------


def test_policy_variants_resolve_identical_tokens(official_dir: Path) -> None:
    reference = transformers.AutoTokenizer.from_pretrained(str(official_dir), trust_remote_code=False)
    narrowed = cookbook_tokenizers.load_tokenizer(str(official_dir))
    explicit_true = cookbook_tokenizers.load_tokenizer(str(official_dir), trust_remote_code=True)
    assert type(narrowed) is type(explicit_true)
    assert narrowed.special_tokens_map == reference.special_tokens_map
    assert getattr(narrowed, "chat_template", None) == getattr(reference, "chat_template", None)
    _assert_battery_equal(narrowed, reference)
    _assert_battery_equal(explicit_true, reference)


@pytest.mark.parametrize("renderer_name_is_resolved", [False, True])
@pytest.mark.parametrize("caller_trust", [None, True])
def test_render_worker_v4_context_cannot_reopen_code_when_identity_changes(
    official_dir: Path, tmp_path: Path, renderer_name_is_resolved: bool, caller_trust: bool | None
) -> None:
    from training.utils import supervised

    directory = tmp_path / "configless-tokenizer"
    original = transformers.AutoTokenizer.from_pretrained(str(official_dir), trust_remote_code=False)
    original.save_pretrained(directory)
    tokenizer_path = directory / "tokenizer.json"
    tokenizer_path.write_text(json.dumps(json.loads(tokenizer_path.read_text()), indent=1))
    assert not is_deepseek_v4_tokenizer_source(str(directory))
    marker = _write_canary(directory, tmp_path / "render_worker_executed")
    state: dict[str, Any] = {}
    error = None
    try:
        supervised.populate_render_worker_state(
            state,
            tokenizer_model=str(directory),
            tokenizer_trust_remote_code=caller_trust,
            renderer_name="deepseek_v4",
            renderer_name_is_resolved=renderer_name_is_resolved,
            max_seq_len=4096,
        )
    except Exception as exc:
        error = exc
    assert not marker.exists(), "Known V4 renderer must prohibit artifact Python even when recognition misses"
    assert error is None, f"Stock tokenizer and renderer must still initialize: {error}"
    _assert_battery_equal(state["tokenizer"], original)
    assert state["renderer"].__class__.__name__ == "DeepseekV4Renderer"
    assert state["max_seq_len"] == 4096


@pytest.fixture
def unidentified_v4_dir(official_dir: Path, tmp_path: Path) -> tuple[Path, Any]:
    """Semantically identical V4 data whose mutable recognition signals miss."""
    original = transformers.AutoTokenizer.from_pretrained(str(official_dir), trust_remote_code=False)
    directory = tmp_path / "unidentified-tokenizer"
    original.save_pretrained(directory)
    tokenizer_path = directory / "tokenizer.json"
    tokenizer_path.write_text(json.dumps(json.loads(tokenizer_path.read_text()), indent=1))
    assert not is_deepseek_v4_tokenizer_source(str(directory))
    return directory, original


@pytest.mark.parametrize("revision", [None, V4_REVISION])
@pytest.mark.parametrize("caller_trust", [None, True])
def test_hf_parity_v4_renderer_context_blocks_unidentified_artifact_code(
    unidentified_v4_dir: tuple[Path, Any], tmp_path: Path, revision: str | None, caller_trust: bool | None
) -> None:
    from training.renderer.verifier.utils import hf_parity

    directory, original = unidentified_v4_dir
    # V4 uses a repository encoder instead of a Jinja oracle. Capture the
    # tokenizer passed to the renderer; parity's later oracle error is expected.
    seen = []
    get_renderer = hf_parity.get_renderer

    def capture_renderer(name, tokenizer):
        seen.append(tokenizer)
        return get_renderer(name, tokenizer)

    marker = _write_canary(directory, tmp_path / "parity_executed")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(hf_parity, "get_renderer", capture_renderer)
        try:
            hf_parity.compare_renderer_to_hf(
                renderer_name="deepseek_v4",
                tokenizer_model=str(directory),
                tokenizer_revision=revision,
                tokenizer_trust_remote_code=caller_trust,
                messages=[{"role": "user", "content": "Hello"}],
            )
        except ValueError as exc:
            assert "chat template" in str(exc).lower() or "chat_template" in str(exc)
    assert not marker.exists(), "Known V4 parity context must prohibit artifact Python"
    assert len(seen) == 1
    _assert_battery_equal(seen[0], original)


def test_unpickle_v4_renderer_blocks_unidentified_artifact_code(
    unidentified_v4_dir: tuple[Path, Any], tmp_path: Path
) -> None:
    import pickle

    from training.renderer import get_renderer

    directory, original = unidentified_v4_dir
    renderer = get_renderer("deepseek_v4", original, model_name=str(directory))
    messages = [{"role": "user", "content": "Hello"}]
    baseline = renderer.build_generation_prompt(messages)
    serialized = pickle.dumps(renderer)
    marker = _write_canary(directory, tmp_path / "unpickle_executed")
    restored = pickle.loads(serialized)
    assert not marker.exists(), "Renderer reconstruction must retain the known V4 loading policy"
    assert type(restored) is type(renderer)
    assert restored.build_generation_prompt(messages) == baseline


def test_verifier_ui_v4_context_blocks_unidentified_artifact_code(
    unidentified_v4_dir: tuple[Path, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from training.renderer.verifier import serve

    directory, original = unidentified_v4_dir
    marker = _write_canary(directory, tmp_path / "ui_executed")
    monkeypatch.setattr(serve, "resolve_dispatch", lambda **kwargs: ("test-model", "test"))
    monkeypatch.setattr(serve, "_client", lambda *args: None)
    monkeypatch.setattr(serve, "run_probe", lambda **kwargs: kwargs)
    result = serve._run_one_probe({"renderer": "deepseek_v4", "tokenizer_model": str(directory), "messages": []})
    assert not marker.exists(), "Verifier UI must pass known V4 context before loading artifact data"
    _assert_battery_equal(result["tokenizer"], original)


@pytest.mark.parametrize("entrypoint", ["cli", "triage"])
def test_verifier_cli_tokenizer_retains_v4_policy(
    unidentified_v4_dir: tuple[Path, Any], tmp_path: Path, entrypoint: str
) -> None:
    from training.renderer.verifier import cli, triage

    directory, original = unidentified_v4_dir
    marker = _write_canary(directory, tmp_path / "cli_executed")
    loader = (cli if entrypoint == "cli" else triage)._build_tokenizer
    tokenizer = loader(str(directory), renderer_name="deepseek_v4")
    assert not marker.exists()
    _assert_battery_equal(tokenizer, original)


def test_verifier_cache_cannot_reuse_other_renderer_remote_code(
    unidentified_v4_dir: tuple[Path, Any], tmp_path: Path
) -> None:
    from training.renderer.verifier import serve

    directory, original = unidentified_v4_dir
    marker = _write_canary(directory, tmp_path / "cache_executed")
    # The next model is held: explicitly preserve its existing permissive
    # policy, then prove its cached object cannot cross into the V4 context.
    custom_tokenizer = serve._tokenizer(str(directory), renderer_name="qwen3")
    assert marker.exists()
    marker.unlink()
    code_free_tokenizer = serve._tokenizer(str(directory), renderer_name="deepseek_v4")
    assert not marker.exists()
    assert type(code_free_tokenizer) is type(original)
    assert code_free_tokenizer is not custom_tokenizer
    _assert_battery_equal(code_free_tokenizer, original)


@pytest.mark.parametrize("loader_name", ["renderer", "verifier"])
def test_processor_v4_context_overrides_environment_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, loader_name: str
) -> None:
    from transformers.models.auto.image_processing_auto import AutoImageProcessor
    from training.renderer import _image_processing_utils
    from training.renderer.verifier.utils import tokenizer as verifier_tokenizers

    calls = []
    processor = object()

    def capture(model, **kwargs):
        calls.append((model, kwargs))
        return processor

    monkeypatch.setenv("HF_TRUST_REMOTE_CODE", "1")
    monkeypatch.setattr(AutoImageProcessor, "from_pretrained", capture)
    # V4 is text-only: this verifies the dormant processor sink's policy,
    # not a claim of supported V4 vision functionality or an image-code repro.
    loader = (
        _image_processing_utils.get_image_processor
        if loader_name == "renderer"
        else verifier_tokenizers.load_image_processor
    )
    assert loader(str(tmp_path), renderer_name="deepseek_v4") is processor
    assert calls[0][1]["trust_remote_code"] is False


# ---------------------------------------------------------------------------
# Family-named local directories: names never substitute for data. A hostile
# family-named dir with mutated artifact identity is NOT recognized and keeps
# the legacy permissive default; only its data (config model_type / digest)
# or an explicitly selected context (reviewed base_model / renderer_name)
# restricts it to installed code.
# ---------------------------------------------------------------------------


def _family_named_hostile_dir(official_dir: Path, tmp_path: Path, dirname: str) -> tuple[Path, Path, Any]:
    """V4 data under a family-named path, digest-mutated and auto_map-armed."""
    original = transformers.AutoTokenizer.from_pretrained(str(official_dir), trust_remote_code=False)
    directory = tmp_path / dirname
    directory.mkdir(parents=True)
    original.save_pretrained(directory)
    tokenizer_path = directory / "tokenizer.json"
    tokenizer_path.write_text(json.dumps(json.loads(tokenizer_path.read_text()), indent=1))
    marker = _write_canary(directory, tmp_path / f"executed-{dirname}")
    return directory, marker, original


def test_family_named_local_dir_decision_follows_data_not_name(official_dir: Path, tmp_path: Path) -> None:
    """Same artifact, same family-shaped name: mutated data is not recognized;
    restoring a V4 config earns recognition. The name contributes nothing."""
    directory, _marker, _original = _family_named_hostile_dir(official_dir, tmp_path, "deepseek-v4-flash-0731-copy")
    assert is_deepseek_v4_tokenizer_source(str(directory)) is False
    (directory / "config.json").write_text(json.dumps({"model_type": "deepseek_v4"}))
    assert is_deepseek_v4_tokenizer_source(str(directory)) is True


def test_family_named_hostile_dir_without_context_keeps_legacy_default(official_dir: Path, tmp_path: Path) -> None:
    """Boundary: a family-named hostile dir with no selected context keeps the
    legacy permissive default -- a directory name no longer restricts."""
    directory, marker, _original = _family_named_hostile_dir(official_dir, tmp_path, "deepseek-v4-flash-0731-copy")
    cookbook_tokenizers.load_tokenizer(str(directory))
    assert marker.exists(), "a family-shaped directory name must not silently restrict"


def test_family_named_hostile_dir_under_selected_context_stays_code_free(official_dir: Path, tmp_path: Path) -> None:
    """Explicit reviewed V4 context + mutated artifacts: no execution, stock parity."""
    directory, marker, original = _family_named_hostile_dir(official_dir, tmp_path, "deepseek-v4-flash-0731-copy")
    tokenizer = cookbook_tokenizers.load_tokenizer(
        str(directory), base_model="accounts/fireworks/models/deepseek-v4-flash-0731"
    )
    assert not marker.exists(), "selected V4 context must restrict the mutated family-named dir"
    _assert_battery_equal(tokenizer, original)


def test_renderer_get_tokenizer_family_named_hostile_dir_policies(official_dir: Path, tmp_path: Path) -> None:
    """renderer get_tokenizer: permissive without context; code-free under a
    selected reviewed base model. After the permissive load staged the canary
    module in sys.modules, an absent marker alone cannot prove the restricted
    load avoided remote dispatch -- the returned class must be stock."""
    directory, marker, original = _family_named_hostile_dir(official_dir, tmp_path, "deepseek-v4-flash-0731-copy")
    _tokenizer_utils.get_tokenizer(str(directory))
    assert marker.exists(), "renderer path without context keeps the legacy default"
    marker.unlink()
    tokenizer = _tokenizer_utils.get_tokenizer(
        str(directory), base_model="accounts/fireworks/models/deepseek-v4-flash-0731"
    )
    assert not marker.exists(), "selected V4 context must restrict the renderer load"
    assert type(tokenizer).__module__.startswith(
        "transformers."
    ), f"restricted load must return the stock class, got {type(tokenizer).__module__}"
    _assert_battery_equal(tokenizer, original)


def test_verifier_load_tokenizer_family_named_hostile_dir_policies(official_dir: Path, tmp_path: Path) -> None:
    """verifier load_tokenizer: permissive without context; code-free under an
    explicit deepseek_v4 renderer selection. Same module-cache hazard as the
    renderer case: the restricted class must be stock, not just token-equal."""
    from training.renderer.verifier.utils.tokenizer import load_tokenizer as verifier_load_tokenizer

    directory, marker, original = _family_named_hostile_dir(official_dir, tmp_path, "deepseek-v4-flash-0731-copy")
    verifier_load_tokenizer(str(directory))
    assert marker.exists(), "verifier path without context keeps the legacy default"
    marker.unlink()
    tokenizer = verifier_load_tokenizer(str(directory), renderer_name="deepseek_v4")
    assert not marker.exists(), "explicit deepseek_v4 renderer must restrict the verifier load"
    assert type(tokenizer).__module__.startswith(
        "transformers."
    ), f"restricted load must return the stock class, got {type(tokenizer).__module__}"
    _assert_battery_equal(tokenizer, original)


@pytest.mark.parametrize("loader_name", ["cookbook", "renderer", "verifier"])
def test_nested_tokenizer_directory_keeps_selected_v4_context(
    official_dir: Path, tmp_path: Path, loader_name: str
) -> None:
    directory, marker, original = _family_named_hostile_dir(official_dir, tmp_path, "deepseek-v4-flash-0731-copy")
    nested = directory / "tokenizer"
    nested.mkdir()
    for source in tuple(directory.iterdir()):
        if source.is_file():
            source.rename(nested / source.name)
    assert is_deepseek_v4_tokenizer_source(str(nested)) is False
    if loader_name == "cookbook":
        tokenizer = cookbook_tokenizers.load_tokenizer(
            str(nested), base_model="accounts/fireworks/models/deepseek-v4-flash-0731"
        )
    elif loader_name == "renderer":
        tokenizer = _tokenizer_utils.get_tokenizer(
            str(nested), base_model="accounts/fireworks/models/deepseek-v4-flash-0731"
        )
    else:
        from training.renderer.verifier.utils.tokenizer import load_tokenizer as verifier_load_tokenizer

        tokenizer = verifier_load_tokenizer(str(nested), renderer_name="deepseek_v4")
    assert not marker.exists(), "selected V4 context must restrict a tokenizer subdirectory"
    _assert_battery_equal(tokenizer, original)


def test_non_family_named_source_keeps_legacy_default(official_dir: Path, tmp_path: Path) -> None:
    """Boundary control: an opaque path with mutated identity still executes
    (the renamed/no-context boundary stays open and is not silently exempted)."""
    directory, marker, _original = _family_named_hostile_dir(official_dir, tmp_path, "acme-custom-tokenizer")
    cookbook_tokenizers.load_tokenizer(str(directory))
    assert marker.exists(), "non-family sources keep the legacy default unchanged"


def test_v41_family_name_stays_outside_the_v4_name_signal(official_dir: Path, tmp_path: Path) -> None:
    """V4.1 control: a V4.1-named hostile dir must NOT be recognized by the
    V4 name signal (exclusion preserved), so the legacy default still runs."""
    directory, marker, _original = _family_named_hostile_dir(official_dir, tmp_path, "deepseek-v4.1-flash")
    assert is_deepseek_v4_tokenizer_source(str(directory)) is False
    cookbook_tokenizers.load_tokenizer(str(directory))
    assert marker.exists(), "V4.1 names must stay outside the V4 name signal"


def test_resolve_renderer_name_recognizes_local_v4_data_without_dirname(official_dir: Path, tmp_path: Path) -> None:
    """A real local V4 artifact keeps the deepseek_v4 renderer from its data
    under any directory name -- including names mimicking other families or a
    canonical ID -- while a canonical-named directory without V4 data does
    not get it."""
    from training.utils.supervised import resolve_renderer_name

    for dirname in (
        "totally-unrelated-name",
        "qwen3.5-checkpoint",
        "accounts/fireworks/models/deepseek-v4-flash-0731",
    ):
        directory = tmp_path / dirname
        directory.mkdir(parents=True)
        shutil.copytree(official_dir, directory, dirs_exist_ok=True)
        assert resolve_renderer_name(str(directory)) == "deepseek_v4", dirname
    bare = tmp_path / "deepseek-v4-flash-0731"
    bare.mkdir()
    with pytest.raises(ValueError, match="Could not infer"):
        resolve_renderer_name(str(bare))
    bare_canonical = tmp_path / "staging/accounts/fireworks/models/deepseek-v4-flash-0731"
    bare_canonical.mkdir(parents=True)
    with pytest.raises(ValueError, match="Could not infer"):
        resolve_renderer_name(str(bare_canonical))


# ---------------------------------------------------------------------------
# Image-processor context: known-V4 base/renderer context must pin explicit
# trust_remote_code=False at the AutoImageProcessor boundary, in the loader's
# cache key, and across renderer pickling.
# ---------------------------------------------------------------------------


def _processor_hostile_bundle(official_dir: Path, tmp_path: Path, name: str) -> tuple[Path, Path, Path]:
    """Opaque (marker-free name) bundle: configless digest-mutated V4 tokenizer
    files plus a stock-type image processor config with an auto_map canary."""
    directory = tmp_path / name
    directory.mkdir(parents=True)
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy2(official_dir / filename, directory / filename)
    tokenizer_path = directory / "tokenizer.json"
    tokenizer_path.write_text(json.dumps(json.loads(tokenizer_path.read_text()), indent=1))
    tok_marker = _write_canary(directory, tmp_path / f"proc-tok-executed-{name}")
    proc_marker = tmp_path / f"proc-img-executed-{name}"
    (directory / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "image_processor_type": "CLIPImageProcessor",
                "auto_map": {"AutoImageProcessor": "canary_processor.CanaryProcessor"},
                "size": {"shortest_edge": 224},
                "crop_size": {"height": 224, "width": 224},
            }
        )
    )
    (directory / "canary_processor.py").write_text(
        "from pathlib import Path as _P\n"
        f"_P({json.dumps(str(proc_marker))}).write_text('x')\n"
        "from transformers import CLIPImageProcessor\n"
        "class CanaryProcessor(CLIPImageProcessor):\n    pass\n"
    )
    return directory, tok_marker, proc_marker


def _populate(official_dir: Path, tmp_path: Path, name: str, base_model: str) -> tuple[dict, Path, Path]:
    from training.utils import supervised

    directory, tok_marker, proc_marker = _processor_hostile_bundle(official_dir, tmp_path, name)
    state: dict[str, Any] = {}
    supervised.populate_render_worker_state(
        state,
        tokenizer_model=str(directory),
        renderer_name="qwen3_vl",
        max_seq_len=4096,
        renderer_name_is_resolved=True,
        base_model=base_model,
    )
    return state, tok_marker, proc_marker


def _stock_processor(directory: Path) -> Any:
    from transformers.models.auto.image_processing_auto import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(str(directory), trust_remote_code=False)


def _tiny_image() -> Any:
    from PIL import Image

    return Image.new("RGB", (8, 8), (10, 20, 30))


def _pixels(pixel_values: Any) -> Any:
    """Normalize pixel output (list, list-of-arrays, or ndarray) for equality."""
    try:
        import numpy as np

        return np.asarray(pixel_values, dtype=float).tolist()
    except Exception:
        return pixel_values.tolist() if hasattr(pixel_values, "tolist") else pixel_values


def _processor_parity(loaded: Any, directory: Path) -> None:
    stock = _stock_processor(directory)
    assert type(loaded) is type(stock)
    assert loaded.to_dict() == stock.to_dict()
    image = _tiny_image()
    assert _pixels(loaded(images=image)["pixel_values"]) == _pixels(stock(images=image)["pixel_values"])


@pytest.mark.parametrize("model_type", ["deepseek_v41", "qwen3", "llama", "new_model_family"])
@pytest.mark.parametrize("boundary", ["renderer", "verifier", "supervised"])
def test_reviewed_tokenizer_bytes_block_foreign_processor_code(
    official_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_type: str, boundary: str
) -> None:
    # lazy: these real processor boundaries need the optional vision dependencies.
    from training.renderer import _image_processing_utils
    from training.renderer.verifier.utils import tokenizer as verifier_tokenizers
    from training.utils import supervised

    directory, _tok_marker, proc_marker = _processor_hostile_bundle(official_dir, tmp_path, "shared-bytes")
    shutil.copy2(official_dir / "tokenizer.json", directory / "tokenizer.json")
    (directory / "config.json").write_text(json.dumps({"model_type": model_type}))
    monkeypatch.setenv("HF_TRUST_REMOTE_CODE", "1")
    if boundary == "renderer":
        processor = _image_processing_utils.get_image_processor(str(directory), renderer_name="qwen3_vl")
    elif boundary == "verifier":
        processor = verifier_tokenizers.load_image_processor(str(directory), renderer_name="qwen3_vl")
    else:
        processor = supervised._get_image_processor_with_remote_code_default(str(directory), renderer_name="qwen3_vl")
    assert not proc_marker.exists()
    _processor_parity(processor, directory)


@pytest.mark.parametrize("env", [None, "1"])
def test_render_worker_processor_known_v4_base_opaque_foreign_renderer(
    official_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env
) -> None:
    if env is None:
        monkeypatch.delenv("HF_TRUST_REMOTE_CODE", raising=False)
    else:
        monkeypatch.setenv("HF_TRUST_REMOTE_CODE", env)
    from transformers.models.auto.image_processing_auto import AutoImageProcessor

    seen: list[dict[str, Any]] = []
    real_from_pretrained = AutoImageProcessor.from_pretrained

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs)
        return real_from_pretrained(*args, **kwargs)

    monkeypatch.setattr(AutoImageProcessor, "from_pretrained", staticmethod(spy))
    from training.utils import supervised

    state: dict[str, Any] = {}
    bundle, tok_marker, proc_marker = _processor_hostile_bundle(official_dir, tmp_path, "acme-vision-main")
    supervised.populate_render_worker_state(
        state,
        tokenizer_model=str(bundle),
        renderer_name="qwen3_vl",
        max_seq_len=4096,
        renderer_name_is_resolved=True,
        base_model="accounts/fireworks/models/deepseek-v4-flash-0731",
    )
    assert not tok_marker.exists(), "tokenizer context must hold"
    assert not proc_marker.exists(), "processor staged Python must not execute"
    assert (
        seen and seen[-1].get("trust_remote_code") is False
    ), f"explicit False required at the AutoImageProcessor boundary, got {seen[-1]!r}"
    _processor_parity(state["renderer"].image_processor, bundle)


def test_image_processor_cache_cannot_reuse_permissive_object_under_restrictive_context(
    official_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cached permissive (custom-class) processor must not be returned when
    the restrictive context differs: the context is part of the cache key."""
    from training.renderer._image_processing_utils import get_image_processor

    bundle, _t, proc_marker = _processor_hostile_bundle(official_dir, tmp_path, "acme-vision-cache")
    monkeypatch.setenv("HF_TRUST_REMOTE_CODE", "1")
    permissive = get_image_processor(str(bundle))
    assert proc_marker.exists(), "precondition: permissive load executes the canary"
    monkeypatch.delenv("HF_TRUST_REMOTE_CODE", raising=False)
    restrictive = get_image_processor(str(bundle), base_model="accounts/fireworks/models/deepseek-v4-flash-0731")
    assert restrictive is not permissive, "restrictive context must not hit the permissive cache entry"
    assert type(restrictive) is not type(permissive)
    _processor_parity(restrictive, bundle)


def test_renderer_pickle_roundtrip_preserves_base_context_under_env_optin(
    official_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pickle a renderer built under known-V4 base context; reconstruction
    under ambient env=1 must keep tokenizer and processor code-free."""
    import pickle

    monkeypatch.delenv("HF_TRUST_REMOTE_CODE", raising=False)
    state, tok_marker, proc_marker = _populate(
        official_dir, tmp_path, "acme-vision-pickle", "accounts/fireworks/models/deepseek-v4-flash-0731"
    )
    assert not proc_marker.exists(), "precondition: fixed build is code-free"
    blob = pickle.dumps(state["renderer"])
    tok_marker.unlink(missing_ok=True)
    proc_marker.unlink(missing_ok=True)
    monkeypatch.setenv("HF_TRUST_REMOTE_CODE", "1")
    try:
        rebuilt = pickle.loads(blob)
    finally:
        monkeypatch.delenv("HF_TRUST_REMOTE_CODE", raising=False)
    assert not tok_marker.exists(), "reconstruction must not execute tokenizer staged Python"
    assert not proc_marker.exists(), "reconstruction must not execute processor staged Python"
    assert getattr(rebuilt, "_tokenizer_base_model", None) == "accounts/fireworks/models/deepseek-v4-flash-0731"


def test_render_worker_processor_v41_restricts_without_changing_unknown_policy(
    official_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unknown families stay permissive; migrated V4.1 context restricts."""
    monkeypatch.delenv("HF_TRUST_REMOTE_CODE", raising=False)
    _state, _t1, proc_marker_nonv4 = _populate(
        official_dir, tmp_path, "acme-vision-nonv4", "accounts/acme/models/custom-x"
    )
    assert proc_marker_nonv4.exists(), "non-V4 policy must remain unchanged"
    _state2, _t2, proc_marker_v41 = _populate(
        official_dir, tmp_path, "acme-vision-v41", "accounts/fireworks/models/deepseek-v4.1-flash"
    )
    assert not proc_marker_v41.exists(), "V4.1 context must restrict its processor load"


def test_processor_default_known_v4_renderer_opaque_source_env_optin(
    official_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Known V4 renderer + opaque source + ambient env=1, driven at the real
    helper signature: renderer context must reach the loader's gate."""
    monkeypatch.setenv("HF_TRUST_REMOTE_CODE", "1")
    from training.utils.supervised import _get_image_processor_with_remote_code_default
    from transformers.models.auto.image_processing_auto import AutoImageProcessor

    seen: list[dict[str, Any]] = []
    real_from_pretrained = AutoImageProcessor.from_pretrained

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs)
        return real_from_pretrained(*args, **kwargs)

    monkeypatch.setattr(AutoImageProcessor, "from_pretrained", staticmethod(spy))
    bundle, _tok_marker, proc_marker = _processor_hostile_bundle(official_dir, tmp_path, "acme-vision-v4renderer")
    processor = _get_image_processor_with_remote_code_default(str(bundle), renderer_name="deepseek_v4")
    assert not proc_marker.exists(), "V4 renderer context must keep the processor code-free under env opt-in"
    assert seen and seen[-1].get("trust_remote_code") is False
    _processor_parity(processor, bundle)


def test_processor_default_foreign_renderer_no_context_env_optin_executes(
    official_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Foreign renderer + opaque source + env=1 WITHOUT any V4 context: the
    legacy permissive path is preserved for custom models."""
    monkeypatch.setenv("HF_TRUST_REMOTE_CODE", "1")
    from training.utils.supervised import _get_image_processor_with_remote_code_default

    bundle, _tok_marker, proc_marker = _processor_hostile_bundle(official_dir, tmp_path, "acme-vision-legacy")
    _get_image_processor_with_remote_code_default(str(bundle), renderer_name="qwen3_vl")
    assert proc_marker.exists(), "custom-model processor contract must remain unchanged"


@pytest.mark.parametrize("base_model", [None, V4_ALIAS])
def test_single_turn_rollout_factory_threads_setup_base_model_to_processor(
    official_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, base_model
) -> None:
    monkeypatch.delenv("HF_TRUST_REMOTE_CODE", raising=False)
    from training.recipes.rl_loop import single_turn_rollout_fn
    from training.utils.rl.rollout.setup import RolloutSetup

    bundle, _tok_marker, proc_marker = _processor_hostile_bundle(official_dir, tmp_path, "acme-vision-factory")
    stock_tokenizer = transformers.AutoTokenizer.from_pretrained(str(bundle), trust_remote_code=False)
    setup = RolloutSetup(
        sampler=SimpleNamespace(close=lambda: None),
        tokenizer=stock_tokenizer,
        tokenizer_id=str(bundle),
        sample_kwargs={},
        completions_per_prompt=1,
        inference_base_url="http://localhost",
        api_key="test-key",
        model="accounts/fireworks/models/deepseek-v4-flash-0731",
        base_model=base_model,
    )
    factory = single_turn_rollout_fn("qwen3_vl")
    factory(setup)  # renderer construction happens at bind time
    assert proc_marker.exists() is (base_model is None)
