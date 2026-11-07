"""Example entrypoints pass selected context through the real tokenizer load."""

from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock

import pytest

from training.tests._v4_tokenizer_data import (
    NON_V4_MODEL,
    RENAMED_ALIAS,
    V4_ALIAS,
    V41_ALIAS,
    assert_stock_tokenizer,
    tampered_v4_dir,
    v4_files as v4_files,
)


class _StopAfterLoad(Exception):
    pass


@pytest.mark.parametrize("entrypoint", ["harbor", "nemo", "distillation"])
@pytest.mark.parametrize(
    "base_model,renderer_name,executes",
    [
        (V4_ALIAS, "kimi_k3", False),
        (RENAMED_ALIAS, "deepseek_v4", False),
        (NON_V4_MODEL, "kimi_k3", True),
        (V41_ALIAS, "kimi_k3", False),
        (RENAMED_ALIAS, "deepseek_v41", False),
    ],
)
def test_example_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    v4_files: dict[str, Path],
    entrypoint: str,
    base_model: str,
    renderer_name: str,
    executes: bool,
) -> None:
    # lazy: each example imports its own optional training dependencies.
    if entrypoint == "harbor":
        from training.examples.harbor_rl import common as module
    elif entrypoint == "nemo":
        from training.examples.rl.nemo_gym import train as module
    else:
        from training.examples.distillation.serverless_k3_privileged import train_serverless_k3_privileged as module

    directory, marker = tampered_v4_dir(v4_files, tmp_path)
    real_load = module.load_tokenizer
    seen = {}

    def load_and_stop(*args: object, **kwargs: object) -> None:
        seen.update(context=kwargs.get("base_model"), tokenizer=real_load(*args, **kwargs))
        raise _StopAfterLoad

    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(module, "load_tokenizer", load_and_stop)
    with pytest.raises(_StopAfterLoad):
        if entrypoint == "harbor":
            with module.deployment_rollout_setup(
                base_model=base_model,
                deployment_id="dep-fake",
                tokenizer_model=str(directory),
                tokenizer_revision=None,
                max_completion_tokens=8,
                max_seq_len=64,
                temperature=1.0,
                sample_timeout=30,
                completions_per_prompt=1,
            ):
                pytest.fail("must stop after loading")
        elif entrypoint == "nemo":
            monkeypatch.setattr(module, "_check_gym_head_available", lambda: None)
            monkeypatch.setattr(module, "_load_rows", lambda path: [{"prompt": "p"}])
            monkeypatch.setattr(module, "get_recommended_renderer_name", lambda model: "simple")
            monkeypatch.setattr(
                module,
                "parse_args",
                lambda: SimpleNamespace(
                    base_model=base_model,
                    tokenizer_model=str(directory),
                    tokenizer_revision=None,
                    renderer_name="",
                    dataset_path="unused.jsonl",
                    resources_server="",
                    max_rows=1,
                    exclude_user_simulator=False,
                    log_path=str(tmp_path),
                ),
            )
            module.run()
        else:
            monkeypatch.setattr(module, "load_dotenv", Mock())
            module.main(
                module.Config(
                    log_path=str(tmp_path),
                    base_model=base_model,
                    renderer_name=renderer_name,
                    tokenizer_model=str(directory),
                )
            )
    assert seen["context"] == base_model
    # Only distillation also has a selected renderer at this load boundary.
    restricted = base_model in {V4_ALIAS, V41_ALIAS} or (entrypoint == "distillation" and not executes)
    assert marker.exists() is (not restricted)
    if restricted:
        assert_stock_tokenizer(seen["tokenizer"], directory)
