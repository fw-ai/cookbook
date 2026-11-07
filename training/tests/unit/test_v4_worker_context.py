"""Parent and DataLoader callbacks preserve context on real tokenizer loads."""

import json
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

from training.recipes import decision_loop, dpo_loop, orpo_loop, sft_loop
from training.tests._v4_tokenizer_data import (
    NON_V4_MODEL,
    RENAMED_ALIAS,
    V4_ALIAS,
    V41_ALIAS,
    assert_stock_tokenizer,
    tampered_v4_dir,
    v4_files as v4_files,
)
from training.utils.supervised import build_renderer_from_resolved_name, render_messages_to_datums

_FOREIGN_RENDERER = "qwen3"
_MESSAGES = [{"role": "user", "content": "Name one prime above 10."}, {"role": "assistant", "content": "11 is prime."}]
_V41_CANONICAL = "accounts/fireworks/models/deepseek-v41-flash"


class _StopAfterSetup(Exception):
    pass


@pytest.fixture(
    params=[
        (V4_ALIAS, "", V4_ALIAS),
        (RENAMED_ALIAS, "deepseek_v4", "deepseek_v4"),
        (_V41_CANONICAL, "", _V41_CANONICAL),
        (RENAMED_ALIAS, "deepseek_v41", "deepseek_v41"),
        (RENAMED_ALIAS, "deepseek_v41_text", "deepseek_v41_text"),
    ]
)
def v4_model_context(request: pytest.FixtureRequest) -> tuple[str, str, str]:
    return request.param


def _stub_common(monkeypatch: pytest.MonkeyPatch, module: ModuleType) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    service = Mock(
        accelerator_type=None,
        accelerator_count=None,
        training_profile=None,
        trainer_job_id="job-fake",
        max_context_length=4096,
    )
    for name in ("setup_wandb", "validate_config", "validate_warm_start_config"):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, lambda *a, **k: None)
    monkeypatch.setattr(module, "build_service_client", lambda **k: service)
    monkeypatch.setattr(module.ReconnectableClient, "from_training_client", lambda *a, **k: Mock())
    monkeypatch.setattr(module, "TrainingCheckpoints", lambda *a, **k: Mock(resume=lambda **k: None))
    if hasattr(module, "RunnerIO"):
        monkeypatch.setattr(module, "RunnerIO", lambda cfg: MagicMock())


@pytest.mark.parametrize("module", [sft_loop, dpo_loop, decision_loop], ids=["sft", "dpo", "decision"])
def test_parent_and_spawn_callback_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    v4_files: dict[str, Path],
    v4_model_context: tuple[str, str, str],
    module: ModuleType,
) -> None:
    staged, marker = tampered_v4_dir(v4_files, tmp_path)
    _stub_common(monkeypatch, module)
    dataset = MagicMock()
    dataset.__len__.return_value = 2
    dataset.approx_row_sizes.return_value = [8, 8]
    seen = {}

    def dataloader(*args: object, **kwargs: object) -> None:
        seen.update(kwargs)
        raise _StopAfterSetup

    monkeypatch.setattr(module, "make_render_dataloader", dataloader)
    if module is sft_loop:
        monkeypatch.setattr(module, "_prepare_datasets", lambda cfg: (dataset, dataset, dataset))
        monkeypatch.setattr(module, "prepare_generation_eval", lambda *a, **k: [])
    elif module is dpo_loop:
        monkeypatch.setattr(module, "_make_pair_dataset", lambda cfg: dataset)
        monkeypatch.setattr(module, "_prepare_eval_pairs", lambda cfg, pairs: (pairs, []))
        monkeypatch.setattr(module, "resolve_renderer_snapshot", lambda **k: _FOREIGN_RENDERER)
    else:
        monkeypatch.setattr(module, "write_running_step", lambda *a, **k: None)
        monkeypatch.setattr(module, "prepare_datasets", lambda cfg: (dataset, [], dataset))

    base_model, base_model_type, expected = v4_model_context
    config = dict(
        log_path=str(tmp_path),
        base_model=base_model,
        base_model_type=base_model_type,
        dataset=str(tmp_path / "rows.jsonl"),
        tokenizer_model=str(staged),
        max_seq_len=4096,
    )
    if module is not dpo_loop:
        config["renderer_name"] = _FOREIGN_RENDERER
    (tmp_path / "rows.jsonl").write_text(
        "".join(
            json.dumps({"state": f"s{i}", "questions": {"q": {"type": "noul", "instructions": "?", "label": "a"}}})
            + "\n"
            for i in range(12)
        )
    )
    state = dpo_loop._pair_worker_state if module is dpo_loop else sft_loop._worker_state
    init = dpo_loop._init_pair_worker if module is dpo_loop else sft_loop._init_render_worker
    state.clear()
    try:
        with pytest.raises(_StopAfterSetup):
            module.main(module.Config(**config))
        assert not marker.exists(), "parent init must not execute staged Python"
        callback = seen["worker_init_fn"]
        assert callback.func is init
        assert callback.keywords["base_model"] == expected
        state.clear()
        callback(0)
        assert not marker.exists(), "spawn callback must not execute staged Python"
        assert state["renderer"] is not None
    finally:
        state.clear()


@pytest.mark.parametrize(
    "base_model,base_model_type,expected,executes",
    [
        (V4_ALIAS, "", V4_ALIAS, False),
        (RENAMED_ALIAS, "deepseek_v4", "deepseek_v4", False),
        (NON_V4_MODEL, "", NON_V4_MODEL, True),
        (V41_ALIAS, "", V41_ALIAS, False),
    ],
)
def test_orpo_context_and_datum_parity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    v4_files: dict[str, Path],
    base_model: str,
    base_model_type: str,
    expected: str,
    executes: bool,
) -> None:
    staged, marker = tampered_v4_dir(v4_files, tmp_path)
    _stub_common(monkeypatch, orpo_loop)
    seen = {}

    def capture(tokenizer: Any, tokenizer_model: str, renderer_name: str, *, base_model: str | None = None) -> None:
        seen.update(tokenizer=tokenizer, renderer=renderer_name, context=base_model)
        raise _StopAfterSetup

    monkeypatch.setattr(orpo_loop, "build_renderer_from_resolved_name", capture)
    dataset = tmp_path / "prefs.jsonl"
    dataset.write_text("{}\n")
    cfg = orpo_loop.Config(
        log_path=str(tmp_path),
        base_model=base_model,
        base_model_type=base_model_type,
        dataset=str(dataset),
        tokenizer_model=str(staged),
        renderer_name=_FOREIGN_RENDERER,
    )
    with pytest.raises(_StopAfterSetup):
        orpo_loop.main(cfg)
    assert marker.exists() is executes
    assert seen["context"] == expected
    assert seen["renderer"] == _FOREIGN_RENDERER
    if not executes:
        # lazy: Transformers is required only by the real-tokenizer fixture.
        import transformers

        assert_stock_tokenizer(seen["tokenizer"], staged)
        stock = transformers.AutoTokenizer.from_pretrained(str(staged), trust_remote_code=False)

        def datums(tokenizer: Any) -> list[Any]:
            renderer = build_renderer_from_resolved_name(tokenizer, str(staged), _FOREIGN_RENDERER)
            return render_messages_to_datums(
                _MESSAGES, renderer=renderer, train_on_what="all_assistant_messages", include_loss_mask=True
            )

        assert [(list(d.token_ids), list(d.token_weights)) for d in datums(seen["tokenizer"])] == [
            (list(d.token_ids), list(d.token_weights)) for d in datums(stock)
        ]
