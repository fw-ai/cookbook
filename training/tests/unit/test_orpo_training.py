"""Exercise ORPO orchestration with real data, loss, and runner files on CPU."""

from __future__ import annotations

import json
import signal
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
from fireworks.training import sdk as tinker
import torch

from training.recipes import orpo_loop
from training.utils.runner import DatasetError, RunnerConfig


@pytest.fixture
def orpo_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[SimpleNamespace]:
    service = MagicMock(
        accelerator_type="CPU",
        accelerator_count=1,
        training_profile=None,
        trainer_job_id="test-trainer",
        max_context_length=32,
    )
    client = MagicMock()
    checkpoints = MagicMock()
    checkpoints.resume.return_value = None
    gradients: list[list[torch.Tensor]] = []

    def forward_backward(
        datums: list[tinker.Datum],
        loss_fn: Callable,
    ) -> SimpleNamespace:
        logprobs = [
            torch.full(
                (len(d.loss_fn_inputs["target_tokens"].data),),
                -0.5 if i % 2 == 0 else -1.5,
                requires_grad=True,
            )
            for i, d in enumerate(datums)
        ]
        loss, metrics = loss_fn(datums, logprobs)
        assert torch.isfinite(loss)
        loss.backward()
        gradients.append([lp.grad.clone() for lp in logprobs])
        return SimpleNamespace(metrics=metrics)

    client.forward_backward_custom.side_effect = forward_backward
    tokenizer = SimpleNamespace(encode=lambda text: list(text.encode("ascii")))
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(orpo_loop, "build_service_client", lambda **kwargs: service)
    monkeypatch.setattr(
        orpo_loop.ReconnectableClient, "from_training_client", lambda *a, **k: client
    )
    monkeypatch.setattr(orpo_loop, "TrainingCheckpoints", lambda *a, **k: checkpoints)
    monkeypatch.setattr(orpo_loop, "load_tokenizer", lambda *a: tokenizer)
    monkeypatch.setattr(
        orpo_loop, "resolve_renderer_snapshot", lambda **k: "test-renderer"
    )
    monkeypatch.setattr(
        orpo_loop, "build_renderer_from_resolved_name", lambda *a: object()
    )
    monkeypatch.setattr(orpo_loop, "setup_wandb", lambda *a: None)
    monkeypatch.setattr(orpo_loop, "wandb_log", lambda *a: None)
    monkeypatch.setattr(orpo_loop, "wandb_finish", lambda: None)
    cfg = orpo_loop.Config(
        log_path=str(tmp_path),
        dataset=str(tmp_path / "pairs.jsonl"),
        tokenizer_model="test-tokenizer",
        batch_size=2,
        runner=RunnerConfig(
            status_file=str(tmp_path / "status.json"),
            metrics_file=str(tmp_path / "metrics.jsonl"),
            metadata_file=str(tmp_path / "metadata.json"),
            output_model_path=str(tmp_path / "output.json"),
        ),
    )
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    yield SimpleNamespace(
        cfg=cfg,
        service=service,
        client=client,
        checkpoints=checkpoints,
        gradients=gradients,
        path=tmp_path,
    )
    for sig, handler in handlers.items():
        signal.signal(sig, handler)


def _write_pairs(path: str, pairs: list[tuple[str | None, str]]) -> None:
    rows = [
        {
            "chosen": {"text": chosen} if chosen is not None else {"messages": []},
            "rejected": {"text": rejected},
        }
        for chosen, rejected in pairs
    ]
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.mark.parametrize("legacy_schedule", [False, True])
def test_orpo_full_and_partial_batches_checkpoint_and_report(
    orpo_run: SimpleNamespace,
    legacy_schedule: bool,
) -> None:
    run = orpo_run
    _write_pairs(run.cfg.dataset, [(f"q:good{i}", f"q:bad{i}") for i in range(3)])
    run.cfg.epochs = 2
    run.cfg.dcp_save_interval = 2
    run.cfg.sampler_save_interval = 3
    run.cfg.output_model_id = "test-orpo"
    if legacy_schedule:
        run.cfg.lr_schedule = "linear"
        run.cfg.warmup_ratio = 0.25
    result = orpo_loop.main(run.cfg)

    assert result == {"steps": 4, "job_id": "test-trainer"}
    batches = [c.args[0] for c in run.client.forward_backward_custom.call_args_list]
    assert [len(batch) for batch in batches] == [4, 2, 4, 2]
    assert run.client.optim_step.call_count == 4
    for batch_grads in run.gradients:
        for i, grad in enumerate(batch_grads):
            assert torch.isfinite(grad).all()
            assert grad[0] == 0  # Prompt logprobs must not contribute to ORPO.
            if i % 2 == 0:
                assert (grad[1:] < 0).all()
            else:
                assert (grad[1:] > 0).all()
    assert run.checkpoints.save.call_args_list == [
        call("step-2", resumable=True, promotable=False, data_consumed=4),
        call("step-3", resumable=False, promotable=True, data_consumed=6),
        call("step-4", resumable=True, promotable=False, data_consumed=8),
        call("step-4", resumable=True, promotable=True, data_consumed=8),
    ]
    run.checkpoints.promote_latest.assert_called_once_with(
        "test-orpo", run.cfg.base_model
    )
    assert json.loads((run.path / "status.json").read_text())["code"] == 0
    assert json.loads((run.path / "output.json").read_text())["checkpoint"] == "step-4"
    metrics = [
        json.loads(line)
        for line in (run.path / "metrics.jsonl").read_text().splitlines()
    ]
    assert [m["step"] for m in metrics] == [1, 2, 3, 4]
    assert [m["train/epoch"] for m in metrics] == [1, 1, 2, 2]
    assert all(m["train/accuracy"] == 1.0 for m in metrics)
    assert (
        json.loads((run.path / "metadata.json").read_text())["metadata"]["tokens"] == 78
    )
    run.service.close.assert_called_once()


def test_orpo_resume_filters_invalid_and_long_pairs(orpo_run: SimpleNamespace) -> None:
    run = orpo_run
    _write_pairs(
        run.cfg.dataset, [(None, "q:bad"), ("x" * 40, "q:bad"), ("q:good", "q:bad")]
    )
    run.checkpoints.resume.return_value = SimpleNamespace(step=5)
    run.cfg.init_from_checkpoint = "tinker://test-checkpoint/weights"
    run.cfg.save_final_checkpoint = False

    assert orpo_loop.main(run.cfg) == {"steps": 6, "job_id": "test-trainer"}
    run.checkpoints.resume.assert_called_once_with(
        init_from_checkpoint=run.cfg.init_from_checkpoint,
        warm_start_from_adapter=None,
    )
    run.client.optim_step.assert_called_once()
    run.checkpoints.save.assert_not_called()
    run.service.close.assert_called_once()


@pytest.mark.parametrize("pairs", [[], [(None, "q:bad"), ("x" * 40, "q:bad")]])
def test_orpo_unusable_dataset_closes_trainer(
    orpo_run: SimpleNamespace,
    pairs: list[tuple[str | None, str]],
) -> None:
    run = orpo_run
    _write_pairs(run.cfg.dataset, pairs)
    with pytest.raises(DatasetError):
        orpo_loop.main(run.cfg)
    run.service.close.assert_called_once()
    run.client.optim_step.assert_not_called()
    run.checkpoints.save.assert_not_called()
    assert json.loads((run.path / "status.json").read_text())["code"] != 0


def test_orpo_failed_optimizer_closes_trainer_without_checkpoint(
    orpo_run: SimpleNamespace,
) -> None:
    run = orpo_run
    _write_pairs(run.cfg.dataset, [("q:good", "q:bad")])
    run.client.optim_step.side_effect = RuntimeError("optimizer failed")
    with pytest.raises(RuntimeError, match="optimizer failed"):
        orpo_loop.main(run.cfg)
    run.service.close.assert_called_once()
    run.checkpoints.save.assert_not_called()
    assert json.loads((run.path / "status.json").read_text())["code"] != 0
