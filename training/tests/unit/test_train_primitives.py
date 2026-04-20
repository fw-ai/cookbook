"""Unit tests for ref_fwd_bwd / finish_step / dump_trajectory_jsonl.

The primitives are thin wrappers around well-known SDK call sequences.
We mock the SDK boundary (policy/reference/weight_syncer) and verify the
primitives orchestrate them in the right order with the right arguments.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from training.utils.rl.losses import PromptGroup
from training.utils.rl.train import (
    TrainContext,
    dump_trajectory_jsonl,
    finish_step,
    ref_fwd_bwd,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _logprob_output(value: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(data=[value])


def _forward_result(n: int) -> SimpleNamespace:
    return SimpleNamespace(
        loss_fn_outputs=[{"logprobs": _logprob_output(-0.1 * i)} for i in range(n)],
    )


def _make_pg(n_data: int = 2, n_ref: int = 0) -> PromptGroup:
    target = SimpleNamespace(data=[1, 2, 3], dtype="int64", shape=[3])
    datum = SimpleNamespace(
        model_input=SimpleNamespace(),
        loss_fn_inputs={"target_tokens": target},
    )
    return PromptGroup(
        data=[datum] * n_data,
        ref_data=[datum] * n_ref,
        advantages=[0.5, -0.5][:n_data],
        ref_logprobs=None,
        prompt_len=2,
        rewards=[1.0, 0.0][:n_data],
        inf_logprobs=[[0.0] * 3 for _ in range(n_data)],
        completion_lens=[1] * n_data,
        truncated=[False] * n_data,
    )


def _make_ctx(
    *,
    builtin: bool = False,
    reference: object | None = None,
    weight_sync_interval: int = 1,
    dcp_save_interval: int = 0,
    completions_per_prompt: int = 2,
    trajectory_dir: str | None = None,
    wandb_log=None,
    log_metrics_json=None,
) -> tuple[TrainContext, MagicMock, MagicMock]:
    policy = MagicMock()
    policy.forward.return_value = _forward_result(2)
    policy.forward_backward.return_value = SimpleNamespace(
        loss_fn_outputs=[{"logprobs": _logprob_output()}] * 2,
        metrics={},
    )
    policy.forward_backward_custom.return_value = SimpleNamespace(
        loss_fn_outputs=[{"logprobs": _logprob_output()}] * 2,
        metrics={},
    )
    policy.optim_step.return_value = SimpleNamespace(
        loss_fn_outputs=[], metrics={"optim/lr": 1e-5},
    )
    weight_syncer = MagicMock()

    ctx = TrainContext(
        policy=policy,
        reference=reference,
        weight_syncer=weight_syncer,
        adam_params=SimpleNamespace(),
        grad_accumulation_normalization=None,
        builtin_server_loss=("kernel", {"k": "v"}) if builtin else None,
        client_loss_builder=lambda *a, **kw: ("client_loss", a, kw),
        tis_config=SimpleNamespace(enabled=False),
        policy_loss="grpo",
        log_path="/tmp/x",
        policy_job_id="job-x",
        completions_per_prompt=completions_per_prompt,
        trajectory_dir=trajectory_dir,
        weight_sync_interval=weight_sync_interval,
        dcp_save_interval=dcp_save_interval,
        wandb_log=wandb_log,
        log_metrics_json=log_metrics_json,
    )
    return ctx, policy, weight_syncer


# ---------------------------------------------------------------------------
# ref_fwd_bwd
# ---------------------------------------------------------------------------


def test_ref_fwd_bwd_uses_client_loss_path_when_no_builtin(monkeypatch):
    monkeypatch.setattr(
        "training.utils.rl.train.combine_prompt_groups",
        lambda groups: ([d for pg in groups for d in pg.data], [], None,
                        [pg.prompt_len for pg in groups], []),
    )
    ctx, policy, _ = _make_ctx(reference=None, builtin=False)
    ref_fwd_bwd(ctx, _make_pg(n_data=2))
    policy.forward.assert_called_once()
    policy.forward_backward_custom.assert_called_once()
    policy.forward_backward.assert_not_called()


def test_ref_fwd_bwd_uses_builtin_path_when_resolved(monkeypatch):
    monkeypatch.setattr(
        "training.utils.rl.train.combine_prompt_groups",
        lambda groups: ([d for pg in groups for d in pg.data], [], None,
                        [pg.prompt_len for pg in groups], []),
    )
    monkeypatch.setattr(
        "training.utils.rl.train.build_builtin_loss_datums",
        lambda *a, **kw: ["packed-datum"] * 2,
    )
    ctx, policy, _ = _make_ctx(builtin=True)
    ref_fwd_bwd(ctx, _make_pg(n_data=2))
    policy.forward_backward.assert_called_once()
    policy.forward_backward_custom.assert_not_called()


def test_ref_fwd_bwd_invokes_reference_forward_when_present(monkeypatch):
    monkeypatch.setattr(
        "training.utils.rl.train.combine_prompt_groups",
        lambda groups: ([d for pg in groups for d in pg.data], [], None,
                        [pg.prompt_len for pg in groups], []),
    )
    reference = MagicMock()
    reference.forward.return_value = _forward_result(2)
    ctx, _, _ = _make_ctx(reference=reference)
    ref_fwd_bwd(ctx, _make_pg(n_data=2, n_ref=2))
    reference.forward.assert_called_once()


def test_ref_fwd_bwd_skips_reference_when_none(monkeypatch):
    monkeypatch.setattr(
        "training.utils.rl.train.combine_prompt_groups",
        lambda groups: ([d for pg in groups for d in pg.data], [], None,
                        [pg.prompt_len for pg in groups], []),
    )
    ctx, _, _ = _make_ctx(reference=None)
    # no error and no reference call (None has no .forward)
    ref_fwd_bwd(ctx, _make_pg(n_data=2))


# ---------------------------------------------------------------------------
# finish_step
# ---------------------------------------------------------------------------


def _stub_metrics(monkeypatch, **values) -> None:
    base = {"rollout/reward": 0.0, "rollout/accuracy": 0.0, "train/mean_kl": 0.0}
    base.update(values)
    monkeypatch.setattr(
        "training.utils.rl.train.compute_step_metrics",
        lambda **kw: dict(base),
    )


def test_finish_step_increments_step_and_calls_optim(monkeypatch):
    _stub_metrics(monkeypatch)
    ctx, policy, weight_syncer = _make_ctx(weight_sync_interval=1)
    new_step, metrics = finish_step(
        ctx, step=4, prompt_groups=[_make_pg()],
        fwd_bwd_results=[SimpleNamespace(metrics={})],
    )
    assert new_step == 5
    assert metrics["train/step"] == 5
    policy.optim_step.assert_called_once()
    # interval=1 → hotload after every step
    weight_syncer.save_and_hotload.assert_called_once_with("step-5")


def test_finish_step_skips_hotload_when_interval_zero(monkeypatch):
    _stub_metrics(monkeypatch)
    ctx, _, weight_syncer = _make_ctx(weight_sync_interval=0)
    finish_step(
        ctx, step=0, prompt_groups=[_make_pg()],
        fwd_bwd_results=[SimpleNamespace(metrics={})],
    )
    weight_syncer.save_and_hotload.assert_not_called()


def test_finish_step_only_hotloads_on_interval(monkeypatch):
    _stub_metrics(monkeypatch)
    ctx, _, weight_syncer = _make_ctx(weight_sync_interval=2)
    # step starts at 0, becomes 1 → 1 % 2 != 0 → no hotload
    finish_step(ctx, step=0, prompt_groups=[_make_pg()],
                fwd_bwd_results=[SimpleNamespace(metrics={})])
    weight_syncer.save_and_hotload.assert_not_called()
    # step starts at 1, becomes 2 → 2 % 2 == 0 → hotload
    finish_step(ctx, step=1, prompt_groups=[_make_pg()],
                fwd_bwd_results=[SimpleNamespace(metrics={})])
    weight_syncer.save_and_hotload.assert_called_once_with("step-2")


def test_finish_step_calls_save_checkpoint_fn_when_dcp_due(monkeypatch):
    _stub_metrics(monkeypatch)
    ctx, _, _ = _make_ctx(weight_sync_interval=0, dcp_save_interval=2)
    save_calls: list[tuple[str, dict]] = []
    finish_step(
        ctx, step=1,
        prompt_groups=[_make_pg()],
        fwd_bwd_results=[SimpleNamespace(metrics={})],
        save_checkpoint_fn=lambda name, extra: save_calls.append((name, extra)),
        step_target=4,
        resume_data_consumed=10,
    )
    # step 1 → 2; 2 % 2 == 0 → save fires
    assert len(save_calls) == 1
    name, extra = save_calls[0]
    assert name == "step-2"
    assert extra["step"] == 2
    assert extra["data_consumed"] == 10 + (2 - 0) * 4
    assert extra["source_job_id"] == "job-x"


def test_finish_step_invokes_optional_logging_callbacks(monkeypatch):
    _stub_metrics(monkeypatch, **{
        "rollout/reward": 0.7, "rollout/accuracy": 0.7, "train/mean_kl": 0.05,
    })
    wandb_calls: list[tuple] = []
    json_calls: list[tuple] = []
    ctx, _, _ = _make_ctx(
        weight_sync_interval=0,
        wandb_log=lambda metrics, step: wandb_calls.append((step, dict(metrics))),
        log_metrics_json=lambda step, **kw: json_calls.append((step, kw)),
    )
    finish_step(ctx, step=0, prompt_groups=[_make_pg()],
                fwd_bwd_results=[SimpleNamespace(metrics={})])
    assert wandb_calls and wandb_calls[0][0] == 1
    assert json_calls == [(1, {"reward": 0.7, "accuracy": 0.7, "kl": 0.05})]


def test_finish_step_dumps_trajectory_when_dir_set(monkeypatch, tmp_path: Path):
    _stub_metrics(monkeypatch)
    ctx, _, _ = _make_ctx(weight_sync_interval=0, trajectory_dir=str(tmp_path))
    pg = _make_pg()
    pg.completions = ["A", "B"]
    finish_step(ctx, step=0, prompt_groups=[pg],
                fwd_bwd_results=[SimpleNamespace(metrics={})])
    out = tmp_path / "step_0001.jsonl"
    assert out.exists()
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(records) == 2


# ---------------------------------------------------------------------------
# dump_trajectory_jsonl (pure)
# ---------------------------------------------------------------------------


def test_dump_trajectory_jsonl_writes_one_record_per_completion(tmp_path: Path):
    pg = PromptGroup(
        data=[],
        ref_data=[],
        advantages=[0.5, -0.5],
        ref_logprobs=None,
        prompt_len=4,
        rewards=[1.0, 0.0],
        completion_lens=[3, 4],
        truncated=[False, True],
        prompt=[{"role": "user", "content": "Q"}],
        completions=["A", "B"],
        row_meta={"ground_truth": "<answer>1</answer>"},
    )
    dump_trajectory_jsonl(str(tmp_path), step=7, prompt_groups=[pg])
    out = tmp_path / "step_0007.jsonl"
    assert out.exists()
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["completion"] == "A"
    assert records[0]["reward"] == 1.0
    assert records[0]["truncated"] is False
    assert records[1]["truncated"] is True
    assert records[0]["ground_truth"] == "<answer>1</answer>"


def test_dump_trajectory_jsonl_handles_missing_metadata(tmp_path: Path):
    pg = PromptGroup(
        data=[],
        ref_data=[],
        advantages=[],
        ref_logprobs=None,
        prompt_len=0,
        rewards=[1.0],
        completion_lens=[],
        truncated=[],
        completions=["A"],
    )
    dump_trajectory_jsonl(str(tmp_path), step=0, prompt_groups=[pg])
    record = json.loads((tmp_path / "step_0000.jsonl").read_text().strip())
    assert record["advantage"] is None
    assert record["completion_len"] is None
    assert record["truncated"] is None
    assert record["ground_truth"] is None
