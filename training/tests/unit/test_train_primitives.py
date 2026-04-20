"""Unit tests for streaming-pipeline primitives in training.utils.rl.train.

Covers TrainContext + ref_fwd_bwd + finish_step + dump_trajectory_jsonl
without standing up real trainer clients.
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


def _fake_loss_fn_output(value: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(data=[value])


def _fake_forward_result(n_outputs: int) -> SimpleNamespace:
    return SimpleNamespace(
        loss_fn_outputs=[
            {"logprobs": _fake_loss_fn_output(-0.1 * i)} for i in range(n_outputs)
        ],
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
) -> tuple[TrainContext, MagicMock, MagicMock, MagicMock]:
    policy = MagicMock()
    policy.forward.return_value = _fake_forward_result(2)
    policy.forward_backward.return_value = SimpleNamespace(
        loss_fn_outputs=[{"logprobs": _fake_loss_fn_output()}] * 2,
        metrics={},
    )
    policy.forward_backward_custom.return_value = SimpleNamespace(
        loss_fn_outputs=[{"logprobs": _fake_loss_fn_output()}] * 2,
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
    )
    return ctx, policy, weight_syncer, policy  # last alias unused; keep symmetry


# ---------------------------------------------------------------------------
# ref_fwd_bwd
# ---------------------------------------------------------------------------


def test_ref_fwd_bwd_skips_reference_when_none(monkeypatch):
    monkeypatch.setattr(
        "training.utils.rl.train.combine_prompt_groups",
        lambda groups: ([d for pg in groups for d in pg.data], [], None, [pg.prompt_len for pg in groups], []),
    )
    monkeypatch.setattr(
        "training.utils.rl.train.build_builtin_loss_datums",
        lambda *a, **kw: ["packed-datum"] * 2,
    )

    ctx, policy, _, _ = _make_ctx(reference=None)
    group = _make_pg(n_data=2)

    ref_fwd_bwd(ctx, group)
    policy.forward.assert_called_once()  # policy_forward step
    policy.forward_backward_custom.assert_called_once()  # client-side path


def test_ref_fwd_bwd_invokes_reference_when_present(monkeypatch):
    monkeypatch.setattr(
        "training.utils.rl.train.combine_prompt_groups",
        lambda groups: ([d for pg in groups for d in pg.data], [], None, [pg.prompt_len for pg in groups], []),
    )
    reference = MagicMock()
    reference.forward.return_value = _fake_forward_result(2)
    ctx, _, _, _ = _make_ctx(reference=reference)
    group = _make_pg(n_data=2, n_ref=2)

    ref_fwd_bwd(ctx, group)
    reference.forward.assert_called_once()


def test_ref_fwd_bwd_uses_builtin_path_when_resolved(monkeypatch):
    monkeypatch.setattr(
        "training.utils.rl.train.combine_prompt_groups",
        lambda groups: ([d for pg in groups for d in pg.data], [], None, [pg.prompt_len for pg in groups], []),
    )
    monkeypatch.setattr(
        "training.utils.rl.train.build_builtin_loss_datums",
        lambda *a, **kw: ["packed-datum"] * 2,
    )

    ctx, policy, _, _ = _make_ctx(builtin=True)
    group = _make_pg(n_data=2)

    ref_fwd_bwd(ctx, group)
    policy.forward_backward.assert_called_once()
    policy.forward_backward_custom.assert_not_called()


# ---------------------------------------------------------------------------
# finish_step
# ---------------------------------------------------------------------------


def test_finish_step_advances_step_and_calls_optim(monkeypatch):
    monkeypatch.setattr(
        "training.utils.rl.train.compute_step_metrics",
        lambda **kw: {"rollout/reward": 0.5, "rollout/accuracy": 0.5, "train/mean_kl": 0.01},
    )
    ctx, policy, weight_syncer, _ = _make_ctx(weight_sync_interval=1)
    groups = [_make_pg() for _ in range(3)]
    fwd_results = [SimpleNamespace(metrics={}) for _ in range(3)]

    new_step, metrics = finish_step(ctx, step=4, prompt_groups=groups, fwd_bwd_results=fwd_results)

    assert new_step == 5
    assert metrics["train/step"] == 5
    policy.optim_step.assert_called_once()
    weight_syncer.save_and_hotload.assert_called_once_with("step-5")


def test_finish_step_skips_hotload_when_interval_zero(monkeypatch):
    monkeypatch.setattr(
        "training.utils.rl.train.compute_step_metrics",
        lambda **kw: {"rollout/reward": 0.0, "rollout/accuracy": 0.0, "train/mean_kl": 0.0},
    )
    ctx, _, weight_syncer, _ = _make_ctx(weight_sync_interval=0)
    finish_step(ctx, step=0, prompt_groups=[_make_pg()], fwd_bwd_results=[SimpleNamespace(metrics={})])
    weight_syncer.save_and_hotload.assert_not_called()


def test_finish_step_calls_save_checkpoint_fn_when_dcp_due(monkeypatch):
    monkeypatch.setattr(
        "training.utils.rl.train.compute_step_metrics",
        lambda **kw: {"rollout/reward": 0.0, "rollout/accuracy": 0.0, "train/mean_kl": 0.0},
    )
    ctx, _, _, _ = _make_ctx(weight_sync_interval=0, dcp_save_interval=2)
    save_calls: list[tuple[str, dict]] = []

    finish_step(
        ctx, step=1,
        prompt_groups=[_make_pg()],
        fwd_bwd_results=[SimpleNamespace(metrics={})],
        save_checkpoint_fn=lambda name, extra: save_calls.append((name, extra)),
        step_target=4,
        resume_data_consumed=10,
    )
    # step incremented from 1 -> 2; 2 % 2 == 0 -> save fires
    assert len(save_calls) == 1
    assert save_calls[0][0] == "step-2"
    assert save_calls[0][1]["step"] == 2
    assert save_calls[0][1]["data_consumed"] == 10 + (2 - 0) * 4


def test_finish_step_invokes_optional_logging_callbacks(monkeypatch):
    monkeypatch.setattr(
        "training.utils.rl.train.compute_step_metrics",
        lambda **kw: {"rollout/reward": 0.7, "rollout/accuracy": 0.7, "train/mean_kl": 0.05},
    )
    wandb_calls: list[tuple] = []
    json_calls: list[tuple] = []

    ctx, _, _, _ = _make_ctx(weight_sync_interval=0)
    ctx.wandb_log = lambda metrics, step: wandb_calls.append((step, dict(metrics)))
    ctx.log_metrics_json = lambda step, **kw: json_calls.append((step, kw))

    finish_step(ctx, step=0, prompt_groups=[_make_pg()], fwd_bwd_results=[SimpleNamespace(metrics={})])

    assert wandb_calls and wandb_calls[0][0] == 1
    assert json_calls and json_calls[0][0] == 1
    assert json_calls[0][1] == {"reward": 0.7, "accuracy": 0.7, "kl": 0.05}


# ---------------------------------------------------------------------------
# dump_trajectory_jsonl
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
    assert records[1]["completion"] == "B"
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
