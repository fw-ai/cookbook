"""Per-step training primitives + trajectory logger.

Library-style helpers for the training side of an RL loop. The primitives
here are *thin wrappers* around well-defined SDK call sequences:

* :func:`ref_fwd_bwd` — for one prompt group, run reference forward,
  policy forward, and forward_backward (server accumulates gradients).
* :func:`finish_step` — after all groups in a step have been processed,
  fire ``optim_step`` + (optional) hotload + metrics + (optional) DCP save.
* :class:`TrainContext` — bundle of trainer-side handles + config so the
  primitives don't take 15 keyword arguments each.
* :func:`dump_trajectory_jsonl` — pure trajectory dump.

Recipes own the loop body — they decide *when* to call ``ref_fwd_bwd``
(per group, per batch, etc.) and *when* to call ``finish_step``. The
cookbook does not ship a runner.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

from training.utils.rl.losses import (
    PromptGroup,
    build_builtin_loss_datums,
    combine_prompt_groups,
)
from training.utils.rl.metrics import compute_step_metrics
from training.utils.timer import flush_timing, timer

logger = logging.getLogger(__name__)

__all__ = [
    "TrainContext",
    "dump_trajectory_jsonl",
    "finish_step",
    "ref_fwd_bwd",
]


# ---------------------------------------------------------------------------
# TrainContext — bundle of objects ref_fwd_bwd / finish_step need
# ---------------------------------------------------------------------------


@dataclass
class TrainContext:
    """Trainer-side handles and config for ``ref_fwd_bwd`` / ``finish_step``.

    Constructed once in the recipe (typically right after infra setup),
    then passed into each per-group / per-step call. The fields are all
    things the recipe already had to construct anyway — bundling them
    just keeps the per-call signatures short.

    Attributes:
        policy: ``ReconnectableClient`` for the policy trainer.
        reference: ``ReconnectableClient`` for the reference trainer
            (or ``None`` when KL is disabled, or LoRA shares the policy
            session).
        weight_syncer: ``WeightSyncer`` for save + hotload.
        adam_params: ``tinker.AdamParams`` passed to ``optim_step``.
        grad_accumulation_normalization: Forwarded to ``optim_step``.
        builtin_server_loss: Resolved builtin loss tuple from
            ``resolve_builtin_loss(...)``, or ``None`` to use the
            client-side loss closure.
        client_loss_builder: Callable that builds the Python loss
            closure for ``forward_backward_custom`` when no builtin is
            eligible.
        tis_config: TIS config; passed into builtin datum construction.
        policy_loss: Loss name (e.g. ``"grpo"``); selects builtin datum shape.
        log_path: Recipe log directory (used when saving checkpoints).
        policy_job_id: Trainer job ID (recorded in checkpoint metadata).
        completions_per_prompt: Echoed into step metrics.
        trajectory_dir: When set, ``finish_step`` dumps per-step JSONL.
        weight_sync_interval: Hotload cadence (1 = every step,
            0 = never).
        dcp_save_interval: DCP checkpoint cadence (0 = never; only the
            ``save_checkpoint_fn`` passed to ``finish_step`` actually fires).
        wandb_log: Optional ``(metrics_dict, step) -> None``.
        log_metrics_json: Optional ``(step, **kwargs) -> None``.
    """

    policy: Any
    reference: Any | None
    weight_syncer: Any
    adam_params: Any
    grad_accumulation_normalization: Any | None

    builtin_server_loss: tuple[Any, dict[str, Any]] | None
    client_loss_builder: Callable[..., Any]
    tis_config: Any
    policy_loss: str

    log_path: str
    policy_job_id: str | None
    completions_per_prompt: int
    trajectory_dir: str | None = None

    weight_sync_interval: int = 1
    dcp_save_interval: int = 0

    wandb_log: Callable[..., None] | None = None
    log_metrics_json: Callable[..., None] | None = None


# ---------------------------------------------------------------------------
# Internals (private — the public surface is ref_fwd_bwd / finish_step)
# ---------------------------------------------------------------------------


def _ref_forward(ctx: TrainContext, groups: list[PromptGroup]) -> None:
    """Compute reference logprobs for the supplied groups (one batched call).

    Mutates each group's ``ref_logprobs`` in place. No-op when
    ``ctx.reference is None`` or when no group carries ref data.
    """
    if ctx.reference is None:
        return
    all_ref_data = [d for pg in groups for d in pg.ref_data]
    if not all_ref_data:
        return
    ref_fwd = ctx.reference.forward(all_ref_data, "cross_entropy")
    idx = 0
    for pg in groups:
        n = len(pg.ref_data)
        pg.ref_logprobs = [
            ref_fwd.loss_fn_outputs[idx + i]["logprobs"].data for i in range(n)
        ]
        idx += n


def _fwd_bwd(ctx: TrainContext, prompt_groups: list[PromptGroup]) -> Any:
    """One forward-backward using the builtin or client-side loss path."""
    if not prompt_groups:
        raise ValueError("fwd_bwd requires at least one prompt group")

    data, adv, ref_lp, prompt_lens, inf_lp = combine_prompt_groups(prompt_groups)

    with timer("policy_forward"):
        prox_fwd = ctx.policy.forward(data, "cross_entropy")
        prox_lp = [
            prox_fwd.loss_fn_outputs[i]["logprobs"].data for i in range(len(data))
        ]

    with timer("fwd_bwd"):
        if ctx.builtin_server_loss is not None:
            kernel_loss, kernel_config = ctx.builtin_server_loss
            rl_datums = build_builtin_loss_datums(
                data, adv, prox_lp, inf_lp, prompt_lens,
                ctx.tis_config, policy_loss=ctx.policy_loss,
            )
            return ctx.policy.forward_backward(
                rl_datums, kernel_loss, loss_fn_config=kernel_config,
            )
        return ctx.policy.forward_backward_custom(
            data,
            ctx.client_loss_builder(adv, ref_lp, prompt_lens, inf_lp, prox_lp),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ref_fwd_bwd(ctx: TrainContext, group: PromptGroup) -> Any:
    """ref_forward + policy_forward + forward_backward for ONE prompt group.

    The trainer accumulates gradients across calls within a step. Call
    :func:`finish_step` after processing all groups in the step to fire
    ``optim_step``.

    Returns the ``forward_backward`` result; the caller typically appends
    these into a list passed to ``finish_step`` for metrics.
    """
    t0 = time.time()
    with timer("ref_forward"):
        _ref_forward(ctx, [group])
    logger.info("ref_forward: done (%.1fs)", time.time() - t0)

    t0 = time.time()
    result = _fwd_bwd(ctx, [group])
    logger.info("fwd_bwd: done (%.1fs)", time.time() - t0)
    return result


def finish_step(
    ctx: TrainContext,
    step: int,
    prompt_groups: list[PromptGroup],
    fwd_bwd_results: list[Any],
    loop_stats: dict | None = None,
    *,
    save_checkpoint_fn: Callable[[str, dict], Any] | None = None,
    checkpoint_extra: dict | None = None,
    step_target: int | None = None,
    resume_data_consumed: int = 0,
    step_offset: int = 0,
) -> tuple[int, dict]:
    """Finalize a training step: optim_step + (optional) hotload + metrics.

    Call after :func:`ref_fwd_bwd` has been invoked for every group in
    the step. Returns ``(new_step, metrics_dict)``.

    The optional ``save_checkpoint_fn`` is invoked when
    ``ctx.dcp_save_interval > 0`` and ``new_step`` is on the interval.
    Signature: ``(name, extra_dict) -> any``.
    """
    t0 = time.time()
    with timer("optim_step"):
        optim_result = ctx.policy.optim_step(
            ctx.adam_params,
            grad_accumulation_normalization=ctx.grad_accumulation_normalization,
        )
    step += 1
    logger.info("[step %d] optim_step: done (%.1fs)", step, time.time() - t0)

    if ctx.weight_sync_interval > 0 and step % ctx.weight_sync_interval == 0:
        logger.info("[step %d] weight_sync: saving + loading...", step)
        t0 = time.time()
        with timer("weight_sync"):
            ctx.weight_syncer.save_and_hotload(f"step-{step}")
        logger.info("[step %d] weight_sync: done (%.1fs)", step, time.time() - t0)

    if (
        ctx.dcp_save_interval > 0
        and step % ctx.dcp_save_interval == 0
        and save_checkpoint_fn is not None
    ):
        target = step_target if step_target is not None else 1
        data_consumed = resume_data_consumed + (step - step_offset) * target
        extra = {
            "step": step,
            "data_consumed": data_consumed,
            "source_job_id": ctx.policy_job_id,
        }
        if checkpoint_extra:
            extra.update(checkpoint_extra)
        save_checkpoint_fn(f"step-{step}", extra)

    metrics = compute_step_metrics(
        prompt_groups=prompt_groups,
        fwd_bwd_results=fwd_bwd_results,
        optim_result=optim_result,
        n_accum=len(fwd_bwd_results),
        timing_metrics=flush_timing(),
        loop_stats=loop_stats,
        completions_per_prompt=ctx.completions_per_prompt,
    )
    metrics["train/step"] = step

    avg_reward = metrics.get("rollout/reward", 0.0)
    avg_acc = metrics.get("rollout/accuracy", 0.0)
    avg_kl = metrics.get("train/mean_kl", 0.0)
    logger.info(
        "Step %d | Reward: %.3f | Acc: %.1f%% | KL: %.4f",
        step, avg_reward, avg_acc * 100, avg_kl,
    )

    if ctx.log_metrics_json is not None:
        ctx.log_metrics_json(step, reward=avg_reward, accuracy=avg_acc, kl=avg_kl)
    if ctx.wandb_log is not None:
        ctx.wandb_log(metrics, step)

    if ctx.trajectory_dir:
        dump_trajectory_jsonl(ctx.trajectory_dir, step, prompt_groups)

    return step, metrics


def dump_trajectory_jsonl(
    trajectory_dir: str,
    step: int,
    prompt_groups: list[PromptGroup],
) -> None:
    """Write per-step trajectory JSONL: one line per individual completion."""
    os.makedirs(trajectory_dir, exist_ok=True)
    path = os.path.join(trajectory_dir, f"step_{step:04d}.jsonl")
    n_records = 0
    with open(path, "w") as f:
        for pg_idx, pg in enumerate(prompt_groups):
            completions = pg.completions or []
            for comp_idx, comp_text in enumerate(completions):
                record = {
                    "step": step,
                    "prompt_group": pg_idx,
                    "completion_index": comp_idx,
                    "prompt": pg.prompt,
                    "completion": comp_text,
                    "reward": pg.rewards[comp_idx] if comp_idx < len(pg.rewards) else None,
                    "advantage": (
                        pg.advantages[comp_idx] if comp_idx < len(pg.advantages) else None
                    ),
                    "completion_len": (
                        pg.completion_lens[comp_idx]
                        if comp_idx < len(pg.completion_lens)
                        else None
                    ),
                    "truncated": (
                        pg.truncated[comp_idx] if comp_idx < len(pg.truncated) else None
                    ),
                    "ground_truth": pg.row_meta.get("ground_truth") if pg.row_meta else None,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_records += 1
    logger.info(
        "[step %d] Saved trajectory to %s (%d completions from %d groups)",
        step, path, n_records, len(prompt_groups),
    )
