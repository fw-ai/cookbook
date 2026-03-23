"""RL training loop orchestration for Fireworks recipes.

Provides:
- ``train_one_step`` -- a single training step shared by sync and async paths.
- ``run_rl_loop`` -- an on-policy loop that samples
  ``prompt_groups_per_step`` prompts per optimizer step, then runs a single
  ``train_step`` callback (1:1 ratio).
"""

from __future__ import annotations

import os
import json
import asyncio
import logging
import itertools
import time as _time
from typing import Any, Callable, Iterable, Coroutine
from dataclasses import dataclass

import tinker

from training.utils.rl.losses import (
    PromptGroup,
    build_builtin_loss_datums,
    combine_prompt_groups,
)
from dataclasses import field as _field
from typing import Coroutine as _Coroutine

DynamicFilterFn = Callable[[PromptGroup], bool]
"""Filter callback: ``(PromptGroup) -> bool``.  Return True to keep."""


@dataclass
class RolloutStats:
    """Statistics from one batch collection round."""
    valid_groups: int = 0
    total_sampled: int = 0
    filter_drops: int = 0
    sample_fails: int = 0
    wall_time: float = 0.0
    raw_rewards: list[float] = _field(default_factory=list)
    version_offsets: list[int] = _field(default_factory=list)


async def collect_sync_batch(
    coros: list[_Coroutine[Any, Any, PromptGroup | None]],
    filter_fn: DynamicFilterFn | None = None,
    target: int = 1,
) -> tuple[list[PromptGroup], RolloutStats]:
    """Submit coroutines concurrently, collect up to *target* accepted groups."""
    import time as _t
    queue: asyncio.Queue[PromptGroup | None] = asyncio.Queue()
    worker_error: BaseException | None = None

    async def _worker(coro):
        nonlocal worker_error
        try:
            result = await coro
            queue.put_nowait(result)
        except BaseException as exc:
            if worker_error is None:
                worker_error = exc
            queue.put_nowait(None)

    for c in coros:
        asyncio.create_task(_worker(c))

    stats = RolloutStats()
    accepted: list[PromptGroup] = []
    t0 = _t.time()

    for _ in range(len(coros)):
        item = await queue.get()
        if worker_error is not None:
            raise RuntimeError(f"Sampling worker failed: {worker_error}") from worker_error
        if item is None:
            stats.sample_fails += 1
            stats.total_sampled += 1
            continue
        stats.total_sampled += 1
        stats.raw_rewards.extend(item.rewards)
        if filter_fn is not None and not filter_fn(item):
            stats.filter_drops += 1
            continue
        accepted.append(item)

    stats.valid_groups = len(accepted)
    stats.wall_time = _t.time() - t0
    return accepted, stats
from training.utils.timer import timer, flush_timing
from training.utils.rl.metrics import compute_step_metrics

logger = logging.getLogger(__name__)

__all__ = [
    "TrainStepFns",
    "TrainContext",
    "DynamicFilterFn",
    "train_one_step",
    "ref_fwd_bwd",
    "train_one_group",  # backward-compatible alias
    "finish_step",
    "run_rl_loop",
]


@dataclass
class TrainStepFns:
    """Training callbacks for the 1:1 loop.

    A single ``train_step`` callback receives all prompt groups for one
    optimizer step and is responsible for ref_forward + fwd_bwd + optim_step.
    """

    train_step: Callable[[int, list[PromptGroup], dict | None], tuple[int, dict]]


@dataclass
class TrainContext:
    """Everything ``train_one_step`` needs, set up once in the recipe.

    Bundles the objects created during infrastructure setup so
    ``train_one_step`` doesn't need 15 keyword arguments.
    """

    policy: Any
    """ReconnectableClient for the policy trainer."""
    reference: Any | None
    """ReconnectableClient for the reference trainer (may be ``None``)."""
    weight_syncer: Any
    """WeightSyncer for saving + hotloading."""
    adam_params: tinker.AdamParams
    grad_accumulation_normalization: Any | None

    # Loss resolution (set once from Config)
    builtin_server_loss: tuple[str, dict[str, Any]] | None
    client_loss_builder: Callable[..., Any]
    tis_config: Any
    policy_loss: str

    # Checkpoint / logging context
    log_path: str
    policy_job_id: str | None
    completions_per_prompt: int
    trajectory_dir: str | None = None

    # Weight sync config
    weight_sync_interval: int = 1
    dcp_save_interval: int = 0

    # Logging callbacks
    wandb_log: Callable[..., None] | None = None
    log_metrics_json: Callable[..., None] | None = None


def _ref_forward(ctx: TrainContext, groups: list[PromptGroup]) -> None:
    """Compute reference logprobs for all prompt groups (one call)."""
    if ctx.reference is None:
        return
    all_ref_data = [d for pg in groups for d in pg.ref_data]
    if not all_ref_data:
        return
    ref_fwd = ctx.reference.forward(all_ref_data, "cross_entropy")
    idx = 0
    for pg in groups:
        n = len(pg.ref_data)
        pg.ref_logprobs = [ref_fwd.loss_fn_outputs[idx + i]["logprobs"].data for i in range(n)]
        idx += n


def _fwd_bwd(ctx: TrainContext, prompt_groups: list[PromptGroup]) -> Any:
    """One forward-backward using the builtin or client-side loss path."""
    if not prompt_groups:
        raise ValueError("fwd_bwd requires at least one prompt group")

    data, adv, ref_lp, prompt_lens, inf_lp = combine_prompt_groups(prompt_groups)

    with timer("policy_forward"):
        prox_fwd = ctx.policy.forward(data, "cross_entropy")
        prox_lp = [prox_fwd.loss_fn_outputs[i]["logprobs"].data for i in range(len(data))]

    with timer("fwd_bwd"):
        if ctx.builtin_server_loss is not None:
            kernel_loss, kernel_config = ctx.builtin_server_loss
            rl_datums = build_builtin_loss_datums(
                data, adv, prox_lp, inf_lp, prompt_lens,
                ctx.tis_config, policy_loss=ctx.policy_loss,
            )
            fwd_bwd_result = ctx.policy.forward_backward(
                rl_datums, kernel_loss, loss_fn_config=kernel_config,
            )
        else:
            fwd_bwd_result = ctx.policy.forward_backward_custom(
                data,
                ctx.client_loss_builder(adv, ref_lp, prompt_lens, inf_lp, prox_lp),
            )
    return fwd_bwd_result


def _dump_trajectory(trajectory_dir: str, step: int, prompt_groups: list[PromptGroup]) -> None:
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
                    "advantage": pg.advantages[comp_idx] if comp_idx < len(pg.advantages) else None,
                    "completion_len": pg.completion_lens[comp_idx] if comp_idx < len(pg.completion_lens) else None,
                    "truncated": pg.truncated[comp_idx] if comp_idx < len(pg.truncated) else None,
                    "ground_truth": pg.row_meta.get("ground_truth") if pg.row_meta else None,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_records += 1
    logger.info(
        "[step %d] Saved trajectory to %s (%d completions from %d groups)",
        step, path, n_records, len(prompt_groups),
    )


def ref_fwd_bwd(ctx: TrainContext, group: PromptGroup) -> Any:
    """ref_forward + fwd_bwd for ONE prompt group.

    The server accumulates gradients across calls.  Call ``finish_step``
    after processing ``step_target`` groups to fire optim_step.
    """
    t0 = _time.time()
    with timer("ref_forward"):
        _ref_forward(ctx, [group])
    logger.info("ref_forward: done (%.1fs)", _time.time() - t0)

    t0 = _time.time()
    result = _fwd_bwd(ctx, [group])
    logger.info("fwd_bwd: done (%.1fs)", _time.time() - t0)
    return result


# Backward-compatible alias
train_one_group = ref_fwd_bwd


def finish_step(
    ctx: TrainContext,
    step: int,
    prompt_groups: list[PromptGroup],
    fwd_bwd_results: list[Any],
    loop_stats: dict | None = None,
    *,
    save_checkpoint_fn: Callable[..., Any] | None = None,
    checkpoint_extra: dict | None = None,
    step_target: int | None = None,
    resume_data_consumed: int = 0,
    step_offset: int = 0,
) -> tuple[int, dict]:
    """Complete a training step: optim_step + weight_sync + metrics.

    Called after ``train_one_group`` has been invoked for every group in
    the step.  The server has accumulated gradients from each call.

    Returns ``(new_step, metrics_dict)``.
    """
    t0 = _time.time()
    with timer("optim_step"):
        optim_result = ctx.policy.optim_step(
            ctx.adam_params,
            grad_accumulation_normalization=ctx.grad_accumulation_normalization,
        )
    step += 1
    logger.info("[step %d] optim_step: done (%.1fs)", step, _time.time() - t0)

    if ctx.weight_sync_interval > 0 and step % ctx.weight_sync_interval == 0:
        logger.info("[step %d] weight_sync: saving + loading...", step)
        t0 = _time.time()
        with timer("weight_sync"):
            ctx.weight_syncer.save_and_hotload(f"step-{step}")
        logger.info("[step %d] weight_sync: done (%.1fs)", step, _time.time() - t0)

    if (
        ctx.dcp_save_interval > 0
        and step % ctx.dcp_save_interval == 0
        and save_checkpoint_fn is not None
    ):
        _target = step_target if step_target is not None else 1
        _data_consumed = resume_data_consumed + (step - step_offset) * _target
        extra = {"step": step, "data_consumed": _data_consumed, "source_job_id": ctx.policy_job_id}
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
        _dump_trajectory(ctx.trajectory_dir, step, prompt_groups)

    return step, metrics


def train_one_step(
    ctx: TrainContext,
    step: int,
    prompt_groups: list[PromptGroup],
    loop_stats: dict | None = None,
    *,
    save_checkpoint_fn: Callable[..., Any] | None = None,
    checkpoint_extra: dict | None = None,
    step_target: int | None = None,
    resume_data_consumed: int = 0,
    step_offset: int = 0,
) -> tuple[int, dict]:
    """Execute one training step: ref_forward + fwd_bwd + optim + weight_sync + metrics.

    Backward-compatible wrapper around ``train_one_group`` + ``finish_step``.

    Returns ``(new_step, metrics_dict)``.
    """
    fwd_bwd_results = [train_one_group(ctx, g) for g in prompt_groups]
    return finish_step(
        ctx, step, prompt_groups, fwd_bwd_results, loop_stats,
        save_checkpoint_fn=save_checkpoint_fn,
        checkpoint_extra=checkpoint_extra,
        step_target=step_target,
        resume_data_consumed=resume_data_consumed,
        step_offset=step_offset,
    )


# ---------------------------------------------------------------------------
# Sync RL loop
# ---------------------------------------------------------------------------


async def run_rl_loop(
    sample_fns: Iterable[Coroutine[Any, Any, PromptGroup | None]],
    *,
    train_fns: TrainStepFns,
    prompt_groups_per_step: int = 1,
    dynamic_filter_fn: DynamicFilterFn | None = None,
    global_step: int = 0,
    metrics_callback: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Run the on-policy RL training loop.

    All ``prompt_groups_per_step`` sampling coroutines fire concurrently.
    Request-level throttling is handled by the ``request_semaphore``
    passed into ``sample_with_tokens`` by the caller -- this loop does
    not limit concurrency itself.
    """
    coros = list(sample_fns)
    coro_iter = iter(coros)

    while True:
        step_coros = list(itertools.islice(coro_iter, prompt_groups_per_step))
        if not step_coros:
            break

        step_prompt_groups, stats = await collect_sync_batch(
            step_coros,
            filter_fn=dynamic_filter_fn,
            target=prompt_groups_per_step,
        )

        if not step_prompt_groups:
            logger.warning("[step %d] no valid prompt groups after filtering, skipping", global_step + 1)
            continue

        logger.info(
            "Sampling complete: %d/%d groups in %.1fs (failed=%d, filtered=%d)",
            stats.valid_groups, prompt_groups_per_step,
            stats.wall_time, stats.sample_fails, stats.filter_drops,
        )
        loop_stats = {
            "valid_prompt_groups": stats.valid_groups,
            "total_sampled": stats.total_sampled,
            "filter_drops": stats.filter_drops,
            "sample_fails": stats.sample_fails,
            "sample_wait_time": stats.wall_time,
            "step_wall_time": stats.wall_time,
            "all_raw_rewards": list(stats.raw_rewards),
        }

        global_step, _ = await asyncio.to_thread(
            train_fns.train_step, global_step, step_prompt_groups, loop_stats,
        )

        if metrics_callback is not None:
            metrics_callback({
                "train/step": global_step,
                "rollout/sample_fails": stats.sample_fails,
                "rollout/filter_drops": stats.filter_drops,
            })

    return global_step
