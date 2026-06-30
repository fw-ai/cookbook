"""Per-run async RL training loop.

Acknowledgements -- prior art referenced while designing this loop:

* AReaL  (https://github.com/inclusionAI/AReaL)
* slime  (https://github.com/THUDM/slime)
* Miles  (https://github.com/radixark/miles)

The user supplies ``rollout_fn(sample_prompt) -> RolloutRun | None`` -- one
trajectory per call.  ``sample_prompt`` is a dataset row's dict re-named
once it crosses into the sampling layer.  The loop fans each dataset row
out to ``completions_per_prompt`` parallel rollout calls, joins them by row
id via :class:`GroupAssembler`, applies the optional dynamic filter on the
assembled :class:`PromptGroup`, and feeds the trainer when a batch fills.

Submission is row-atomic; the gate accounts in LLM-call slots, with one row
consuming ``completions_per_prompt`` slots against the staleness budget.
Because rows are submitted whole, every rollout run in a row carries the
same submit version, so the group's accountable version is unambiguous.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Hashable, Iterable, Iterator, List

from training.utils.data import compute_advantages
from training.utils.rl.losses import PromptGroup
from training.utils.rl.rollout.group_assembler import (
    AdvantageFn,
    GroupAssembler,
    RowResolution,
)
from training.utils.rl.rollout.types import RolloutRun
from training.train_loop import DynamicFilterFn

logger = logging.getLogger(__name__)

__all__ = ["AsyncTrainStepFn", "RowRequest", "run_async_rl_loop"]


RunFactory = Callable[[int], Awaitable[RolloutRun | None]]
AsyncTrainStepFn = Callable[
    [int, list[PromptGroup], dict | None, bool],
    tuple[int, dict],
]
PostTrainMetricsFn = Callable[[dict[str, Any]], None]


@dataclass
class RowRequest:
    """One dataset row's fan-out factory.

    ``run_factory(sub_index)`` is called by the loop for each
    ``sub_index in [0, completions_per_prompt)`` and must return a fresh
    coroutine resolving to a :class:`RolloutRun` or ``None``.
    ``None`` counts as one lost run within the row's group (the row
    can still produce a valid PromptGroup if at least
    ``GroupAssembler.min_group_size`` runs land).

    ``on_resolved`` fires exactly once per row, when the loop decides the
    row's fate: ``"accepted"``, ``"filter"`` (assembled group rejected
    by ``dynamic_filter_fn``), or ``"none"`` (no surviving runs /
    advantage_fn produced non-finite values).
    """

    row_id: Hashable
    run_factory: RunFactory
    row_meta: dict | None = None
    on_resolved: Callable[[str], None] | None = None


@dataclass
class _StalenessController:
    """Sample-level (LLM-call) capacity gate.

    Bookkeeping is in samples to match ``deployment.max_batch_size``.
    One prompt submission consumes ``completions_per_prompt`` samples;
    a prompt resolution releases the same amount (re-credited to
    ``accepted_samples`` on accept, freed on reject).
    """

    batch_size_samples: int  # = prompt_groups_per_step * completions_per_prompt
    completions_per_prompt: int  # samples per prompt
    max_staleness: int  # versions
    max_concurrent_samples: int | None
    version: int = 0
    accepted_samples: int = 0
    running_samples: int = 0
    sample_fails: int = 0
    filter_drops: int = 0
    rejected_count: int = 0
    # Wall-clock blocked on the staleness budget; ~0 in healthy async runs.
    sampler_wait_for_trainer_total: float = 0.0
    _sampler_wait_for_trainer_start: float | None = field(default=None, repr=False)

    def staleness_capacity(self) -> int:
        return (
            (self.max_staleness + self.version + 1) * self.batch_size_samples
            - (self.accepted_samples + self.running_samples)
        )

    def concurrency_capacity(self) -> int | None:
        if self.max_concurrent_samples is None:
            return None
        return self.max_concurrent_samples - self.running_samples

    def capacity(self) -> int:
        """Binding admit budget (min of staleness and concurrency), in samples."""
        s = self.staleness_capacity()
        c = self.concurrency_capacity()
        cap = s if c is None else min(c, s)
        return max(0, cap)

    def is_staleness_bound(self) -> bool:
        """True iff one more prompt is blocked by staleness, not concurrency.

        Returning True when both budgets are zero would wrongly attribute
        deployment-saturation wall time as ``sampler_wait_for_trainer``.
        """
        cpp = self.completions_per_prompt
        if self.staleness_capacity() >= cpp:
            return False
        if self.max_concurrent_samples is None:
            return True
        return self.concurrency_capacity() >= cpp

    def mark_sampler_wait_for_trainer_start(self, now: float) -> None:
        if self._sampler_wait_for_trainer_start is None:
            self._sampler_wait_for_trainer_start = now

    def mark_sampler_wait_for_trainer_end(self, now: float) -> None:
        if self._sampler_wait_for_trainer_start is not None:
            self.sampler_wait_for_trainer_total += now - self._sampler_wait_for_trainer_start
            self._sampler_wait_for_trainer_start = None

    def submit(self) -> None:
        self.running_samples += self.completions_per_prompt

    def accept(self) -> None:
        self.running_samples -= self.completions_per_prompt
        self.accepted_samples += self.completions_per_prompt

    def reject(self, reason: str) -> None:
        self.running_samples -= self.completions_per_prompt
        self.rejected_count += 1
        if reason == "none":
            self.sample_fails += 1
        elif reason == "filter":
            self.filter_drops += 1

    def advance_version(self) -> None:
        self.version += 1

    def resolved_count(self, offset: int) -> int:
        # Resolved prompts since start = accepted (in prompts) + rejected.
        return offset + (self.accepted_samples // self.completions_per_prompt) + self.rejected_count


@dataclass
class _RowState:
    request: RowRequest
    submit_version: int
    run_tasks: List[asyncio.Task] = field(default_factory=list)


@dataclass(frozen=True)
class _TrainBatchPlan:
    run_optimizer_step: bool
    chunk_idx: int
    chunks_per_step: int
    accumulated_groups: int
    optimizer_target_groups: int


@dataclass
class _PipelineChunkState:
    """Scheduler-level optimizer-window state.

    This loop never waits to fill a scheduler chunk. It drains completed
    prompt groups as they arrive, capped by ``prompt_groups_per_chunk`` and the
    remaining groups in the current optimizer window. API/trainer continuous
    batching owns execution-level coalescing and microbatching.
    """

    prompt_groups_per_step: int
    requested_chunks_per_step: int
    prompt_groups_per_chunk: int = field(init=False)
    configured_chunks_per_step: int = field(init=False)
    accumulated_groups: int = 0
    chunk_idx: int = 0

    def __post_init__(self) -> None:
        requested = max(1, self.requested_chunks_per_step)
        self.prompt_groups_per_chunk = max(
            1,
            math.ceil(self.prompt_groups_per_step / requested),
        )
        self.configured_chunks_per_step = max(
            1,
            math.ceil(self.prompt_groups_per_step / self.prompt_groups_per_chunk),
        )

    def ready_batch_size(self, *, buffer_len: int) -> int | None:
        if buffer_len == 0:
            return None
        remaining = max(0, self.prompt_groups_per_step - self.accumulated_groups)
        if remaining == 0:
            return None
        return min(buffer_len, self.prompt_groups_per_chunk, remaining)

    def plan_dispatch(
        self,
        *,
        batch_size: int,
        draining: bool,
    ) -> _TrainBatchPlan:
        self.chunk_idx += 1
        accumulated_groups = self.accumulated_groups + batch_size
        run_optimizer_step = (
            accumulated_groups >= self.prompt_groups_per_step or draining
        )
        optimizer_batch_groups = (
            accumulated_groups
            if run_optimizer_step and accumulated_groups < self.prompt_groups_per_step
            else self.prompt_groups_per_step
        )
        chunks_per_step = max(
            self.chunk_idx,
            math.ceil(optimizer_batch_groups / self.prompt_groups_per_chunk),
        )
        return _TrainBatchPlan(
            run_optimizer_step=run_optimizer_step,
            chunk_idx=self.chunk_idx,
            chunks_per_step=max(1, chunks_per_step),
            accumulated_groups=accumulated_groups,
            optimizer_target_groups=optimizer_batch_groups,
        )

    def mark_dispatched(self, plan: _TrainBatchPlan) -> None:
        if plan.run_optimizer_step:
            self.accumulated_groups = 0
            self.chunk_idx = 0
            return
        self.accumulated_groups = plan.accumulated_groups


async def run_async_rl_loop(
    rows: Iterable[RowRequest],
    *,
    train_step_fn: AsyncTrainStepFn,
    completions_per_prompt: int,
    prompt_groups_per_step: int,
    max_head_offpolicy_versions: int,
    advantage_fn: AdvantageFn = compute_advantages,
    with_reference: bool = False,
    router_replay_completion_only: bool = False,
    min_group_size: int = 1,
    weight_sync_fn: Callable[[int], None] | None = None,
    weight_sync_interval: int = 1,
    max_concurrent: int | None = None,
    dynamic_filter_fn: DynamicFilterFn | None = None,
    global_step: int = 0,
    resolved_rows_offset: int = 0,
    resolved_rows_fn: Callable[[], int] | None = None,
    return_final_stats: bool = False,
    pipeline_chunks_per_step: int = 1,
    post_train_metrics_fn: PostTrainMetricsFn | None = None,
) -> int | tuple[int, dict[str, Any]]:
    """Run the per-run async RL loop.

    Args:
        rows: Iterable of :class:`RowRequest`, one per dataset row.  The
            loop calls each row's ``run_factory`` ``completions_per_prompt``
            times to materialize the per-run coroutines.
        train_step_fn: Training callback.  The final bool is control-plane
            data: whether this dispatch should also run an optimizer step.
        completions_per_prompt: Number of rollout runs drawn per row.
        prompt_groups_per_step: Number of accepted rows that form one
            optimizer step.
        max_head_offpolicy_versions: Off-policy budget in sampler
            (policy) versions; the gate's version increments once per
            ``weight_sync_fn`` call, not per optimizer step. ``0`` is
            strict on-policy.
        advantage_fn: Group-level advantage computation.  Default is
            GRPO z-score normalization.  Output is validated for
            non-finite values; a row producing NaN/inf advantages is
            dropped.
        with_reference: Pack reference-model datums alongside the
            policy datums (needed for KL).
        router_replay_completion_only: When a segment carries
            ``routing_matrices``, zero out prompt-position routing so
            only completion-token routing is replayed.
        min_group_size: Minimum surviving runs for a row to emit a
            PromptGroup.  Rows with fewer surviving runs are dropped.
        weight_sync_fn: Called after every ``weight_sync_interval``
            optimizer steps.  Must bump the sampler version; the
            loop increments its internal version counter on return.
        weight_sync_interval: Fire ``weight_sync_fn`` every N steps.
        max_concurrent: Hard cap on **samples (LLM calls)** in flight --
            this is the same unit the deployment's ``max_batch_size``
            gates on, so the recipe-side cap and the deployment cap are
            directly comparable.  Must be ``>= completions_per_prompt``
            (one row's worth) or the gate deadlocks.  ``None`` lets the
            staleness budget alone bound concurrency.
        dynamic_filter_fn: Post-assembly filter on :class:`PromptGroup`.
            ``False`` drops the row from the trainer buffer (without
            charging the gate beyond the row slot already consumed).
        global_step: Initial optimizer-step counter.
        resolved_rows_offset: Resume cursor offset.
        resolved_rows_fn: Optional durable cursor provider.
        return_final_stats: When ``True``, return ``(global_step, stats)``.
        pipeline_chunks_per_step: Pipeline-mode scheduler chunks
            per global optimizer batch.  The scheduler does not wait to fill
            these chunks; it sends ready prompt groups immediately and leaves
            execution-level coalescing/microbatching to the trainer.
        post_train_metrics_fn: Optional logging callback called after each
            train dispatch returns.  The loop uses this only to expose
            scheduler/trainer overlap data; it does not read callback output.

    Returns:
        Final ``global_step``, optionally with a stats dict.
    """
    if completions_per_prompt < 1:
        raise ValueError("completions_per_prompt must be >= 1")
    if prompt_groups_per_step < 1:
        raise ValueError("prompt_groups_per_step must be >= 1")
    if max_head_offpolicy_versions < 0:
        raise ValueError("max_head_offpolicy_versions must be >= 0")
    if weight_sync_interval < 1:
        raise ValueError("weight_sync_interval must be >= 1")
    if pipeline_chunks_per_step < 1:
        raise ValueError("pipeline_chunks_per_step must be >= 1")
    if max_concurrent is not None and max_concurrent < completions_per_prompt:
        # Sample-level cap: must fit at least one row's worth (cpp samples).
        # Anything smaller would deadlock the gate.
        raise ValueError(
            f"max_concurrent (samples) = {max_concurrent} must be "
            f">= completions_per_prompt ({completions_per_prompt}); "
            "smaller values would cap row concurrency at <1 (deadlock)."
        )
    if min_group_size < 1:
        raise ValueError("min_group_size must be >= 1")
    if min_group_size > completions_per_prompt:
        raise ValueError(
            f"min_group_size ({min_group_size}) must be "
            f"<= completions_per_prompt ({completions_per_prompt})",
        )
    if weight_sync_interval > max_head_offpolicy_versions + 1:
        raise ValueError(
            f"weight_sync_interval ({weight_sync_interval}) must be "
            f"<= max_head_offpolicy_versions + 1 "
            f"({max_head_offpolicy_versions + 1}); otherwise the async "
            "gate stalls before the next weight sync because all rollouts at "
            "the current version are exhausted.  Either lower "
            "weight_sync_interval to <= max_head_offpolicy_versions + 1 "
            "(so a weight sync arrives before capacity hits zero), or raise "
            "max_head_offpolicy_versions to >= weight_sync_interval - 1 "
            "(so the head budget covers every step between weight syncs)."
        )

    staleness = _StalenessController(
        batch_size_samples=prompt_groups_per_step * completions_per_prompt,
        completions_per_prompt=completions_per_prompt,
        max_staleness=max_head_offpolicy_versions,
        max_concurrent_samples=max_concurrent,
    )
    assembler = GroupAssembler(
        completions_per_prompt=completions_per_prompt,
        advantage_fn=advantage_fn,
        with_reference=with_reference,
        router_replay_completion_only=router_replay_completion_only,
        min_group_size=min_group_size,
    )

    # Each in-flight rollout task -> (row_id, sub_index).
    in_flight: dict[asyncio.Task, tuple[Hashable, int]] = {}
    # Row state by row_id; lifetime spans first sample submit -> row resolution.
    rows_state: dict[Hashable, _RowState] = {}
    buffer: list[tuple[PromptGroup, int, RowRequest]] = []

    rows_iter: Iterator[RowRequest] = iter(rows)
    iterator_exhausted = False
    chunk_state = _PipelineChunkState(
        prompt_groups_per_step=prompt_groups_per_step,
        requested_chunks_per_step=pipeline_chunks_per_step,
    )

    def _resolved_rows(fallback: int) -> int:
        if resolved_rows_fn is not None:
            return resolved_rows_fn()
        return resolved_rows_offset + fallback

    def _resolve(request: RowRequest, reason: str) -> None:
        if request.on_resolved is not None:
            request.on_resolved(reason)

    def _submit_row(request: RowRequest) -> None:
        rows_state[request.row_id] = _RowState(
            request=request,
            submit_version=staleness.version,
        )
        staleness.submit()
        for sub_index in range(completions_per_prompt):
            assembler.note_started(
                request.row_id,
                submit_version=staleness.version,
                row_meta=request.row_meta if sub_index == 0 else None,
            )
            coro = request.run_factory(sub_index)
            task = asyncio.ensure_future(coro)
            in_flight[task] = (request.row_id, sub_index)
            rows_state[request.row_id].run_tasks.append(task)

    def _refill() -> None:
        nonlocal iterator_exhausted
        now = time.monotonic()
        if iterator_exhausted:
            # Draining, not blocked on the trainer.
            staleness.mark_sampler_wait_for_trainer_end(now)
            return
        slots = staleness.capacity() // completions_per_prompt
        if slots == 0:
            if staleness.is_staleness_bound():
                staleness.mark_sampler_wait_for_trainer_start(now)
            return
        staleness.mark_sampler_wait_for_trainer_end(now)
        for _ in range(slots):
            try:
                request = next(rows_iter)
            except StopIteration:
                iterator_exhausted = True
                return
            _submit_row(request)

    def _can_make_batch() -> bool:
        return chunk_state.ready_batch_size(
            buffer_len=len(buffer),
        ) is not None

    def _has_outstanding_work() -> bool:
        return bool(in_flight) or _can_make_batch() or not iterator_exhausted

    def _on_row_resolved(row_id: Hashable, resolution: RowResolution) -> None:
        """Drive row-level outcome bookkeeping when the GroupAssembler
        settles a row (returned ``RowResolution``)."""
        state = rows_state.pop(row_id, None)
        if state is None:
            return
        if resolution.pg is None:
            staleness.reject("none")
            _resolve(state.request, "none")
            return
        if dynamic_filter_fn is not None and not dynamic_filter_fn(resolution.pg):
            staleness.reject("filter")
            _resolve(state.request, "filter")
            return
        staleness.accept()
        buffer.append((resolution.pg, resolution.min_submit_version, state.request))

    _refill()
    # Gap from here to the next ``step_start`` is ``trainer_wait_for_sampler_time``.
    last_step_end = time.monotonic()
    prev_sampler_wait_for_trainer_total = staleness.sampler_wait_for_trainer_total
    rollout_tasks_completed_during_train_total = 0
    rollout_tasks_available_during_train_total = 0
    train_dispatch_wall_time_total = 0.0

    while _has_outstanding_work():
        if not _can_make_batch():
            if not in_flight:
                if iterator_exhausted:
                    break
                _refill()
                if not in_flight:
                    logger.warning(
                        "Async loop stalled: capacity=0 with no in-flight "
                        "tasks and the row iterator still open. Increase "
                        "max_head_offpolicy_versions or supply a "
                        "weight_sync_fn so versions advance.",
                    )
                    iterator_exhausted = True
                    break
                continue

            done, _ = await asyncio.wait(
                set(in_flight), return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                row_id, _sub = in_flight.pop(task)
                exc = task.exception()
                if exc is not None:
                    for other in in_flight:
                        other.cancel()
                    in_flight.clear()
                    raise exc
                run = task.result()
                if run is None:
                    resolution = assembler.note_dropped(row_id)
                else:
                    resolution = assembler.add_run(row_id, run)
                if resolution is not None:
                    _on_row_resolved(row_id, resolution)
            _refill()
            continue

        batch_size = chunk_state.ready_batch_size(
            buffer_len=len(buffer),
        )
        assert batch_size is not None
        batch_pairs = buffer[:batch_size]
        buffer = buffer[batch_size:]
        batch = [pg for pg, _, _ in batch_pairs]
        versions = [v for _, v, _ in batch_pairs]
        offsets = [staleness.version - v for v in versions]
        batch_plan = chunk_state.plan_dispatch(
            batch_size=len(batch),
            draining=iterator_exhausted and not in_flight and not buffer,
        )

        step_start = time.monotonic()
        for _, _, request in batch_pairs:
            _resolve(request, "accepted")
        resolved_rows = (
            resolved_rows_fn()
            if resolved_rows_fn is not None
            else staleness.resolved_count(resolved_rows_offset)
        )

        sampler_wait_for_trainer_time = (
            staleness.sampler_wait_for_trainer_total
            - prev_sampler_wait_for_trainer_total
        )
        prev_sampler_wait_for_trainer_total = staleness.sampler_wait_for_trainer_total
        trainer_wait_for_sampler_time = step_start - last_step_end
        cc = staleness.concurrency_capacity()
        train_start_tasks = list(in_flight)
        train_in_flight_at_start = len(train_start_tasks)
        train_done_at_start = sum(1 for task in train_start_tasks if task.done())
        logger.info(
            "[batch-ready v=%d] in_flight=%d running=%d accepted=%d buffer=%d "
            "staleness_cap=%d concurrency_cap=%s "
            "wait_for_sampler=%.1fs wait_for_trainer=%.1fs "
            "(staleness_bound=%s)",
            staleness.version,
            len(in_flight),
            staleness.running_samples,
            staleness.accepted_samples,
            len(buffer),
            staleness.staleness_capacity(),
            "unbounded" if cc is None else cc,
            trainer_wait_for_sampler_time,
            sampler_wait_for_trainer_time,
            staleness.is_staleness_bound(),
        )
        extra_metrics: dict[str, Any] = {
            "async/version_offset_mean": sum(offsets) / len(offsets),
            "async/version_offset_max": max(offsets),
            "async/version_offset_min": min(offsets),
            "async/in_flight": len(in_flight),
            "async/in_flight_at_train_start": train_in_flight_at_start,
            "async/in_flight_done_at_train_start": train_done_at_start,
            "async/sample_fails": staleness.sample_fails,
            "async/filter_drops": staleness.filter_drops,
            "async/stale_drops": 0,
            "all_raw_rewards": [r for pg in batch for r in pg.rewards],
            "valid_prompt_groups": len(batch),
            "total_sampled": (
                staleness.accepted_samples
                + staleness.rejected_count * completions_per_prompt
            ),
            "filter_drops": staleness.filter_drops,
            "sample_fails": staleness.sample_fails,
            "stale_drops": 0,
            "resolved_rows": resolved_rows,
            "trainer_wait_for_sampler_time": trainer_wait_for_sampler_time,
            "sampler_wait_for_trainer_time": sampler_wait_for_trainer_time,
            "async/running_samples": staleness.running_samples,
            "async/accepted_samples": staleness.accepted_samples,
            "async/staleness_capacity_at_step": staleness.staleness_capacity(),
            "ctx/current_version": staleness.version,
            "async/concurrency_capacity_at_step": (
                -1 if cc is None else cc
            ),
            "pipeline/chunk_idx": batch_plan.chunk_idx,
            "pipeline/chunk_prompt_groups": len(batch),
            "pipeline/prompt_groups_accumulated": batch_plan.accumulated_groups,
            "pipeline/prompt_groups_per_step": prompt_groups_per_step,
            "pipeline/prompt_groups_per_chunk": chunk_state.prompt_groups_per_chunk,
            "pipeline/chunks_per_step": batch_plan.chunks_per_step,
            "pipeline/configured_chunks_per_step": (
                chunk_state.configured_chunks_per_step
            ),
            "pipeline/requested_chunks_per_step": pipeline_chunks_per_step,
            "batch/optimizer_prompt_groups": batch_plan.optimizer_target_groups,
        }

        prev_global_step = global_step
        train_dispatch_start = time.monotonic()
        global_step, _step_metrics = await asyncio.to_thread(
            train_step_fn,
            global_step,
            batch,
            extra_metrics,
            batch_plan.run_optimizer_step,
        )
        last_step_end = time.monotonic()
        train_dispatch_wall_time = last_step_end - train_dispatch_start
        train_done_after = sum(1 for task in train_start_tasks if task.done())
        completed_during_train = max(0, train_done_after - train_done_at_start)
        available_during_train = max(0, train_in_flight_at_start - train_done_at_start)
        completion_ratio = (
            completed_during_train / available_during_train
            if available_during_train > 0
            else 0.0
        )
        rollout_tasks_completed_during_train_total += completed_during_train
        rollout_tasks_available_during_train_total += available_during_train
        train_dispatch_wall_time_total += train_dispatch_wall_time
        overlap_metrics: dict[str, Any] = {
            "rollout/step": prev_global_step + 1,
            "train/step": global_step if batch_plan.run_optimizer_step else prev_global_step + 1,
            "train/run_optimizer_step": int(batch_plan.run_optimizer_step),
            "pipeline/chunk_idx": batch_plan.chunk_idx,
            "pipeline/chunks_per_step": batch_plan.chunks_per_step,
            "async/in_flight_at_train_start": train_in_flight_at_start,
            "async/in_flight_done_at_train_start": train_done_at_start,
            "async/in_flight_done_after_train": train_done_after,
            "async/rollout_tasks_available_during_train": available_during_train,
            "async/rollout_tasks_completed_during_train": completed_during_train,
            "perf/train_dispatch_wall_time": train_dispatch_wall_time,
            "perf/train_rollout_overlap_completion_ratio": completion_ratio,
        }
        logger.info(
            "[train-overlap v=%d] chunk=%d/%d in_flight_start=%d "
            "done_start=%d done_after=%d completed_during_train=%d "
            "ratio=%.3f train_wall=%.1fs",
            staleness.version,
            batch_plan.chunk_idx,
            batch_plan.chunks_per_step,
            train_in_flight_at_start,
            train_done_at_start,
            train_done_after,
            completed_during_train,
            completion_ratio,
            train_dispatch_wall_time,
        )
        if post_train_metrics_fn is not None:
            post_train_metrics_fn(overlap_metrics)

        chunk_state.mark_dispatched(batch_plan)

        if (
            weight_sync_fn is not None
            and global_step > prev_global_step
            and batch_plan.run_optimizer_step
            and global_step % weight_sync_interval == 0
        ):
            await asyncio.to_thread(weight_sync_fn, global_step)
            staleness.advance_version()

        _refill()

    pending_tasks = list(in_flight)
    for task in pending_tasks:
        task.cancel()
    results = await asyncio.gather(*pending_tasks, return_exceptions=True)
    for task, result in zip(pending_tasks, results):
        if isinstance(result, asyncio.CancelledError):
            continue
        if isinstance(result, BaseException):
            logger.warning(
                "Async loop drained an in-flight rollout sample with an "
                "unhandled exception: %r", result,
            )
    in_flight.clear()
    rows_state.clear()

    staleness.mark_sampler_wait_for_trainer_end(time.monotonic())

    accepted_prompts = staleness.accepted_samples // completions_per_prompt
    resolved_this_run = accepted_prompts + staleness.rejected_count
    final_stats = {
        "sample_fails": staleness.sample_fails,
        "filter_drops": staleness.filter_drops,
        "stale_drops": 0,
        "total_accepted": accepted_prompts,
        "resolved_rows": _resolved_rows(resolved_this_run),
        "sampler_wait_for_trainer_time_total": staleness.sampler_wait_for_trainer_total,
        "rollout_tasks_available_during_train_total": rollout_tasks_available_during_train_total,
        "rollout_tasks_completed_during_train_total": rollout_tasks_completed_during_train_total,
        "train_dispatch_wall_time_total": train_dispatch_wall_time_total,
    }
    if return_final_stats:
        return global_step, final_stats
    return global_step
