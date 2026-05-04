"""Per-sample async RL training loop.

The user supplies ``rollout_fn(sample_prompt) -> RolloutSample | None`` --
one trajectory per call.  ``sample_prompt`` is a dataset row's dict
re-named once it crosses into the sampling layer.  The loop fans each
dataset row out to ``completions_per_prompt`` parallel sample calls,
joins them by row id via :class:`GroupAssembler`, applies the optional
dynamic filter on the assembled :class:`PromptGroup`, and feeds the
trainer when a batch fills.

Off-policy gating is per dataset row: one row = ``completions_per_prompt``
samples consumed against the staleness budget.  The accountable "version"
of a group is the oldest submit version among its samples (samples within
a row may straddle a weight-sync boundary; the older side dominates the
staleness reading).
"""

from __future__ import annotations

import asyncio
import logging
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
from training.utils.rl.rollout.types import RolloutSample
from training.utils.rl.train import DynamicFilterFn, TrainStepFns

logger = logging.getLogger(__name__)

__all__ = ["RowRequest", "run_async_rl_loop"]


SampleFactory = Callable[[int], Awaitable[RolloutSample | None]]


@dataclass
class RowRequest:
    """One dataset row's fan-out factory.

    ``sample_factory(sub_index)`` is called by the loop for each
    ``sub_index in [0, completions_per_prompt)`` and must return a fresh
    coroutine resolving to a :class:`RolloutSample` or ``None``.
    ``None`` counts as one lost sample within the row's group (the row
    can still produce a valid PromptGroup if at least
    ``GroupAssembler.min_group_size`` samples land).

    ``on_resolved`` fires exactly once per row, when the loop decides the
    row's fate: ``"accepted"``, ``"filter"`` (assembled group rejected
    by ``dynamic_filter_fn``), or ``"none"`` (no surviving samples /
    advantage_fn produced non-finite values).
    """

    row_id: Hashable
    sample_factory: SampleFactory
    row_meta: dict | None = None
    on_resolved: Callable[[str], None] | None = None


@dataclass
class _StalenessController:
    """Sample-level (LLM-call) capacity gate.

    All capacity bookkeeping is in **samples** (≈ HTTP requests to the
    inference deployment).  One row consumes ``completions_per_prompt``
    samples on submit, and one row resolution releases the same amount
    (re-credited to ``accepted_samples`` on accept, fully freed on reject).

    Unifying on the sample axis matches the deployment's natural unit
    (``max_batch_size``) and removes the implicit ``// completions_per_prompt``
    division that the previous form required at the recipe seam.  All gate
    state and arithmetic is in samples; the loop translates submission
    intent (one prompt = ``completions_per_prompt`` samples) to gate units
    at the call site.

    ``prompt_groups_per_step`` enters only as
    ``batch_size_samples = prompt_groups_per_step * completions_per_prompt``.
    """

    batch_size_samples: int            # = prompt_groups_per_step * completions_per_prompt
    completions_per_prompt: int        # samples per prompt
    max_staleness: int                 # versions
    max_concurrent_samples: int | None
    version: int = 0
    accepted_samples: int = 0
    running_samples: int = 0
    sample_fails: int = 0
    filter_drops: int = 0
    rejected_count: int = 0
    # Cumulative wall-clock time the rollout pipeline was provably
    # blocked by the off-policy version budget (i.e. waiting for the
    # trainer's next ``advance_version`` to free staleness slots).
    # Mirrors the trainer-side ``trainer_wait_for_sampler_time`` so the two sides'
    # idle-on-each-other accounting is symmetric.  For a healthy async
    # pipeline (concurrency cap < staleness budget), this stays 0.
    sampler_wait_for_trainer_total: float = 0.0
    _sampler_wait_for_trainer_start: float | None = field(default=None, repr=False)

    def staleness_capacity(self) -> int:
        # Budget = (max_off_versions + 1) batches × samples per batch per
        # version completed, minus samples already accepted into the buffer
        # or still running.
        return (
            (self.max_staleness + self.version + 1) * self.batch_size_samples
            - (self.accepted_samples + self.running_samples)
        )

    def concurrency_capacity(self) -> int | None:
        if self.max_concurrent_samples is None:
            return None
        return self.max_concurrent_samples - self.running_samples

    def capacity(self) -> int:
        """Available admissions in samples (binding min of staleness ∧ concurrency)."""
        s = self.staleness_capacity()
        c = self.concurrency_capacity()
        cap = s if c is None else min(c, s)
        return max(0, cap)

    def is_staleness_bound(self) -> bool:
        """True iff staleness is the binding constraint preventing one more prompt.

        Used to attribute idle time: a prompt's worth of staleness budget
        (cpp samples) is unavailable but concurrency could still admit one
        = waiting for the trainer to advance the version.  When both
        budgets fall below cpp samples, advancing the version alone would
        not free a slot -- the rollout pipeline is concurrency-bound, not
        trainer-bound.  Returning True there would over-attribute deployment-
        saturation wall time as ``sampler_wait_for_trainer``.
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
    sample_tasks: List[asyncio.Task] = field(default_factory=list)


async def run_async_rl_loop(
    rows: Iterable[RowRequest],
    *,
    train_fns: TrainStepFns,
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
    synchronous_training: bool = False,
) -> int | tuple[int, dict[str, Any]]:
    """Run the per-sample async RL loop.

    Args:
        rows: Iterable of :class:`RowRequest`, one per dataset row.  The
            loop calls each row's ``sample_factory`` ``completions_per_prompt``
            times to materialize the per-sample coroutines.
        train_fns: Training callbacks (see :class:`TrainStepFns`).
        completions_per_prompt: Number of samples drawn per row.
        prompt_groups_per_step: Number of accepted rows that form one
            optimizer step.
        max_head_offpolicy_versions: Off-policy budget.  ``0`` is strict
            on-policy.
        advantage_fn: Group-level advantage computation.  Default is
            GRPO z-score normalization.  Output is validated for
            non-finite values; a row producing NaN/inf advantages is
            dropped.
        with_reference: Pack reference-model datums alongside the
            policy datums (needed for KL).
        router_replay_completion_only: When a sample carries
            ``routing_matrices``, zero out prompt-position routing so
            only completion-token routing is replayed.
        min_group_size: Minimum surviving samples for a row to emit a
            PromptGroup.  Rows with fewer surviving samples are dropped.
        weight_sync_fn: Called after every ``weight_sync_interval``
            optimizer steps.  Must bump the deployment version; the
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
            "gate stalls before the next sync because all rollouts at "
            "the current version are exhausted.  Either lower "
            "weight_sync_interval to <= max_head_offpolicy_versions + 1 "
            "(so a sync arrives before capacity hits zero), or raise "
            "max_head_offpolicy_versions to >= weight_sync_interval - 1 "
            "(so the head budget covers every step between syncs)."
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

    # Each in-flight sample task -> (row_id, sub_index).
    in_flight: dict[asyncio.Task, tuple[Hashable, int]] = {}
    # Row state by row_id; lifetime spans first sample submit -> row resolution.
    rows_state: dict[Hashable, _RowState] = {}
    buffer: list[tuple[PromptGroup, int, RowRequest]] = []

    rows_iter: Iterator[RowRequest] = iter(rows)
    iterator_exhausted = False

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
            coro = request.sample_factory(sub_index)
            task = asyncio.ensure_future(coro)
            in_flight[task] = (request.row_id, sub_index)
            rows_state[request.row_id].sample_tasks.append(task)

    def _refill() -> None:
        nonlocal iterator_exhausted
        now = time.monotonic()
        if iterator_exhausted:
            # No more rows to submit -- not waiting on the trainer either,
            # just draining.  Close any open wait window.
            staleness.mark_sampler_wait_for_trainer_end(now)
            return
        # ``capacity()`` returns the binding sample-level admit budget.
        # Each prompt submission consumes ``completions_per_prompt`` samples,
        # so floor-divide here to recover how many prompts we can admit.
        slots = staleness.capacity() // completions_per_prompt
        if slots == 0:
            # Cannot fit one more prompt's worth of samples.  ``is_staleness_bound``
            # distinguishes the trainer-induced subset from ordinary deployment
            # backpressure.
            if staleness.is_staleness_bound():
                staleness.mark_sampler_wait_for_trainer_start(now)
            return
        # We can submit -- if a wait window was open, the trainer has
        # since freed budget; close it now.
        staleness.mark_sampler_wait_for_trainer_end(now)
        for _ in range(slots):
            try:
                request = next(rows_iter)
            except StopIteration:
                iterator_exhausted = True
                return
            _submit_row(request)

    def _can_make_batch() -> bool:
        if len(buffer) >= prompt_groups_per_step:
            return True
        if iterator_exhausted and not in_flight and buffer:
            return True
        return False

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
    # End time of the previous train_step (or loop start) -- the gap to the
    # next ``step_start`` is the trainer's wait-for-batch time, surfaced as
    # ``perf/wait_time_ratio`` in metrics.
    last_step_end = time.monotonic()
    # Cumulative ``sampler_wait_for_trainer_total`` snapshot at the previous step
    # boundary, used to compute the per-step delta (``sampler_wait_for_trainer_time``)
    # symmetric to ``trainer_wait_for_sampler_time``.
    prev_sampler_wait_for_trainer_total = staleness.sampler_wait_for_trainer_total

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
                sample = task.result()
                if sample is None:
                    resolution = assembler.note_dropped(row_id)
                else:
                    resolution = assembler.add_sample(row_id, sample)
                if resolution is not None:
                    _on_row_resolved(row_id, resolution)
            _refill()
            continue

        batch_pairs = buffer[:prompt_groups_per_step]
        buffer = buffer[prompt_groups_per_step:]
        batch = [pg for pg, _, _ in batch_pairs]
        versions = [v for _, v, _ in batch_pairs]
        offsets = [staleness.version - v for v in versions]

        step_start = time.monotonic()
        for _, _, request in batch_pairs:
            _resolve(request, "accepted")
        resolved_rows = (
            resolved_rows_fn()
            if resolved_rows_fn is not None
            else staleness.resolved_count(resolved_rows_offset)
        )

        # Synchronous-mode blocker: drain every remaining in-flight rollout
        # task before invoking ``train_step``.  This kills the async overlap
        # (rollouts cannot run in parallel with training) and folds any
        # straggler completions back into the buffer/staleness counters
        # naturally.  The wait-window is then opened explicitly so that
        # ``perf/sampler_wait_for_trainer_time`` reflects the train+sync wall time
        # instead of leaning on staleness saturation to detect the block.
        if synchronous_training and in_flight:
            while in_flight:
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
                    sample = task.result()
                    if sample is None:
                        resolution = assembler.note_dropped(row_id)
                    else:
                        resolution = assembler.add_sample(row_id, sample)
                    if resolution is not None:
                        _on_row_resolved(row_id, resolution)
        if synchronous_training:
            staleness.mark_sampler_wait_for_trainer_start(time.monotonic())

        # Per-step delta of "rollouts blocked on trainer to advance
        # version".  Should stay 0 in healthy async operation where the
        # concurrency cap binds before the staleness budget.  Non-zero
        # = the version budget saturated and rollouts idled waiting
        # for the previous train_step (+ weight_sync) to release slots.
        # In synchronous_training mode this captures the previous step's
        # train+sync wall time as the wait.
        sampler_wait_for_trainer_time = staleness.sampler_wait_for_trainer_total - prev_sampler_wait_for_trainer_total
        prev_sampler_wait_for_trainer_total = staleness.sampler_wait_for_trainer_total
        trainer_wait_for_sampler_time = step_start - last_step_end
        # Debug: surface the gating attribution at the step boundary so we
        # can reason about why the two wait metrics may both be large.  The
        # capacity numbers here are *post-batch-pop* and *sample-level* --
        # they reflect the state the loop is about to drive after train_step
        # returns, in the same unit as ``deployment.max_batch_size`` so you
        # can directly judge deployment-capacity utilization.
        cc = staleness.concurrency_capacity()
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
            "async/sample_fails": staleness.sample_fails,
            "async/filter_drops": staleness.filter_drops,
            "async/stale_drops": 0,
            "all_raw_rewards": [r for pg in batch for r in pg.rewards],
            "valid_prompt_groups": len(batch),
            "total_sampled": staleness.accepted_samples + staleness.rejected_count * completions_per_prompt,
            "filter_drops": staleness.filter_drops,
            "sample_fails": staleness.sample_fails,
            "stale_drops": 0,
            "resolved_rows": resolved_rows,
            "trainer_wait_for_sampler_time": trainer_wait_for_sampler_time,
            "sampler_wait_for_trainer_time": sampler_wait_for_trainer_time,
            # Sample-level gating telemetry (matches deployment.max_batch_size
            # unit -- direct utilization read).
            "async/running_samples": staleness.running_samples,
            "async/accepted_samples": staleness.accepted_samples,
            "async/staleness_capacity_at_step": staleness.staleness_capacity(),
            "async/concurrency_capacity_at_step": (
                -1 if cc is None else cc
            ),
        }

        global_step, _step_metrics = await asyncio.to_thread(
            train_fns.train_step, global_step, batch, extra_metrics,
        )
        last_step_end = time.monotonic()

        if weight_sync_fn is not None and global_step % weight_sync_interval == 0:
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

    # Close any wait window left open at shutdown (e.g. iterator drained
    # while staleness was saturated) so the cumulative total reflects
    # all measured idle-on-trainer time.
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
    }
    if return_final_stats:
        return global_step, final_stats
    return global_step
