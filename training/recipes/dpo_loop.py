#!/usr/bin/env python3
"""DPO training loop with pipelined ref/training overlap and fused train steps.

Optimisations:

  - **Streaming render**: preference rows are rendered to Datums on the fly
    inside DataLoader workers (PyTorch ``spawn``), so the orchestrator
    never holds the full tokenized dataset in RAM. Same machinery as
    ``sft_loop``; see fw-ai/cookbook#371.
  - **Pipelined ref/training**: reference logprobs are computed
    concurrently with policy training via a producer/consumer pipeline.
    The reference trainer is deleted as soon as all ref forwards
    complete -- even while training continues.
  - **Append-only ref cache**: for multi-epoch runs, enriched pairs
    (datums + ref logprobs) are spilled to a tiny pickle log on disk so
    epochs 1+ stream them sequentially without re-running the reference
    forward. The log is sequential-only -- no random access, no offset
    table -- which is sufficient for "iterate front-to-back per epoch".
  - **Client-side fused train steps**: all datums for one optimizer window
    are sent in a single ``forward_backward_custom`` call so the backend
    can batch the whole step more efficiently.

Architecture:
    - Policy RLOR job:    forward_backward_custom + optim_step (trainable)
    - Reference RLOR job: frozen base model runtime for KL baseline
    - Epoch 0: DataLoader → ref forward → train via unbounded asyncio.Queue
    - Epochs 1+: ref logprobs streamed from AppendOnlyPickleLog, no ref GPU

Usage:
    export FIREWORKS_API_KEY=...
    python -m recipes.dpo_loop
"""

from __future__ import annotations

import array
import asyncio
import functools
import logging
import os
import signal
import tempfile
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Callable

import tinker
import torch
from fireworks.training.sdk.training_spec import (
    LRSchedulerSpec,
    compute_lr,
    default_constant_schedule,
    normalize_lr_scheduler_spec,
)
from tqdm import tqdm

from training.utils import (
    DEFAULT_ADAM,
    DEFAULT_RENDER_WORKERS,
    AppendOnlyPickleLog,
    DeployConfig,
    TrainerConfig,
    JsonlRenderDataset,
    RawRowCursor,
    ReconnectableClient,
    RunnerConfig,
    RunnerIO,
    RunStatus,
    WandBConfig,
    build_service_client,
    log_metrics_json,
    make_batch_dpo_loss_fn,
    make_render_dataloader,
    normalize_preference_row,
    populate_render_worker_state,
    read_api_extra_headers_env,
    render_preference_pair,
    resolve_renderer_snapshot,
    setup_wandb,
    validate_config,
    wandb_finish,
    wandb_log,
)
from training.utils.checkpoints import TrainingCheckpoints, validate_warm_start_config
from training.utils.runner_state import write_completed, write_running_step
from training.utils.timer import flush_timing, timer

logger = logging.getLogger(__name__)

# Fixed seed for the length-grouped batch order. DPO builds the ref cache in
# producer order in epoch 0 and replays it in epochs 1+, and epoch-0 resume
# re-streams from the start, so the loader order MUST be reproducible. A
# constant seed keeps the grouped order deterministic across runs/resume while
# still de-correlating batch length from training step.
_DPO_LENGTH_GROUP_SEED = 0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    log_path: str
    """Directory for checkpoints and logs. Required, no default."""

    base_model: str = "accounts/fireworks/models/qwen3-8b"
    dataset: str = ""
    tokenizer_model: str = ""  # HuggingFace model name for client-side tokenization
    tokenizer_revision: str = ""  # Optional HuggingFace revision for client-side tokenization
    tokenizer_trust_remote_code: bool | None = None
    """Reviewed remote-code policy; ``None`` retains legacy behavior."""
    renderer_name: str = ""
    renderer_name_is_resolved: bool = False
    """Whether ``renderer_name`` is a Managed Training materialized snapshot."""
    thinking_trace_history_mode: str = ""
    """Semantic effective mode (``interleaved``/``preserved``) for auditability."""

    beta: float = 0.1
    learning_rate: float = 1e-5
    weight_decay: float = DEFAULT_ADAM["weight_decay"]
    """Adam weight decay. Defaults to the shared cookbook Adam default."""
    lr_scheduler: LRSchedulerSpec = field(default_factory=default_constant_schedule)
    """Per-step LR scheduler spec for managed and local DPO runs."""

    epochs: int = 1
    batch_size: int = 4
    """Number of preference pairs per optimizer step. For managed (V2) jobs
    this is set from ``BaseTrainingConfig.batch_size_samples`` via the
    cookbook orchestrator."""
    max_seq_len: int | None = None
    max_pairs: int | None = None
    """Cap on *valid rendered pairs* after schema/length filtering."""
    lora_rank: int = 0
    ref_cache_concurrency: int = 16
    """Max concurrent reference forward passes during cache warm-up."""
    ref_cache_batch_size: int = 1
    """Number of preference pairs per reference forward call during caching."""
    render_workers: int = DEFAULT_RENDER_WORKERS
    """Number of DataLoader workers for streaming render. <=1 = in-process."""

    group_by_length: bool = False
    """Compose each batch from similarly-sized preference pairs (bucket-then-
    shuffle on the whole-row byte-length proxy, which covers chosen+rejected)
    instead of file order. Cuts padding waste and, under context parallel, lets
    most batches run at a low CP degree. Bucket *order* is shuffled with a fixed
    seed so the ref-cache producer order stays deterministic and resumable.
    Keep off for base64-image multimodal data where byte length is a poor proxy.
    Batch count and resume/cursor semantics are unchanged."""

    length_group_factor: int = 50
    """Mega-batch multiplier for ``group_by_length``: a permutation is cut into
    windows of ``batch_size * length_group_factor`` that are sorted by length
    before chunking into batches. Larger -> tighter length homogeneity (less
    padding / lower CP) but weaker shuffling; smaller -> looser grouping."""

    step_timeout: int = 0
    """Timeout in seconds for forward_backward / optim_step calls.
    0 = use DEFAULT_TIMEOUT_S from training.utils.client."""

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    deployment: DeployConfig = field(default_factory=DeployConfig)
    dcp_save_interval: int = 0
    """Save DCP checkpoints every N steps. 0 disables."""
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="dpo-tinker"))
    init_from_checkpoint: str | None = None
    warm_start_from_adapter: str | None = None
    """GCS URI of an HF PEFT adapter directory. When set, initializes LoRA
    weights from the adapter at training start (weights-only, fresh optimizer).
    Mutually exclusive with ``init_from_checkpoint``. Requires ``lora_rank > 0``."""
    output_model_id: str | None = None
    save_final_checkpoint: bool = True
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    cleanup_on_exit: bool = True
    """Clean up SDK-created trainer resources on close."""

    release_reference_after_cache: bool = True
    """Release SDK-owned separate references after reference logprobs are cached.

    Leave enabled for normal DPO runs to reduce GPU cost. Sequential live E2E
    runs may disable it together with
    ``trainer.cleanup_reference_on_close=False`` so a full-param reference
    trainer can be reused by later phases via ``TrainerConfig.reference_job_id``.
    """


def _validate_dpo_beta(beta: float) -> None:
    if not 0 < beta < 0.5:
        raise ValueError("Config.beta must be > 0 and < 0.5")


# ---------------------------------------------------------------------------
# Per-worker render: tokenizer + renderer cached in module-level state
# ---------------------------------------------------------------------------


# Populated once per process: in the parent (for num_workers <= 1 fallback)
# and in each spawn worker via ``_init_pair_worker``. Mirrors sft_loop.py.
_pair_worker_state: dict = {}


def _init_pair_worker(
    tokenizer_model: str,
    renderer_name: str,
    max_seq_len: int,
    tokenizer_revision: str = "",
    tokenizer_trust_remote_code: bool | None = None,
    _worker_id: int | None = None,
) -> None:
    """DataLoader ``worker_init_fn`` for DPO preference-pair rendering.

    Module-level (so spawn workers can pickle it) and accepts
    ``_worker_id`` so it can be used as a DataLoader ``worker_init_fn``.
    """
    populate_render_worker_state(
        _pair_worker_state,
        tokenizer_model=tokenizer_model,
        tokenizer_revision=tokenizer_revision,
        tokenizer_trust_remote_code=tokenizer_trust_remote_code,
        renderer_name=renderer_name,
        max_seq_len=max_seq_len,
        renderer_name_is_resolved=True,
    )


def _render_pair_worker(row: dict[str, Any]) -> dict[str, Any] | None:
    """Render one JSONL row to a Datum-pair dict, or ``None`` to drop.

    Combines schema normalisation (chosen / rejected / samples / OpenAI),
    rendering, and over-length filtering. Returning ``None`` covers all
    drop reasons; the DataLoader's collate filters Nones out of each batch.
    """
    pair = normalize_preference_row(row)
    if pair is None:
        return None
    rendered = render_preference_pair(
        pair["chosen"], pair["rejected"],
        renderer=_pair_worker_state["renderer"],
        tokenizer=_pair_worker_state["tokenizer"],
    )
    if rendered is None:
        return None
    max_seq_len = _pair_worker_state["max_seq_len"]
    if (
        len(rendered.chosen_tokens) > max_seq_len
        or len(rendered.rejected_tokens) > max_seq_len
    ):
        return None
    return {
        "chosen_tokens_len": len(rendered.chosen_tokens),
        "rejected_tokens_len": len(rendered.rejected_tokens),
        "response_start": rendered.response_start,
        "chosen_datum": rendered.chosen_datum,
        "rejected_datum": rendered.rejected_datum,
    }


# ---------------------------------------------------------------------------
# Reference forward
# ---------------------------------------------------------------------------


async def _ref_forward_batch(
    pairs: list[dict[str, Any]],
    reference: ReconnectableClient,
    semaphore: asyncio.Semaphore,
    ref_batch_size: int,
) -> list[dict[str, Any]]:
    """Compute reference logprobs for *pairs*, sub-batched by *ref_batch_size*.

    Returns enriched pairs with ``ref_chosen`` / ``ref_rejected`` logprobs
    attached as :class:`array.array` of ``'f'`` (4 bytes/token vs 28+ for a
    Python ``list[float]``) — both ``torch.tensor`` and pickle handle them
    natively. Returned pairs preserve input order so multi-epoch ref-cache
    iteration keeps producer-order consistent with epoch 0.
    """
    sub_batches = [pairs[i:i + ref_batch_size] for i in range(0, len(pairs), ref_batch_size)]

    async def _process_sub_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        datums: list[tinker.Datum] = []
        for pair_data in batch:
            datums.append(pair_data["chosen_datum"])
            datums.append(pair_data["rejected_datum"])

        async with semaphore:
            fwd = await asyncio.to_thread(
                lambda d=datums: reference.forward(d, "cross_entropy")
            )

        results: list[dict[str, Any]] = []
        for j, pair_data in enumerate(batch):
            results.append({
                "chosen_tokens_len": pair_data["chosen_tokens_len"],
                "rejected_tokens_len": pair_data["rejected_tokens_len"],
                "chosen_datum": pair_data["chosen_datum"],
                "rejected_datum": pair_data["rejected_datum"],
                "ref_chosen": array.array("f", fwd.loss_fn_outputs[2 * j]["logprobs"].data),
                "ref_rejected": array.array("f", fwd.loss_fn_outputs[2 * j + 1]["logprobs"].data),
                "response_start": pair_data["response_start"],
            })
        return results

    batch_results = await asyncio.gather(*[_process_sub_batch(b) for b in sub_batches])
    return [pair for batch in batch_results for pair in batch]


# ---------------------------------------------------------------------------
# Batched training loop
# ---------------------------------------------------------------------------


def _forward_backward_pairs(
    batch_pairs: list[dict[str, Any]],
    policy: ReconnectableClient,
    beta: float,
) -> Any:
    """Run forward_backward_custom on a batch of preference pairs.

    Arranges datums as [chosen_0, rejected_0, chosen_1, rejected_1, ...].
    """
    datums: list[tinker.Datum] = []
    ref_chosen_list: list[list[float]] = []
    ref_rejected_list: list[list[float]] = []
    response_starts: list[int] = []

    for cached in batch_pairs:
        datums.append(cached["chosen_datum"])
        datums.append(cached["rejected_datum"])
        ref_chosen_list.append(cached["ref_chosen"])
        ref_rejected_list.append(cached["ref_rejected"])
        response_starts.append(cached["response_start"])

    loss_fn = make_batch_dpo_loss_fn(
        ref_chosen_list, ref_rejected_list, response_starts, beta,
    )
    return policy.forward_backward_custom(datums, loss_fn)


_DONE = object()


async def _train_loop(
    pair_dataset: JsonlRenderDataset,
    ref_cache_log: AppendOnlyPickleLog | None,
    reference: ReconnectableClient,
    policy: ReconnectableClient,
    adam_params: tinker.AdamParams,
    cfg: Config,
    step_offset: int,
    *,
    cursor: RawRowCursor,
    ckpt: TrainingCheckpoints | None = None,
    worker_init_fn: Callable[[int], None] | None = None,
    on_ref_done: Callable[[], None] | None = None,
    runner: RunnerIO | None = None,
    training_shape_id: str | None = None,
) -> int:
    """Pipelined DPO training -- streaming render + ref forward overlap policy.

    Epoch 0 runs a producer/consumer pipeline with an unbounded queue:
    a DataLoader streams rendered preference batches; the producer
    computes reference logprobs at full speed (concurrent via semaphore)
    and pushes enriched batches to the consumer that runs train steps.
    Once the producer finishes, *on_ref_done* fires to delete the
    reference trainer immediately -- even while training continues.

    For multi-epoch runs, ``ref_cache_log`` (must be opened by the caller)
    captures every enriched pair in producer order so epochs 1+ stream
    them sequentially without holding the cache in RAM.
    """
    multi_epoch = cfg.epochs > 1
    if multi_epoch and ref_cache_log is None:
        raise ValueError("ref_cache_log is required when cfg.epochs > 1")

    batch_size = cfg.batch_size
    step = step_offset
    rough_pairs_per_epoch = (
        min(len(pair_dataset), cfg.max_pairs)
        if cfg.max_pairs is not None
        else len(pair_dataset)
    )
    total_steps = ((rough_pairs_per_epoch + batch_size - 1) // batch_size) * cfg.epochs
    lr_scheduler = normalize_lr_scheduler_spec(cfg.lr_scheduler)
    total_raw_rows = len(pair_dataset)
    total_raw_batches = (total_raw_rows + batch_size - 1) // batch_size
    progress_interval = max(1, total_raw_batches // 20) if total_raw_batches else 1
    raw_rows_consumed = 0
    rendered_pairs = 0
    pairs_per_epoch = 0

    if runner is None:
        runner = RunnerIO()
    adam_kwargs = adam_params.model_dump()
    adam_kwargs.pop("learning_rate", None)
    pipe: asyncio.Queue = asyncio.Queue()
    sem = asyncio.Semaphore(cfg.ref_cache_concurrency)

    def _run_train_step(
        epoch: int,
        step_pairs: list[dict[str, Any]],
        *,
        data_consumed: int | None = None,
    ) -> None:
        nonlocal step
        step_t0 = time.monotonic()
        step_tokens = sum(
            p["chosen_tokens_len"] + p["rejected_tokens_len"]
            for p in step_pairs
        )
        # Epoch 0: ref model processed these same sequences; epochs 1+ use cached ref logprobs.
        ref_tokens = step_tokens if epoch == 0 else 0
        total_tokens = step_tokens + ref_tokens

        with timer("fwd_bwd"):
            fwd_bwd_result = _forward_backward_pairs(step_pairs, policy, cfg.beta)
        step_lr = compute_lr(
            lr_scheduler,
            step=step + 1,
            base_lr=cfg.learning_rate,
            total_steps=total_steps,
        )
        step_adam_params = tinker.AdamParams(learning_rate=step_lr, **adam_kwargs)
        optim_result = policy.optim_step(step_adam_params)
        step += 1

        step_metrics: dict[str, Any] = {}
        if optim_result and hasattr(optim_result, "metrics") and optim_result.metrics:
            for k, v in optim_result.metrics.items():
                step_metrics[f"train/{k}"] = v

        if cfg.dcp_save_interval > 0 and step % cfg.dcp_save_interval == 0:
            with timer("dcp_save"):
                ckpt.save(
                    f"step-{step}",
                    resumable=True,
                    promotable=False,
                    data_consumed=(
                        cursor.value if data_consumed is None else data_consumed
                    ),
                )

        step_elapsed = time.monotonic() - step_t0
        tokens_per_sec = step_tokens / step_elapsed if step_elapsed > 0 else 0.0
        step_metrics.update(flush_timing())

        fwd_metrics = fwd_bwd_result.metrics
        avg_loss = fwd_metrics["dpo_loss"]
        avg_margin = fwd_metrics["margin"]
        avg_acc = fwd_metrics["accuracy"]
        logger.info(
            "Step %d/%d | Loss: %.4f | Margin: %+.4f | Acc: %.1f%% | "
            "policy=%d ref=%d tok | %.1f tok/s (%.1fs)",
            step, total_steps, avg_loss, avg_margin, avg_acc * 100,
            step_tokens, ref_tokens, tokens_per_sec, step_elapsed,
        )
        current_lr = step_lr
        log_metrics_json(step, dpo_loss=avg_loss, margin=avg_margin, accuracy=avg_acc,
                         tokens_per_sec=tokens_per_sec, lr=current_lr)
        step_metrics.update({
            "train/step": step,
            "train/dpo_loss": avg_loss,
            "train/loss": avg_loss,
            "train/margin": avg_margin,
            "train/accuracy": avg_acc,
            "train/epoch": epoch + 1,
            "train/tokens_per_sec": tokens_per_sec,
            "train/step_time_sec": step_elapsed,
            "train/step_tokens": step_tokens,
            "train/ref_tokens": ref_tokens,
            "train/total_tokens": total_tokens,
        })
        step_metrics["train/lr"] = current_lr
        wandb_log(step_metrics, step)
        write_running_step(
            runner,
            step=step,
            total_steps=total_steps,
            metrics=step_metrics,
            tokens=step_tokens,
        )

    # -- Epoch 0: stream render → ref forward → training -----------------------

    pbar = tqdm(total=total_steps, desc="DPO training", unit="step")
    # Default path: stable file order (shuffle=False) so the ref cache producer
    # order is reproducible. Length-grouped path: bucket by the byte-length
    # proxy and shuffle batch *order* with a FIXED seed -- still fully
    # deterministic (so ref-cache replay + epoch-0 resume stay valid) but
    # length is de-correlated from training step instead of file order.
    group_by_length = cfg.group_by_length
    loader_generator = (
        torch.Generator().manual_seed(_DPO_LENGTH_GROUP_SEED) if group_by_length else None
    )
    loader = make_render_dataloader(
        pair_dataset,
        batch_size=batch_size,
        num_workers=cfg.render_workers,
        shuffle=group_by_length,  # grouped path uses seeded shuffle; else file order
        generator=loader_generator,
        worker_init_fn=worker_init_fn,
        group_by_length=group_by_length,
        length_group_factor=cfg.length_group_factor,
        sizes=pair_dataset.approx_row_sizes() if group_by_length else None,
    )

    async def _ref_producer() -> None:
        nonlocal raw_rows_consumed, rendered_pairs, pairs_per_epoch
        # The DataLoader is synchronous; pull each batch via to_thread so
        # render workers and ref forwards can overlap on the event loop.
        loader_iter = iter(loader)
        sentinel = object()
        batches_consumed = 0
        pending_pairs: list[dict[str, Any]] = []

        async def _emit_enriched(
            chunk: list[dict[str, Any]],
            *,
            data_consumed: int,
        ) -> None:
            nonlocal pairs_per_epoch
            enriched = await _ref_forward_batch(
                chunk, reference, sem, cfg.ref_cache_batch_size,
            )
            pairs_per_epoch += len(enriched)
            if multi_epoch:
                for pair in enriched:
                    ref_cache_log.append(pair)
            await pipe.put((enriched, data_consumed))

        while True:
            batch = await asyncio.to_thread(next, loader_iter, sentinel)
            if batch is sentinel:
                break
            batches_consumed += 1
            delta = min(batch_size, total_raw_rows - raw_rows_consumed)
            raw_rows_consumed += delta
            cursor.record(delta)
            if (
                total_raw_rows > 0
                and (
                    batches_consumed % progress_interval == 0
                    or raw_rows_consumed == total_raw_rows
                )
            ):
                runner.report_rendering_progress(
                    raw_rows_consumed,
                    total_raw_rows,
                    label="rendering/ref cache",
                )
            rendered_pairs += len(batch)
            if not batch:  # all rows in this DataLoader batch rendered to None
                continue
            if cfg.max_pairs is not None:
                remaining = cfg.max_pairs - pairs_per_epoch - len(pending_pairs)
                if remaining <= 0:
                    break
                batch = batch[:remaining]
                if not batch:
                    break
            pending_pairs.extend(batch)
            while len(pending_pairs) >= batch_size:
                await _emit_enriched(
                    pending_pairs[:batch_size],
                    data_consumed=cursor.value,
                )
                pending_pairs = pending_pairs[batch_size:]
            if cfg.max_pairs is not None and pairs_per_epoch + len(pending_pairs) >= cfg.max_pairs:
                break
        if pending_pairs:
            await _emit_enriched(pending_pairs, data_consumed=cursor.value)
        await pipe.put(_DONE)

    async def _trainer() -> None:
        while True:
            item = await pipe.get()
            if item is _DONE:
                break
            step_pairs, data_consumed = item
            await asyncio.to_thread(
                _run_train_step,
                0,
                step_pairs,
                data_consumed=data_consumed,
            )
            pbar.update(1)

    producer = asyncio.create_task(_ref_producer())
    consumer = asyncio.create_task(_trainer())
    await producer
    if on_ref_done is not None:
        await asyncio.to_thread(on_ref_done)
    await consumer

    filtered_count = max(0, raw_rows_consumed - rendered_pairs)
    if filtered_count > 0:
        logger.info(
            "Seq-length / format filter: %d/%d raw rows filtered",
            filtered_count,
            raw_rows_consumed,
        )
    if rendered_pairs == 0 or pairs_per_epoch == 0:
        raise RuntimeError("No valid pairs after tokenization")

    total_steps = ((pairs_per_epoch + batch_size - 1) // batch_size) * cfg.epochs
    pbar.total = total_steps
    pbar.refresh()

    # -- Epochs 1+: stream cached ref logprobs from disk -----------------------

    if multi_epoch:
        ref_cache_log.close_write()
        n_cached = len(ref_cache_log)
        logger.info(
            "Ref cache built: %d pairs / %.1f GiB on disk",
            n_cached, ref_cache_log.disk_size_bytes() / (1024 ** 3),
        )
        for epoch in range(1, cfg.epochs):
            chunk: list[dict[str, Any]] = []
            for pair in ref_cache_log:
                chunk.append(pair)
                if len(chunk) == batch_size:
                    _run_train_step(epoch, chunk)
                    pbar.update(1)
                    chunk = []
            if chunk:
                _run_train_step(epoch, chunk)
                pbar.update(1)
            # Replay re-consumes source rows; cache holds only post-filter pairs.
            cursor.record(total_raw_rows)

    pbar.close()
    return step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    config: Config,
):
    cfg = config
    _validate_dpo_beta(cfg.beta)

    def _signal_handler(signum, frame):
        name = signal.Signals(signum).name
        logger.warning("Received %s — raising SystemExit for cleanup", name)
        raise SystemExit(f"Terminated by {name}")

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    validate_config(
        cfg.base_model,
        cfg.dataset,
        deploy=cfg.deployment,
        output_model_id=cfg.output_model_id,
    )
    validate_warm_start_config(
        warm_start_from_adapter=cfg.warm_start_from_adapter,
        init_from_checkpoint=cfg.init_from_checkpoint,
        lora_rank=cfg.lora_rank,
    )
    if not cfg.tokenizer_model:
        raise ValueError(
            "Config.tokenizer_model is required for client-side tokenization. "
            "Set it to the HuggingFace model name (e.g. 'Qwen/Qwen3-1.7B')."
        )
    # Resolve a direct request or reuse the persisted concrete renderer before
    # provisioning either GPU service. Chosen and rejected are then rendered by
    # the same concrete renderer in every worker.
    resolved_renderer_name = resolve_renderer_snapshot(
        tokenizer_model=cfg.tokenizer_model,
        renderer_name=cfg.renderer_name,
        thinking_trace_history_mode=cfg.thinking_trace_history_mode,
        renderer_name_is_resolved=cfg.renderer_name_is_resolved,
    )

    lr_scheduler = normalize_lr_scheduler_spec(cfg.lr_scheduler)
    cfg.lr_scheduler = lr_scheduler
    setup_wandb(cfg.wandb, {
        "beta": cfg.beta,
        "lr": cfg.learning_rate,
        "lr_schedule": lr_scheduler.type,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
    })

    # -- SDK-managed Tinker clients ----------------------------------------

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    additional_headers = read_api_extra_headers_env()

    runner = RunnerIO(cfg.runner)
    runner.write_status(RunStatus.PENDING, message="provisioning")

    with runner, ExitStack() as stack:
        service = build_service_client(
            api_key=api_key,
            base_url=base_url,
            additional_headers=additional_headers,
            base_model=cfg.base_model,
            tokenizer_model=cfg.tokenizer_model,
            lora_rank=cfg.lora_rank,
            max_context_length=cfg.max_seq_len,
            learning_rate=cfg.learning_rate,
            trainer=cfg.trainer,
            reference_required=True,
            cleanup_trainer_on_close=cfg.cleanup_on_exit,
        )
        stack.callback(service.close)
        training_client = service.create_training_client(cfg.base_model, lora_rank=cfg.lora_rank)
        runner.set_accelerator_info(
            service.accelerator_type,
            service.accelerator_count,
            profile=service.training_profile,
        )
        policy_job_id = service.trainer_job_id
        max_seq_len = service.max_context_length

        policy = ReconnectableClient.from_training_client(
            training_client,
            base_model=cfg.base_model,
            lora_rank=cfg.lora_rank,
            job_id=policy_job_id,
            default_timeout=cfg.step_timeout or 3600,
            service=service,
        )
        # DPO always needs a reference. The SDK owns the shared-vs-separate
        # decision: LoRA without an explicit reference shape reuses the policy
        # session; full-param (or an explicit reference_training_shape_id)
        # provisions a separate frozen reference trainer that `service` owns.
        # Backend trainer creation selects a LoRA-capable shape unless
        # cfg.trainer.reference_training_shape_id pins a LoRA-capable shape.
        reference = ReconnectableClient.from_training_client(
            service.create_reference_client(cfg.base_model, lora_rank=cfg.lora_rank),
            base_model=cfg.base_model,
            lora_rank=0,
            job_id=service.reference_client_job_id,
            default_timeout=cfg.step_timeout or 3600,
            service=service,
            base_only=True,
        )
        reference_job_id = service.reference_trainer_job_id

        ckpt = TrainingCheckpoints(
            policy,
            service,
            trainer_id=policy_job_id,
            log_path=cfg.log_path,
            lora_rank=cfg.lora_rank,
        )

        resume_info = ckpt.resume(
            init_from_checkpoint=cfg.init_from_checkpoint,
            warm_start_from_adapter=cfg.warm_start_from_adapter,
        )
        step_offset = resume_info.step if resume_info else 0
        wandb_log({"train/step": step_offset}, step_offset)
        adam_kwargs = dict(DEFAULT_ADAM)
        adam_kwargs["weight_decay"] = cfg.weight_decay
        adam_params = tinker.AdamParams(learning_rate=cfg.learning_rate, **adam_kwargs)

        # -- Stream-render dataset + (optional) ref cache ---------------------
        #
        # Render preference rows on the fly inside DataLoader workers so the
        # orchestrator never holds the full tokenized dataset in RAM. For
        # multi-epoch runs we additionally spool enriched pairs (datums +
        # ref logprobs) to a small append-only pickle log so the reference
        # trainer can be released after epoch 0 and epochs 1+ stream the
        # cache from disk. See fw-ai/cookbook#371 / #373 for OOM context.

        init_args = (
            cfg.tokenizer_model,
            resolved_renderer_name,
            max_seq_len,
            cfg.tokenizer_revision,
            cfg.tokenizer_trust_remote_code,
        )
        _init_pair_worker(*init_args)
        worker_init_fn = functools.partial(_init_pair_worker, *init_args)

        pair_dataset = JsonlRenderDataset(cfg.dataset, _render_pair_worker)
        if len(pair_dataset) == 0:
            raise RuntimeError(f"No data found in {cfg.dataset}")

        logger.info(
            "Streaming %d raw preference rows from %s (renderer=%s, workers=%d%s)",
            len(pair_dataset), cfg.dataset,
            resolved_renderer_name,
            cfg.render_workers,
            (
                f", max_pairs={cfg.max_pairs}"
                if cfg.max_pairs is not None
                else ""
            ),
        )

        ref_cache_log: AppendOnlyPickleLog | None = None
        if cfg.epochs > 1:
            ref_cache_dir = stack.enter_context(
                tempfile.TemporaryDirectory(prefix="dpo_ref_")
            )
            ref_cache_log = stack.enter_context(
                AppendOnlyPickleLog(os.path.join(ref_cache_dir, "ref_cache.pkl"))
            )

        def _on_ref_done() -> None:
            # Free the separate reference trainer (if any) as soon as all
            # reference forwards are done, while policy training continues.
            # No-op when the reference shared the policy session (LoRA).
            nonlocal reference_job_id
            if not cfg.release_reference_after_cache:
                return
            service.release_references()
            reference_job_id = None

        cursor = RawRowCursor(max_rows=len(pair_dataset) * cfg.epochs)
        cursor.resume(resume_info.data_consumed if resume_info else None)
        runner.start_training()
        step = asyncio.run(
            _train_loop(
                pair_dataset, ref_cache_log,
                reference, policy, adam_params, cfg, step_offset,
                cursor=cursor,
                ckpt=ckpt,
                worker_init_fn=worker_init_fn,
                on_ref_done=_on_ref_done,
                runner=runner,
                training_shape_id=cfg.trainer.training_shape_id,
            )
        )

        # -- Final checkpoint --------------------------------------------------

        if cfg.save_final_checkpoint and step > step_offset:
            cp_name = f"step-{step}"
            ckpt.save(
                cp_name,
                resumable=True,
                promotable=True,
                data_consumed=cursor.value,
            )

            if getattr(cfg, "output_model_id", None):
                ckpt.promote_latest(cfg.output_model_id, cfg.base_model)
                runner.write_output_model(
                    model_id=cfg.output_model_id, checkpoint=cp_name, job_id=policy_job_id,
                )

        total_steps = step
        write_completed(runner, step=step, total_steps=total_steps)
        logger.info("Training complete: %d optimizer steps (%d new)", step, step - step_offset)
        wandb_finish()
        return {"steps": step, "policy_job_id": policy_job_id, "reference_job_id": reference_job_id}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(Config(log_path="./dpo_logs"))
