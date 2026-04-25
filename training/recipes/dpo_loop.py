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
    - Reference RLOR job: forward only (frozen base model, for KL baseline)
    - Epoch 0: DataLoader → ref forward → train via unbounded asyncio.Queue
    - Epochs 1+: ref logprobs streamed from AppendOnlyPickleLog, no ref GPU

Usage:
    export FIREWORKS_API_KEY=...
    python -m recipes.dpo_loop
"""

from __future__ import annotations

import array
import asyncio
import logging
import os
import signal
import tempfile
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Callable

import tinker
from tqdm import tqdm

from fireworks.training.sdk import DeploymentManager, TrainerJobManager
from training.utils import (
    AppendOnlyPickleLog,
    DEFAULT_ADAM,
    DEFAULT_RENDER_WORKERS,
    DeployConfig,
    InfraConfig,
    JsonlRenderDataset,
    ReconnectableClient,
    ResourceCleanup,
    RunStatus,
    RunnerConfig,
    RunnerIO,
    WandBConfig,
    log_metrics_json,
    make_batch_dpo_loss_fn,
    make_render_dataloader,
    normalize_preference_row,
    populate_render_worker_state,
    read_api_extra_headers_env,
    render_preference_pair,
    resolve_renderer_name,
    setup_render_worker,
    setup_wandb,
    validate_config,
    wandb_finish,
    wandb_log,
)
from training.utils.checkpoint_utils import (
    CheckpointKind,
    resolve_resume,
    save_checkpoint,
    validate_warm_start_config,
)
from training.utils.rl import setup_infra
from training.utils.timer import flush_timing, timer

logger = logging.getLogger(__name__)


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
    renderer_name: str = ""

    beta: float = 0.1
    learning_rate: float = 1e-5
    epochs: int = 1
    batch_size: int = 4
    """Number of preference pairs per optimizer step."""
    grad_accum: int = 1
    """Deprecated. Ignored. Use ``batch_size`` to control the effective batch."""
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

    step_timeout: int = 0
    """Timeout in seconds for forward_backward / optim_step calls.
    0 = use DEFAULT_TIMEOUT_S from training.utils.client."""

    infra: InfraConfig = field(default_factory=InfraConfig)
    deployment: DeployConfig = field(default_factory=DeployConfig)
    dcp_save_interval: int = 0
    """Save DCP checkpoints every N steps. 0 disables."""
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="dpo-tinker"))
    policy_job_id: str | None = None
    reference_job_id: str | None = None
    init_from_checkpoint: str | None = None
    warm_start_from_adapter: str | None = None
    """GCS URI of an HF PEFT adapter directory. When set, initializes LoRA
    weights from the adapter at training start (weights-only, fresh optimizer).
    Mutually exclusive with ``init_from_checkpoint``. Requires ``lora_rank > 0``."""
    output_model_id: str | None = None
    save_final_checkpoint: bool = True
    runner: RunnerConfig = field(default_factory=RunnerConfig)


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
    _worker_id: int | None = None,
) -> None:
    """DataLoader ``worker_init_fn`` for DPO preference-pair rendering.

    Module-level (so spawn workers can pickle it) and accepts
    ``_worker_id`` so it can be passed directly after binding the other
    args via :func:`setup_render_worker`.
    """
    populate_render_worker_state(
        _pair_worker_state,
        tokenizer_model=tokenizer_model,
        renderer_name=renderer_name,
        max_seq_len=max_seq_len,
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
    total_raw_rows = len(pair_dataset)
    total_raw_batches = (total_raw_rows + batch_size - 1) // batch_size
    progress_interval = max(1, total_raw_batches // 20) if total_raw_batches else 1
    raw_rows_consumed = 0
    rendered_pairs = 0
    pairs_per_epoch = 0

    if runner is None:
        runner = RunnerIO()
    pipe: asyncio.Queue = asyncio.Queue()
    sem = asyncio.Semaphore(cfg.ref_cache_concurrency)

    def _run_train_step(epoch: int, step_pairs: list[dict[str, Any]]) -> None:
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
        optim_result = policy.optim_step(adam_params)
        step += 1

        step_metrics: dict[str, Any] = {}
        if optim_result and hasattr(optim_result, "metrics") and optim_result.metrics:
            for k, v in optim_result.metrics.items():
                step_metrics[f"train/{k}"] = v

        if cfg.dcp_save_interval > 0 and step % cfg.dcp_save_interval == 0:
            with timer("dcp_save"):
                save_checkpoint(
                    policy, f"step-{step}", cfg.log_path,
                    {
                        "step": step,
                        "data_consumed": (step - step_offset) * cfg.batch_size,
                        "source_job_id": policy.job_id,
                    },
                    kind=CheckpointKind.STATE,
                    base_model=cfg.base_model,
                    training_shape=training_shape_id,
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
        log_metrics_json(step, dpo_loss=avg_loss, margin=avg_margin, accuracy=avg_acc,
                         tokens_per_sec=tokens_per_sec)
        step_metrics.update({
            "train/step": step,
            "train/dpo_loss": avg_loss,
            "train/loss": avg_loss,  # alias for frontend compatibility
            "train/margin": avg_margin,
            "train/accuracy": avg_acc,
            "train/epoch": epoch + 1,
            "train/tokens_per_sec": tokens_per_sec,
            "train/step_time_sec": step_elapsed,
            "train/step_tokens": step_tokens,
            "train/ref_tokens": ref_tokens,
            "train/total_tokens": total_tokens,
        })
        wandb_log(step_metrics, step)
        runner.append_metrics(step, step_metrics, tokens=total_tokens)
        runner.write_status(RunStatus.RUNNING, step=step, total_steps=total_steps, message="training")
        runner.write_metadata()

    # -- Epoch 0: stream render → ref forward → training -----------------------

    pbar = tqdm(total=total_steps, desc="DPO training", unit="step")
    loader = make_render_dataloader(
        pair_dataset,
        batch_size=batch_size,
        num_workers=cfg.render_workers,
        shuffle=False,  # ref-cache iteration depends on stable producer order
        worker_init_fn=worker_init_fn,
    )

    async def _ref_producer() -> None:
        nonlocal raw_rows_consumed, rendered_pairs, pairs_per_epoch
        # The DataLoader is synchronous; pull each batch via to_thread so
        # render workers and ref forwards can overlap on the event loop.
        loader_iter = iter(loader)
        sentinel = object()
        batches_consumed = 0
        pending_pairs: list[dict[str, Any]] = []

        async def _emit_enriched(chunk: list[dict[str, Any]]) -> None:
            nonlocal pairs_per_epoch
            enriched = await _ref_forward_batch(
                chunk, reference, sem, cfg.ref_cache_batch_size,
            )
            pairs_per_epoch += len(enriched)
            if multi_epoch:
                for pair in enriched:
                    ref_cache_log.append(pair)
            await pipe.put(enriched)

        while True:
            batch = await asyncio.to_thread(next, loader_iter, sentinel)
            if batch is sentinel:
                break
            batches_consumed += 1
            raw_rows_consumed = min(batches_consumed * batch_size, total_raw_rows)
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
                await _emit_enriched(pending_pairs[:batch_size])
                pending_pairs = pending_pairs[batch_size:]
            if cfg.max_pairs is not None and pairs_per_epoch + len(pending_pairs) >= cfg.max_pairs:
                break
        if pending_pairs:
            await _emit_enriched(pending_pairs)
        await pipe.put(_DONE)

    async def _trainer() -> None:
        while True:
            item = await pipe.get()
            if item is _DONE:
                break
            await asyncio.to_thread(_run_train_step, 0, item)
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

    pbar.close()
    return step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    config: Config,
    rlor_mgr: TrainerJobManager | None = None,
    deploy_mgr: DeploymentManager | None = None,
):
    cfg = config

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

    if cfg.grad_accum > 1:
        logger.warning(
            "grad_accum is deprecated and ignored. "
            "Increase batch_size instead for larger effective batches."
        )

    setup_wandb(cfg.wandb, {
        "beta": cfg.beta,
        "lr": cfg.learning_rate,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
    })

    # -- Setup infrastructure ----------------------------------------------

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    additional_headers = read_api_extra_headers_env()

    if rlor_mgr is None:
        rlor_mgr = TrainerJobManager(
            api_key=api_key, base_url=base_url, additional_headers=additional_headers,
        )
    if deploy_mgr is None:
        deploy_mgr = DeploymentManager(
            api_key=api_key, base_url=base_url, additional_headers=additional_headers,
        )

    runner = RunnerIO(cfg.runner)
    runner.write_status(RunStatus.PENDING, message="provisioning")

    def _on_trainer_status(msg: str) -> None:
        runner.write_status(RunStatus.PENDING, message=msg)

    with runner, ResourceCleanup(rlor_mgr) as cleanup, ExitStack() as stack:
        # One call: shapes + trainers + clients. DPO doesn't need a
        # deployment/sampler/weight_syncer (needs_inference=False).
        # needs_reference=True drives the LoRA shared-session optimisation:
        # when lora_rank > 0, only the policy trainer is provisioned and
        # reference logprobs come from policy.create_base_reference().
        infra = setup_infra(
            rlor_mgr=rlor_mgr,
            deploy_mgr=None,
            base_model=cfg.base_model,
            infra_cfg=cfg.infra,
            lora_rank=cfg.lora_rank,
            max_seq_len=cfg.max_seq_len,
            learning_rate=cfg.learning_rate,
            step_timeout=cfg.step_timeout,
            policy_job_id=cfg.policy_job_id,
            reference_job_id=cfg.reference_job_id,
            needs_reference=True,
            needs_inference=False,
            role_prefix="dpo",
            api_key=api_key,
            cleanup=cleanup,
            on_status=_on_trainer_status,
        )
        for closeable in infra.closeables:
            stack.callback(closeable.close)
        runner.set_accelerator_info(profile=infra.policy_profile)

        policy = infra.policy
        reference = infra.reference
        policy_job_id = infra.policy_job_id
        reference_job_id = infra.reference_job_id

        resume_info = resolve_resume(
            policy,
            cfg.log_path,
            cfg.init_from_checkpoint,
            cfg.warm_start_from_adapter,
        )
        step_offset = resume_info.step if resume_info else 0
        wandb_log({"train/step": step_offset}, step_offset)
        adam_params = tinker.AdamParams(learning_rate=cfg.learning_rate, **DEFAULT_ADAM)

        # -- Stream-render dataset + (optional) ref cache ---------------------
        #
        # Render preference rows on the fly inside DataLoader workers so the
        # orchestrator never holds the full tokenized dataset in RAM. For
        # multi-epoch runs we additionally spool enriched pairs (datums +
        # ref logprobs) to a small append-only pickle log so the reference
        # trainer can be released after epoch 0 and epochs 1+ stream the
        # cache from disk. See fw-ai/cookbook#371 / #373 for OOM context.

        max_seq_len = infra.max_seq_len
        worker_init_fn = setup_render_worker(
            _init_pair_worker,
            cfg.tokenizer_model, cfg.renderer_name, max_seq_len,
        )

        pair_dataset = JsonlRenderDataset(cfg.dataset, _render_pair_worker)
        if len(pair_dataset) == 0:
            raise RuntimeError(f"No data found in {cfg.dataset}")

        logger.info(
            "Streaming %d raw preference rows from %s (renderer=%s, workers=%d%s)",
            len(pair_dataset), cfg.dataset,
            resolve_renderer_name(cfg.tokenizer_model, cfg.renderer_name),
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

        def _on_ref_done():
            nonlocal reference_job_id
            # LoRA shared-session: no separate trainer. The base-only handle
            # is closed via the ExitStack callback; nothing to cancel here.
            if reference_job_id is None:
                return
            logger.info("Reference forward complete — closing reference client before trainer cleanup")
            try:
                if reference is not None and hasattr(reference, "close"):
                    reference.close()
                logger.info("Reference forward complete — canceling reference trainer to free GPU")
                cleanup.cancel_trainer(reference_job_id)
                reference_job_id = None
            except Exception as e:
                logger.warning("Early cleanup of reference job %s failed: %s", reference_job_id, e)

        runner.start_training()
        step = asyncio.run(
            _train_loop(
                pair_dataset, ref_cache_log,
                reference, policy, adam_params, cfg, step_offset,
                worker_init_fn=worker_init_fn,
                on_ref_done=_on_ref_done,
                runner=runner,
                training_shape_id=infra.training_shape_id,
            )
        )

        # -- Final checkpoint --------------------------------------------------

        if cfg.save_final_checkpoint and step > step_offset:
            cp_name = f"step-{step}"
            paths = save_checkpoint(
                policy, cp_name, cfg.log_path,
                {
                    "step": step,
                    "data_consumed": (step - step_offset) * cfg.batch_size,
                    "source_job_id": policy_job_id,
                },
                kind=CheckpointKind.BOTH,
                base_model=cfg.base_model,
                training_shape=infra.training_shape_id,
            )

            if getattr(cfg, "output_model_id", None):
                if not paths.get("sampler_path"):
                    raise RuntimeError("Failed to save final base checkpoint for promotion")
                rlor_mgr.promote_checkpoint(
                    policy_job_id,
                    paths["sampler_path"],
                    cfg.output_model_id,
                    cfg.base_model,
                )
                runner.write_output_model(
                    model_id=cfg.output_model_id, checkpoint=cp_name, job_id=policy_job_id,
                )

        total_steps = step
        runner.write_status(RunStatus.COMPLETED, step=step, total_steps=total_steps, message="done")
        runner.write_metadata()
        logger.info("Training complete: %d optimizer steps (%d new)", step, step - step_offset)
        wandb_finish()
        return {"steps": step, "policy_job_id": policy_job_id, "reference_job_id": reference_job_id}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(Config(log_path="./dpo_logs"))
