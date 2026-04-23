#!/usr/bin/env python3
"""DPO training loop with pipelined ref/training overlap and fused train steps.

Optimisations:

  - **Pipelined ref/training**: reference logprobs are computed
    concurrently with policy training via a producer/consumer pipeline.
    The reference trainer is deleted as soon as all ref forwards
    complete -- even while training continues.
  - **Client-side fused train steps**: all datums for one optimizer window
    are sent in a single ``forward_backward_custom`` call so the backend can
    batch the whole step more efficiently.

Architecture:
    - Policy RLOR job:    forward_backward_custom + optim_step (trainable)
    - Reference RLOR job: forward only (frozen base model, for KL baseline)
    - Epoch 0: ref forward and training overlap via unbounded asyncio.Queue
    - Epochs 1+: ref logprobs cached from epoch 0, no ref GPU needed

Usage:
    export FIREWORKS_API_KEY=...
    python -m recipes.dpo_loop
"""

from __future__ import annotations

import array
import os
import tempfile
import time
import signal
import asyncio
import logging
from contextlib import ExitStack
from typing import Any, Callable
import dataclasses
from dataclasses import field, dataclass
from concurrent.futures import ThreadPoolExecutor

import tinker
import transformers
from tqdm import tqdm

from fireworks.training.sdk import DeploymentManager, TrainerJobManager
from training.utils import (
    DEFAULT_ADAM,
    DEFAULT_RENDER_CHUNKSIZE,
    DEFAULT_RENDER_WORKERS,
    DiskBackedDatumStore,
    InfraConfig,
    MemTracer,
    ResourceCleanup,
    RunnerConfig,
    RunnerIO,
    RunStatus,
    WandBConfig,
    DeployConfig,
    ReconnectableClient,
    count_jsonl_rows,
    iter_preference_examples,
    stream_render_to_store,
    wandb_log,
    setup_wandb,
    wandb_finish,
    validate_config,
    log_metrics_json,
    make_batch_dpo_loss_fn,
    read_api_extra_headers_env,
    build_renderer,
    render_preference_pair,
    resolve_renderer_name,
)
from training.utils.checkpoint_utils import (
    resolve_resume,
    save_checkpoint,
    validate_warm_start_config,
    CheckpointKind,
)
from training.utils.rl import setup_infra
from training.utils.timer import timer, flush_timing

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
    lora_rank: int = 0

    ref_cache_concurrency: int = 16
    """Max concurrent reference forward passes during cache warm-up."""
    ref_cache_batch_size: int = 1
    """Number of preference pairs per reference forward call during caching."""

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
    """Optional orchestration outputs written during training.

    Paths can be set here or via environment variables:
      COOKBOOK_STATUS_FILE      -- training status + progress (JSON, overwritten each step)
      COOKBOOK_METADATA_FILE    -- tokens processed + accelerator-seconds (JSON)
      COOKBOOK_METRICS_FILE     -- per-step metrics (JSONL, appended each step)
      COOKBOOK_OUTPUT_MODEL_PATH -- final model info written on completion (JSON)

    All paths are optional; unset paths are silently skipped.
    See training/utils/runner.py for file format details.
    """


# ---------------------------------------------------------------------------
# Tokenization and reference forward
# ---------------------------------------------------------------------------


# Module-level worker state for the spawn pool. Populated once per worker via
# `_init_pair_worker` so we don't re-pickle the tokenizer on every task.
# Mirrors the sft_loop.py pattern; see fw-ai/cookbook#371.
_pair_worker_state: dict = {}


def _init_pair_worker(tokenizer_model: str, renderer_name: str, max_seq_len: int) -> None:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_model, trust_remote_code=True,
    )
    _pair_worker_state.update(
        renderer=build_renderer(tokenizer, tokenizer_model, renderer_name),
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
    )


def _render_pair(
    example: dict[str, Any], *, renderer: Any, tokenizer: Any, max_seq_len: int,
) -> dict[str, Any] | None:
    """Render a preference example to a Datum-pair dict, or None if invalid / too long.

    Combines the original ``"filtered"`` (too long) and ``None`` (render failed)
    cases into a single None return so this fits the
    :func:`stream_render_to_store` ``T | None`` contract; the caller logs the
    combined drop count.

    Stores ``chosen_tokens_len`` / ``rejected_tokens_len`` instead of the full
    token lists — the train loop only needs the lengths for tokens-per-second
    accounting, and the actual tokens already live inside the datums.
    """
    pair = render_preference_pair(
        example["chosen"], example["rejected"],
        renderer=renderer, tokenizer=tokenizer,
    )
    if pair is None:
        return None
    if len(pair.chosen_tokens) > max_seq_len or len(pair.rejected_tokens) > max_seq_len:
        return None
    return {
        "chosen_tokens_len": len(pair.chosen_tokens),
        "rejected_tokens_len": len(pair.rejected_tokens),
        "response_start": pair.response_start,
        "chosen_datum": pair.chosen_datum,
        "rejected_datum": pair.rejected_datum,
    }


def _render_pair_worker(example: dict[str, Any]) -> dict[str, Any] | None:
    return _render_pair(example, **_pair_worker_state)


async def _ref_forward_batch(
    pairs: list[dict[str, Any]],
    reference: ReconnectableClient,
    semaphore: asyncio.Semaphore,
    ref_batch_size: int,
) -> list[dict[str, Any]]:
    """Compute reference logprobs for *pairs*, sub-batched by *ref_batch_size*.

    Uses the semaphore for concurrency control. Returns enriched pairs
    with ``ref_chosen`` / ``ref_rejected`` logprobs attached as
    :class:`array.array` of ``'f'`` (4 bytes/token vs 28+ bytes for a Python
    ``list[float]``) — both ``torch.tensor`` and pickle handle them natively.

    Returned pairs preserve the input order. The producer relies on this
    when appending to ``ref_cache_store`` so that epochs 1+ can iterate the
    cache sequentially and get the same ordering as epoch 0.
    """
    sub_batches = [pairs[i:i + ref_batch_size] for i in range(0, len(pairs), ref_batch_size)]

    async def _process_sub_batch(
        batch: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
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
        ref_chosen_list,
        ref_rejected_list,
        response_starts,
        beta,
    )
    return policy.forward_backward_custom(datums, loss_fn)


_DONE = object()


async def _train_loop(
    tokenized_store: DiskBackedDatumStore,
    ref_cache_store: DiskBackedDatumStore | None,
    reference: ReconnectableClient,
    policy: ReconnectableClient,
    adam_params: tinker.AdamParams,
    cfg: Config,
    step_offset: int,
    on_ref_done: Callable[[], None] | None = None,
    runner: RunnerIO | None = None,
    training_shape_id: str | None = None,
) -> int:
    """Pipelined DPO training -- ref forward overlaps with policy training.

    Epoch 0 runs a producer/consumer pipeline with an unbounded queue:
    the producer pulls tokenized pair chunks from ``tokenized_store``,
    computes reference logprobs at full speed (concurrent via semaphore),
    and pushes enriched chunks to the consumer that runs train steps.
    Once the producer finishes, *on_ref_done* fires to delete the reference
    trainer immediately -- even while training continues.

    For multi-epoch runs, ``ref_cache_store`` (must be opened by the caller)
    captures every enriched pair in producer order so epochs 1+ can stream
    them sequentially without holding the cache in RAM.
    """
    multi_epoch = cfg.epochs > 1
    if multi_epoch and ref_cache_store is None:
        raise ValueError("ref_cache_store is required when cfg.epochs > 1")

    batch_size = cfg.batch_size
    step = step_offset
    total_steps = len(tokenized_store) * cfg.epochs // batch_size

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

    # -- Epoch 0: pipelined ref forward + training -----------------------------

    n_pairs = len(tokenized_store)
    n_batches_epoch0 = (n_pairs + batch_size - 1) // batch_size
    pbar = tqdm(total=total_steps, desc="DPO training", unit="step")

    async def _ref_producer() -> None:
        for start in range(0, n_pairs, batch_size):
            stop = min(start + batch_size, n_pairs)
            chunk = [tokenized_store[i] for i in range(start, stop)]
            enriched = await _ref_forward_batch(
                chunk, reference, sem, cfg.ref_cache_batch_size,
            )
            if multi_epoch:
                # Append in producer order = input order (see _ref_forward_batch
                # docstring); epochs 1+ then iterate the store sequentially.
                for pair in enriched:
                    ref_cache_store.append(pair)
            await pipe.put(enriched)
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

    # -- Epochs 1+: stream cached ref logprobs from disk -----------------------

    if multi_epoch:
        ref_cache_store.close_write()
        n_cached = len(ref_cache_store)
        logger.info(
            "Ref cache built: %d pairs / %.1f GiB on disk",
            n_cached, ref_cache_store.disk_size_bytes() / (1024 ** 3),
        )
        for epoch in range(1, cfg.epochs):
            for start in range(0, n_cached, batch_size):
                stop = min(start + batch_size, n_cached)
                chunk = [ref_cache_store[i] for i in range(start, stop)]
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
            api_key=api_key,
            base_url=base_url,
            additional_headers=additional_headers,
        )
    if deploy_mgr is None:
        deploy_mgr = DeploymentManager(
            api_key=api_key,
            base_url=base_url,
            additional_headers=additional_headers,
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

        # -- Tokenize (streaming) + pipelined training -------------------------
        #
        # Stream-render preference pairs to a disk-backed store so peak RAM
        # stays bounded by ``num_workers * per_worker_render_footprint``
        # instead of scaling with dataset size. For multi-epoch runs we also
        # open a second disk-backed store for the ref-cache; together this
        # eliminates the two largest in-memory tables (`tokenized_pairs` and
        # `ref_cache`) that pushed the orchestrator over the node memory
        # limit on long-context DPO runs.
        # See docs/engineering/sft-v2-orchestrator-oom-debug.md.

        # ``count_jsonl_rows`` is an upper bound (the ``samples`` schema may
        # collapse multiple rows into 0 or 1 pair) but it's good enough for
        # progress reporting.
        total_raw = count_jsonl_rows(cfg.dataset, cfg.max_pairs)
        if total_raw == 0:
            raise RuntimeError(f"No data found in {cfg.dataset}")
        max_seq_len = infra.max_seq_len
        num_workers = min(os.cpu_count() or 1, DEFAULT_RENDER_WORKERS)
        logger.info(
            "Streaming %d preference rows from %s (renderer=%s, workers=%d,"
            " chunksize=%d)",
            total_raw, cfg.dataset,
            resolve_renderer_name(cfg.tokenizer_model, cfg.renderer_name),
            num_workers, DEFAULT_RENDER_CHUNKSIZE,
        )

        # Initialise _pair_worker_state in the parent too so the same
        # render_fn (_render_pair_worker) is used regardless of whether the
        # helper picks the in-process loop (num_workers == 1) or the spawn
        # pool. Mirrors the sft_loop.py pattern.
        init_args = (cfg.tokenizer_model, cfg.renderer_name, max_seq_len)
        _init_pair_worker(*init_args)

        render_cache_dir = stack.enter_context(
            tempfile.TemporaryDirectory(prefix="dpo_render_")
        )

        def _open_store(name: str) -> DiskBackedDatumStore:
            return stack.enter_context(
                DiskBackedDatumStore(os.path.join(render_cache_dir, name))
            )

        tokenized_store = _open_store("tokenized_pairs.bin")
        # Open the ref-cache store eagerly only for multi-epoch runs; for
        # single-epoch runs we skip the disk traffic entirely.
        ref_cache_store: DiskBackedDatumStore | None = (
            _open_store("ref_cache.bin") if cfg.epochs > 1 else None
        )

        mem_tracer = MemTracer(
            store_callback=lambda: (
                len(tokenized_store), tokenized_store.disk_size_bytes(),
            ),
            log=logger,
        )
        mem_tracer.log("before_rendering", 0, total_raw)

        log_interval = max(1, total_raw // 20)       # ~5% for runner status
        mem_log_interval = max(1, total_raw // 200)  # ~0.5% for memory tracing

        def _on_render_progress(i: int, _pair: dict | None) -> None:
            if i % log_interval == 0 or i == total_raw:
                runner.report_rendering_progress(i, total_raw)
            if i % mem_log_interval == 0 or i == total_raw:
                mem_tracer.log("rendering", i, total_raw)

        filtered_count = stream_render_to_store(
            iter_preference_examples(cfg.dataset, cfg.max_pairs),
            render_fn=_render_pair_worker,
            store=tokenized_store,
            num_workers=num_workers,
            chunksize=DEFAULT_RENDER_CHUNKSIZE,
            initializer=_init_pair_worker,
            initargs=init_args,
            on_progress=_on_render_progress,
        )
        tokenized_store.close_write()
        mem_tracer.log("after_rendering", total_raw, total_raw)

        if filtered_count > 0:
            logger.info(
                "Seq-length filter: %d/%d pairs dropped "
                "(chosen or rejected > %d tokens, or render failed)",
                filtered_count, total_raw, max_seq_len,
            )
        logger.info("Prepared %d preference pairs", len(tokenized_store))
        if len(tokenized_store) == 0:
            raise RuntimeError("No valid pairs after tokenization")

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
                tokenized_store, ref_cache_store,
                reference, policy, adam_params, cfg, step_offset,
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

        total_steps = len(tokenized_store) * cfg.epochs // cfg.batch_size
        runner.write_status(RunStatus.COMPLETED, step=step, total_steps=total_steps, message="done")
        runner.write_metadata()
        logger.info("Training complete: %d optimizer steps (%d new)", step, step - step_offset)
        wandb_finish()
        return {"steps": step, "policy_job_id": policy_job_id, "reference_job_id": reference_job_id}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(Config(log_path="./dpo_logs"))
