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

import os
import time
import signal
import asyncio
import logging
from typing import Any, Callable
from dataclasses import field, dataclass
from concurrent.futures import ThreadPoolExecutor

import tinker
from tqdm import tqdm

from fireworks.training.sdk import DeploymentManager, TrainerJobManager
from training.utils import (
    DEFAULT_ADAM,
    InfraConfig,
    ResourceCleanup,
    RunnerConfig,
    RunnerIO,
    RunStatus,
    WandBConfig,
    DeployConfig,
    WeightSyncConfig,
    ReconnectableClient,
    wandb_log,
    setup_wandb,
    wandb_finish,
    validate_config,
    log_metrics_json,
    make_batch_dpo_loss_fn,
    setup_deployment,
    create_trainer_job,
    load_preference_dataset,
    build_renderer,
    render_preference_pair,
    resolve_renderer_name,
)
from fireworks.training.sdk.deployment import DEFAULT_DELTA_COMPRESSION
from fireworks.training.sdk.weight_syncer import WeightSyncer
from training.utils.checkpoint_utils import resolve_resume
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

    infra: InfraConfig = field(default_factory=InfraConfig)
    deployment: DeployConfig = field(default_factory=DeployConfig)
    weight_sync: WeightSyncConfig = field(default_factory=lambda: WeightSyncConfig(weight_sync_interval=0))
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="dpo-tinker"))
    init_from_checkpoint: str | None = None
    output_model_id: str | None = None
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


def _tokenize_pair(
    example: dict[str, Any],
    tokenizer: Any,
    renderer: Any,
    max_seq_len: int,
) -> dict[str, Any] | None:
    """Tokenize a single preference pair. Returns None if invalid, 'filtered' if too long."""
    pair = render_preference_pair(
        example["chosen"],
        example["rejected"],
        renderer=renderer,
        tokenizer=tokenizer,
    )
    if pair is None:
        return None
    if len(pair.chosen_tokens) > max_seq_len or len(pair.rejected_tokens) > max_seq_len:
        return "filtered"

    return {
        "chosen_tokens": pair.chosen_tokens,
        "rejected_tokens": pair.rejected_tokens,
        "response_start": pair.response_start,
        "chosen_datum": pair.chosen_datum,
        "rejected_datum": pair.rejected_datum,
    }


def _tokenize_pairs(
    raw_data: list[dict[str, Any]],
    tokenizer: Any,
    renderer: Any,
    max_seq_len: int,
) -> tuple[list[tuple[int, dict[str, Any]]], int]:
    """Tokenize all preference pairs (CPU only).

    Returns ``(tokenized, filtered_count)`` where each entry is
    ``(original_index, pair_data_dict)``.
    """
    tokenized: list[tuple[int, dict[str, Any]]] = []
    filtered_count = 0
    for i, example in enumerate(raw_data):
        result = _tokenize_pair(example, tokenizer, renderer, max_seq_len)
        if result == "filtered":
            filtered_count += 1
        elif result is not None:
            tokenized.append((i, result))
    return tokenized, filtered_count


async def _ref_forward_batch(
    pairs: list[tuple[int, dict[str, Any]]],
    reference: ReconnectableClient,
    semaphore: asyncio.Semaphore,
    ref_batch_size: int,
) -> list[tuple[int, dict[str, Any]]]:
    """Compute reference logprobs for *pairs*, sub-batched by *ref_batch_size*.

    Uses the semaphore for concurrency control.  Returns enriched pairs
    with ``ref_chosen`` / ``ref_rejected`` logprobs attached.
    """
    sub_batches = [pairs[i:i + ref_batch_size] for i in range(0, len(pairs), ref_batch_size)]

    async def _process_sub_batch(
        batch: list[tuple[int, dict[str, Any]]],
    ) -> list[tuple[int, dict[str, Any]]]:
        datums: list[tinker.Datum] = []
        for _, pair_data in batch:
            datums.append(pair_data["chosen_datum"])
            datums.append(pair_data["rejected_datum"])

        async with semaphore:
            fwd = await asyncio.to_thread(
                lambda d=datums: reference.forward(d, "cross_entropy")
            )

        results: list[tuple[int, dict[str, Any]]] = []
        for j, (idx, pair_data) in enumerate(batch):
            results.append((idx, {
                "chosen_tokens": pair_data["chosen_tokens"],
                "rejected_tokens": pair_data["rejected_tokens"],
                "chosen_datum": pair_data["chosen_datum"],
                "rejected_datum": pair_data["rejected_datum"],
                "ref_chosen": fwd.loss_fn_outputs[2 * j]["logprobs"].data,
                "ref_rejected": fwd.loss_fn_outputs[2 * j + 1]["logprobs"].data,
                "response_start": pair_data["response_start"],
            }))
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
    tokenized_pairs: list[tuple[int, dict[str, Any]]],
    reference: ReconnectableClient,
    policy: ReconnectableClient,
    adam_params: tinker.AdamParams,
    weight_syncer: WeightSyncer,
    cfg: Config,
    step_offset: int,
    on_ref_done: Callable[[], None] | None = None,
    runner: RunnerIO | None = None,
) -> int:
    """Pipelined DPO training -- ref forward overlaps with policy training.

    Epoch 0 runs a producer/consumer pipeline with an unbounded queue:
    the producer computes reference logprobs at full speed (concurrent
    via semaphore) while the consumer trains.  Once the producer finishes,
    *on_ref_done* fires to delete the reference trainer immediately --
    even while training continues.

    Epochs 1+ reuse cached ref logprobs (no ref GPU needed).
    """
    batch_size = cfg.batch_size
    step = step_offset
    total_steps = len(tokenized_pairs) * cfg.epochs // batch_size

    if runner is None:
        runner = RunnerIO()
    ref_cache: dict[int, dict[str, Any]] = {}
    pipe: asyncio.Queue = asyncio.Queue()
    sem = asyncio.Semaphore(cfg.ref_cache_concurrency)

    def _run_train_step(epoch: int, step_pairs: list[dict[str, Any]]) -> None:
        nonlocal step
        step_t0 = time.monotonic()
        step_tokens = sum(
            len(p["chosen_tokens"]) + len(p["rejected_tokens"])
            for p in step_pairs
        )

        with timer("fwd_bwd"):
            fwd_bwd_result = _forward_backward_pairs(step_pairs, policy, cfg.beta)
        optim_result = policy.optim_step(adam_params)
        step += 1

        step_metrics: dict[str, Any] = {}
        if optim_result and hasattr(optim_result, "metrics") and optim_result.metrics:
            for k, v in optim_result.metrics.items():
                step_metrics[f"train/{k}"] = v

        hl = cfg.weight_sync
        if hl.weight_sync_interval > 0 and step % hl.weight_sync_interval == 0:
            with timer("weight_sync"):
                weight_syncer.save_and_hotload(f"step-{step}")
        if hl.dcp_save_interval > 0 and step % hl.dcp_save_interval == 0:
            with timer("dcp_save"):
                weight_syncer.save_dcp(f"step-{step}")

        step_elapsed = time.monotonic() - step_t0
        tokens_per_sec = step_tokens / step_elapsed if step_elapsed > 0 else 0.0
        step_metrics.update(flush_timing())

        fwd_metrics = fwd_bwd_result.metrics
        avg_loss = fwd_metrics["dpo_loss"]
        avg_margin = fwd_metrics["margin"]
        avg_acc = fwd_metrics["accuracy"]
        logger.info(
            "Step %d/%d | Loss: %.4f | Margin: %+.4f | Acc: %.1f%% | %.1f tok/s (%.1fs)",
            step, total_steps, avg_loss, avg_margin, avg_acc * 100,
            tokens_per_sec, step_elapsed,
        )
        log_metrics_json(step, dpo_loss=avg_loss, margin=avg_margin, accuracy=avg_acc,
                         tokens_per_sec=tokens_per_sec)
        step_metrics.update({
            "train/step": step,
            "train/dpo_loss": avg_loss,
            "train/margin": avg_margin,
            "train/accuracy": avg_acc,
            "train/epoch": epoch + 1,
            "train/tokens_per_sec": tokens_per_sec,
            "train/step_time_sec": step_elapsed,
            "train/step_tokens": step_tokens,
        })
        wandb_log(step_metrics, step)
        runner.append_metrics(step, step_metrics, tokens=step_tokens)
        runner.write_status(RunStatus.RUNNING, step=step, total_steps=total_steps, message="training")
        runner.write_metadata()

    # -- Epoch 0: pipelined ref forward + training -----------------------------

    multi_epoch = cfg.epochs > 1

    n_batches_epoch0 = (len(tokenized_pairs) + batch_size - 1) // batch_size
    pbar = tqdm(total=total_steps, desc="DPO training", unit="step")

    async def _ref_producer() -> None:
        for start in range(0, len(tokenized_pairs), batch_size):
            chunk = tokenized_pairs[start:start + batch_size]
            enriched = await _ref_forward_batch(
                chunk, reference, sem, cfg.ref_cache_batch_size,
            )
            if multi_epoch:
                for idx, pair in enriched:
                    ref_cache[idx] = pair
            await pipe.put([pair for _, pair in enriched])
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

    # -- Epochs 1+: iterate cached ref logprobs --------------------------------

    for epoch in range(1, cfg.epochs):
        ordered_pairs = [ref_cache[idx] for idx, _ in tokenized_pairs]
        for start in range(0, len(ordered_pairs), batch_size):
            chunk = ordered_pairs[start:start + batch_size]
            _run_train_step(epoch, chunk)
            pbar.update(1)

    pbar.close()
    ref_cache.clear()
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
    runner = RunnerIO(cfg.runner)

    def _signal_handler(signum, frame):
        name = signal.Signals(signum).name
        logger.warning("Received %s — raising SystemExit for cleanup", name)
        raise SystemExit(f"Terminated by {name}")

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    validate_config(
        cfg.base_model,
        cfg.dataset,
        cfg.weight_sync,
        cfg.deployment,
        output_model_id=cfg.output_model_id,
    )
    if not cfg.tokenizer_model:
        raise ValueError(
            "Config.tokenizer_model is required for client-side tokenization. "
            "Set it to the HuggingFace model name (e.g. 'Qwen/Qwen3-1.7B')."
        )

    if cfg.grad_accum > 1:
        from training.utils.deprecation import warn_deprecated_param
        warn_deprecated_param("grad_accum", "batch_size", extra="grad_accum is ignored.")

    setup_wandb(cfg.wandb, {
        "beta": cfg.beta,
        "lr": cfg.learning_rate,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
    })

    # -- Setup infrastructure ----------------------------------------------

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")

    if rlor_mgr is None:
        rlor_mgr = TrainerJobManager(api_key=api_key, base_url=base_url)
    if deploy_mgr is None:
        deploy_mgr = DeploymentManager(api_key=api_key, base_url=base_url)

    if cfg.deployment.deployment_id:
        setup_deployment(deploy_mgr, cfg.deployment, cfg.base_model, cfg.infra)

    profile = None
    if cfg.infra.training_shape_id:
        profile = rlor_mgr.resolve_training_profile(cfg.infra.training_shape_id)

    ref_profile = None
    if cfg.infra.ref_training_shape_id:
        ref_profile = rlor_mgr.resolve_training_profile(cfg.infra.ref_training_shape_id)
    elif profile is not None:
        raise ValueError(
            "ref_training_shape_id must be set when training_shape_id is set. "
            "DPO always requires a reference model. Set it explicitly "
            "(can be the same as training_shape_id)."
        )

    if profile and getattr(profile, "base_model", ""):
        if cfg.base_model and cfg.base_model != profile.base_model:
            from training.utils.deprecation import warn_deprecated_param
            warn_deprecated_param(
                "base_model", "profile.base_model (from training shape)",
                extra=f"The training shape specifies '{profile.base_model}'.",
            )
        cfg.base_model = profile.base_model
    if profile and cfg.max_seq_len is None:
        cfg.max_seq_len = profile.max_supported_context_length
        logger.info("max_seq_len from training shape: %d", cfg.max_seq_len)

    if cfg.max_seq_len is None:
        raise ValueError(
            "max_seq_len is required. Set it in Config, or use a training shape "
            "(InfraConfig.training_shape_id) to auto-populate it."
        )

    with ResourceCleanup(rlor_mgr) as cleanup:
        with ThreadPoolExecutor(max_workers=2) as pool:
            pol_fut = pool.submit(
                create_trainer_job,
                rlor_mgr,
                base_model=cfg.base_model,
                infra=cfg.infra,
                profile=profile,
                lora_rank=cfg.lora_rank,
                max_seq_len=cfg.max_seq_len,
                learning_rate=cfg.learning_rate,
                display_name="dpo-policy",
                hot_load_deployment_id=cfg.deployment.deployment_id,  # weight sync target deployment
                cleanup=cleanup,
            )
            ref_fut = pool.submit(
                create_trainer_job,
                rlor_mgr,
                base_model=cfg.base_model,
                infra=cfg.infra,
                profile=ref_profile,
                lora_rank=cfg.lora_rank,
                max_seq_len=cfg.max_seq_len,
                learning_rate=cfg.learning_rate,
                display_name="dpo-reference",
                forward_only=True,
                cleanup=cleanup,
            )
            policy_ep = pol_fut.result()
            reference_ep = ref_fut.result()

        policy_job_id = policy_ep.job_id
        reference_job_id = reference_ep.job_id

        policy = ReconnectableClient(rlor_mgr, policy_ep.job_id, cfg.base_model, cfg.lora_rank)
        reference = ReconnectableClient(rlor_mgr, reference_ep.job_id, cfg.base_model, cfg.lora_rank)

        weight_syncer = WeightSyncer(
            policy_client=policy.inner,
            deploy_mgr=deploy_mgr,
            deployment_id=cfg.deployment.deployment_id,
            base_model=cfg.base_model,
            hotload_timeout=cfg.weight_sync.weight_sync_timeout,
            first_checkpoint_type=cfg.weight_sync.first_checkpoint_type,
            compression_format=DEFAULT_DELTA_COMPRESSION,
        )

        resume_info = resolve_resume(policy, cfg.log_path, cfg.init_from_checkpoint)
        step_offset = resume_info.step if resume_info else 0
        wandb_log({"train/step": step_offset}, step_offset)
        adam_params = tinker.AdamParams(learning_rate=cfg.learning_rate, **DEFAULT_ADAM)

        # -- Tokenize + pipelined training -------------------------------------

        import transformers

        tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.tokenizer_model, trust_remote_code=True)
        renderer = build_renderer(tokenizer, cfg.tokenizer_model, cfg.renderer_name)
        logger.info(
            "Using renderer=%s for preference tokenization",
            resolve_renderer_name(cfg.tokenizer_model, cfg.renderer_name),
        )

        raw_data = load_preference_dataset(cfg.dataset, cfg.max_pairs)
        if not raw_data:
            raise RuntimeError(f"No data loaded from {cfg.dataset}")

        tokenized_pairs, filtered_count = _tokenize_pairs(
            raw_data, tokenizer, renderer, cfg.max_seq_len,
        )
        if filtered_count > 0:
            logger.info(
                "Seq-length filter: %d/%d pairs filtered (chosen or rejected > %d tokens)",
                filtered_count, len(raw_data), cfg.max_seq_len,
            )
        logger.info("Prepared %d preference pairs", len(tokenized_pairs))
        if not tokenized_pairs:
            raise RuntimeError("No valid pairs after tokenization")

        runner.set_accelerator_info(cfg.infra.accelerator_type, cfg.infra.accelerator_count)
        runner.write_status(RunStatus.RUNNING, message="provisioning")

        def _on_ref_done():
            nonlocal reference_job_id
            logger.info("Reference forward complete — deleting reference trainer to free GPU")
            try:
                cleanup.delete_trainer(reference_job_id)
                reference_job_id = None
            except Exception as e:
                logger.warning("Early cleanup of reference job %s failed: %s", reference_job_id, e)

        runner.start_training()
        with runner:
            step = asyncio.run(
                _train_loop(
                    tokenized_pairs, reference, policy, adam_params, weight_syncer, cfg, step_offset,
                    on_ref_done=_on_ref_done,
                    runner=runner,
                )
            )

        # -- Final checkpoint --------------------------------------------------

        hl = cfg.weight_sync
        if step > step_offset:
            dcp_name = f"step-{step}"
            weight_syncer.save_dcp(dcp_name)
            final_sampler_checkpoint_id: str | None = None
            if getattr(cfg, "output_model_id", None):
                final_cp_name = f"final-step-{step}"
                final_sampler_checkpoint_id = weight_syncer.save_only(
                    final_cp_name,
                    checkpoint_type="base",
                )
                if hl.weight_sync_interval > 0 and final_sampler_checkpoint_id:
                    weight_syncer.hotload(final_sampler_checkpoint_id)
            elif hl.weight_sync_interval > 0:
                final_cp_name = f"final-step-{step}"
                weight_syncer.save_and_hotload(final_cp_name)

            if getattr(cfg, "output_model_id", None):
                if not final_sampler_checkpoint_id:
                    raise RuntimeError("Failed to save final base checkpoint for promotion")
                rlor_mgr.promote_checkpoint(
                    policy_job_id,
                    final_sampler_checkpoint_id,
                    cfg.output_model_id,
                )
                runner.write_output_model(
                    model_id=cfg.output_model_id, checkpoint=f"final-step-{step}", job_id=policy_job_id,
                )

        runner.write_status(RunStatus.COMPLETED, step=step, message="done")
        runner.write_metadata()
        logger.info("Training complete: %d optimizer steps (%d new)", step, step - step_offset)
        wandb_finish()
        return {"steps": step, "policy_job_id": policy_job_id, "reference_job_id": reference_job_id}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(Config(log_path="./dpo_logs"))
