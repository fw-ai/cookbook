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
from contextlib import ExitStack
from typing import Any, Callable
import dataclasses
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
    ReconnectableClient,
    wandb_log,
    setup_wandb,
    wandb_finish,
    validate_config,
    log_metrics_json,
    make_batch_dpo_loss_fn,
    create_trainer_job,
    load_preference_dataset,
    build_renderer,
    auto_select_training_shape,
    render_preference_pair,
    resolve_renderer_name,
)
from training.utils.checkpoint_utils import resolve_resume, save_checkpoint, CheckpointKind
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
    dcp_save_interval: int = 0
    """Save DCP checkpoints every N steps. 0 disables."""
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="dpo-tinker"))
    policy_job_id: str | None = None
    reference_job_id: str | None = None
    init_from_checkpoint: str | None = None
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
                    training_shape=cfg.infra.training_shape_id,
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

    if rlor_mgr is None:
        rlor_mgr = TrainerJobManager(api_key=api_key, base_url=base_url)
    if deploy_mgr is None:
        deploy_mgr = DeploymentManager(api_key=api_key, base_url=base_url)

    if not cfg.infra.training_shape_id:
        cfg.infra.training_shape_id = auto_select_training_shape(
            rlor_mgr, base_model=cfg.base_model, trainer_role="policy",
            lora_rank=cfg.lora_rank, max_seq_len=cfg.max_seq_len,
        )
        logger.info("Auto-selected policy training shape: %s", cfg.infra.training_shape_id)
    if not cfg.infra.ref_training_shape_id:
        cfg.infra.ref_training_shape_id = auto_select_training_shape(
            rlor_mgr, base_model=cfg.base_model, trainer_role="reference",
            lora_rank=cfg.lora_rank, max_seq_len=cfg.max_seq_len,
        )
        logger.info("Auto-selected reference training shape: %s", cfg.infra.ref_training_shape_id)

    policy_profile = rlor_mgr.resolve_training_profile(cfg.infra.training_shape_id)
    ref_profile = rlor_mgr.resolve_training_profile(cfg.infra.ref_training_shape_id)

    if cfg.max_seq_len is None:
        cfg.max_seq_len = policy_profile.max_supported_context_length

    runner = RunnerIO(cfg.runner)
    runner.set_accelerator_info(profile=policy_profile)
    runner.write_status(RunStatus.PENDING, message="provisioning")

    def _on_trainer_status(msg: str) -> None:
        runner.write_status(RunStatus.PENDING, message=msg)

    with runner, ResourceCleanup(rlor_mgr) as cleanup, ExitStack() as stack:
        # -- Create trainer jobs first (trainer owns the hot-load bucket) ------
        _on_trainer_status("provisioning policy and reference trainers")
        with ThreadPoolExecutor(max_workers=2) as pool:
            pol_fut = pool.submit(
                create_trainer_job,
                rlor_mgr,
                base_model=cfg.base_model,
                infra=cfg.infra,
                profile=policy_profile,
                lora_rank=cfg.lora_rank,
                max_seq_len=cfg.max_seq_len,
                learning_rate=cfg.learning_rate,
                display_name="dpo-policy",
                job_id=cfg.policy_job_id,
                cleanup=cleanup,
                on_status=_on_trainer_status,
            )
            # Reference trainer only runs forward — strip --full-oom-check
            # (which runs a backward warmup that OOMs on smaller ref shapes)
            # and don't request LoRA adapters.
            ref_infra_extra = [
                a for a in (cfg.infra.extra_args or [])
                if a != "--full-oom-check"
            ] or None
            ref_infra = dataclasses.replace(cfg.infra, extra_args=ref_infra_extra)
            ref_fut = pool.submit(
                create_trainer_job,
                rlor_mgr,
                base_model=cfg.base_model,
                infra=ref_infra,
                profile=ref_profile,
                lora_rank=0,
                max_seq_len=cfg.max_seq_len,
                learning_rate=cfg.learning_rate,
                display_name="dpo-reference",
                forward_only=True,
                job_id=cfg.reference_job_id,
                cleanup=cleanup,
                on_status=_on_trainer_status,
            )
            # Collect both results so that if both fail we report
            # both errors instead of swallowing the second one.
            errors: list[str] = []
            policy_ep = reference_ep = None
            try:
                policy_ep = pol_fut.result()
            except Exception as e:
                errors.append(f"Policy trainer: {e}")
            try:
                reference_ep = ref_fut.result()
            except Exception as e:
                errors.append(f"Reference trainer: {e}")
            if errors:
                raise RuntimeError(
                    "Trainer creation failed:\n" + "\n".join(errors)
                )

        policy_job_id = policy_ep.job_id
        reference_job_id = reference_ep.job_id

        policy = ReconnectableClient(rlor_mgr, policy_ep.job_id, cfg.base_model, cfg.lora_rank)
        # Match the ref trainer's lora_rank=0 (set above).
        reference = ReconnectableClient(rlor_mgr, reference_ep.job_id, cfg.base_model, 0)
        if hasattr(policy, "close"):
            stack.callback(policy.close)
        if hasattr(reference, "close"):
            stack.callback(reference.close)

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

        def _on_ref_done():
            nonlocal reference_job_id
            logger.info("Reference forward complete — closing reference client before trainer cleanup")
            try:
                if hasattr(reference, "close"):
                    reference.close()
                logger.info("Reference forward complete — canceling reference trainer to free GPU")
                cleanup.cancel_trainer(reference_job_id)
                reference_job_id = None
            except Exception as e:
                logger.warning("Early cleanup of reference job %s failed: %s", reference_job_id, e)

        runner.start_training()
        step = asyncio.run(
            _train_loop(
                tokenized_pairs, reference, policy, adam_params, cfg, step_offset,
                on_ref_done=_on_ref_done,
                runner=runner,
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
                training_shape=cfg.infra.training_shape_id,
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

        total_steps = len(tokenized_pairs) * cfg.epochs // cfg.batch_size
        runner.write_status(RunStatus.COMPLETED, step=step, total_steps=total_steps, message="done")
        runner.write_metadata()
        logger.info("Training complete: %d optimizer steps (%d new)", step, step - step_offset)
        wandb_finish()
        return {"steps": step, "policy_job_id": policy_job_id, "reference_job_id": reference_job_id}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(Config(log_path="./dpo_logs"))
