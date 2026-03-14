#!/usr/bin/env python3
"""DPO training loop with concurrent reference caching and pipelined training.

Optimisations:

  - **Concurrent ref caching**: ``gather_with_progress`` computes reference
    logprobs for all pairs in parallel instead of sequentially.
  - **Pipelined training**: ``forward_backward_custom`` and ``optim_step``
    futures are issued back-to-back so they land on the same trainer
    clock cycle, matching tinker-cookbook's ``train_step`` pattern.

Architecture:
    - Policy RLOR job:    forward_backward_custom + optim_step (trainable)
    - Reference RLOR job: forward only (frozen base model, for KL baseline)
    - Reference logprobs cached at initialisation from the frozen reference

Usage:
    export FIREWORKS_API_KEY=...
    export FIREWORKS_ACCOUNT_ID=...
    python -m recipes.dpo_loop
"""

from __future__ import annotations

import os
import signal
import asyncio
import logging
from typing import Any, TypeVar, Awaitable, Iterable
from dataclasses import field, dataclass
from concurrent.futures import ThreadPoolExecutor

import tinker
from tqdm import tqdm

from fireworks.training.sdk import DeploymentManager, TrainerJobManager
from training.utils import (
    DEFAULT_ADAM,
    InfraConfig,
    ResourceCleanup,
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
_T = TypeVar("_T")


async def gather_with_progress(
    awaitables: Iterable[Awaitable[_T]],
    *,
    desc: str,
) -> list[_T]:
    """Await a collection of awaitables while showing a progress bar."""
    tasks = [asyncio.create_task(awaitable) for awaitable in awaitables]
    if not tasks:
        return []

    results: list[_T] = []
    try:
        with tqdm(total=len(tasks), desc=desc) as pbar:
            for task in asyncio.as_completed(tasks):
                results.append(await task)
                pbar.update(1)
    except Exception:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    return results

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
    batch_size: int = 1
    """Number of preference pairs per forward_backward_custom call."""
    grad_accum: int = 4
    max_seq_len: int | None = None
    max_pairs: int | None = None
    lora_rank: int = 0

    ref_cache_concurrency: int = 16
    """Max concurrent reference forward passes during cache warm-up."""
    ref_cache_batch_size: int = 1
    """Number of preference pairs per reference forward call during caching."""

    grad_accumulation_normalization: str | None = "num_sequences"
    """Normalization mode for accumulated gradients at optim_step.
    Defaults to "num_sequences" so gradients are correctly averaged
    across all accumulation steps regardless of grad_accum setting."""

    infra: InfraConfig = field(default_factory=InfraConfig)
    deployment: DeployConfig = field(default_factory=DeployConfig)
    weight_sync: WeightSyncConfig = field(default_factory=lambda: WeightSyncConfig(weight_sync_interval=0))
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="dpo-tinker"))
    init_from_checkpoint: str | None = None
    output_model_id: str | None = None


# ---------------------------------------------------------------------------
# Concurrent reference caching
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


async def _cache_ref_logprobs(
    raw_data: list[dict[str, Any]],
    reference: ReconnectableClient,
    tokenizer: Any,
    renderer: Any,
    max_seq_len: int,
    concurrency: int = 16,
    batch_size: int = 1,
) -> tuple[dict[int, dict[str, Any]], int]:
    """Compute reference logprobs concurrently using ``gather_with_progress``.

    When ``batch_size > 1``, multiple pairs are sent in a single reference
    forward call (2 * batch_size datums), reducing per-call overhead.

    Returns ``(ref_cache, filtered_count)``.
    """
    tokenized: list[tuple[int, dict[str, Any]]] = []
    filtered_count = 0
    for i, example in enumerate(raw_data):
        result = _tokenize_pair(example, tokenizer, renderer, max_seq_len)
        if result == "filtered":
            filtered_count += 1
        elif result is not None:
            tokenized.append((i, result))

    batches: list[list[tuple[int, dict[str, Any]]]] = []
    for start in range(0, len(tokenized), batch_size):
        batches.append(tokenized[start : start + batch_size])

    semaphore = asyncio.Semaphore(concurrency)

    async def _process_batch(
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

    batch_results = await gather_with_progress(
        (_process_batch(b) for b in batches),
        desc="Caching reference logprobs",
    )

    ref_cache: dict[int, dict[str, Any]] = {}
    for batch_result in batch_results:
        for idx, pair_data in batch_result:
            ref_cache[idx] = pair_data

    return ref_cache, filtered_count


# ---------------------------------------------------------------------------
# Pipelined training loop
# ---------------------------------------------------------------------------


def _flush_batch(
    batch_pairs: list[dict[str, Any]],
    policy: ReconnectableClient,
    beta: float,
    raw_sum: bool = False,
) -> Any:
    """Send a batch of pairs through forward_backward_custom.

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
        raw_sum=raw_sum,
    )
    return policy.forward_backward_custom(datums, loss_fn)


async def _train_loop(
    ref_cache: dict[int, dict[str, Any]],
    valid_indices: list[int],
    policy: ReconnectableClient,
    adam_params: tinker.AdamParams,
    weight_syncer: WeightSyncer,
    cfg: Config,
    step_offset: int,
) -> int:
    """DPO training with batched forward_backward + optim_step.

    Accumulates ``batch_size`` pairs into a single ``forward_backward_custom``
    call (matching SFT's batching pattern), then runs ``grad_accum`` such
    micro-batches before each ``optim_step``.
    """
    batch_size = cfg.batch_size
    step = step_offset
    total_steps = len(valid_indices) * cfg.epochs // (cfg.grad_accum * batch_size)
    accum_count = 0
    agg: dict[str, float] = {"dpo_loss": 0.0, "margin": 0.0, "accuracy": 0.0, "count": 0}
    # NOTE: raw_sum=True when server-side normalization is active to
    # avoid double-normalization (client divides by count AND server
    # divides again). raw_sum=False only with "none".
    use_raw_sum = cfg.grad_accumulation_normalization != "none"

    fwd_bwd_futures: list[Any] = []

    def _do_optim_step(epoch: int) -> None:
        nonlocal step, accum_count, agg, fwd_bwd_futures

        optim_result = policy.optim_step(
            adam_params,
            grad_accumulation_normalization=cfg.grad_accumulation_normalization,
        )

        step_metrics: dict[str, Any] = {}
        for result in fwd_bwd_futures:
            fwd_metrics = result.metrics
            agg["dpo_loss"] += fwd_metrics["dpo_loss"]
            agg["margin"] += fwd_metrics["margin"]
            agg["accuracy"] += fwd_metrics["accuracy"]
            agg["count"] += 1
        fwd_bwd_futures = []
        step += 1
        accum_count = 0

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

        step_metrics.update(flush_timing())

        n = agg["count"]
        if n > 0:
            avg_loss = agg["dpo_loss"] / n
            avg_margin = agg["margin"] / n
            avg_acc = agg["accuracy"] / n
            logger.info(
                "Step %d/%d | Loss: %.4f | Margin: %+.4f | Acc: %.1f%%",
                step, total_steps, avg_loss, avg_margin, avg_acc * 100,
            )
            log_metrics_json(step, dpo_loss=avg_loss, margin=avg_margin, accuracy=avg_acc)
            step_metrics.update({
                "train/step": step,
                "train/dpo_loss": avg_loss,
                "train/margin": avg_margin,
                "train/accuracy": avg_acc,
                "train/epoch": epoch + 1,
            })
            wandb_log(step_metrics, step)

        agg = {"dpo_loss": 0.0, "margin": 0.0, "accuracy": 0.0, "count": 0}

    for epoch in range(cfg.epochs):
        batch_buffer: list[dict[str, Any]] = []
        for idx in valid_indices:
            batch_buffer.append(ref_cache[idx])

            if len(batch_buffer) >= batch_size:
                with timer("fwd_bwd"):
                    fwd_bwd_result = _flush_batch(batch_buffer, policy, cfg.beta, raw_sum=use_raw_sum)
                fwd_bwd_futures.append(fwd_bwd_result)
                batch_buffer = []
                accum_count += 1

                if accum_count >= cfg.grad_accum:
                    _do_optim_step(epoch)

        if batch_buffer:
            with timer("fwd_bwd"):
                fwd_bwd_result = _flush_batch(batch_buffer, policy, cfg.beta, raw_sum=use_raw_sum)
            fwd_bwd_futures.append(fwd_bwd_result)
            batch_buffer = []
            accum_count += 1

        if accum_count > 0:
            for result in fwd_bwd_futures:
                metrics = result.metrics
                agg["dpo_loss"] += metrics["dpo_loss"]
                agg["margin"] += metrics["margin"]
                agg["accuracy"] += metrics["accuracy"]
                agg["count"] += 1
            fwd_bwd_futures = []

            policy.optim_step(
                adam_params,
                grad_accumulation_normalization=cfg.grad_accumulation_normalization,
            )
            step += 1
            accum_count = 0

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

    validate_config(cfg.base_model, cfg.dataset, cfg.weight_sync, cfg.deployment)
    if not cfg.tokenizer_model:
        raise ValueError(
            "Config.tokenizer_model is required for client-side tokenization. "
            "Set it to the HuggingFace model name (e.g. 'Qwen/Qwen3-1.7B')."
        )
    setup_wandb(cfg.wandb, {
        "beta": cfg.beta,
        "lr": cfg.learning_rate,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "grad_accum": cfg.grad_accum,
    })

    # -- Setup infrastructure ----------------------------------------------

    api_key = os.environ["FIREWORKS_API_KEY"]
    account = os.environ.get("FIREWORKS_ACCOUNT_ID", "")
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")

    if rlor_mgr is None:
        rlor_mgr = TrainerJobManager(api_key=api_key, account_id=account, base_url=base_url)
    if deploy_mgr is None:
        deploy_mgr = DeploymentManager(api_key=api_key, account_id=account, base_url=base_url)

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
            )
            policy_ep = pol_fut.result()
            reference_ep = ref_fut.result()

        policy_job_id = policy_ep.job_id
        reference_job_id = reference_ep.job_id
        cleanup.trainer(policy_job_id)
        cleanup.trainer(reference_job_id)

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

        # -- Cache reference logprobs concurrently ------------------------------

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

        logger.info("Computing reference logprobs for %d pairs...", len(raw_data))
        ref_cache, filtered_count = asyncio.run(
            _cache_ref_logprobs(
                raw_data, reference, tokenizer, renderer, cfg.max_seq_len,
                concurrency=cfg.ref_cache_concurrency,
                batch_size=cfg.ref_cache_batch_size,
            )
        )

        valid_indices = list(ref_cache.keys())
        if filtered_count > 0:
            logger.info(
                "Seq-length filter: %d/%d pairs filtered (chosen or rejected > %d tokens)",
                filtered_count, len(raw_data), cfg.max_seq_len,
            )
        logger.info("Prepared %d preference pairs", len(valid_indices))
        if not valid_indices:
            raise RuntimeError("No valid pairs after tokenization")

        # -- Training loop (pipelined) -----------------------------------------

        step = asyncio.run(
            _train_loop(
                ref_cache, valid_indices, policy, adam_params, weight_syncer, cfg, step_offset,
            )
        )

        # -- Final checkpoint --------------------------------------------------

        hl = cfg.weight_sync
        if step > step_offset:
            cp_name = f"step-{step}"
            weight_syncer.save_dcp(cp_name)
            if hl.weight_sync_interval > 0:
                cp_name = f"final-step-{step}"
                weight_syncer.save_and_hotload(cp_name)
                
            if getattr(cfg, "output_model_id", None):
                from training.utils.checkpoint_utils import promote_checkpoint
                promote_checkpoint(
                    rlor_mgr,
                    policy_job_id,
                    cp_name,
                    cfg.output_model_id,
                )

        logger.info("Training complete: %d optimizer steps (%d new)", step, step - step_offset)
        wandb_finish()
        return {"steps": step, "policy_job_id": policy_job_id, "reference_job_id": reference_job_id}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(Config(log_path="./dpo_logs"))
