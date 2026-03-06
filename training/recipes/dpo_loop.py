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
    WandBConfig,
    DeployConfig,
    ResumeConfig,
    HotloadConfig,
    ReconnectableClient,
    wandb_log,
    setup_wandb,
    extract_text,
    setup_resume,
    wandb_finish,
    validate_config,
    log_metrics_json,
    make_batch_dpo_loss_fn,
    setup_deployment,
    create_trainer_job,
    load_preference_dataset,
    find_common_prefix_length,
)
from fireworks.training.sdk.deployment import DEFAULT_DELTA_COMPRESSION
from fireworks.training.sdk.weight_syncer import WeightSyncer
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
    base_model: str = "accounts/fireworks/models/qwen3-8b"
    dataset: str = ""
    tokenizer_model: str = ""  # HuggingFace model name for client-side tokenization

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

    infra: InfraConfig = field(default_factory=InfraConfig)
    deployment: DeployConfig = field(default_factory=DeployConfig)
    hotload: HotloadConfig = field(default_factory=lambda: HotloadConfig(hot_load_interval=0))
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="dpo-tinker"))
    resume: ResumeConfig = field(default_factory=ResumeConfig)


# ---------------------------------------------------------------------------
# Concurrent reference caching
# ---------------------------------------------------------------------------


def _tokenize_pair(
    example: dict[str, Any],
    tokenizer: Any,
    max_seq_len: int,
) -> dict[str, Any] | None:
    """Tokenize a single preference pair. Returns None if invalid, 'filtered' if too long."""
    chosen_text = extract_text(example["chosen"])
    rejected_text = extract_text(example["rejected"])
    if not chosen_text or not rejected_text:
        return None

    chosen_tokens = tokenizer.encode(chosen_text)
    rejected_tokens = tokenizer.encode(rejected_text)
    if len(chosen_tokens) > max_seq_len or len(rejected_tokens) > max_seq_len:
        return "filtered"
    if len(chosen_tokens) < 2 or len(rejected_tokens) < 2:
        return None

    prompt_len = find_common_prefix_length(chosen_tokens, rejected_tokens)
    return {
        "chosen_tokens": chosen_tokens,
        "rejected_tokens": rejected_tokens,
        "prompt_len": prompt_len,
    }


def _make_datum(tokens: list[int]) -> tinker.Datum:
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=tokens[1:], dtype="int64", shape=[len(tokens) - 1]
            )
        },
    )


async def _cache_ref_logprobs(
    raw_data: list[dict[str, Any]],
    reference: ReconnectableClient,
    tokenizer: Any,
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
        result = _tokenize_pair(example, tokenizer, max_seq_len)
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
            datums.append(_make_datum(pair_data["chosen_tokens"]))
            datums.append(_make_datum(pair_data["rejected_tokens"]))

        async with semaphore:
            fwd = await asyncio.to_thread(
                lambda d=datums: reference.forward(d, "cross_entropy")
            )

        results: list[tuple[int, dict[str, Any]]] = []
        for j, (idx, pair_data) in enumerate(batch):
            results.append((idx, {
                "chosen_tokens": pair_data["chosen_tokens"],
                "rejected_tokens": pair_data["rejected_tokens"],
                "ref_chosen": fwd.loss_fn_outputs[2 * j]["logprobs"].data,
                "ref_rejected": fwd.loss_fn_outputs[2 * j + 1]["logprobs"].data,
                "prompt_len": pair_data["prompt_len"],
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
) -> Any:
    """Send a batch of pairs through forward_backward_custom.

    Arranges datums as [chosen_0, rejected_0, chosen_1, rejected_1, ...].
    """
    datums: list[tinker.Datum] = []
    ref_chosen_list: list[list[float]] = []
    ref_rejected_list: list[list[float]] = []
    response_starts: list[int] = []

    for cached in batch_pairs:
        response_start = max(0, cached["prompt_len"] - 1)
        datums.append(_make_datum(cached["chosen_tokens"]))
        datums.append(_make_datum(cached["rejected_tokens"]))
        ref_chosen_list.append(cached["ref_chosen"])
        ref_rejected_list.append(cached["ref_rejected"])
        response_starts.append(response_start)

    loss_fn = make_batch_dpo_loss_fn(
        ref_chosen_list, ref_rejected_list, response_starts, beta,
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

    fwd_bwd_futures: list[Any] = []

    def _do_optim_step(epoch: int) -> None:
        nonlocal step, accum_count, agg, fwd_bwd_futures

        optim_result = policy.optim_step(adam_params)

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

        hl = cfg.hotload
        if hl.hot_load_interval > 0 and step % hl.hot_load_interval == 0:
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
                    fwd_bwd_result = _flush_batch(batch_buffer, policy, cfg.beta)
                fwd_bwd_futures.append(fwd_bwd_result)
                batch_buffer = []
                accum_count += 1

                if accum_count >= cfg.grad_accum:
                    _do_optim_step(epoch)

        if batch_buffer:
            with timer("fwd_bwd"):
                fwd_bwd_result = _flush_batch(batch_buffer, policy, cfg.beta)
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

            policy.optim_step(adam_params)
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

    validate_config(cfg.base_model, cfg.dataset, cfg.hotload, cfg.deployment, cfg.infra, cfg.resume)
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

    if cfg.max_seq_len is None:
        raise ValueError(
            "max_seq_len is required. Set it in Config, or use a training shape "
            "(InfraConfig.training_shape_id) to auto-populate it."
        )

    ref_extra = list(cfg.infra.extra_args or [])
    if "--forward-only" not in ref_extra:
        ref_extra.append("--forward-only")
    if "--no-compile" not in ref_extra:
        ref_extra.append("--no-compile")

    policy_job_id: str | None = None
    reference_job_id: str | None = None

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            pol_fut = pool.submit(
                create_trainer_job,
                rlor_mgr,
                base_model=cfg.base_model,
                infra=cfg.infra,
                lora_rank=cfg.lora_rank,
                max_seq_len=cfg.max_seq_len,
                learning_rate=cfg.learning_rate,
                display_name="dpo-policy",
                hot_load_deployment_id=cfg.deployment.deployment_id,
            )
            ref_fut = pool.submit(
                create_trainer_job,
                rlor_mgr,
                base_model=cfg.base_model,
                infra=cfg.infra,
                lora_rank=cfg.lora_rank,
                max_seq_len=cfg.max_seq_len,
                learning_rate=cfg.learning_rate,
                display_name="dpo-reference",
                extra_args=ref_extra,
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
            hotload_timeout=cfg.hotload.hot_load_timeout,
            first_checkpoint_type=cfg.hotload.first_checkpoint_type,
            compression_format=DEFAULT_DELTA_COMPRESSION,
        )

        step_offset, _ = setup_resume(policy, cfg.resume)
        adam_params = tinker.AdamParams(learning_rate=cfg.learning_rate, **DEFAULT_ADAM)

        # -- Cache reference logprobs concurrently ------------------------------

        import transformers

        tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.tokenizer_model, trust_remote_code=True)

        raw_data = load_preference_dataset(cfg.dataset, cfg.max_pairs)
        if not raw_data:
            raise RuntimeError(f"No data loaded from {cfg.dataset}")

        logger.info("Computing reference logprobs for %d pairs...", len(raw_data))
        ref_cache, filtered_count = asyncio.run(
            _cache_ref_logprobs(
                raw_data, reference, tokenizer, cfg.max_seq_len,
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

        hl = cfg.hotload
        if step > step_offset and (hl.hot_load_interval > 0 or hl.dcp_save_interval > 0):
            weight_syncer.save_and_hotload(f"final-step-{step}")

        logger.info("Training complete: %d optimizer steps (%d new)", step, step - step_offset)
        return {"steps": step, "policy_job_id": policy_job_id, "reference_job_id": reference_job_id}
    finally:
        wandb_finish()
        if policy_job_id:
            try:
                logger.info("Cleanup: deleting policy trainer job %s", policy_job_id)
                rlor_mgr.delete(policy_job_id)
            except Exception as e:
                logger.warning("Cleanup: failed to delete policy job %s: %s", policy_job_id, e)
        if reference_job_id:
            try:
                logger.info("Cleanup: deleting reference trainer job %s", reference_job_id)
                rlor_mgr.delete(reference_job_id)
            except Exception as e:
                logger.warning("Cleanup: failed to delete reference job %s: %s", reference_job_id, e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(Config())
