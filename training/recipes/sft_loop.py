#!/usr/bin/env python3
"""Minimal SFT (Supervised Fine-Tuning) training loop.

A readable, modifiable fine-tuning loop using the Fireworks RLOR API.
Uses a single RLOR trainer job with cross-entropy loss on response tokens.

Dataset format (JSONL, OpenAI chat format):
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Usage:
    export FIREWORKS_API_KEY=...
    export FIREWORKS_ACCOUNT_ID=...
    python cookbook/recipes/sft_loop.py --dataset /path/to/chat_data.jsonl
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List
from dataclasses import field, dataclass

import torch
import tinker

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
    setup_resume,
    wandb_finish,
    validate_config,
    log_metrics_json,
    setup_deployment,
    create_trainer_job,
    make_batch_sft_loss_fn,
)
from fireworks.training.sdk.deployment import DEFAULT_DELTA_COMPRESSION
from fireworks.training.sdk.weight_syncer import WeightSyncer
from training.utils.timer import timer, flush_timing

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    base_model: str = "accounts/fireworks/models/qwen3-8b"
    dataset: str = ""
    tokenizer_model: str = ""  # HuggingFace model name for chat template, e.g. "Qwen/Qwen3-1.7B"

    learning_rate: float = 1e-4
    epochs: int = 3
    batch_size: int = 32
    grad_accum: int = 4
    max_seq_len: int | None = None
    max_examples: int | None = None
    lora_rank: int = 0

    infra: InfraConfig = field(default_factory=InfraConfig)
    deployment: DeployConfig = field(default_factory=DeployConfig)
    hotload: HotloadConfig = field(default_factory=lambda: HotloadConfig(hot_load_interval=0))
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="sft-tinker"))
    resume: ResumeConfig = field(default_factory=ResumeConfig)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main(
    config: Config,
    rlor_mgr: TrainerJobManager | None = None,
    deploy_mgr: DeploymentManager | None = None,
):
    cfg = config

    validate_config(cfg.base_model, cfg.dataset, cfg.hotload, cfg.deployment, cfg.infra, cfg.resume)
    setup_wandb(
        cfg.wandb,
        {
            "lr": cfg.learning_rate,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "grad_accum": cfg.grad_accum,
        },
    )

    if not cfg.tokenizer_model:
        raise ValueError(
            "Config.tokenizer_model is required for chat template formatting. "
            "Set it to the HuggingFace model name (e.g. 'Qwen/Qwen3-1.7B')."
        )

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

    endpoint = create_trainer_job(
        rlor_mgr,
        base_model=cfg.base_model,
        infra=cfg.infra,
        lora_rank=cfg.lora_rank,
        max_seq_len=cfg.max_seq_len,
        learning_rate=cfg.learning_rate,
        grad_accum=cfg.grad_accum,
        display_name="sft-trainer",
        hot_load_deployment_id=cfg.deployment.deployment_id,
    )
    client = ReconnectableClient(rlor_mgr, endpoint.job_id, cfg.base_model, cfg.lora_rank)

    weight_syncer = WeightSyncer(
        policy_client=client.inner,
        deploy_mgr=deploy_mgr,
        deployment_id=cfg.deployment.deployment_id,
        base_model=cfg.base_model,
        hotload_timeout=cfg.hotload.hot_load_timeout,
        first_checkpoint_type=cfg.hotload.first_checkpoint_type,
        compression_format=DEFAULT_DELTA_COMPRESSION,
    )

    # -- Prepare data ------------------------------------------------------

    import json

    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.tokenizer_model, trust_remote_code=True)

    raw_data: List[Dict[str, Any]] = []
    with open(cfg.dataset) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_data.append(json.loads(line))
            if cfg.max_examples and len(raw_data) >= cfg.max_examples:
                break
    logger.info("Loaded %d examples from %s", len(raw_data), cfg.dataset)

    training_data: List[Dict[str, Any]] = []
    filtered_count = 0
    for row in raw_data:
        messages = row.get("messages", [])
        if not messages:
            continue

        full_tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)
        if len(full_tokens) > cfg.max_seq_len or len(full_tokens) < 2:
            filtered_count += 1
            continue

        prompt_messages = [m for m in messages if m.get("role") != "assistant"]
        prompt_tokens = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        training_data.append({"tokens": full_tokens, "prompt_len": len(prompt_tokens)})

    if filtered_count > 0:
        logger.info(
            "Seq-length filter: %d/%d examples filtered (len > %d or len < 2)",
            filtered_count,
            len(raw_data),
            cfg.max_seq_len,
        )
    logger.info("Prepared %d training examples", len(training_data))
    if not training_data:
        raise RuntimeError("No valid training examples after tokenization")

    step_offset, _ = setup_resume(client, cfg.resume)
    adam_params = tinker.AdamParams(learning_rate=cfg.learning_rate, **DEFAULT_ADAM)

    # -- Training loop (batched) -------------------------------------------

    batch_size = cfg.batch_size
    step = step_offset
    total_steps = len(training_data) * cfg.epochs // (cfg.grad_accum * batch_size)
    accum = 0
    agg_loss_sum = 0.0
    agg_resp_tokens = 0

    def _create_datum(tokens: list[int]) -> tinker.Datum:
        return tinker.Datum(
            model_input=tinker.ModelInput.from_ints(tokens[:-1]),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData(data=tokens[1:], dtype="int64", shape=[len(tokens) - 1]),
            },
        )

    def _flush_batch(batch_buf: list[dict], step: int, accum: int) -> tuple[int, int]:
        """Send a batch through forward_backward_custom and return (step, accum)."""
        nonlocal agg_loss_sum, agg_resp_tokens

        datums = [_create_datum(ex["tokens"]) for ex in batch_buf]
        prompt_counts = [ex["prompt_len"] for ex in batch_buf]

        loss_fn = make_batch_sft_loss_fn(prompt_counts)
        with timer("fwd_bwd"):
            result = client.forward_backward_custom(datums, loss_fn)

        fwd_metrics = result.metrics
        agg_loss_sum += fwd_metrics.get("ce_loss_sum", 0.0)
        agg_resp_tokens += fwd_metrics.get("response_tokens", 0)
        accum += 1

        if accum >= cfg.grad_accum:
            with timer("optim_step"):
                optim_result = client.optim_step(adam_params)
            step += 1
            accum = 0

            hl = cfg.hotload
            if hl.hot_load_interval > 0 and step % hl.hot_load_interval == 0:
                with timer("weight_sync"):
                    weight_syncer.save_and_hotload(f"step-{step}")
            if hl.dcp_save_interval > 0 and step % hl.dcp_save_interval == 0:
                with timer("dcp_save"):
                    weight_syncer.save_dcp(f"step-{step}")

            step_metrics: Dict[str, Any] = flush_timing()

            if optim_result and hasattr(optim_result, "metrics") and optim_result.metrics:
                for k, v in optim_result.metrics.items():
                    step_metrics[f"train/{k}"] = v

            if agg_resp_tokens > 0:
                avg_loss = agg_loss_sum / agg_resp_tokens
                ppl = torch.exp(torch.tensor(avg_loss)).item()
                logger.info(
                    "Step %d/%d | Loss: %.4f | PPL: %.2f",
                    step,
                    total_steps,
                    avg_loss,
                    ppl,
                )
                log_metrics_json(step, ce_loss=avg_loss, ppl=ppl)
                step_metrics.update({
                    "train/step": step,
                    "train/ce_loss": avg_loss,
                    "train/ppl": ppl,
                })
                wandb_log(step_metrics, step)

            agg_loss_sum = 0.0
            agg_resp_tokens = 0

        return step, accum

    for _epoch in range(cfg.epochs):
        batch_buffer: list[dict] = []
        for ex in training_data:
            batch_buffer.append(ex)
            if len(batch_buffer) >= batch_size:
                step, accum = _flush_batch(batch_buffer, step, accum)
                batch_buffer = []

        if batch_buffer:
            step, accum = _flush_batch(batch_buffer, step, accum)

    if accum > 0:
        client.optim_step(adam_params)
        step += 1

    # -- Final checkpoint --------------------------------------------------

    if step > step_offset:
        weight_syncer.save_dcp(f"step-{step}")
    if cfg.hotload.hot_load_interval > 0 or cfg.deployment.deployment_id:
        weight_syncer.save_and_hotload(f"final-step-{step}")

    logger.info("Training complete: %d optimizer steps", step)
    wandb_finish()
    return {"steps": step, "job_id": client.job_id}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(Config())
