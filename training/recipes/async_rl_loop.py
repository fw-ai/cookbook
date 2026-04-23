#!/usr/bin/env python3
"""Async RL recipe -- training mechanics only.

One extension point: ``rollout_fn(row, ctx) -> Rollout | None``.  The user
owns everything about the rollout (sampling, grading, multi-turn, remote
agents, per-turn logging).  The recipe owns everything about the training
side (infra provisioning, loss, optimizer, weight sync, gate-native async
off-policy scheduling).

The recipe always uses the client-side ``forward_backward_custom`` loss
path so the same code works for every supported loss (GRPO, DAPO, GSPO,
CISPO, REINFORCE, etc.), multi-turn masking, and custom corrections.
Users who want the server-side builtin kernel for a supported loss can
replace the ``fwd_bwd_one`` body with ``resolve_builtin_loss`` +
``build_builtin_loss_datums`` + ``policy.forward_backward``; see
``training/recipes/rl_loop.py`` for the dual-path pattern.

The recipe is intentionally stripped-down: no resume, no intra-training
checkpoint ladder, no orchestrator telemetry, no pre-training hotload.
Fork it and add whatever your production harness needs on top; see
``training/examples/`` for working scaffolds.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time as _time
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import tinker
import transformers

from fireworks.training.sdk import DeploymentManager, TrainerJobManager
from fireworks.training.sdk.client import GradAccNormalization
from fireworks.training.sdk.deployment import (
    AdaptiveConcurrencyController,
    DeploymentSampler,
)
from fireworks.training.sdk.weight_syncer import WeightSyncer
from training.utils import (
    DEFAULT_ADAM,
    ConcurrencyConfig,
    DeployConfig,
    InfraConfig,
    WandBConfig,
    WeightSyncConfig,
    load_jsonl_dataset,
    read_api_extra_headers_env,
    setup_wandb,
    validate_config,
    wandb_log,
)
from training.utils.checkpoint_utils import CheckpointKind, save_checkpoint
from training.utils.rl import PromptGroup, setup_infra
from training.utils.rl.async_train import run_async_rl_loop
from training.utils.rl.cispo import CISPOConfig
from training.utils.rl.dapo import DAPOConfig
from training.utils.rl.gspo import GSPOConfig
from training.utils.rl.losses import build_loss_fn, combine_prompt_groups
from training.utils.rl.metrics import compute_step_metrics
from training.utils.rl.tis import TISConfig
from training.utils.rl.train import DynamicFilterFn, TrainStepFns
from training.utils.rl.rollout import Rollout, rollout_to_prompt_group
from training.utils.timer import flush_timing, timer

logger = logging.getLogger(__name__)

__all__ = ["Config", "RolloutContext", "RolloutFn", "main"]


@dataclass
class Config:
    log_path: str
    base_model: str = "accounts/fireworks/models/qwen3-8b"
    dataset: str | None = None
    """JSONL dataset path or URL.  Optional -- ``main(..., rows=...)`` also
    accepts rows directly for users building their dataset in Python."""

    learning_rate: float = 1e-5
    kl_beta: float = 0.001
    completions_per_prompt: int = 4
    max_completion_tokens: int = 1024
    temperature: float = 1.0
    epochs: int = 1
    max_rows: int = 100
    max_seq_len: int | None = None
    lora_rank: int = 0

    prompt_groups_per_step: int = 1
    max_head_offpolicy_versions: int = 0
    """Staleness budget in optimizer-step versions.  ``0`` = strict on-policy."""
    sample_max_concurrency: int | None = None
    weight_sync_interval: int = 1

    grad_accumulation_normalization: GradAccNormalization | str | None = (
        GradAccNormalization.NUM_LOSS_TOKENS
    )

    policy_loss: str = "grpo"
    dapo: DAPOConfig = field(default_factory=DAPOConfig)
    gspo: GSPOConfig = field(default_factory=GSPOConfig)
    cispo: CISPOConfig = field(default_factory=CISPOConfig)
    eps_clip: float = 0.2
    eps_clip_high: float | None = None
    ratio_log_cap: float = 20.0
    tis: TISConfig = field(default_factory=TISConfig)

    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)

    infra: InfraConfig = field(default_factory=InfraConfig)
    deployment: DeployConfig = field(default_factory=DeployConfig)
    weight_sync: WeightSyncConfig = field(default_factory=WeightSyncConfig)
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="rl-async"))


@dataclass
class RolloutContext:
    """What a ``rollout_fn`` receives to do its job.

    All fields are live: ``current_version()`` returns the up-to-date version
    counter at call time, so multi-turn rollouts that span a hotload see the
    new version on later segments.
    """

    sampler: DeploymentSampler
    tokenizer: Any
    completions_per_prompt: int
    sample_kwargs: dict[str, Any]
    inference_url: str
    api_key: str
    model: str
    current_version: Callable[[], int]


RolloutFn = Callable[[dict, RolloutContext], Awaitable[Rollout | None]]


def main(
    config: Config,
    *,
    rollout_fn: RolloutFn,
    dynamic_filter_fn: DynamicFilterFn | None = None,
    rows: list[dict] | None = None,
) -> None:
    """Run the async RL loop with a user-supplied rollout function."""
    cfg = config

    def _signal_handler(signum, _):
        name = signal.Signals(signum).name
        raise SystemExit(f"Terminated by {name}")

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    if rows is None and not cfg.dataset:
        raise ValueError("Provide either cfg.dataset or rows= to main().")
    if not cfg.deployment.tokenizer_model:
        raise ValueError("deployment.tokenizer_model is required.")

    validate_config(cfg.base_model, cfg.dataset or "", cfg.weight_sync, cfg.deployment)
    setup_wandb(
        cfg.wandb,
        {
            "completions_per_prompt": cfg.completions_per_prompt,
            "prompt_groups_per_step": cfg.prompt_groups_per_step,
            "max_head_offpolicy_versions": cfg.max_head_offpolicy_versions,
            "kl_beta": cfg.kl_beta,
            "lr": cfg.learning_rate,
        },
    )

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    additional_headers = read_api_extra_headers_env()

    rlor_mgr = TrainerJobManager(
        api_key=api_key, base_url=base_url, additional_headers=additional_headers,
    )
    deploy_mgr = DeploymentManager(
        api_key=api_key, base_url=base_url, additional_headers=additional_headers,
    )

    with ExitStack() as stack:
        infra = setup_infra(
            rlor_mgr=rlor_mgr,
            deploy_mgr=deploy_mgr,
            base_model=cfg.base_model,
            infra_cfg=cfg.infra,
            deploy_cfg=cfg.deployment,
            lora_rank=cfg.lora_rank,
            max_seq_len=cfg.max_seq_len,
            learning_rate=cfg.learning_rate,
            needs_reference=(cfg.kl_beta > 0),
            needs_inference=True,
            role_prefix="rl-async",
            api_key=api_key,
            cleanup=None,
        )
        for closeable in infra.closeables:
            stack.callback(closeable.close)

        wandb_log(infra.boot_metrics, step=0)

        policy = infra.policy
        reference = infra.reference

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            cfg.deployment.tokenizer_model, trust_remote_code=True,
        )
        initial_window = cfg.concurrency.initial_window or (8 * infra.deployment_gpu_count)
        concurrency_controller = AdaptiveConcurrencyController(
            initial_window=initial_window,
            min_window=cfg.concurrency.min_window,
            max_window=cfg.concurrency.max_window,
            prefill_queue_target=cfg.concurrency.prefill_queue_target,
        )
        sampler = DeploymentSampler(
            inference_url=deploy_mgr.inference_url,
            model=infra.inference_model,
            api_key=api_key,
            tokenizer=tokenizer,
            concurrency_controller=concurrency_controller,
        )
        weight_syncer = WeightSyncer(
            policy_client=policy.inner,
            deploy_mgr=deploy_mgr,
            deployment_id=infra.deployment_id,
            base_model=cfg.base_model,
            hotload_timeout=cfg.weight_sync.weight_sync_timeout,
            first_checkpoint_type=cfg.weight_sync.first_checkpoint_type,
            lora_rank=cfg.lora_rank,
        )

        current_version = 0

        if rows is None:
            raw_dataset = load_jsonl_dataset(cfg.dataset, cfg.max_rows)
            rows = raw_dataset * cfg.epochs

        adam_params = tinker.AdamParams(learning_rate=cfg.learning_rate, **DEFAULT_ADAM)
        client_loss_builder = build_loss_fn(
            policy_loss=cfg.policy_loss,
            kl_beta=cfg.kl_beta,
            dapo_config=cfg.dapo,
            gspo_config=cfg.gspo,
            cispo_config=cfg.cispo,
            ratio_log_cap=cfg.ratio_log_cap,
            tis_config=cfg.tis,
            eps_clip=cfg.eps_clip,
            eps_clip_high=cfg.eps_clip_high,
        )

        sample_kwargs: dict = dict(
            max_tokens=cfg.max_completion_tokens,
            temperature=cfg.temperature,
            max_seq_len=infra.max_seq_len,
            http_timeout=cfg.deployment.sample_timeout,
            logprobs=True,
        )

        ctx = RolloutContext(
            sampler=sampler,
            tokenizer=tokenizer,
            completions_per_prompt=cfg.completions_per_prompt,
            sample_kwargs=sample_kwargs,
            inference_url=deploy_mgr.inference_url,
            api_key=api_key,
            model=infra.inference_model,
            current_version=lambda: current_version,
        )

        async def sample_one_prompt(row: dict) -> PromptGroup | None:
            try:
                rollout = await rollout_fn(row, ctx)
            except Exception as e:
                logger.warning("rollout_fn failed: %s", e)
                return None
            if rollout is None:
                return None
            return rollout_to_prompt_group(
                rollout, with_reference=(reference is not None),
            )

        def ref_forward(groups: list[PromptGroup]) -> None:
            if reference is None:
                return
            all_ref_data = [d for pg in groups for d in pg.ref_data]
            ref_fwd = reference.forward(all_ref_data, "cross_entropy")
            idx = 0
            for pg in groups:
                n = len(pg.ref_data)
                pg.ref_logprobs = [
                    ref_fwd.loss_fn_outputs[idx + i]["logprobs"].data for i in range(n)
                ]
                idx += n

        def fwd_bwd_one(prompt_groups: list[PromptGroup]):
            # Always use forward_backward_custom -- supports every registered
            # policy_loss, kl_beta > 0, and per-token loss_mask for multi-turn.
            # For the server-side builtin kernel pattern see rl_loop.py.
            data, adv, ref_lp, prompt_lens, inf_lp = combine_prompt_groups(prompt_groups)
            prox_fwd = policy.forward(data, "cross_entropy")
            prox_lp = [prox_fwd.loss_fn_outputs[i]["logprobs"].data for i in range(len(data))]
            return policy.forward_backward_custom(
                data, client_loss_builder(adv, ref_lp, prompt_lens, inf_lp, prox_lp),
            )

        def train_step(
            step: int, prompt_groups: list[PromptGroup], loop_stats: dict | None = None,
        ) -> tuple[int, dict]:
            t0 = _time.time()
            ref_forward(prompt_groups)
            logger.info("[step %d] ref_forward (%.1fs)", step + 1, _time.time() - t0)

            t0 = _time.time()
            fwd_bwd_result = fwd_bwd_one(prompt_groups)
            logger.info("[step %d] fwd_bwd (%.1fs)", step + 1, _time.time() - t0)

            t0 = _time.time()
            optim_result = policy.optim_step(
                adam_params,
                grad_accumulation_normalization=cfg.grad_accumulation_normalization,
            )
            step += 1
            logger.info("[step %d] optim_step (%.1fs)", step, _time.time() - t0)

            metrics = compute_step_metrics(
                prompt_groups=prompt_groups,
                fwd_bwd_results=[fwd_bwd_result],
                optim_result=optim_result,
                n_accum=1,
                timing_metrics=flush_timing(),
                loop_stats=loop_stats,
                completions_per_prompt=cfg.completions_per_prompt,
            )
            metrics["train/step"] = step
            logger.info(
                "Step %d | Reward %.3f | Acc %.1f%% | KL %.4f",
                step,
                metrics.get("rollout/reward", 0.0),
                metrics.get("rollout/accuracy", 0.0) * 100,
                metrics.get("train/mean_kl", 0.0),
            )
            wandb_log(metrics, step)
            return step, metrics

        def _weight_sync(step: int) -> None:
            nonlocal current_version
            with timer("weight_sync"):
                weight_syncer.save_and_hotload(f"step-{step}")
            current_version = step

        def _metrics_cb(loop_metrics: dict) -> None:
            for k, v in concurrency_controller.step_completed().items():
                loop_metrics[f"concurrency/{k}"] = v
            wandb_log(loop_metrics, step=loop_metrics.get("train/step", 0))

        global_step = asyncio.run(
            run_async_rl_loop(
                sample_fns=(sample_one_prompt(row) for row in rows),
                train_fns=TrainStepFns(train_step=train_step),
                prompt_groups_per_step=cfg.prompt_groups_per_step,
                max_head_offpolicy_versions=cfg.max_head_offpolicy_versions,
                weight_sync_fn=_weight_sync,
                weight_sync_interval=cfg.weight_sync_interval,
                max_concurrent=cfg.sample_max_concurrency,
                dynamic_filter_fn=dynamic_filter_fn,
                metrics_callback=_metrics_cb,
            )
        )

        if global_step > 0:
            save_checkpoint(
                policy, f"step-{global_step}", cfg.log_path,
                {"step": global_step},
                kind=CheckpointKind.BOTH,
                base_model=cfg.base_model,
                training_shape=infra.training_shape_id,
            )
