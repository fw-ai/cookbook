#!/usr/bin/env python3
"""GRPO training loop with concurrent rollout.

A readable, modifiable RL training loop using the Fireworks RLOR API.
Fork this script and customise the reward function, loss, or sampling
strategy to fit your task.

Each optimizer step samples ``prompt_groups_per_step`` prompts concurrently,
then runs a single training update + ``optim_step`` (1:1 ratio).

RL losses can execute in two places, picked **explicitly** by ``cfg.loss_path``:
- ``"builtin"``: ``forward_backward(...)`` with a server-side fused kernel,
  configured via :func:`training.utils.rl.losses.get_builtin_loss_config`.
- ``"client"``: ``forward_backward_custom(...)`` with a Python loss closure
  built by :func:`training.utils.rl.losses.build_loss_fn`.

``validate_loss_path`` runs at startup and raises with an actionable message
if the chosen path is incompatible with this run's config/profile (e.g.
``"builtin"`` + ``kl_beta>0``).

Usage:
    export FIREWORKS_API_KEY=...
    python -m recipes.rl_loop
"""

from __future__ import annotations

import os
import re
import json
import math
import signal
import asyncio
import logging
from contextlib import ExitStack
from typing import Any, Awaitable, Callable, List, Optional
from dataclasses import field, dataclass

import tinker

from training.utils.client import GradAccNormalization
from training.utils import (
    CLEANUP_DEPLOYMENT_ON_CLOSE_SCALE_TO_ZERO,
    DEFAULT_ADAM,
    AdaptiveConcurrencyController,
    ConcurrencyConfig,
    TrainerConfig,
    RunnerConfig,
    RunnerIO,
    RunStatus,
    WandBConfig,
    DeployConfig,
    RawRowCursor,
    RLPromptDataset,
    build_service_client,
    wandb_log,
    setup_wandb,
    wandb_finish,
    validate_config,
    log_metrics_json,
    compute_advantages,
    load_deployment_tokenizer,
    read_api_extra_headers_env,
    load_jsonl_dataset,
    prepare_sampling_messages,
    build_renderer,
    ReconnectableClient,
)
from tinker_cookbook.renderers import get_text_content
from training.utils.checkpoints import TrainingCheckpoints, validate_warm_start_config
from training.utils.rl import PromptGroup
from training.utils.rl.common import align_sample_logprobs_to_target_tokens
from training.utils.rl.tis import TISConfig
from training.utils.timer import timer, flush_timing
import time as _time
from training.utils.rl.dapo import DAPOConfig
from training.utils.rl.dro import DROConfig
from training.utils.rl.gspo import GSPOConfig
from training.utils.rl.cispo import CISPOConfig
from training.train_loop import TrainStepFns, raw_rows_from_stats, run_batched_training_loop
from training.utils.rl.losses import (
    LossPath,
    PolicyLoss,
    build_builtin_loss_datums,
    build_loss_fn,
    combine_prompt_groups,
    get_builtin_loss_config,
    validate_loss_path,
)
from training.utils.rl.metrics import compute_step_metrics
from training.utils.rl.router_replay import build_r3_routing_matrices
from training.utils.runner_state import start_running, write_completed, write_running_step

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    log_path: str
    """Directory for checkpoints and logs. Required, no default."""

    base_model: str = "accounts/fireworks/models/qwen3-8b"
    dataset: str = "https://raw.githubusercontent.com/eval-protocol/python-sdk/main/development/gsm8k_sample.jsonl"

    learning_rate: float = 1e-5
    kl_beta: float = 0.001
    completions_per_prompt: int = 4
    max_completion_tokens: int = 1024
    temperature: float = 1.0
    epochs: int = 1
    max_rows: int = 100
    max_seq_len: int | None = None
    """Max sequence length for sampling and training.  When using training
    shapes, this is auto-populated from the shape's
    ``max_supported_context_length``.  Must be set manually on the
    manual path (no training shape)."""
    lora_rank: int = 0
    lora_alpha: int | None = 32
    """LoRA alpha scaling factor. Ignored when ``lora_rank == 0``.

    Defaults to ``32`` to match Tinker and the Training SDK
    ``DEFAULT_LORA_ALPHA``. Override when you need a different scaling factor."""
    prompt_groups_per_step: int = 1
    """Number of prompt groups per optimizer step.

    All groups are collected before a single ``forward_backward_custom`` +
    ``optim_step`` pair fires (1:1 ratio)."""

    router_replay: bool = False
    router_replay_completion_only: bool = True
    """Keep R3 on completion-only replay by default.

    Full-sequence replay requires echo=True, which currently causes a
    significant serving slowdown. Do not set this to False until the echo
    performance bug is fixed."""

    grad_accumulation_normalization: GradAccNormalization | str | None = None
    """Optional server-side normalization for accumulated gradients.
    ``None`` leaves accumulated gradients unchanged."""

    grad_clip_norm: float = 0.0
    """Max gradient norm for clipping. 0 disables clipping."""

    policy_loss: PolicyLoss = "grpo"
    """One of the registered RL policy losses (see :data:`PolicyLoss`)."""

    loss_path: LossPath = "client"
    """Which forward/backward path to use:

    - ``"builtin"`` -- server-side ``forward_backward(...)`` with a fused
      kernel. Faster, but cannot apply KL (``kl_beta`` must be 0).
    - ``"client"`` -- client-side ``forward_backward_custom(...)``. Always
      works; slower because of an extra forward pass for old-policy logprobs.

    Validated at startup by :func:`validate_loss_path` -- mismatches raise,
    they no longer silently fall back.
    """

    dapo: DAPOConfig = field(default_factory=DAPOConfig)
    dro: DROConfig = field(default_factory=DROConfig)
    gspo: GSPOConfig = field(default_factory=GSPOConfig)
    cispo: CISPOConfig = field(default_factory=CISPOConfig)
    eps_clip: float = 0.2
    """PPO clip epsilon for the off-policy ratio (GRPO only)."""
    eps_clip_high: float | None = None
    """Asymmetric upper clip bound (GRPO only)."""
    ratio_log_cap: float = 20.0
    """Log-ratio clamp for ``policy_loss="importance_sampling"``."""
    ppo_n_minibatches: int = 1
    """Number of inner PPO minibatches per rollout batch.

    Each rollout batch snapshots ``old_policy_logprobs`` followed by
    ``ppo_n_minibatches`` × (``forward_backward`` + ``optim_step``). When
    >1, the policy drifts across inner steps, so ``old_policy_logprobs`` anchors the
    PPO ratio and the clip does real work. ``1`` uses the default 1:1
    behavior."""
    tis: TISConfig = field(default_factory=TISConfig)
    """TIS (Train-Inference IS) weight correction config."""

    separate_tis: bool = False
    """Train-inference correction strategy for ``policy_loss='importance_sampling'``.

    - ``False`` (default — Tinker on-policy IS): anchor the ratio directly on the
      inference/sampler logprobs, ``ratio = exp(pi - inf)`` (bounded only by
      ``ratio_log_cap``). The train-inference gap is corrected *inside* the ratio,
      two-sided and unclamped; the per-step old-policy forward is skipped and the
      ``TISConfig`` (``cap``/``level``) has no effect. Requires
      ``ppo_n_minibatches == 1``.
    - ``True`` (separate TIS): snapshot old-policy logprobs from a trainer forward
      so the ratio measures policy drift ``exp(pi - old_policy)``, and correct the
      train-inference gap with a *separate, clamped* weight
      ``clamp(exp(old_policy - inf), max=TISConfig.cap)`` applied per ``TISConfig.level``
      (token or sequence). This is the only mode where ``TISConfig`` matters and
      the only mode compatible with ``ppo_n_minibatches > 1``.

    Note the two modes coincide exactly while the TIS clamp does not bind:
    ``exp(pi - old_policy) * exp(old_policy - inf) == exp(pi - inf)``. They differ only when
    ``|old_policy - inf|`` exceeds the cap, where the separate-TIS path clips the
    correction (asymmetrically when ``cap <= 1``) and the default does not.

    Ignored for non-IS losses (GRPO/DAPO/GSPO/CISPO), which always snapshot
    old-policy logprobs for the PPO ratio and apply TIS as a composable weight."""

    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    """Concurrency control for inference sampling.  ``"fixed"`` (default)
    uses a static semaphore; ``"adaptive"`` adjusts the window based on
    server-side prefill queue latency.  Adaptive mode requires ``stream=True``."""

    trajectory_dir: str | None = None
    """Directory to save per-step trajectory JSONL files.  Each file contains
    prompts, completions, and rewards for every prompt group in that step."""

    init_from_checkpoint: str | None = None
    """Load pretrained DCP weights on a fresh dataset. Supports cross-job
    format ``"job_id:checkpoint_name"``."""

    warm_start_from_adapter: str | None = None
    """GCS URI of an HF PEFT adapter directory. When set, initializes LoRA
    weights from the adapter at training start (weights-only, fresh optimizer).
    Mutually exclusive with ``init_from_checkpoint``. Requires ``lora_rank > 0``."""

    output_model_id: str | None = None
    save_final_checkpoint: bool = True

    step_timeout: int = 0
    """Timeout in seconds for forward_backward / optim_step calls.
    0 = use DEFAULT_TIMEOUT_S from training.utils.client."""

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    deployment: DeployConfig = field(default_factory=DeployConfig)
    weight_sync_interval: int = 1
    weight_sync_before_training: bool = False
    weight_sync_timeout: int = 600
    dcp_save_interval: int = 0
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="grpo-tinker"))
    cleanup_on_exit: bool = True
    """Clean up SDK-created trainer/deployment resources on close."""

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
# Reward function -- customise this for your task
# ---------------------------------------------------------------------------


def extract_answer(text: str) -> Optional[str]:
    match = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    digits = re.search(r"(-?\d+)", match.group(1))
    return digits.group(1) if digits else None


def reward_fn(completion: str, row: dict) -> float:
    """Return 1.0 if the model's numeric answer matches the ground truth."""
    predicted = extract_answer(completion)
    truth = extract_answer(str(row.get("ground_truth", "")))
    if predicted is None or truth is None:
        return 0.0
    return 1.0 if predicted == truth else 0.0


def _response_text_for_grading(renderer, sampled) -> str:
    """Return the model's response-channel text for reward grading.

    Parses the completion TOKENS through the renderer's ``parse_response``
    (which restores the prompt-prefilled ``<think>`` via
    ``_normalize_response_tokens`` and splits off the think block) and returns
    ``get_text_content`` -- the post-think answer, matching how the chat
    template structures generations. This grades the model's response channel
    rather than the raw completion text (which, for thinking models, still
    contains the reasoning because the prefilled ``<think>`` lives in the
    prompt).
    """
    message, _termination = renderer.parse_response(
        sampled.full_tokens[sampled.prompt_len :]
    )
    return get_text_content(message)


# ---------------------------------------------------------------------------
# Rollout filter -- customise this for your task
# ---------------------------------------------------------------------------


def should_accept(pg: PromptGroup) -> bool:
    """Reject groups where all rewards are identical (zero-variance).

    Passed to ``run_rl_loop`` as a pluggable filter.  Replace with your
    own logic (e.g. minimum reward threshold, response length filter).
    """
    return len(set(pg.rewards)) > 1


# ---------------------------------------------------------------------------
# Trajectory logging
# ---------------------------------------------------------------------------


def _dump_trajectory(trajectory_dir: str, step: int, prompt_groups: list[PromptGroup]) -> None:
    """Write per-step trajectory JSONL: one line per individual completion."""
    os.makedirs(trajectory_dir, exist_ok=True)
    path = os.path.join(trajectory_dir, f"step_{step:04d}.jsonl")
    n_records = 0
    with open(path, "w") as f:
        for pg_idx, pg in enumerate(prompt_groups):
            completions = pg.completions or []
            for comp_idx, comp_text in enumerate(completions):
                record = {
                    "step": step,
                    "prompt_group": pg_idx,
                    "completion_index": comp_idx,
                    "prompt": pg.prompt,
                    "completion": comp_text,
                    "reward": pg.rewards[comp_idx] if comp_idx < len(pg.rewards) else None,
                    "advantage": pg.advantages[comp_idx] if comp_idx < len(pg.advantages) else None,
                    "completion_len": pg.completion_lens[comp_idx] if comp_idx < len(pg.completion_lens) else None,
                    "truncated": pg.truncated[comp_idx] if comp_idx < len(pg.truncated) else None,
                    "ground_truth": pg.row_meta.get("ground_truth") if pg.row_meta else None,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_records += 1
    logger.info(
        "[step %d] Saved trajectory to %s (%d completions from %d groups)", step, path, n_records, len(prompt_groups)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    config: Config,
    *,
    sample_prompt_fn: Callable[..., Awaitable[PromptGroup | None]] | None = None,
):
    cfg = config
    runner = RunnerIO(cfg.runner)
    uses_recipe_sampler = sample_prompt_fn is None

    # Convert SIGTERM/SIGINT into exceptions so the finally block runs cleanup.
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
    if cfg.policy_loss == "importance_sampling" and not cfg.separate_tis and cfg.ppo_n_minibatches > 1:
        raise ValueError(
            "importance_sampling with the default inference-anchored ratio requires "
            "ppo_n_minibatches == 1 (the inference anchor cannot stand in for an old-policy "
            "snapshot across drifting inner steps). Set separate_tis=True to use a trainer "
            "old-policy snapshot with ppo_n_minibatches > 1."
        )
    completions_per_prompt = cfg.completions_per_prompt
    prompt_groups_per_step = cfg.prompt_groups_per_step
    if not cfg.deployment.tokenizer_model:
        raise ValueError(
            "deployment.tokenizer_model is required for client-side tokenization. "
            "Set it to the HuggingFace model name (e.g. 'Qwen/Qwen3-1.7B')."
        )
    setup_wandb(
        cfg.wandb,
        {
            "completions_per_prompt": completions_per_prompt,
            "prompt_groups_per_step": prompt_groups_per_step,
            "kl_beta": cfg.kl_beta,
            "loss_path": cfg.loss_path,
            "lr": cfg.learning_rate,
        },
    )

    # -- SDK-managed Tinker clients -----------------------------------------

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    additional_headers = read_api_extra_headers_env()

    runner.write_status(RunStatus.PENDING, message="provisioning")

    with runner, ExitStack() as stack:
        service = build_service_client(
            api_key=api_key,
            base_url=base_url,
            additional_headers=additional_headers,
            base_model=cfg.base_model,
            tokenizer_model=cfg.deployment.tokenizer_model,
            lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            max_context_length=cfg.max_seq_len,
            learning_rate=cfg.learning_rate,
            trainer=cfg.trainer,
            deployment=cfg.deployment,
            hotload_timeout_s=cfg.weight_sync_timeout,
            cleanup_trainer_on_close=cfg.cleanup_on_exit,
            cleanup_deployment_on_close=(
                CLEANUP_DEPLOYMENT_ON_CLOSE_SCALE_TO_ZERO
                if cfg.cleanup_on_exit
                else None
            ),
            reference_required=cfg.kl_beta > 0,
        )
        stack.callback(service.close)
        training_client = service.create_training_client(
            cfg.base_model,
            lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
        )
        runner.set_accelerator_info(
            service.accelerator_type,
            service.accelerator_count,
            profile=service.training_profile,
        )

        policy_job_id = service.trainer_job_id
        deployment_id = service.deployment_id

        policy = ReconnectableClient.from_training_client(
            training_client,
            base_model=cfg.base_model,
            lora_rank=cfg.lora_rank,
            job_id=policy_job_id,
            default_timeout=cfg.step_timeout or 3600,
            service=service,
        )
        policy_profile = None
        max_seq_len = service.max_context_length

        # The KL reference is optional in RL (only needed when kl_beta > 0).
        # The SDK owns the shared-vs-separate decision: LoRA without an explicit
        # reference shape reuses the policy session; full-param (or an explicit
        # reference_training_shape_id) provisions a separate frozen reference
        # trainer that `service` owns. Backend trainer creation selects a
        # LoRA-capable shape unless a LoRA-capable shape is pinned.
        # reference_job_id mirrors the policy job when shared, else the
        # separate reference trainer's id.
        reference = None
        reference_job_id = None
        if cfg.kl_beta > 0:
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

        tokenizer = load_deployment_tokenizer(cfg.deployment)
        # Renderer used to grade the model's response channel (see
        # _response_text_for_grading): restores the prompt-prefilled <think>
        # and strips the think block so reward sees the post-think answer.
        response_renderer = build_renderer(tokenizer, cfg.deployment.tokenizer_model)
        concurrency_controller = None
        sampler = None
        if uses_recipe_sampler:
            # Adaptive concurrency — window adjusts based on server-side prefill queue.
            # For fixed (no rate limiting), use FixedConcurrencyController instead.
            # Fallback scales with deployment replicas (see ConcurrencyConfig.initial_window).
            initial_window = cfg.concurrency.initial_window or (
                8 * (cfg.deployment.replica_count or 1)
            )
            concurrency_controller = AdaptiveConcurrencyController(
                initial_window=initial_window,
                min_window=cfg.concurrency.min_window,
                max_window=cfg.concurrency.max_window,
                prefill_queue_target=cfg.concurrency.prefill_queue_target,
            )
            logger.info(
                "Concurrency: adaptive (initial=%d, range=%d-%d, target_pq=%.2fs)",
                initial_window,
                cfg.concurrency.min_window,
                cfg.concurrency.max_window,
                cfg.concurrency.prefill_queue_target,
            )
            sampler = service.create_deployment_sampler(
                tokenizer=tokenizer, concurrency_controller=concurrency_controller,
            )

        ckpt = TrainingCheckpoints(
            policy,
            service,
            trainer_id=policy_job_id,
            log_path=cfg.log_path,
            lora_rank=cfg.lora_rank,
        )

        logger.info(
            "Training: prompt_groups_per_step=%d | completions_per_prompt=%d",
            prompt_groups_per_step,
            completions_per_prompt,
        )

        # -- Resume ---------------------------------------------------------------

        resume_info = ckpt.resume(
            init_from_checkpoint=cfg.init_from_checkpoint,
            warm_start_from_adapter=cfg.warm_start_from_adapter,
        )
        step_offset = resume_info.step if resume_info else 0
        wandb_log({"train/step": step_offset}, step_offset)

        if cfg.weight_sync_before_training or service.requires_initial_sampler_sync():
            logger.info("[step %d] weight sync: saving + loading...", step_offset)
            t0 = _time.time()
            with timer("weight_sync"):
                saved = policy.save_weights_for_sampler(
                    f"step-{step_offset}",
                    checkpoint_type="base",
                )
                service.hotload_sampler_snapshot(saved.path)
            logger.info("[step %d] weight sync: done (%.1fs)", step_offset, _time.time() - t0)

        # -- Prepare sampling and training --------------------------------------

        raw_dataset = load_jsonl_dataset(cfg.dataset, cfg.max_rows)
        all_rows = raw_dataset * cfg.epochs
        rl_dataset = RLPromptDataset(all_rows, prompts_per_step=prompt_groups_per_step)
        cursor = RawRowCursor(max_rows=len(all_rows))
        adam_kwargs = dict(DEFAULT_ADAM)
        adam_kwargs["grad_clip_norm"] = cfg.grad_clip_norm
        adam_params = tinker.AdamParams(learning_rate=cfg.learning_rate, **adam_kwargs)
        sample_kwargs: dict = dict(
            max_tokens=cfg.max_completion_tokens,
            temperature=cfg.temperature,
            # Full-distribution on-policy sampling. Without explicit top_p/top_k
            # the serving stack applies the model's generation_config.json
            # defaults (e.g. Qwen3.5: top_k=20/top_p=0.95), which truncate
            # rollouts and bias the policy-gradient estimator.
            top_p=1.0,
            top_k=0,
            max_seq_len=max_seq_len,
            http_timeout=cfg.deployment.sample_timeout,
        )
        if cfg.router_replay:
            sample_kwargs.update(
                include_routing_matrix=True,
                echo=not cfg.router_replay_completion_only,
                logprobs=True,
            )
        sample_kwargs["logprobs"] = True

        # -- Sample one prompt (VISIBLE -- customise this) ----------------------

        async def sample_one_prompt(row: dict, *, cursor_index: int) -> PromptGroup | None:
            """Sample completions for one prompt and return a PromptGroup."""
            if sample_prompt_fn is not None:
                return await sample_prompt_fn(row, cursor_index=cursor_index)
            if sampler is None:
                raise RuntimeError("live sampling requires a deployment sampler")
            messages = row.get("messages", [])
            input_messages = prepare_sampling_messages(messages)
            if not input_messages:
                return None

            try:
                sampled = await sampler.sample_with_tokens(
                    messages=input_messages,
                    n=completions_per_prompt,
                    **sample_kwargs,
                )
            except Exception as e:
                # HTTP 425 during deployment hot-load is counted as a sample failure.
                logger.warning("Sampling failed: %s", e)
                return None

            if not sampled or len(sampled) < completions_per_prompt:
                return None

            rewards = [reward_fn(_response_text_for_grading(response_renderer, s), row) for s in sampled]
            advantages = compute_advantages(rewards)

            prompt_len = sampled[0].prompt_len
            policy_data: List[tinker.Datum] = []
            reference_data: List[tinker.Datum] = []
            adv_filtered: List[float] = []
            inf_logprobs_aligned: List[List[float]] = []
            raw_inf_logprobs_aligned: List[List[float]] = []

            for idx, s in enumerate(sampled):
                tokens = s.full_tokens
                if len(tokens) < 2:
                    continue
                model_input_len = len(tokens) - 1

                rm = None
                if cfg.router_replay:
                    rm = build_r3_routing_matrices(
                        s.routing_matrices,
                        s.prompt_len,
                        model_input_len,
                        completion_only=cfg.router_replay_completion_only,
                    )

                policy_datum = tinker.Datum(
                    model_input=tinker.ModelInput.from_ints(tokens[:-1], routing_matrices=rm),
                    loss_fn_inputs={
                        "target_tokens": tinker.TensorData(data=tokens[1:], dtype="int64", shape=[model_input_len]),
                    },
                )
                policy_data.append(policy_datum)

                if reference is not None:
                    reference_datum = tinker.Datum(
                        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
                        loss_fn_inputs={
                            "target_tokens": tinker.TensorData(
                                data=tokens[1:], dtype="int64", shape=[model_input_len]
                            ),
                        },
                    )
                    reference_data.append(reference_datum)

                adv_filtered.append(advantages[idx])

                rollout_logprobs = align_sample_logprobs_to_target_tokens(
                    s,
                    attr="sampling_logprobs",
                    source="rollout_logprobs",
                    sample_idx=idx,
                    required=True,
                )
                raw_logprobs = align_sample_logprobs_to_target_tokens(
                    s,
                    attr="inference_logprobs",
                    source="raw inference logprobs",
                    sample_idx=idx,
                    required=False,
                )
                inf_logprobs_aligned.append(rollout_logprobs)
                raw_inf_logprobs_aligned.append(raw_logprobs or [])

            if not policy_data:
                return None

            comp_lens = [len(s.full_tokens) - s.prompt_len for s in sampled]
            trunc = [s.finish_reason == "length" for s in sampled]

            return PromptGroup(
                data=policy_data,
                ref_data=reference_data,
                advantages=adv_filtered,
                ref_logprobs=None,
                prompt_len=prompt_len,
                rewards=rewards,
                inf_logprobs=inf_logprobs_aligned,
                raw_inf_logprobs=raw_inf_logprobs_aligned,
                completion_lens=comp_lens,
                truncated=trunc,
                prompt=input_messages if cfg.trajectory_dir else None,
                completions=[s.text for s in sampled] if cfg.trajectory_dir else None,
                row_meta={"ground_truth": row.get("ground_truth", "")} if cfg.trajectory_dir else None,
            )

        # -- Training callbacks ----------------------------------------------------

        def ref_forward(groups: list[PromptGroup]) -> None:
            """Compute reference logprobs for all prompt groups (one call)."""
            if reference is None:
                return
            all_ref_data = [d for pg in groups for d in pg.ref_data]
            try:
                ref_fwd = reference.forward(all_ref_data, "cross_entropy")
            except Exception as e:
                raise RuntimeError(
                    f"Reference forward failed (batch of {len(all_ref_data)} datums): {e}\n"
                    "Possible causes: reference trainer crashed, NCCL timeout, or "
                    "request exceeded the default timeout. Check reference trainer "
                    "logs and consider increasing the client timeout."
                ) from e
            idx = 0
            for pg in groups:
                n = len(pg.ref_data)
                pg.ref_logprobs = [ref_fwd.loss_fn_outputs[idx + i]["logprobs"].data for i in range(n)]
                idx += n

        validate_loss_path(cfg, policy_profile)
        if cfg.loss_path == "builtin":
            builtin_loss = get_builtin_loss_config(cfg)
            client_loss_builder = None
            logger.info(
                "policy_loss=%s loss_path=builtin (server-side loss=%s)",
                cfg.policy_loss,
                builtin_loss[0],
            )
        else:
            builtin_loss = None
            client_loss_builder = build_loss_fn(cfg)
            logger.info(
                "policy_loss=%s loss_path=client (forward_backward_custom)",
                cfg.policy_loss,
            )
        use_rollout_logprobs = (
            cfg.policy_loss == "importance_sampling" and not cfg.separate_tis
        )
        logger.info(
            "use_rollout_logprobs=%s (derived from policy_loss=%s separate_tis=%s)",
            use_rollout_logprobs,
            cfg.policy_loss,
            cfg.separate_tis,
        )

        def fwd_bwd_minibatch(
            data,
            adv,
            ref_lp,
            prompt_lens,
            inf_lp,
            raw_inf_lp,
            old_policy_logprobs,
        ):
            """One inner PPO minibatch using builtin or client-side loss path.

            Callers pre-compute ``old_policy_logprobs`` once per rollout batch and pass
            a slice of the flattened rollout tensors for this minibatch.
            """
            if builtin_loss is not None:
                loss_name, loss_cfg = builtin_loss
                rl_datums = build_builtin_loss_datums(
                    data,
                    adv,
                    old_policy_logprobs,
                    inf_lp,
                    prompt_lens,
                    cfg.tis,
                    policy_loss=cfg.policy_loss,
                )
                return policy.forward_backward(
                    rl_datums,
                    loss_name,
                    loss_fn_config=loss_cfg,
                )

            assert client_loss_builder is not None
            return policy.forward_backward_custom(
                data,
                client_loss_builder(
                    adv,
                    ref_lp,
                    prompt_lens,
                    inf_lp,
                    old_policy_logprobs,
                    raw_inf_lp,
                ),
            )

        def train_step(
            step: int,
            prompt_groups: list[PromptGroup],
            loop_stats: dict | None = None,
        ) -> tuple[int, dict]:
            """ref_forward + old_policy_logprobs snapshot + num_minibatches x (fwd_bwd + optim_step) + metrics.

            ``num_minibatches = cfg.ppo_n_minibatches``. ``old_policy_logprobs``
            is snapshotted once per rollout batch and reused across every inner
            optim step so the PPO ratio measures genuine policy drift. DCP
            checkpoints fire only at rollout boundaries (cadence in rollout
            batches, not optim steps) so resume accounting is independent of
            the minibatch count.
            """
            if not prompt_groups:
                raise ValueError("train_step requires at least one prompt group")

            train_step_start = _time.time()
            t0 = _time.time()
            ref_forward(prompt_groups)
            logger.info("[step %d] ref_forward: done (%.1fs)", step + 1, _time.time() - t0)

            data, adv, ref_lp, prompt_lens, inf_lp, raw_inf_lp = combine_prompt_groups(
                prompt_groups,
                include_raw=True,
            )
            # Historical sync-loop behavior: IS without separate TIS uses
            # rollout_logprobs; other modes snapshot on the trainer.
            if use_rollout_logprobs:
                old_policy_logprobs = inf_lp
                logger.info(
                    "[step %d] old_policy_logprobs: using rollout_logprobs",
                    step + 1,
                )
            else:
                t0 = _time.time()
                old_policy_fwd = policy.forward(data, "cross_entropy")
                old_policy_logprobs = [
                    old_policy_fwd.loss_fn_outputs[i]["logprobs"].data
                    for i in range(len(data))
                ]
                logger.info("[step %d] old_policy_forward: done (%.1fs)", step + 1, _time.time() - t0)

            n = len(data)
            num_minibatches = max(1, cfg.ppo_n_minibatches)
            minibatch_size = max(1, math.ceil(n / num_minibatches))
            fwd_bwd_results: list = []
            optim_result: Any = None
            for minibatch_idx in range(num_minibatches):
                minibatch_start = minibatch_idx * minibatch_size
                minibatch_end = min(minibatch_start + minibatch_size, n)
                if minibatch_start >= minibatch_end:
                    break

                t0 = _time.time()
                fwd_bwd_results.append(fwd_bwd_minibatch(
                    data[minibatch_start:minibatch_end],
                    adv[minibatch_start:minibatch_end],
                    ref_lp[minibatch_start:minibatch_end],
                    prompt_lens[minibatch_start:minibatch_end],
                    inf_lp[minibatch_start:minibatch_end],
                    raw_inf_lp[minibatch_start:minibatch_end],
                    old_policy_logprobs[minibatch_start:minibatch_end],
                ))
                logger.info(
                    "[step %d] fwd_bwd (mb %d/%d): done (%.1fs)",
                    step + 1, minibatch_idx + 1, num_minibatches, _time.time() - t0,
                )

                t0 = _time.time()
                optim_result = policy.optim_step(
                    adam_params,
                    grad_accumulation_normalization=cfg.grad_accumulation_normalization,
                )
                step += 1
                logger.info(
                    "[step %d] optim_step (mb %d/%d): done (%.1fs)",
                    step, minibatch_idx + 1, num_minibatches, _time.time() - t0,
                )

            cursor.record(raw_rows_from_stats(loop_stats, accepted_rows=len(prompt_groups)))

            rollouts_completed = (step - step_offset) // num_minibatches
            dcp_interval = cfg.dcp_save_interval
            if dcp_interval > 0 and rollouts_completed > 0 and rollouts_completed % dcp_interval == 0:
                logger.info("[step %d] dcp_save...", step)
                t0 = _time.time()
                ckpt.save(
                    f"step-{step}",
                    resumable=True,
                    promotable=False,
                    data_consumed=cursor.value,
                )
                logger.info("[step %d] dcp_save: done (%.1fs)", step, _time.time() - t0)

            if loop_stats is not None:
                train_wall_time = _time.time() - train_step_start
                loop_stats["train_wall_time"] = train_wall_time
                rollout_wall_time = float(loop_stats.get("rollout_batch_wall_time", 0.0))
                loop_stats["scheduler_step_wall_time"] = rollout_wall_time + train_wall_time

            metrics = compute_step_metrics(
                prompt_groups=prompt_groups,
                fwd_bwd_results=fwd_bwd_results,
                optim_result=optim_result,
                n_accum=len(fwd_bwd_results),
                timing_metrics=flush_timing(),
                loop_stats=loop_stats,
                completions_per_prompt=completions_per_prompt,
            )
            metrics["train/step"] = step

            step_tokens = sum(
                len(d.loss_fn_inputs["target_tokens"].data) for pg in prompt_groups for d in pg.data
            )
            avg_reward = metrics.get("rollout/reward", 0.0)
            avg_acc = metrics.get("rollout/accuracy", 0.0)
            avg_ref_kl = metrics.get("train/ref_kl", 0.0)
            logger.info(
                "Step %d | Reward: %.3f | Acc: %.1f%% | RefKL: %.4f",
                step,
                avg_reward,
                avg_acc * 100,
                avg_ref_kl,
            )
            log_metrics_json(step, reward=avg_reward, accuracy=avg_acc, ref_kl=avg_ref_kl)
            wandb_log(metrics, step)

            total_rl_steps = len(rl_dataset) * max(1, cfg.ppo_n_minibatches) - step_offset
            write_running_step(
                runner,
                step=step,
                total_steps=total_rl_steps,
                metrics=metrics,
                tokens=step_tokens,
            )

            if cfg.trajectory_dir:
                _dump_trajectory(cfg.trajectory_dir, step, prompt_groups)

            return step, metrics

        # -- Run ----------------------------------------------------------------

        def _loop_metrics_callback(loop_metrics: dict) -> None:
            """Called by run_rl_loop after each train step with loop-level metrics."""
            if concurrency_controller is not None:
                cc_summary = concurrency_controller.step_completed()
                for k, v in cc_summary.items():
                    loop_metrics[f"concurrency/{k}"] = v
            wandb_log(loop_metrics, step=loop_metrics.get("train/step", 0))

        train_fns = TrainStepFns(train_step=train_step)

        # Prefer the persisted raw-row cursor; fall back to step-derived
        # progress for older checkpoints.
        rollouts_done = step_offset // max(1, cfg.ppo_n_minibatches)
        cursor.resume(
            resume_info.data_consumed if resume_info else None,
            fallback=rollouts_done * prompt_groups_per_step,
        )
        remaining_rows = all_rows[cursor.value:]

        total_rl_steps = len(rl_dataset) * max(1, cfg.ppo_n_minibatches) - step_offset
        start_running(runner, total_steps=total_rl_steps)

        global_step = asyncio.run(
            run_batched_training_loop(
                sample_fns=(
                    sample_one_prompt(row, cursor_index=cursor.value + offset)
                    for offset, row in enumerate(remaining_rows)
                ),
                train_fns=train_fns,
                prompt_groups_per_step=prompt_groups_per_step,
                dynamic_filter_fn=should_accept,
                global_step=step_offset,
                metrics_callback=_loop_metrics_callback,
                weight_sync_fn=(
                    lambda step: service.hotload_sampler_snapshot(
                        policy.save_weights_for_sampler(f"step-{step}").path
                    )
                    if cfg.weight_sync_interval > 0
                    else None
                ),
                weight_sync_interval=cfg.weight_sync_interval,
            )
        )

        # -- Final checkpoint ----------------------------------------------------

        if cfg.save_final_checkpoint and global_step > step_offset:
            try:
                cp_name = f"step-{global_step}"
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
            except Exception as e:
                logger.warning("Failed to save final checkpoint: %s", e)

        write_completed(runner, step=global_step, total_steps=total_rl_steps)
        logger.info("Training complete: %d steps", global_step)
        wandb_finish()
        return {
            "steps": global_step,
            "policy_job_id": policy_job_id,
            "reference_job_id": reference_job_id,
            "deployment_id": deployment_id,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config(
        log_path="./rl_logs",
        base_model="accounts/fireworks/models/qwen3-8b",
        deployment=DeployConfig(
            tokenizer_model="Qwen/Qwen3-8B",
        ),
    )
    main(cfg)
