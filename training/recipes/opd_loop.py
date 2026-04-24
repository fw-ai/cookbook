#!/usr/bin/env python3
"""On-Policy Distillation (OPD) training loop.

Implements the training loop from arXiv:2604.13016. A student model learns
from a stronger teacher model via on-policy knowledge distillation.

The student generates rollouts via a Fireworks inference deployment.
The teacher scores the same sequences via a separate deployment
(``echo=True, logprobs=True, top_logprobs=k``).  The student trains on a
sampled-token reverse-KL loss and logs all paper metrics (overlap ratio,
overlap advantage, entropy gap, overlap mass, per-position entropy).

Usage:
    export FIREWORKS_API_KEY=...
    python -m recipes.opd_loop
"""

from __future__ import annotations

import os
import math
import signal
import asyncio
import logging
from contextlib import ExitStack
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import field, dataclass

import torch
import tinker
import transformers

from fireworks.training.sdk import DeploymentManager, TrainerJobManager
from fireworks.training.sdk.client import GradAccNormalization
from fireworks.training.sdk.deployment import AdaptiveConcurrencyController, DeploymentSampler
from fireworks.training.sdk.weight_syncer import WeightSyncer
from training.utils import (
    DEFAULT_ADAM,
    ConcurrencyConfig,
    InfraConfig,
    ResourceCleanup,
    RunnerConfig,
    RunnerIO,
    RunStatus,
    WandBConfig,
    DeployConfig,
    WeightSyncConfig,
    RLPromptDataset,
    wandb_log,
    setup_wandb,
    wandb_finish,
    validate_config,
    log_metrics_json,
    read_api_extra_headers_env,
    load_jsonl_dataset,
    prepare_sampling_messages,
)
from training.utils.checkpoint_utils import (
    resolve_resume,
    save_checkpoint,
    CheckpointKind,
)
from training.utils.rl import PromptGroup, setup_infra
from training.utils.rl.train import TrainStepFns, run_rl_loop
from training.utils.rl.losses import combine_prompt_groups
from training.utils.timer import timer, flush_timing
from training.utils.opd_metrics import compute_opd_metrics, PositionLogprobs
import time as _time

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    log_path: str
    """Directory for checkpoints and logs. Required, no default."""

    base_model: str = "accounts/fireworks/models/qwen3-8b"
    rollout_base_model: str | None = None
    dataset: str = "https://raw.githubusercontent.com/eval-protocol/python-sdk/main/development/gsm8k_sample.jsonl"

    teacher_model: str = ""
    """Teacher model ID on Fireworks (e.g. ``accounts/fireworks/models/qwen3-235b-a22b``)."""

    teacher_deployment_id: str | None = None
    """Reuse an existing teacher deployment instead of creating one."""

    opd_mode: str = "sampled_token"
    """``"sampled_token"`` (Eq. 3) or ``"top_k"`` (Eq. 5).
    Start with sampled_token -- paper Section 6.3 shows it matches top_k."""

    opd_top_k: int = 16
    """Number of top logprobs to request from teacher for metrics computation."""

    teacher_temperature: float = 1.0
    """Temperature applied to teacher logprobs during scoring."""

    learning_rate: float = 1e-5
    completions_per_prompt: int = 1
    max_completion_tokens: int = 1024
    temperature: float = 1.0
    epochs: int = 1
    max_rows: int = 100
    max_seq_len: int | None = None
    lora_rank: int = 0

    prompt_groups_per_step: int = 1

    grad_accumulation_normalization: GradAccNormalization | str | None = GradAccNormalization.NUM_LOSS_TOKENS

    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)

    policy_job_id: str | None = None
    init_from_checkpoint: str | None = None
    output_model_id: str | None = None
    save_final_checkpoint: bool = True
    step_timeout: int = 0

    infra: InfraConfig = field(default_factory=InfraConfig)
    deployment: DeployConfig = field(default_factory=DeployConfig)
    weight_sync: WeightSyncConfig = field(default_factory=WeightSyncConfig)
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="opd-tinker"))
    runner: RunnerConfig = field(default_factory=RunnerConfig)


# ---------------------------------------------------------------------------
# Teacher scoring
# ---------------------------------------------------------------------------


def _extract_teacher_top_logprobs(
    response_content: list[dict],
) -> Tuple[list[float], PositionLogprobs]:
    """Extract per-token logprobs and top-k logprobs from API response content.

    ``response_content`` is the ``logprobs.content`` list from the completions
    API response. Each entry has ``logprob``, ``token_id``, and optionally
    ``top_logprobs`` (list of ``{token_id, logprob}`` dicts).
    """
    token_logprobs: list[float] = []
    top_k_logprobs: PositionLogprobs = []
    for entry in response_content:
        token_logprobs.append(entry.get("logprob", 0.0))
        top_lps = entry.get("top_logprobs")
        if top_lps and isinstance(top_lps, list):
            position_topk = [(tp.get("token_id", 0), tp.get("logprob", 0.0)) for tp in top_lps]
            top_k_logprobs.append(position_topk)
        else:
            top_k_logprobs.append([(entry.get("token_id", 0), entry.get("logprob", 0.0))])
    return token_logprobs, top_k_logprobs


async def score_with_teacher(
    sampler: DeploymentSampler,
    token_ids: list[int],
    *,
    top_logprobs: int = 16,
) -> Tuple[list[float], PositionLogprobs] | None:
    """Score a token sequence with the teacher deployment.

    Uses ``echo=True`` to get teacher logprobs on the student's sequence
    without generating new tokens.
    """
    try:
        result, _ = await sampler._stream_call(
            prompt_ids=token_ids,
            n=1,
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=top_logprobs,
            echo=True,
            raw_output=True,
        )
        choices = result.get("choices", [])
        if not choices:
            return None
        lp_data = choices[0].get("logprobs", {})
        content = lp_data.get("content", [])
        if not content:
            return None
        # Drop the first (unconditional) logprob to align with training
        content = content[1:]
        return _extract_teacher_top_logprobs(content)
    except Exception as e:
        logger.warning("Teacher scoring failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# OPD loss
# ---------------------------------------------------------------------------


def make_opd_loss_fn(
    teacher_logprobs: List[List[float]],
    prompt_lens: List[int],
) -> Any:
    """Sampled-token OPD loss (Eq. 3): minimize reverse KL at sampled tokens.

    ``loss = (1/N) * sum_t [log p_student(y_t) - log p_teacher(y_t)]``
    """
    def loss_fn(
        data: List[tinker.Datum],
        logprobs_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = torch.zeros((), dtype=torch.float32)
        total_tokens = 0
        total_student_nll = 0.0
        total_teacher_nll = 0.0

        for i, (datum, student_lp) in enumerate(zip(data, logprobs_list)):
            prompt_len = prompt_lens[i] if i < len(prompt_lens) else 1
            response_start = max(0, prompt_len - 1)
            resp_lp = student_lp[response_start:]

            teacher_lp = teacher_logprobs[i] if i < len(teacher_logprobs) else []
            teacher_resp_lp = teacher_lp[response_start:]

            n = min(len(resp_lp), len(teacher_resp_lp))
            if n == 0:
                continue

            student_resp = resp_lp[:n]
            teacher_resp = torch.tensor(
                teacher_resp_lp[:n],
                dtype=student_resp.dtype,
                device=student_resp.device,
            )

            # Sampled-token OPD loss: log p_student - log p_teacher
            token_loss = student_resp - teacher_resp
            total_loss = total_loss + token_loss.sum()
            total_tokens += n
            total_student_nll += float((-student_resp).sum().item())
            total_teacher_nll += float((-teacher_resp).sum().item())

        denom = max(total_tokens, 1)
        metrics = {
            "mean_loss": float(total_loss.item()) / denom,
            "student_nll": total_student_nll / denom,
            "teacher_nll": total_teacher_nll / denom,
            "active_tokens": total_tokens,
        }
        return total_loss, metrics

    return loss_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    config: Config,
    rlor_mgr: TrainerJobManager | None = None,
    deploy_mgr: DeploymentManager | None = None,
    cancel_on_exit: bool = False,
):
    cfg = config
    runner = RunnerIO(cfg.runner)

    def _signal_handler(signum, frame):
        name = signal.Signals(signum).name
        logger.warning("Received %s -- raising SystemExit for cleanup", name)
        raise SystemExit(f"Terminated by {name}")

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    if not cfg.teacher_model:
        raise ValueError("Config.teacher_model is required for OPD training.")

    validate_config(
        cfg.base_model,
        cfg.dataset,
        cfg.weight_sync,
        cfg.deployment,
        output_model_id=cfg.output_model_id,
    )
    completions_per_prompt = cfg.completions_per_prompt
    prompt_groups_per_step = cfg.prompt_groups_per_step
    if not cfg.deployment.tokenizer_model:
        raise ValueError(
            "deployment.tokenizer_model is required for client-side tokenization."
        )
    setup_wandb(
        cfg.wandb,
        {
            "teacher_model": cfg.teacher_model,
            "opd_mode": cfg.opd_mode,
            "opd_top_k": cfg.opd_top_k,
            "lr": cfg.learning_rate,
        },
    )

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

    runner.write_status(RunStatus.PENDING, message="provisioning")

    def _on_trainer_status(msg: str) -> None:
        runner.write_status(RunStatus.PENDING, message=msg)

    with runner, ResourceCleanup(rlor_mgr, deploy_mgr) as cleanup, ExitStack() as stack:
        infra = setup_infra(
            rlor_mgr=rlor_mgr,
            deploy_mgr=deploy_mgr,
            base_model=cfg.base_model,
            infra_cfg=cfg.infra,
            deploy_cfg=cfg.deployment,
            lora_rank=cfg.lora_rank,
            max_seq_len=cfg.max_seq_len,
            learning_rate=cfg.learning_rate,
            step_timeout=cfg.step_timeout,
            policy_job_id=cfg.policy_job_id,
            needs_reference=False,
            needs_inference=True,
            role_prefix="opd",
            api_key=api_key,
            cleanup=cleanup if cancel_on_exit else None,
            on_status=_on_trainer_status,
        )
        for closeable in infra.closeables:
            stack.callback(closeable.close)

        runner.set_accelerator_info(profile=infra.policy_profile)
        wandb_log(infra.boot_metrics, step=0)

        policy = infra.policy

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            cfg.deployment.tokenizer_model, trust_remote_code=True,
        )

        # Student sampler (for rollout generation)
        initial_window = cfg.concurrency.initial_window or (8 * infra.deployment_gpu_count)
        concurrency_controller = AdaptiveConcurrencyController(
            initial_window=initial_window,
            min_window=cfg.concurrency.min_window,
            max_window=cfg.concurrency.max_window,
            prefill_queue_target=cfg.concurrency.prefill_queue_target,
        )
        student_sampler = DeploymentSampler(
            inference_url=deploy_mgr.inference_url,
            model=infra.inference_model,
            api_key=api_key,
            tokenizer=tokenizer,
            concurrency_controller=concurrency_controller,
        )

        # Teacher sampler (for scoring student sequences)
        teacher_sampler = DeploymentSampler(
            inference_url=base_url,
            model=cfg.teacher_model,
            api_key=api_key,
            tokenizer=tokenizer,
        )

        weight_syncer = WeightSyncer(
            policy_client=policy.inner,
            deploy_mgr=deploy_mgr,
            deployment_id=infra.deployment_id,
            base_model=cfg.rollout_base_model or cfg.base_model,
            hotload_timeout=cfg.weight_sync.weight_sync_timeout,
            first_checkpoint_type=cfg.weight_sync.first_checkpoint_type,
            lora_rank=cfg.lora_rank,
        )

        logger.info(
            "OPD Training: teacher=%s | opd_mode=%s | top_k=%d",
            cfg.teacher_model, cfg.opd_mode, cfg.opd_top_k,
        )

        # -- Resume ------------------------------------------------------------

        resume_info = resolve_resume(
            policy, cfg.log_path, cfg.init_from_checkpoint, None,
        )
        step_offset = resume_info.step if resume_info else 0
        wandb_log({"train/step": step_offset}, step_offset)

        if cfg.weight_sync.weight_sync_before_training and infra.deployment_id:
            name = f"resume-{step_offset}-base" if step_offset > 0 else "step-0-base"
            weight_syncer.save_and_hotload(name, checkpoint_type="base")

        # -- Prepare dataset ---------------------------------------------------

        raw_dataset = load_jsonl_dataset(cfg.dataset, cfg.max_rows)
        all_rows = raw_dataset * cfg.epochs
        rl_dataset = RLPromptDataset(all_rows, prompts_per_step=prompt_groups_per_step)
        adam_params = tinker.AdamParams(learning_rate=cfg.learning_rate, **DEFAULT_ADAM)

        sample_kwargs: dict = dict(
            max_tokens=cfg.max_completion_tokens,
            temperature=cfg.temperature,
            max_seq_len=infra.max_seq_len,
            http_timeout=cfg.deployment.sample_timeout,
            logprobs=True,
        )

        # Storage for teacher logprob data needed during train_step
        _teacher_data_by_step: dict[int, list[Tuple[list[float], PositionLogprobs]]] = {}
        _current_step_counter = [step_offset]

        # -- Sample one prompt with teacher scoring ----------------------------

        async def sample_one_prompt(row: dict) -> PromptGroup | None:
            messages = row.get("messages", [])
            input_messages = prepare_sampling_messages(messages)
            if not input_messages:
                return None

            try:
                sampled = await student_sampler.sample_with_tokens(
                    messages=input_messages,
                    n=completions_per_prompt,
                    **sample_kwargs,
                )
            except Exception as e:
                logger.warning("Student sampling failed: %s", e)
                return None

            if not sampled or len(sampled) < completions_per_prompt:
                return None

            prompt_len = sampled[0].prompt_len
            policy_data: List[tinker.Datum] = []
            teacher_results: list[Tuple[list[float], PositionLogprobs]] = []
            inf_logprobs_aligned: List[List[float]] = []

            for s in sampled:
                tokens = s.full_tokens
                if len(tokens) < 2:
                    continue
                model_input_len = len(tokens) - 1

                # Score with teacher
                teacher_result = await score_with_teacher(
                    teacher_sampler, tokens, top_logprobs=cfg.opd_top_k,
                )
                if teacher_result is None:
                    continue
                teacher_token_lp, teacher_topk_lp = teacher_result
                teacher_results.append((teacher_token_lp, teacher_topk_lp))

                policy_datum = tinker.Datum(
                    model_input=tinker.ModelInput.from_ints(tokens[:-1]),
                    loss_fn_inputs={
                        "target_tokens": tinker.TensorData(
                            data=tokens[1:], dtype="int64", shape=[model_input_len],
                        ),
                    },
                )
                policy_data.append(policy_datum)

                if s.inference_logprobs:
                    response_start = max(0, prompt_len - 1)
                    aligned = [0.0] * response_start + list(s.inference_logprobs)
                    inf_logprobs_aligned.append(aligned)
                else:
                    inf_logprobs_aligned.append([])

            if not policy_data:
                return None

            # Stash teacher data for train_step to use
            step_key = _current_step_counter[0]
            if step_key not in _teacher_data_by_step:
                _teacher_data_by_step[step_key] = []
            _teacher_data_by_step[step_key].extend(teacher_results)

            comp_lens = [len(s.full_tokens) - s.prompt_len for s in sampled if len(s.full_tokens) >= 2]
            trunc = [s.finish_reason == "length" for s in sampled if len(s.full_tokens) >= 2]

            return PromptGroup(
                data=policy_data,
                advantages=[0.0] * len(policy_data),
                ref_logprobs=None,
                prompt_len=prompt_len,
                rewards=[0.0] * len(policy_data),
                inf_logprobs=inf_logprobs_aligned,
                completion_lens=comp_lens[:len(policy_data)],
                truncated=trunc[:len(policy_data)],
            )

        # -- Training step -----------------------------------------------------

        def train_step(
            step: int,
            prompt_groups: list[PromptGroup],
            loop_stats: dict | None = None,
        ) -> tuple[int, dict]:
            if not prompt_groups:
                raise ValueError("train_step requires at least one prompt group")

            data, _, _, prompt_lens, inf_lp = combine_prompt_groups(prompt_groups)

            # Gather teacher data
            step_key = _current_step_counter[0]
            teacher_data = _teacher_data_by_step.pop(step_key, [])
            teacher_token_logprobs = [td[0] for td in teacher_data]
            teacher_topk_logprobs = [td[1] for td in teacher_data]

            # Build student top-k logprobs for metrics from inference logprobs
            student_topk_logprobs: PositionLogprobs = []
            for lp_list in inf_lp:
                for lp_val in lp_list:
                    student_topk_logprobs.append([(0, lp_val)])

            # Teacher top-k logprobs (flattened across samples for metrics)
            teacher_topk_flat: PositionLogprobs = []
            for sample_topk in teacher_topk_logprobs:
                teacher_topk_flat.extend(sample_topk)

            # OPD loss via forward_backward_custom
            t0 = _time.time()
            opd_loss = make_opd_loss_fn(teacher_token_logprobs, prompt_lens)
            fwd_bwd_result = policy.forward_backward_custom(data, opd_loss)
            logger.info("[step %d] fwd_bwd: done (%.1fs)", step + 1, _time.time() - t0)

            t0 = _time.time()
            optim_result = policy.optim_step(
                adam_params,
                grad_accumulation_normalization=cfg.grad_accumulation_normalization,
            )
            step += 1
            logger.info("[step %d] optim_step: done (%.1fs)", step, _time.time() - t0)

            _current_step_counter[0] = step

            # Compute OPD paper metrics
            n_metrics_positions = min(len(student_topk_logprobs), len(teacher_topk_flat))
            opd_paper_metrics: Dict[str, float] = {}
            if n_metrics_positions > 0:
                opd_paper_metrics = compute_opd_metrics(
                    student_topk_logprobs[:n_metrics_positions],
                    teacher_topk_flat[:n_metrics_positions],
                )

            # Assemble metrics
            metrics = dict(flush_timing())
            metrics["train/step"] = step

            if fwd_bwd_result and hasattr(fwd_bwd_result, "metrics"):
                for k, v in fwd_bwd_result.metrics.items():
                    metrics[f"train/{k}"] = v

            if optim_result and hasattr(optim_result, "metrics") and optim_result.metrics:
                for k, v in optim_result.metrics.items():
                    metrics[f"train/{k}"] = v

            metrics.update(opd_paper_metrics)

            avg_loss = metrics.get("train/mean_loss", 0.0)
            overlap = opd_paper_metrics.get("distill/overlap_ratio", 0.0)
            ent_gap = opd_paper_metrics.get("distill/entropy_gap", 0.0)
            logger.info(
                "Step %d | Loss: %.4f | Overlap: %.3f | EntropyGap: %.4f",
                step, avg_loss, overlap, ent_gap,
            )
            log_metrics_json(step, loss=avg_loss, overlap_ratio=overlap, entropy_gap=ent_gap)
            wandb_log(metrics, step)

            step_tokens = sum(
                len(d.loss_fn_inputs["target_tokens"].data)
                for pg in prompt_groups for d in pg.data
            )
            total_rl_steps = len(rl_dataset) - step_offset
            runner.append_metrics(step, metrics, tokens=step_tokens)
            runner.write_status(
                RunStatus.RUNNING, step=step, total_steps=total_rl_steps, message="training",
            )
            runner.write_metadata()

            return step, metrics

        # -- Run ---------------------------------------------------------------

        def _weight_sync(step: int) -> None:
            logger.info("[step %d] weight_sync: saving + loading...", step)
            t0 = _time.time()
            with timer("weight_sync"):
                weight_syncer.save_and_hotload(f"step-{step}")
            logger.info("[step %d] weight_sync: done (%.1fs)", step, _time.time() - t0)

        def _loop_metrics_callback(loop_metrics: dict) -> None:
            if concurrency_controller is not None:
                cc_summary = concurrency_controller.step_completed()
                for k, v in cc_summary.items():
                    loop_metrics[f"concurrency/{k}"] = v
            wandb_log(loop_metrics, step=loop_metrics.get("train/step", 0))

        train_fns = TrainStepFns(train_step=train_step)

        remaining_rows = []
        for i_step in range(step_offset, len(rl_dataset)):
            remaining_rows.extend(rl_dataset.get_batch(i_step))

        total_rl_steps = len(rl_dataset) - step_offset
        runner.start_training()
        runner.write_status(RunStatus.RUNNING, total_steps=total_rl_steps, message="training")

        global_step = asyncio.run(
            run_rl_loop(
                sample_fns=(sample_one_prompt(row) for row in remaining_rows),
                train_fns=train_fns,
                prompt_groups_per_step=prompt_groups_per_step,
                global_step=step_offset,
                metrics_callback=_loop_metrics_callback,
                weight_sync_fn=_weight_sync if cfg.weight_sync.weight_sync_interval > 0 else None,
                weight_sync_interval=cfg.weight_sync.weight_sync_interval,
            )
        )

        # -- Final checkpoint --------------------------------------------------

        if cfg.save_final_checkpoint and global_step > step_offset:
            try:
                cp_name = f"step-{global_step}"
                _data_consumed = (resume_info.data_consumed if resume_info else 0) + (
                    (global_step - step_offset) * prompt_groups_per_step
                )
                paths = save_checkpoint(
                    policy,
                    cp_name,
                    cfg.log_path,
                    {
                        "step": global_step,
                        "data_consumed": _data_consumed,
                        "source_job_id": infra.policy_job_id,
                    },
                    kind=CheckpointKind.BOTH,
                    base_model=cfg.base_model,
                    training_shape=infra.training_shape_id,
                )

                if getattr(cfg, "output_model_id", None):
                    rlor_mgr.promote_checkpoint(
                        infra.policy_job_id,
                        paths["sampler_path"],
                        cfg.output_model_id,
                        cfg.base_model,
                    )
                    runner.write_output_model(
                        model_id=cfg.output_model_id,
                        checkpoint=cp_name,
                        job_id=infra.policy_job_id,
                    )
            except Exception as e:
                logger.warning("Failed to save final checkpoint: %s", e)

        runner.write_status(
            RunStatus.COMPLETED, step=global_step, total_steps=total_rl_steps, message="done",
        )
        runner.write_metadata()
        logger.info("OPD training complete: %d steps", global_step)
        wandb_finish()
        return {
            "steps": global_step,
            "policy_job_id": infra.policy_job_id,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config(
        log_path="./opd_logs",
        base_model="accounts/fireworks/models/qwen3-8b",
        teacher_model="accounts/fireworks/models/qwen3-235b-a22b",
        infra=InfraConfig(
            training_shape_id="accounts/fireworks/trainingShapes/qwen3-8b-128k-h200",
        ),
        deployment=DeployConfig(
            tokenizer_model="Qwen/Qwen3-8B",
        ),
    )
    main(cfg)
