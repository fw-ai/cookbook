#!/usr/bin/env python3
"""On-policy distillation training loop.

Distill a fine-tuned teacher model into a smaller student via reverse-KL
on student-sampled trajectories. Mirrors the structure of ``rl_loop.py``
with a single additional step: between rollout and ``forward_backward``,
fetch the teacher's logprobs at the student's sampled tokens, estimate
per-sample reverse KL by Monte Carlo, and fold ``-coef * KL`` into the
advantage signal.

The on-policy framing means the full-vocab/top-k logit-distillation
problem disappears: reverse KL ``KL[p_student || p_teacher]`` is
estimated as ``E_{y ~ p_student}[log p_S(y) - log p_T(y)]``, which only
requires each model's logprob at the *realized* sampled tokens -- a
single scalar per position, which is exactly what the SDK's
``forward(data, "cross_entropy")`` returns.

Architectural notes
-------------------
* **Teacher hosting (v1).** The teacher is loaded via the existing
  ``reference`` slot in ``setup_infra``. ``setup_infra`` currently
  builds the reference trainer from the same ``base_model`` as the
  policy, so for distillation the user must pre-create a teacher
  trainer job (pointing at the *teacher* model) and pass its job ID via
  ``teacher_job_id``. A follow-up to ``setup_infra`` to accept a
  separate ``reference_base_model`` would make this turnkey.
* **KL granularity.** Per-token reverse-KL contributions are stored on
  ``PromptGroup.kl_per_token_advantages`` and folded into the per-token
  advantage tensor by ``build_builtin_loss_datums``. Server-side builtin
  loss kernels see a single combined per-token advantage and require no
  changes. The client-side custom-loss path (used only when the chosen
  ``policy_loss`` has no builtin kernel) is not currently per-token-KL
  aware; the recipe falls back to scalar KL on that path with a warning.
* **Tokenizer match.** Reverse-KL on student-sampled tokens is only
  defined when teacher and student share a tokenizer. This is a hard
  pre-condition; the recipe asserts it at startup.

Usage
-----
    export FIREWORKS_API_KEY=...
    python -m recipes.distillation_loop
"""

from __future__ import annotations

import os
import json
import signal
import asyncio
import logging
from contextlib import ExitStack
from typing import List, Optional, Callable
from dataclasses import field, dataclass

import tinker
import torch
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
    compute_advantages,
    read_api_extra_headers_env,
    load_jsonl_dataset,
    prepare_sampling_messages,
)
from training.utils.checkpoints import TrainingCheckpoints, validate_warm_start_config
from training.utils.rl import PromptGroup, setup_infra
from training.utils.rl.tis import TISConfig
from training.utils.timer import timer, flush_timing
import time as _time
from training.utils.rl.dapo import DAPOConfig
from training.utils.rl.dro import DROConfig
from training.utils.rl.gspo import GSPOConfig
from training.utils.rl.cispo import CISPOConfig
from training.utils.rl.train import TrainStepFns, run_rl_loop
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    log_path: str
    """Directory for checkpoints and logs."""

    # Student (policy) -- the smaller model being trained.
    base_model: str = "accounts/fireworks/models/qwen3-1p7b"
    rollout_base_model: str | None = None

    # Teacher -- the fine-tuned model whose behavior is being distilled.
    teacher_base_model: str = "accounts/fireworks/models/qwen3-8b"
    """Resource name of the teacher (typically a customer fine-tuned model).

    Must share a tokenizer with ``base_model``. The recipe asserts this at
    startup by comparing tokenizer vocab against
    ``teacher_tokenizer_model`` (HF) below.
    """
    teacher_tokenizer_model: str = "Qwen/Qwen3-8B"
    """HuggingFace tokenizer name for the teacher. Used only to assert that
    teacher and student share a tokenizer (reverse-KL on sampled tokens is
    only well-defined under tokenizer equivalence)."""
    teacher_job_id: str | None = None
    """Pre-created trainer job ID for the teacher.

    The current ``setup_infra`` builds the reference trainer from the same
    ``base_model`` as the policy. For distillation that's the wrong model,
    so for v1 you must pre-create a trainer job pointing at the teacher
    and pass its ID here. See module docstring for the follow-up that
    removes this requirement.
    """

    dataset: str = "https://raw.githubusercontent.com/eval-protocol/python-sdk/main/development/gsm8k_sample.jsonl"
    """Prompt-only dataset. Student samples completions on-policy; teacher
    scores them. No target responses required."""

    learning_rate: float = 1e-5
    completions_per_prompt: int = 4
    max_completion_tokens: int = 1024
    temperature: float = 1.0
    epochs: int = 1
    max_rows: int = 100
    max_seq_len: int | None = None
    lora_rank: int = 0

    prompt_groups_per_step: int = 1

    # ----- Distillation knobs --------------------------------------------------
    kl_penalty_coef: float = 1.0
    """Coefficient on the reverse-KL penalty added to advantages.

    Drives the magnitude of the distillation signal. For pure distillation
    (no task reward) this is the only learning signal; tune in tandem with
    ``learning_rate``.
    """
    kl_discount_factor: float = 0.0
    """Optional future-KL discount factor in [0, 1).

    When > 0, each token's KL contribution is discounted by powers of this
    factor when summed (a la return-style aggregation). 0.0 = simple sum,
    matching the tinker reference recipe default.
    """
    pure_distillation: bool = True
    """When True, the task-reward signal is zeroed and advantages are
    driven entirely by the KL penalty. Set False to combine distillation
    with a task verifier (provide ``reward_fn`` accordingly)."""
    # ---------------------------------------------------------------------------

    # Loss type. Tinker's on-policy distillation defaults to "importance_sampling";
    # any RL loss the rl_loop supports is fair game here.
    policy_loss: PolicyLoss = "importance_sampling"
    loss_path: LossPath = "builtin"
    """Server-side builtin path is the default here because the distillation KL
    is folded into per-token advantages (not into a kl_beta term inside the loss).
    ``kl_beta=0`` is enforced by ``validate_loss_path`` when ``loss_path='builtin'``;
    switch to ``'client'`` if you want a separate base-policy KL term as well."""
    kl_beta: float = 0.0
    """Reference-vs-policy KL coefficient inside the RL loss itself. Must be 0 on
    the builtin path. The distillation KL penalty (teacher-vs-policy) is folded
    into advantages and is governed by ``kl_penalty_coef`` above."""

    dapo: DAPOConfig = field(default_factory=DAPOConfig)
    dro: DROConfig = field(default_factory=DROConfig)
    gspo: GSPOConfig = field(default_factory=GSPOConfig)
    cispo: CISPOConfig = field(default_factory=CISPOConfig)
    eps_clip: float = 0.2
    eps_clip_high: float | None = None
    ratio_log_cap: float = 20.0
    tis: TISConfig = field(default_factory=TISConfig)

    grad_accumulation_normalization: GradAccNormalization | str | None = GradAccNormalization.NUM_LOSS_TOKENS

    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    trajectory_dir: str | None = None

    policy_job_id: str | None = None
    init_from_checkpoint: str | None = None
    warm_start_from_adapter: str | None = None

    output_model_id: str | None = None
    save_final_checkpoint: bool = True

    step_timeout: int = 0

    infra: InfraConfig = field(default_factory=InfraConfig)
    deployment: DeployConfig = field(default_factory=DeployConfig)
    weight_sync: WeightSyncConfig = field(default_factory=WeightSyncConfig)
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="distillation-tinker"))
    runner: RunnerConfig = field(default_factory=RunnerConfig)


# ---------------------------------------------------------------------------
# Reward function -- pure distillation by default; customise to combine
# distillation with a task verifier.
# ---------------------------------------------------------------------------


def zero_reward_fn(completion: str, row: dict) -> float:
    """Default reward for pure on-policy distillation: every rollout gets 0.

    With all-zero rewards, ``compute_advantages`` returns all-zero advantages
    (its std-clipping path), so the KL penalty alone drives learning. Override
    by setting ``cfg.pure_distillation = False`` and passing a real
    ``reward_fn`` into :func:`main`.
    """
    return 0.0


# ---------------------------------------------------------------------------
# Rollout filter
# ---------------------------------------------------------------------------


def accept_all(_pg: PromptGroup) -> bool:
    """Distillation does not require reward variance across rollouts -- the
    KL penalty signal is per-trajectory and well-defined even when rewards
    are constant. Override if combining with a task verifier and you want
    GRPO-style variance filtering."""
    return True


# ---------------------------------------------------------------------------
# Teacher: compute logprobs at the student's sampled token positions
# ---------------------------------------------------------------------------


def _discounted_future_sum(values: torch.Tensor, gamma: float) -> torch.Tensor:
    """Vectorised reverse cumulative sum: r_t = v_t + gamma * r_{t+1}.

    Returns a tensor of the same shape as ``values``. Used to discount future
    KL contributions when ``kl_discount_factor > 0``.
    """
    if gamma <= 0.0:
        return values
    out = torch.zeros_like(values)
    running = 0.0
    # Walk backwards through the sequence.
    for i in range(values.shape[0] - 1, -1, -1):
        running = float(values[i]) + gamma * running
        out[i] = running
    return out


def incorporate_kl_penalty(
    prompt_groups: List[PromptGroup],
    teacher,
    kl_penalty_coef: float,
    kl_discount_factor: float,
) -> dict:
    """Compute per-response-token reverse-KL-to-teacher and stash on prompt groups.

    For each rollout in each group:
      1. Build a teacher datum with the same (input, target) as the student.
      2. ``teacher.forward(data, "cross_entropy")`` returns the teacher's
         logprob at every sampled token (one scalar per position).
      3. Student's per-token logprobs come from the inference deployment,
         already aligned and stored on the prompt group as ``inf_logprobs``.
      4. Per-token reverse KL = student_logp - teacher_logp (MC estimate of
         ``KL[p_student || p_teacher]`` under the student's sampling distribution).
      5. Per-token advantage contribution = ``-coef * (discounted) per_token_KL``,
         stored on ``pg.kl_per_token_advantages`` (response-aligned, one list
         per rollout).

    ``build_builtin_loss_datums`` consumes this field downstream and adds the
    per-position values into the server-side per-token advantage tensor.

    Returns a dict of diagnostic metrics for logging.
    """
    if kl_penalty_coef == 0.0:
        return {}

    # Batch one teacher forward across all rollouts in all groups,
    # mirroring rl_loop.ref_forward.
    teacher_data: List[tinker.Datum] = []
    response_starts: List[int] = []
    response_lens: List[int] = []  # resp_len = n_tokens - response_start
    for pg in prompt_groups:
        response_start = max(0, pg.prompt_len - 1)
        for datum in pg.data:
            n_positions = len(datum.loss_fn_inputs["target_tokens"].data)
            teacher_data.append(
                tinker.Datum(
                    model_input=datum.model_input,
                    loss_fn_inputs={
                        "target_tokens": datum.loss_fn_inputs["target_tokens"],
                    },
                )
            )
            response_starts.append(response_start)
            response_lens.append(max(0, n_positions - response_start))

    if not teacher_data:
        return {}

    t0 = _time.time()
    try:
        teacher_fwd = teacher.forward(teacher_data, "cross_entropy")
    except Exception as e:
        raise RuntimeError(f"Teacher forward failed ({len(teacher_data)} datums): {e}") from e
    logger.info("teacher_forward: done (%.1fs)", _time.time() - t0)

    flat_idx = 0
    total_kl_sum = 0.0
    total_response_tokens = 0
    for pg in prompt_groups:
        kl_advantages_for_group: List[List[float]] = []
        for sample_idx in range(len(pg.data)):
            start = response_starts[flat_idx]
            resp_len = response_lens[flat_idx]
            teacher_logp_full = teacher_fwd.loss_fn_outputs[flat_idx]["logprobs"].data
            teacher_logp = torch.tensor(
                teacher_logp_full[start:start + resp_len], dtype=torch.float32,
            )

            student_inf = pg.inf_logprobs[sample_idx]
            student_logp = torch.tensor(
                student_inf[start:start + resp_len], dtype=torch.float32,
            )

            # Safety against off-by-one drift between inference and teacher
            # outputs: restrict to the common prefix.
            n = min(student_logp.shape[0], teacher_logp.shape[0])
            student_logp = student_logp[:n]
            teacher_logp = teacher_logp[:n]

            per_token_reverse_kl = student_logp - teacher_logp
            if kl_discount_factor > 0:
                contribution = _discounted_future_sum(per_token_reverse_kl, kl_discount_factor)
            else:
                contribution = per_token_reverse_kl

            # Negative because lower KL is better; the loss treats advantages
            # as "more is better".
            per_token_advantage_offset = (-kl_penalty_coef * contribution).tolist()
            # Pad to resp_len with zeros in the (rare) case n < resp_len.
            if n < resp_len:
                per_token_advantage_offset = per_token_advantage_offset + [0.0] * (resp_len - n)
            kl_advantages_for_group.append(per_token_advantage_offset)

            total_kl_sum += float(per_token_reverse_kl.sum())
            total_response_tokens += int(n)
            flat_idx += 1

        pg.kl_per_token_advantages = kl_advantages_for_group

    metrics = {
        "distill/teacher_kl_mean": (
            total_kl_sum / total_response_tokens if total_response_tokens > 0 else 0.0
        ),
        "distill/response_tokens": total_response_tokens,
    }
    return metrics


# ---------------------------------------------------------------------------
# Trajectory logging
# ---------------------------------------------------------------------------


def _dump_trajectory(trajectory_dir: str, step: int, prompt_groups: list[PromptGroup]) -> None:
    os.makedirs(trajectory_dir, exist_ok=True)
    path = os.path.join(trajectory_dir, f"step_{step:04d}.jsonl")
    with open(path, "w") as f:
        for pg_idx, pg in enumerate(prompt_groups):
            completions = pg.completions or []
            kl_pt = pg.kl_per_token_advantages or []
            for comp_idx, comp_text in enumerate(completions):
                record = {
                    "step": step,
                    "prompt_group": pg_idx,
                    "completion_index": comp_idx,
                    "prompt": pg.prompt,
                    "completion": comp_text,
                    "reward": pg.rewards[comp_idx] if comp_idx < len(pg.rewards) else None,
                    "advantage": pg.advantages[comp_idx] if comp_idx < len(pg.advantages) else None,
                    "kl_per_token_advantage": kl_pt[comp_idx] if comp_idx < len(kl_pt) else None,
                    "completion_len": pg.completion_lens[comp_idx] if comp_idx < len(pg.completion_lens) else None,
                    "truncated": pg.truncated[comp_idx] if comp_idx < len(pg.truncated) else None,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("[step %d] Saved trajectory to %s", step, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    config: Config,
    rlor_mgr: TrainerJobManager | None = None,
    deploy_mgr: DeploymentManager | None = None,
    cancel_on_exit: bool = False,
    reward_fn: Callable[[str, dict], float] | None = None,
):
    cfg = config
    if cfg.teacher_job_id is None:
        # Degenerate case: setup_infra(needs_reference=True) will create a
        # reference trainer from cfg.base_model, so teacher == student-base.
        # No useful learning signal (KL ~= 0), but exercises the full code
        # path -- useful for smoke testing.
        logger.warning(
            "teacher_job_id not provided; teacher will be created from base_model "
            "(%s). Reverse-KL will be ~0; this is only useful for smoke testing. "
            "For real distillation, pre-create a trainer job pointing at the "
            "fine-tuned teacher and pass its ID via teacher_job_id.",
            cfg.base_model,
        )

    runner = RunnerIO(cfg.runner)

    def _signal_handler(signum, frame):
        name = signal.Signals(signum).name
        logger.warning("Received %s -- raising SystemExit for cleanup", name)
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
    validate_warm_start_config(
        warm_start_from_adapter=cfg.warm_start_from_adapter,
        init_from_checkpoint=cfg.init_from_checkpoint,
        lora_rank=cfg.lora_rank,
    )
    if not cfg.deployment.tokenizer_model:
        raise ValueError("deployment.tokenizer_model is required.")

    # Effective reward function: zero by default for pure distillation.
    effective_reward = reward_fn or (zero_reward_fn if cfg.pure_distillation else None)
    if effective_reward is None:
        raise ValueError(
            "Set cfg.pure_distillation=True for KL-only training, or pass a "
            "reward_fn to combine distillation with a task verifier."
        )

    setup_wandb(
        cfg.wandb,
        {
            "completions_per_prompt": cfg.completions_per_prompt,
            "prompt_groups_per_step": cfg.prompt_groups_per_step,
            "kl_penalty_coef": cfg.kl_penalty_coef,
            "kl_discount_factor": cfg.kl_discount_factor,
            "lr": cfg.learning_rate,
            "teacher_base_model": cfg.teacher_base_model,
        },
    )

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    additional_headers = read_api_extra_headers_env()

    if rlor_mgr is None:
        rlor_mgr = TrainerJobManager(api_key=api_key, base_url=base_url, additional_headers=additional_headers)
    if deploy_mgr is None:
        deploy_mgr = DeploymentManager(api_key=api_key, base_url=base_url, additional_headers=additional_headers)

    runner.write_status(RunStatus.PENDING, message="provisioning")

    def _on_trainer_status(msg: str) -> None:
        runner.write_status(RunStatus.PENDING, message=msg)

    with runner, ResourceCleanup(rlor_mgr, deploy_mgr) as cleanup, ExitStack() as stack:
        # The teacher rides in the ``reference`` slot via reference_job_id.
        # setup_infra(needs_reference=True) wires a trainer client that
        # supports .forward(data, "cross_entropy") -- exactly what we need
        # for teacher logprob extraction. See module docstring for the
        # follow-up that removes the pre-create requirement.
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
            reference_job_id=cfg.teacher_job_id,
            needs_reference=True,
            needs_inference=True,
            role_prefix="distill",
            api_key=api_key,
            cleanup=cleanup if cancel_on_exit else None,
            on_status=_on_trainer_status,
        )
        for closeable in infra.closeables:
            stack.callback(closeable.close)

        runner.set_accelerator_info(profile=infra.policy_profile)
        wandb_log(infra.boot_metrics, step=0)

        policy = infra.policy
        teacher = infra.reference  # see comment above
        policy_profile = infra.policy_profile
        policy_job_id = infra.policy_job_id

        if teacher is None:
            raise RuntimeError("Teacher client was not provisioned (infra.reference is None).")

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            cfg.deployment.tokenizer_model, trust_remote_code=True,
        )
        # Reverse-KL on student-sampled tokens is only defined when teacher
        # and student share a tokenizer. Compare vocabs as a fast structural
        # check.
        teacher_tokenizer = transformers.AutoTokenizer.from_pretrained(
            cfg.teacher_tokenizer_model, trust_remote_code=True,
        )
        if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
            raise ValueError(
                f"Student tokenizer ({cfg.deployment.tokenizer_model}) and teacher "
                f"tokenizer ({cfg.teacher_tokenizer_model}) have different vocabs. "
                "Distillation requires a shared tokenizer."
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
            base_model=cfg.rollout_base_model or cfg.base_model,
            hotload_timeout=cfg.weight_sync.weight_sync_timeout,
            first_checkpoint_type=cfg.weight_sync.first_checkpoint_type,
            lora_rank=cfg.lora_rank,
        )

        ckpt = TrainingCheckpoints(
            policy,
            rlor_mgr,
            trainer_id=policy_job_id,
            log_path=cfg.log_path,
            lora_rank=cfg.lora_rank,
        )
        resume_info = ckpt.resume(
            init_from_checkpoint=cfg.init_from_checkpoint,
            warm_start_from_adapter=cfg.warm_start_from_adapter,
        )
        step_offset = resume_info.step if resume_info else 0
        wandb_log({"train/step": step_offset}, step_offset)

        if cfg.weight_sync.weight_sync_before_training and infra.deployment_id:
            name = f"resume-{step_offset}-base" if step_offset > 0 else "step-0-base"
            weight_syncer.save_and_hotload(name, checkpoint_type="base")

        raw_dataset = load_jsonl_dataset(cfg.dataset, cfg.max_rows)
        all_rows = raw_dataset * cfg.epochs
        rl_dataset = RLPromptDataset(all_rows, prompts_per_step=cfg.prompt_groups_per_step)
        adam_params = tinker.AdamParams(learning_rate=cfg.learning_rate, **DEFAULT_ADAM)

        client_loss_builder = build_loss_fn(cfg)

        sample_kwargs: dict = dict(
            max_tokens=cfg.max_completion_tokens,
            temperature=cfg.temperature,
            max_seq_len=infra.max_seq_len,
            http_timeout=cfg.deployment.sample_timeout,
            logprobs=True,  # required: student's per-token logprobs feed the KL estimate
        )

        async def sample_one_prompt(row: dict) -> PromptGroup | None:
            """Same shape as rl_loop.sample_one_prompt -- nothing distillation-specific
            happens here. The KL injection runs later, in train_step."""
            messages = row.get("messages", [])
            input_messages = prepare_sampling_messages(messages)
            if not input_messages:
                return None

            try:
                sampled = await sampler.sample_with_tokens(
                    messages=input_messages,
                    n=cfg.completions_per_prompt,
                    **sample_kwargs,
                )
            except Exception as e:
                logger.warning("Sampling failed: %s", e)
                return None

            if not sampled or len(sampled) < cfg.completions_per_prompt:
                return None

            rewards = [effective_reward(s.text, row) for s in sampled]
            advantages = compute_advantages(rewards)

            prompt_len = sampled[0].prompt_len
            policy_data: List[tinker.Datum] = []
            ref_data: List[tinker.Datum] = []
            adv_filtered: List[float] = []
            inf_logprobs_aligned: List[List[float]] = []

            for idx, s in enumerate(sampled):
                tokens = s.full_tokens
                if len(tokens) < 2:
                    continue
                model_input_len = len(tokens) - 1

                policy_data.append(
                    tinker.Datum(
                        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
                        loss_fn_inputs={
                            "target_tokens": tinker.TensorData(
                                data=tokens[1:], dtype="int64", shape=[model_input_len]
                            ),
                        },
                    )
                )
                # ref_data is consumed by the teacher in incorporate_kl_penalty
                # (and could also feed a separate KL-to-base reference, though
                # this recipe doesn't use one by default).
                ref_data.append(
                    tinker.Datum(
                        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
                        loss_fn_inputs={
                            "target_tokens": tinker.TensorData(
                                data=tokens[1:], dtype="int64", shape=[model_input_len]
                            ),
                        },
                    )
                )

                adv_filtered.append(advantages[idx])

                if not s.inference_logprobs:
                    raise RuntimeError(
                        f"Inference logprobs missing for sample {idx}; "
                        "distillation needs student per-token logprobs."
                    )
                response_start = max(0, prompt_len - 1)
                echoed = getattr(s, "logprobs_echoed", False)
                aligned = (
                    list(s.inference_logprobs)
                    if echoed
                    else [0.0] * response_start + list(s.inference_logprobs)
                )
                inf_logprobs_aligned.append(aligned)

            if not policy_data:
                return None

            return PromptGroup(
                data=policy_data,
                ref_data=ref_data,
                advantages=adv_filtered,
                ref_logprobs=None,
                prompt_len=prompt_len,
                rewards=rewards,
                inf_logprobs=inf_logprobs_aligned,
                completion_lens=[len(s.full_tokens) - s.prompt_len for s in sampled],
                truncated=[s.finish_reason == "length" for s in sampled],
                prompt=input_messages if cfg.trajectory_dir else None,
                completions=[s.text for s in sampled] if cfg.trajectory_dir else None,
                row_meta={"ground_truth": row.get("ground_truth", "")} if cfg.trajectory_dir else None,
            )

        # Validate the chosen loss_path against this run's config. Raises with
        # an actionable message if the user picked 'builtin' with kl_beta>0,
        # PP>1, or a client-only loss -- no silent fallback.
        validate_loss_path(cfg, policy_profile)
        if cfg.loss_path == "builtin":
            builtin_server_loss = get_builtin_loss_config(cfg)
            logger.info(
                "policy_loss=%s loss_path=builtin (server-side loss=%s)",
                cfg.policy_loss, builtin_server_loss[0],
            )
        else:
            builtin_server_loss = None
            logger.info(
                "policy_loss=%s loss_path=client (forward_backward_custom)",
                cfg.policy_loss,
            )

        def fwd_bwd_one(prompt_groups: list[PromptGroup]):
            if not prompt_groups:
                raise ValueError("fwd_bwd_one requires at least one prompt group")

            data, adv, ref_lp, prompt_lens, inf_lp = combine_prompt_groups(prompt_groups)

            # Flatten per-token KL advantages in the same rollout order as
            # combine_prompt_groups flattens data. None-out the whole list
            # when no group carries KL, so build_builtin_loss_datums takes
            # its scalar-only fast path.
            kl_per_token_flat: List[List[float]] | None = None
            if any(pg.kl_per_token_advantages is not None for pg in prompt_groups):
                kl_per_token_flat = []
                for pg in prompt_groups:
                    if pg.kl_per_token_advantages is None:
                        kl_per_token_flat.extend([] for _ in pg.data)
                    else:
                        kl_per_token_flat.extend(pg.kl_per_token_advantages)

            t0 = _time.time()
            prox_fwd = policy.forward(data, "cross_entropy")
            prox_lp = [prox_fwd.loss_fn_outputs[i]["logprobs"].data for i in range(len(data))]
            logger.info("policy_forward: done (%.1fs)", _time.time() - t0)

            t0 = _time.time()
            if builtin_server_loss is not None:
                kernel_loss, kernel_config = builtin_server_loss
                rl_datums = build_builtin_loss_datums(
                    data, adv, prox_lp, inf_lp, prompt_lens, cfg.tis,
                    policy_loss=cfg.policy_loss,
                    kl_per_token_advantages=kl_per_token_flat,
                )
                fwd_bwd_result = policy.forward_backward(
                    rl_datums, kernel_loss, loss_fn_config=kernel_config,
                )
            else:
                # Client-side custom-loss path is not currently per-token-KL
                # aware. Fall back to scalar KL contribution by summing the
                # per-token offsets into each rollout's scalar advantage.
                logger.warning(
                    "policy_loss=%r has no builtin server-side kernel for the current "
                    "profile; falling back to scalar-KL contribution on the client-side "
                    "custom path. Per-token KL credit assignment is disabled.",
                    cfg.policy_loss,
                )
                adv_scalar_fallback = list(adv)
                if kl_per_token_flat is not None:
                    for i, kl_seq in enumerate(kl_per_token_flat):
                        if kl_seq and i < len(adv_scalar_fallback):
                            adv_scalar_fallback[i] = adv_scalar_fallback[i] + float(sum(kl_seq))
                fwd_bwd_result = policy.forward_backward_custom(
                    data,
                    client_loss_builder(adv_scalar_fallback, ref_lp, prompt_lens, inf_lp, prox_lp),
                )
            logger.info("fwd_bwd: done (%.1fs)", _time.time() - t0)
            return fwd_bwd_result

        def train_step(
            step: int,
            prompt_groups: list[PromptGroup],
            loop_stats: dict | None = None,
        ) -> tuple[int, dict]:
            """teacher_forward + incorporate_kl_penalty + fwd_bwd + optim_step.

            The single architectural difference from rl_loop.train_step is the
            KL-penalty step inserted before fwd_bwd. Everything downstream is
            unchanged because the KL adjustment lands inside ``pg.advantages``.
            """
            t0 = _time.time()
            kl_metrics = incorporate_kl_penalty(
                prompt_groups,
                teacher,
                kl_penalty_coef=cfg.kl_penalty_coef,
                kl_discount_factor=cfg.kl_discount_factor,
            )
            logger.info("[step %d] kl_penalty: done (%.1fs)", step + 1, _time.time() - t0)

            t0 = _time.time()
            fwd_bwd_result = fwd_bwd_one(prompt_groups)
            logger.info("[step %d] fwd_bwd: done (%.1fs)", step + 1, _time.time() - t0)

            optim_result = policy.optim_step(
                adam_params,
                grad_accumulation_normalization=cfg.grad_accumulation_normalization,
            )
            step += 1

            if cfg.weight_sync.dcp_save_interval > 0 and step % cfg.weight_sync.dcp_save_interval == 0:
                _data_consumed = (resume_info.data_consumed if resume_info else 0) + (
                    step - step_offset
                ) * cfg.prompt_groups_per_step
                ckpt.save(
                    f"step-{step}",
                    resumable=True,
                    promotable=False,
                    data_consumed=_data_consumed,
                )

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
            metrics.update(kl_metrics)

            step_tokens = sum(
                len(d.loss_fn_inputs["target_tokens"].data) for pg in prompt_groups for d in pg.data
            )
            logger.info(
                "Step %d | teacher_kl: %.4f | adv_mean: %.3f",
                step,
                metrics.get("distill/teacher_kl_mean", 0.0),
                sum(a for pg in prompt_groups for a in pg.advantages)
                / max(1, sum(len(pg.advantages) for pg in prompt_groups)),
            )
            log_metrics_json(
                step,
                kl=metrics.get("distill/teacher_kl_mean", 0.0),
                reward=metrics.get("rollout/reward", 0.0),
            )
            wandb_log(metrics, step)

            total_steps = len(rl_dataset) - step_offset
            runner.append_metrics(step, metrics, tokens=step_tokens)
            runner.write_status(RunStatus.RUNNING, step=step, total_steps=total_steps, message="training")
            runner.write_metadata()

            if cfg.trajectory_dir:
                _dump_trajectory(cfg.trajectory_dir, step, prompt_groups)

            return step, metrics

        def _weight_sync(step: int) -> None:
            with timer("weight_sync"):
                weight_syncer.save_and_hotload(f"step-{step}")

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

        total_steps = len(rl_dataset) - step_offset
        runner.start_training()
        runner.write_status(RunStatus.RUNNING, total_steps=total_steps, message="training")

        global_step = asyncio.run(
            run_rl_loop(
                sample_fns=(sample_one_prompt(row) for row in remaining_rows),
                train_fns=train_fns,
                prompt_groups_per_step=cfg.prompt_groups_per_step,
                dynamic_filter_fn=accept_all,
                global_step=step_offset,
                metrics_callback=_loop_metrics_callback,
                weight_sync_fn=_weight_sync if cfg.weight_sync.weight_sync_interval > 0 else None,
                weight_sync_interval=cfg.weight_sync.weight_sync_interval,
            )
        )

        if cfg.save_final_checkpoint and global_step > step_offset:
            _data_consumed = (resume_info.data_consumed if resume_info else 0) + (
                global_step - step_offset
            ) * cfg.prompt_groups_per_step
            cp_name = f"step-{global_step}"
            ckpt.save(
                cp_name,
                resumable=True,
                promotable=True,
                data_consumed=_data_consumed,
            )
            if cfg.output_model_id:
                ckpt.promote_latest(cfg.output_model_id, cfg.base_model)
                runner.write_output_model(
                    model_id=cfg.output_model_id, checkpoint=cp_name, job_id=policy_job_id,
                )

            runner.write_status(RunStatus.COMPLETED, step=global_step, total_steps=total_steps, message="done")
            runner.write_metadata()
            wandb_finish()
            return {
                "steps": global_step,
                "policy_job_id": policy_job_id,
                "teacher_job_id": cfg.teacher_job_id,
            }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config(
        log_path="./distillation_logs",
        # Smaller student, larger fine-tuned teacher (both same family / tokenizer).
        base_model="accounts/fireworks/models/qwen3-1p7b",
        teacher_base_model="accounts/fireworks/models/qwen3-8b",
        teacher_tokenizer_model="Qwen/Qwen3-8B",
        # Required: pre-create a teacher trainer job and paste its ID here.
        teacher_job_id=os.environ.get("DISTILL_TEACHER_JOB_ID"),
        infra=InfraConfig(
            training_shape_id="accounts/fireworks/trainingShapes/qwen3-1p7b-128k-h200",
        ),
        deployment=DeployConfig(
            tokenizer_model="Qwen/Qwen3-1.7B",
        ),
    )
    main(cfg)
