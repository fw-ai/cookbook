#!/usr/bin/env python3
"""Privileged self-distillation on serverless Kimi K3.

A model such as Kimi K3 shows a behavior only when steered by an instruction
``I`` that will not be present at deployment. On-policy reverse-KL
self-distillation moves that behavior into LoRA weights without SFT on
``I``-conditioned data (which shifts the data distribution and risks forgetting):

    teacher = frozen base weights, prompt = system I + question  (privileged)
    student = current LoRA weights, prompt = question             (deployable)
    per-token advantage = teacher_logprob - sampling_logprob     (sampled reverse KL)
    loss = server-side ``importance_sampling``

It runs on Fireworks serverless training: one pooled session provides the LoRA
training client and Tinker sampling clients on saved snapshots, so there is no
trainer job or deployment. The teacher is a sampler on the step-0 snapshot,
which equals the base weights because LoRA starts at zero.

Why a loop here instead of a recipe: ``recipes/distillation_loop.py`` trains the
same objective with the same ``training.utils.distillation`` datums and metrics,
but serves the student and teacher from dedicated deployments with weight sync.
This example implements a serverless privileged-instruction run using the
``distillation_loop`` step with the
serving layer swapped: a sampler snapshot of the student replaces the weight
sync, and the step-0 snapshot replaces the teacher deployment.

Usage (from the cookbook root)::

    export FIREWORKS_API_KEY=fw_...
    python -m training.examples.distillation.serverless_k3_privileged.train_serverless_k3_privileged \\
        --steps 4 --prompt-groups-per-step 4 --completions-per-prompt 4
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fireworks.training import sdk
from fireworks.training.sdk import FiretitanServiceClient
from training.renderer import get_renderer
from training.utils import (
    DEFAULT_ADAM,
    WandBConfig,
    load_jsonl_dataset,
    log_metrics,
    setup_wandb,
    wandb_finish,
)
from training.utils.distillation import (
    OPDPromptGroup,
    build_opd_datums,
    build_opd_prompt_group,
)
from training.utils.distillation.sampling import (
    completions_from_sample_response,
    score_teacher_logprobs_with_sampling_client,
)
from training.utils.metrics import forward_backward_metrics, optimizer_metrics
from training.utils.serverless import serverless_base_url
from training.utils.timer import elapsed_timer, flush_timing
from training.utils.tokenizers import load_tokenizer

logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "gsm8k_privileged.jsonl"

WANDB_PROJECT = "serverless-distillation"

# The privileged instruction I: only the teacher sees it.
PRIVILEGED_INSTRUCTION = (
    "You are a meticulous mathematician. Reason step by step, keep the reasoning "
    "concise, and verify each arithmetic step before moving on. End your response "
    "with exactly one line of the form 'Final: <answer>'."
)


@dataclass
class Config:
    log_path: str
    base_model: str = "accounts/fireworks/models/kimi-k3"
    tokenizer_model: str = "moonshotai/Kimi-K3"
    renderer_name: str = "kimi_k3"
    dataset: str = str(DEFAULT_DATASET)
    """JSONL rows with a ``question`` field."""
    privileged_instruction: str = PRIVILEGED_INSTRUCTION
    lora_rank: int = 8

    steps: int = 4
    prompt_groups_per_step: int = 4
    completions_per_prompt: int = 4
    max_completion_tokens: int = 1024
    temperature: float = 1.0
    learning_rate: float = 2e-5
    loss_scale: float = 1.0
    """Scalar multiplier on ``teacher_logprob - sampling_logprob``."""
    sampling_timeout_s: float = 900.0
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project=WANDB_PROJECT))


# -- Task: the student and teacher prompts -------------------------------------


def student_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    return [{"role": "user", "content": row["question"]}]


def teacher_messages(row: dict[str, Any], instruction: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": row["question"]},
    ]


# -- Loop ----------------------------------------------------------------------


def main(cfg: Config) -> dict[str, Any]:
    load_dotenv()
    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    metrics_file = os.path.join(cfg.log_path, "metrics.jsonl")

    tokenizer = load_tokenizer(cfg.tokenizer_model)
    renderer = get_renderer(cfg.renderer_name, tokenizer)
    rows = load_jsonl_dataset(cfg.dataset)
    if not rows:
        raise ValueError(f"dataset is empty: {cfg.dataset}")
    row_iterator = itertools.cycle(rows)
    sampling_params = sdk.SamplingParams(
        max_tokens=cfg.max_completion_tokens,
        temperature=cfg.temperature,
        stop=renderer.get_stop_sequences(),
    )
    adam_params = sdk.AdamParams(learning_rate=cfg.learning_rate, **DEFAULT_ADAM)
    setup_wandb(cfg.wandb, {"base_model": cfg.base_model, "lr": cfg.learning_rate, "loss_scale": cfg.loss_scale})

    with ExitStack() as stack:
        service = FiretitanServiceClient(api_key=api_key, base_url=serverless_base_url(base_url))
        stack.callback(service.close)
        training_client = service.create_lora_training_client(base_model=cfg.base_model, rank=cfg.lora_rank)

        def snapshot_sampler(name: str) -> tuple[Any, str]:
            path = training_client.save_weights_for_sampler(name).result().path
            return service.create_sampling_client(model_path=path, tokenizer=tokenizer), path

        # The frozen teacher: the step-0 snapshot, served for the whole run.
        teacher, teacher_path = snapshot_sampler("sdft-teacher-step0")
        stack.callback(teacher.close)
        logger.info("Teacher snapshot: %s", teacher_path)

        def score_prompt_group(row: dict[str, Any], prompt: sdk.ModelInput, response: Any) -> OPDPromptGroup | None:
            """Pair one prompt's student samples with teacher scores under the privileged prompt."""
            sampled = completions_from_sample_response(prompt.to_ints(), response)
            teacher_prompt = renderer.build_generation_prompt(
                teacher_messages(row, cfg.privileged_instruction)
            ).to_ints()
            teacher_logprobs = score_teacher_logprobs_with_sampling_client(
                teacher, teacher_prompt, sampled, timeout=cfg.sampling_timeout_s
            )
            return build_opd_prompt_group(sampled, teacher_logprobs)

        for step in range(1, cfg.steps + 1):
            batch = list(itertools.islice(row_iterator, cfg.prompt_groups_per_step))
            prompts = [renderer.build_generation_prompt(student_messages(row)) for row in batch]

            # 1. Student rollouts on the current weights (every prompt in
            #    flight), each scored token by token by the teacher.
            with elapsed_timer("weight_sync"):
                student, _ = snapshot_sampler(f"sdft-student-{step:04d}")
            with elapsed_timer("rollout"):
                try:
                    futures = [
                        student.sample(
                            prompt=prompt,
                            num_samples=cfg.completions_per_prompt,
                            sampling_params=sampling_params,
                        )
                        for prompt in prompts
                    ]
                    groups = [
                        score_prompt_group(row, prompt, future.result(timeout=cfg.sampling_timeout_s))
                        for row, prompt, future in zip(batch, prompts, futures)
                    ]
                finally:
                    student.close()
            prompt_groups = [group for group in groups if group is not None]
            if not prompt_groups:
                logger.warning("[step %d] no scored samples; skipping update", step)
                continue

            # 2. One server-side importance-sampling step on the
            #    teacher-minus-student advantages.
            datums, opd_metrics = build_opd_datums(prompt_groups, loss_scale=cfg.loss_scale)
            with elapsed_timer("fwd_bwd"):
                fwd_bwd = training_client.forward_backward(datums, "importance_sampling").result()
            with elapsed_timer("optim_step"):
                optim = training_client.optim_step(adam_params).result()

            metrics = flush_timing()
            metrics.update(forward_backward_metrics([fwd_bwd]))
            metrics.update(optimizer_metrics(optim))
            metrics.update({f"train/{k}": v for k, v in opd_metrics.items()})
            metrics["train/step"] = step
            logger.info(
                "Step %d | sampled reverse KL: %.4f | OPD advantage: %.4f | tokens=%d",
                step,
                opd_metrics["opd_sampled_reverse_kl"],
                opd_metrics["opd_advantage"],
                opd_metrics["opd_active_tokens"],
            )
            log_metrics(metrics, step=step, metrics_file=metrics_file)

        # The trained student, servable from a new session on the same account.
        final_path = training_client.save_weights_for_sampler("sdft-student-final").result().path
        logger.info("Final student snapshot: %s", final_path)

    wandb_finish(metrics_file=metrics_file)
    return {"teacher_snapshot": teacher_path, "final_snapshot": final_path, "metrics_file": metrics_file}


def parse_args(argv: list[str] | None = None) -> Config:
    p = argparse.ArgumentParser(description="Serverless privileged self-distillation on Kimi K3")
    p.add_argument("--log-path", default=f"./serverless_distillation_logs/{time.strftime('%Y%m%d-%H%M%S')}")
    p.add_argument("--base-model", default=Config.base_model)
    p.add_argument("--tokenizer-model", default=Config.tokenizer_model)
    p.add_argument("--renderer-name", default=Config.renderer_name)
    p.add_argument("--dataset", default=Config.dataset)
    p.add_argument("--privileged-instruction", default=Config.privileged_instruction)
    p.add_argument("--lora-rank", type=int, default=Config.lora_rank)
    p.add_argument("--steps", type=int, default=Config.steps)
    p.add_argument("--prompt-groups-per-step", type=int, default=Config.prompt_groups_per_step)
    p.add_argument("--completions-per-prompt", type=int, default=Config.completions_per_prompt)
    p.add_argument("--max-completion-tokens", type=int, default=Config.max_completion_tokens)
    p.add_argument("--temperature", type=float, default=Config.temperature)
    p.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    p.add_argument("--loss-scale", type=float, default=Config.loss_scale)
    p.add_argument("--wandb-entity", default=None, help="Enables W&B logging.")
    args = vars(p.parse_args(argv))
    wandb = WandBConfig(entity=args.pop("wandb_entity"), project=WANDB_PROJECT)
    return Config(**args, wandb=wandb)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(parse_args())
