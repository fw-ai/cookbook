#!/usr/bin/env python3
"""SDFT (Self-Distillation Fine-Tuning) on Fireworks serverless training.

Implements `"Self-Distillation Enables Continual Learning"
<https://arxiv.org/abs/2601.19897>`_ on a shared serverless trainer: there is no
trainer job or inference deployment to provision. The student (a LoRA model)
learns from a frozen copy of the base model that sees each question together
with a golden answer.

The training loop is ``sdft_serverless.main()`` in
``training/_vendor/tinker_cookbook_fw`` (branched from the SDFT recipe in the
Fireworks fork of tinker-cookbook; see its README). It needs the ``sdft``
extra::

    uv pip install -e '.[sdft]'

Usage::

    export FIREWORKS_API_KEY=...
    python -m training.recipes.sdft_loop

Edit the ``Config`` at the bottom of this file, or import ``Config`` / ``main``
and pass your own. Re-running with the same ``log_path`` resumes from the last
checkpoint.
"""

from __future__ import annotations

# Fails fast with an install hint when the ``sdft`` extra is missing.
import training._vendor.tinker_cookbook_fw  # noqa: F401

import asyncio
import logging
from dataclasses import dataclass
from functools import partial

from dotenv import load_dotenv
from tinker_cookbook.recipes.sdft.datasets import SDFTDataset, load_sciknoweval
from tinker_cookbook.recipes.sdft.eval import SciKnowEvalEvaluator

from training._vendor.tinker_cookbook_fw.distillation import sdft_serverless
from training.renderer import get_renderer
from training.utils.supervised import resolve_renderer_name
from training.utils.tokenizers import load_tokenizer


@dataclass
class Config:
    log_path: str
    """Directory for metrics, checkpoints.jsonl and logs."""

    # Model
    base_model: str = "accounts/fireworks/models/glm-5p3-flash"
    """Fireworks model trained on the serverless pool (also the frozen teacher)."""
    tokenizer_model: str = "zai-org/GLM-5.3-Flash"
    """Hugging Face tokenizer matching ``base_model``."""
    renderer_name: str = ""
    """Cookbook renderer; empty = infer from ``tokenizer_model``."""
    lora_rank: int = 32
    """Serverless training is LoRA-only, so this must be > 0."""

    # Dataset
    sciknoweval_domain: str = "Chemistry"
    """SciKnowEval domain to train and evaluate on."""

    # Training
    learning_rate: float = 5e-4
    """LoRA learning rate; the recipe's range is 5e-4 to 1e-3."""
    groups_per_batch: int = 32
    """Questions per optimizer step."""
    max_tokens: int = 2048
    """Max tokens per student completion (and eval answer)."""
    max_steps: int | None = None
    """``None`` = one pass over the training set."""
    topk: int = 20
    """Teacher top-K for distillation."""

    # Evaluation and checkpointing
    eval_every: int = 20
    """Evaluate held-out SciKnowEval accuracy every N steps; 0 disables."""
    max_eval_examples: int | None = 64
    save_every: int = 20
    """Save resumable state every N steps; the final state is always saved."""

    # Logging
    wandb_project: str | None = None
    wandb_name: str | None = None


def main(cfg: Config) -> None:
    load_dotenv()
    renderer_name = resolve_renderer_name(cfg.tokenizer_model, cfg.renderer_name)
    renderer = get_renderer(renderer_name, load_tokenizer(cfg.tokenizer_model))

    # (question, golden answer) pairs. To train on another dataset, swap the
    # loader (e.g. ``load_toolalpaca``) and its evaluator.
    train_q, train_a, test_q, test_a = load_sciknoweval(domain=cfg.sciknoweval_domain)
    train_dataset = SDFTDataset(
        questions=train_q,
        golden_answers=train_a,
        batch_size=cfg.groups_per_batch,
        group_size=1,
        renderer=renderer,
        dataset_name="sciknoweval",
    )
    test_q, test_a = test_q[: cfg.max_eval_examples], test_a[: cfg.max_eval_examples]
    evaluator = partial(
        SciKnowEvalEvaluator,
        prompts=[[{"role": "user", "content": q}] for q in test_q],
        answers=test_a,
        renderer=renderer,
        max_tokens=cfg.max_tokens,
    )

    asyncio.run(
        sdft_serverless.main(
            sdft_serverless.Config(
                model_name=cfg.tokenizer_model,
                recipe_name="sdft_loop",
                fireworks_base_model=cfg.base_model,
                renderer_name=renderer_name,
                lora_rank=cfg.lora_rank,
                learning_rate=cfg.learning_rate,
                max_tokens=cfg.max_tokens,
                topk=cfg.topk,
                evaluator_builders=[evaluator] if test_q else [],
                eval_every=cfg.eval_every,
                save_every=cfg.save_every,
                log_path=cfg.log_path,
                wandb_project=cfg.wandb_project,
                wandb_name=cfg.wandb_name,
                max_steps=cfg.max_steps,
            ),
            train_dataset,
        )
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(
        Config(
            log_path="./sdft_logs",
            base_model="accounts/fireworks/models/glm-5p3-flash",
            tokenizer_model="zai-org/GLM-5.3-Flash",
            sciknoweval_domain="Chemistry",
            lora_rank=32,
            learning_rate=5e-4,
            eval_every=0,
        )
    )
