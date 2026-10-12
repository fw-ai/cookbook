"""Save paired base/LoRA completions on held-out SFT conversations."""

from __future__ import annotations

import json
import logging
from contextlib import ExitStack
from typing import Any

import tinker

from training.renderer.message_weights import _mode_targets
from training.utils import fileio
from training.utils.runner import DatasetError, RunStatus
from training.utils.supervised import build_tool_prefixed_messages, normalize_messages, parse_train_on_what

logger = logging.getLogger(__name__)
GENERATION_EVAL_MAX_TOKENS = 1024


def _message_text(message) -> str:
    content = message.get("content")
    if isinstance(content, str) and not message.get("tool_calls"):
        return content
    return json.dumps(message, ensure_ascii=False)


def prepare_generation_eval(
    dataset, *, renderer, max_examples: int, max_seq_len: int, train_on_what: str = "all_assistant_messages"
):
    """Use the last trainable assistant turn in each row as the reference.

    Earlier turns remain context; the reference and subsequent turns never
    enter the generation prompt. The session sampler currently accepts text
    tokens only, so reject multimodal prompts instead of dropping their images.
    """
    examples = []
    for i in range(min(len(dataset), max_examples)):
        row = dataset.raw_row(i)
        messages = row.get("messages", [])
        normalized = normalize_messages(messages)
        targets = _mode_targets(normalized, parse_train_on_what(train_on_what))
        target = next(
            (
                j
                for j in reversed(range(len(messages)))
                if j in targets and normalized[j]["role"] == "assistant"
                and normalized[j].get("trainable", True)
            ),
            None,
        )
        if target is None or target == 0:
            raise DatasetError("Generation evaluation requires a prompt and an assistant reference in each row.")
        context = messages[:target]
        prompt = renderer.build_generation_prompt(
            build_tool_prefixed_messages(context, renderer=renderer, tools=row.get("tools"))
        )
        try:
            tokens = prompt.to_ints()
        except ValueError as exc:
            raise DatasetError("Generation evaluation currently supports text-only prompts.") from exc
        if not tokens or len(tokens) >= max_seq_len:
            raise DatasetError("Generation evaluation prompts must leave room for a completion within max_seq_len.")
        examples.append(
            (
                prompt,
                {
                    "messages": context,
                    "tools": row.get("tools", []),
                    "prompt": "\n\n".join(f"{message['role']}: {_message_text(message)}" for message in context),
                    "reference": _message_text(messages[target]),
                },
            )
        )
    if not examples:
        raise DatasetError("Generation evaluation needs a nonempty evaluation dataset.")
    return examples


def run_generation_eval(
    *,
    cfg,
    examples,
    renderer,
    tokenizer,
    service,
    client,
    runner,
    step,
) -> None:
    """Keep the training session alive until results (including failures) persist."""
    result: dict[str, Any] = {
        "base_model": cfg.base_model,
        "step": step,
        "sampling": {"temperature": 0.0, "max_tokens": GENERATION_EVAL_MAX_TOKENS, "seed": cfg.seed},
        "checkpoints": ["Original", "Your model"],
        "examples": [],
    }
    runner.write_status(
        RunStatus.EVALUATION, step=step, total_steps=step, message="generating baseline and trained-model comparisons"
    )
    try:
        sampler_path = client.save_weights_for_sampler(f"eval-step-{step}").path
        result["trained_checkpoint"] = sampler_path
        with ExitStack() as stack:
            baseline = service.create_sampling_client(base_model=cfg.base_model, tokenizer=tokenizer)
            stack.callback(baseline.close)
            trained = service.create_sampling_client(model_path=sampler_path, tokenizer=tokenizer)
            stack.callback(trained.close)
            for prompt, record in examples:
                record = {**record, "cells": [{"output": ""}, {"output": ""}]}
                result["examples"].append(record)
                params = tinker.SamplingParams(
                    temperature=0.0,
                    max_tokens=min(GENERATION_EVAL_MAX_TOKENS, cfg.max_seq_len - len(prompt.to_ints())),
                    seed=cfg.seed,
                    stop=renderer.get_stop_sequences(),
                )
                for cell, sampler in zip(record["cells"], (baseline, trained)):
                    response = sampler.sample(prompt, num_samples=1, sampling_params=params).result(
                        timeout=cfg.step_timeout or 600,
                    )
                    sequence = response.sequences[0]
                    cell.update(
                        output=tokenizer.decode(sequence.tokens),
                        tokens=list(sequence.tokens),
                        stop_reason=sequence.stop_reason,
                    )
        result["status"] = "completed"
    except Exception:
        # Training and its exported model remain usable if optional inference
        # fails. Persist partial responses and make the failure visible.
        logger.exception("Generation evaluation failed")
        result.update(status="failed", error="Generation evaluation failed. The trained model is still available.")
    # Persist once, outside the inference handler: artifact failures must propagate.
    fileio.write_json(fileio.join(cfg.log_path, "generation_eval.json"), result)
