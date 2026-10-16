"""Save paired base/LoRA completions on held-out SFT conversations."""

from __future__ import annotations

import json
import logging
from contextlib import ExitStack
from typing import Any

from fireworks.training import sdk

from training.renderer.message_weights import _mode_targets
from training.utils import fileio
from training.utils.runner import DatasetError, RunStatus
from training.utils.supervised import (
    build_tool_prefixed_messages,
    normalize_messages,
    parse_train_on_what,
)

logger = logging.getLogger(__name__)
GENERATION_EVAL_MAX_TOKENS = 1024


def _sampling_prompt(prompt, record, renderer, tokenizer) -> tuple[Any, list[str] | None]:
    """Project image prompts into the completions token-and-image contract.

    Text prompts are sampled unchanged. Image prompts reuse the exact
    tool-prefixed messages that built the prompt, strip sampler-owned
    affixes, and emit one image data URL per image. ``prompt.length`` stays
    the expanded visual-token budget used for ``max_tokens``.
    """
    if all(isinstance(chunk, sdk.types.EncodedTextChunk) for chunk in prompt.chunks):
        return prompt, None
    # lazy: the RL rollout helper is only needed for image prompts
    from training.utils.rl.rollout.renderer import build_multimodal_completions_prompt_token_ids

    # Prefer the exact tool-prefixed sequence prepare_generation_eval stored;
    # fall back to messages for callers that pass records directly.
    generation_messages = record.get("generation_messages", record.get("messages"))
    token_ids, images = build_multimodal_completions_prompt_token_ids(
        list(generation_messages or []),
        prompt,
        tokenizer,
        renderer=renderer,
    )
    return sdk.ModelInput.from_ints(token_ids), images


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
    enter the generation prompt. Text and image prompts are both sampled.
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
                if j in targets and normalized[j]["role"] == "assistant" and normalized[j].get("trainable", True)
            ),
            None,
        )
        if target is None or target == 0:
            raise DatasetError("Generation evaluation requires a prompt and an assistant reference in each row.")
        context = messages[:target]
        generation_messages = build_tool_prefixed_messages(context, renderer=renderer, tools=row.get("tools"))
        prompt = renderer.build_generation_prompt(generation_messages)
        # Image chunks have no token ids. Their length is the visual-token
        # budget training already uses, so the same prompt can be sampled.
        prompt_length = prompt.length
        if prompt_length <= 0 or prompt_length >= max_seq_len:
            raise DatasetError("Generation evaluation prompts must leave room for a completion within max_seq_len.")
        examples.append(
            (
                prompt,
                {
                    "messages": context,
                    "generation_messages": generation_messages,
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
    """Sample base and trained answers. Log inference or artifact errors and return."""
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
                sampling_prompt, images = _sampling_prompt(prompt, record, renderer, tokenizer)
                record = {
                    **{key: value for key, value in record.items() if key != "generation_messages"},
                    "cells": [{"output": ""}, {"output": ""}],
                }
                result["examples"].append(record)
                params = sdk.SamplingParams(
                    temperature=0.0,
                    max_tokens=min(GENERATION_EVAL_MAX_TOKENS, cfg.max_seq_len - prompt.length),
                    seed=cfg.seed,
                    stop=renderer.get_stop_sequences(),
                )
                sample_kwargs = {"images": images} if images else {}
                for cell, sampler in zip(record["cells"], (baseline, trained)):
                    response = sampler.sample(
                        sampling_prompt,
                        num_samples=1,
                        sampling_params=params,
                        **sample_kwargs,
                    ).result(
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
    try:
        fileio.write_json(fileio.join(cfg.log_path, "generation_eval.json"), result)
    except Exception:
        # A missing artifact must not discard a model that already trained.
        # Cancellation is BaseException and still propagates.
        logger.exception("Failed to write generation_eval.json")
