"""Verified demonstrations -> serverless SFT, using the shared cookbook recipe.

Run from the cookbook environment. This command starts paid training only when
invoked; it never provisions a dedicated deployment.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from prepare_sft import read_rows
from lipogram_rewards import e_free


def validate_rows(rows, mode="reasoning"):
    if not rows:
        raise ValueError("Empty SFT dataset")
    for i, row in enumerate(rows):
        messages = row["messages"]
        if not messages or any(m["role"] == "system" for m in messages) or messages[-1]["role"] != "assistant":
            raise ValueError(f"Row {i} must omit the instruction and end in an assistant answer")
        answer = messages[-1]["content"]
        if not answer.strip() or not e_free(answer):
            raise ValueError(f"Row {i} is not a nonempty, compliant final answer")
        reasoning = messages[-1].get("reasoning_content", "")
        if not isinstance(reasoning, str):
            raise ValueError(f"Row {i} has invalid reasoning_content")
        if mode == "reasoning" and not reasoning.strip():
            raise ValueError(f"Row {i} is missing reasoning; use --mode answer-only for the comparison")
        if mode == "answer-only" and reasoning:
            raise ValueError(f"Row {i} contains reasoning; use sft_answer_only.jsonl for the comparison")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["reasoning", "answer-only"], default="reasoning")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--log-path", required=True)
    ap.add_argument("--output-model-id", default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--max-seq-len", type=int, default=65536)
    a = ap.parse_args()
    rows = read_rows(a.dataset)
    try:
        validate_rows(rows, a.mode)
    except ValueError as exc:
        ap.error(str(exc))
    from training.recipes import sft_loop
    sft_loop.main(sft_loop.Config(
        serverless=True, base_model="accounts/fireworks/models/glm-5p3-flash",
        tokenizer_model="zai-org/GLM-5.3-Flash", lora_rank=64, lora_alpha=128,
        dataset=str(a.dataset), log_path=a.log_path, max_seq_len=a.max_seq_len,
        epochs=a.epochs, batch_size=a.batch_size, learning_rate=a.learning_rate,
        train_on_what="all_assistant_messages", dcp_save_interval=5,
        output_model_id=a.output_model_id,
    ))


if __name__ == "__main__":
    main()
