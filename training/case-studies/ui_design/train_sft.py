"""Serverless SFT on a UI-design pack. Every setting is an argument, so a notebook's plan is what runs.

    python train_sft.py --dataset pack/smoke/train.jsonl --output-model-id recorded-model-06
    python train_sft.py ... --dry-run     # print the resolved config, start nothing
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output-model-id", required=True)
    ap.add_argument("--base-model", default="accounts/fireworks/models/glm-5p3-flash")
    ap.add_argument("--tokenizer-model", default="zai-org/GLM-5.3-Flash")
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=None, help="default 2x rank")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8, help="rows are ~4k tokens of code; 8 gives ~30k tokens per step")
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--max-seq-len", type=int, default=32768)
    ap.add_argument("--log-path", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.lora_alpha is None:
        a.lora_alpha = 2 * a.lora_rank
    rows = [json.loads(l) for l in a.dataset.read_text().splitlines() if l.strip()]
    cfg = dict(serverless=True, base_model=a.base_model, tokenizer_model=a.tokenizer_model, lora_rank=a.lora_rank, lora_alpha=a.lora_alpha,
               dataset=str(a.dataset), log_path=a.log_path or f"runs/sft-{a.output_model_id}", max_seq_len=a.max_seq_len,
               epochs=a.epochs, batch_size=a.batch_size, learning_rate=a.learning_rate, train_on_what="all_assistant_messages",
               dcp_save_interval=5, output_model_id=a.output_model_id)
    steps = a.epochs * -(-len(rows) // a.batch_size)
    print(json.dumps({"rows": len(rows), "steps": steps, **cfg}, indent=2))
    if a.dry_run:
        print("dry run: nothing started, nothing billed"); return
    from training.recipes import sft_loop
    sft_loop.main(sft_loop.Config(**cfg))


if __name__ == "__main__":
    main()
