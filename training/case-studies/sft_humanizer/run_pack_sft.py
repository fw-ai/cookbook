#!/usr/bin/env python3
"""Train one Humanizer adapter on the self-distillation pack, serverless, and promote it.

Wraps the cookbook's shared serverless SFT recipe (`recipes/sft_loop.py` with
`serverless=True`) unchanged: shuffle the pack with a seed, train for N epochs,
save DCP state, promote the final sampler checkpoint. No in-session sampling,
so this touches only the pooled trainer. Evaluate the promoted adapter
afterwards with evaluate_pack.py.

The gallery adapters were trained by the recipe's predecessor at a fixed step
count (100 steps at batch 4 on 203 rows, about 2 epochs); the recipe counts
epochs, so `--epochs 2` is the matching setting.

Usage:
    export FIREWORKS_API_KEY=...
    python run_pack_sft.py --seed 29 --epochs 2 --output-model-id glm53-humanizer-s29
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from training.recipes import sft_loop

HERE = Path(__file__).resolve().parent
PACK = HERE / "pack" / "humanizer_train.jsonl"
BASE_MODEL = "accounts/fireworks/models/glm-5p3-flash"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--epochs", type=int, default=2, help="203 rows / batch 4 = 51 steps per epoch")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--output-model-id", required=True)
    ap.add_argument("--run-dir", default="")
    a = ap.parse_args()

    rows = [json.loads(l) for l in PACK.read_text().splitlines() if l.strip()]
    random.Random(a.seed).shuffle(rows)
    run_dir = Path(a.run_dir) if a.run_dir else HERE / "results" / time.strftime("%Y-%m-%d") / f"pack-sft-s{a.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    shuffled = run_dir / "train_shuffled.jsonl"
    shuffled.write_text("".join(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n" for r in rows))
    steps_per_epoch = -(-len(rows) // a.batch_size)
    (run_dir / "plan.json").write_text(json.dumps({
        "pack_sha256": hashlib.sha256(PACK.read_bytes()).hexdigest(), "rows": len(rows), "seed": a.seed,
        "epochs": a.epochs, "batch_size": a.batch_size, "steps": a.epochs * steps_per_epoch,
        "learning_rate": a.learning_rate, "lora_rank": a.lora_rank, "lora_alpha": a.lora_alpha,
        "base_model": BASE_MODEL, "output_model_id": a.output_model_id,
    }, indent=2) + "\n")

    cfg = sft_loop.Config(
        serverless=True, log_path=str(run_dir / "log"), base_model=BASE_MODEL, tokenizer_model="zai-org/GLM-5.3-Flash",
        dataset=str(shuffled), seed=a.seed, epochs=a.epochs, batch_size=a.batch_size, learning_rate=a.learning_rate,
        lora_rank=a.lora_rank, lora_alpha=a.lora_alpha, max_seq_len=32768, dcp_save_interval=20,
        output_model_id=a.output_model_id,
    )
    result = sft_loop.main(cfg)
    (run_dir / "summary.json").write_text(json.dumps({**result, "output_model_id": a.output_model_id}, indent=2) + "\n")
    print(f"done: {result['steps']} steps, job {result['job_id']}; run_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main()
