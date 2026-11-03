#!/usr/bin/env python3
"""Serverless LoRA SFT on answer-only lipogram rows, one promotable sampler checkpoint per epoch. Paid.

Wraps the cookbook's shared recipe (`recipes/sft_loop.py`, serverless=True) the way
`sft_caveman/run_pack_sft.py` does, with two changes: `sampler_save_interval` = steps per epoch so every
epoch leaves a promotable checkpoint (promote them with promote.py), and a "Reasoning Effort: Low" renderer
so the training prompt matches the effort the bar is measured at. Defaults reproduce the recorded run
(10,481 rows, lr 1e-4, batch 8; epoch 1 became arxm-lipo-p2-lr1e4-e1).

    python run_sft.py --pack data/sft_p2.jsonl --run-dir runs/sft-p2
    python promote.py --run-dir runs/sft-p2 --prefix lipo-sft
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from training.recipes import sft_loop
from training.renderer import register_renderer
from training.renderer.glm5 import GLM53FlashRenderer

# The stock glm53_flash renderer writes "Reasoning Effort: Max"; the lipogram bar is measured at low,
# so train with the low line. Registered at import time: the recipe's spawned data workers re-import
# this module, so they see the name too.
register_renderer("lipo_text_low",   # no "glm53_flash" in the name: that marker makes the recipe load an image processor
                  lambda tokenizer, image_processor=None: GLM53FlashRenderer(tokenizer, reasoning_effort="low"))

HERE = Path(__file__).resolve().parent
BASE_MODEL = "accounts/fireworks/models/glm-5p3-flash"
TRAIN_PRICE_PER_M = 8.89


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--max-seq-len", type=int, default=16384)
    ap.add_argument("--renderer-name", default="lipo_text_low")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-model-id", default="")
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--init-from-checkpoint", default="",
                    help="serverless cross-run DCP ref <account>/<run>/<step-N>; with --run-dir pointing at the "
                         "original run, its dataloader.json restores step, data cursor and optimizer state")
    ap.add_argument("--dry-run", action="store_true", help="write the plan and cost estimate, train nothing")
    a = ap.parse_args()

    pack = Path(a.pack)
    rows = [json.loads(l) for l in pack.read_text().splitlines() if l.strip()]
    random.Random(a.seed).shuffle(rows)
    run_dir = Path(a.run_dir) if a.run_dir else HERE / "runs" / f"{pack.stem}-{time.strftime('%m%d-%H%M')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    shuffled = run_dir / "train_shuffled.jsonl"
    shuffled.write_text("".join(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n" for r in rows))
    steps_per_epoch = -(-len(rows) // a.batch_size)
    approx_tokens = sum(int(1.4 * len(" ".join(m["content"] for m in r["messages"]).split())) + 40 for r in rows)
    plan = {
        "pack": str(pack), "pack_sha256": hashlib.sha256(pack.read_bytes()).hexdigest(), "rows": len(rows),
        "seed": a.seed, "epochs": a.epochs, "batch_size": a.batch_size, "steps_per_epoch": steps_per_epoch,
        "steps": a.epochs * steps_per_epoch, "learning_rate": a.learning_rate, "lora_rank": a.lora_rank,
        "lora_alpha": a.lora_alpha, "max_seq_len": a.max_seq_len, "renderer_name": a.renderer_name,
        "base_model": BASE_MODEL, "output_model_id": a.output_model_id or None,
        "approx_tokens_per_epoch": approx_tokens,
        "approx_train_cost_usd": round(approx_tokens * a.epochs * TRAIN_PRICE_PER_M / 1e6, 2),
    }
    (run_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2), flush=True)
    if a.dry_run:
        return

    cfg = sft_loop.Config(
        serverless=True, log_path=str(run_dir / "log"), base_model=BASE_MODEL, tokenizer_model="zai-org/GLM-5.3-Flash",
        renderer_name=a.renderer_name, dataset=str(shuffled), seed=a.seed, epochs=a.epochs,
        batch_size=a.batch_size, learning_rate=a.learning_rate, lora_rank=a.lora_rank, lora_alpha=a.lora_alpha,
        max_seq_len=a.max_seq_len, sampler_save_interval=steps_per_epoch, dcp_save_interval=steps_per_epoch,
        output_model_id=a.output_model_id or None,
        init_from_checkpoint=a.init_from_checkpoint or None,
    )
    started = time.time()
    result = sft_loop.main(cfg)
    result = {**result, "wall_seconds": round(time.time() - started, 1)}
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(f"done: {json.dumps(result, default=str)}; run_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main()
