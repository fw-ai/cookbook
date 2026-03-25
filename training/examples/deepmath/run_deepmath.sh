#!/usr/bin/env bash
set -euo pipefail

export FIREWORKS_API_KEY="fw_E6eAVuDZ7tPy1vgEDZYzU9"
export FIREWORKS_ACCOUNT_ID="pyroworks"
export FIREWORKS_BASE_URL="https://api.fireworks.ai"
export WANDB_ENTITY="myh97"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3.11 "$HERE/train_deepmath.py" \
    --dataset-path "/home/chengxi/worksapce_evenUp/cookbook/training/examples/deepmath/deepmath_probability_hard.jsonl" \
    --training-shape "ts-qwen3-30b-a3b-instruct-64k-rft-dev-cp8ep8-v1"
