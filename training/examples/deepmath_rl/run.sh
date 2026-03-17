#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

python "$HERE/train_deepmath.py" \
    --base-model accounts/fireworks/models/qwen3-4b \
    --tokenizer-model Qwen/Qwen3-4B \
    --dataset-path "$HERE/dataset.jsonl" \
    --training-shape accounts/fireworks/trainingShapes/qwen3-4b-minimum-h200 \
    --ref-training-shape accounts/fireworks/trainingShapes/qwen3-4b-minimum-h200-forward \
    --deployment-id deepmath-qwen3-4b-$(date +%s) \
    --region US_VIRGINIA_1 \
    --max-rows 500 \
    --epochs 3 \
    --completions-per-prompt 8 \
    --learning-rate 1e-5 \
    --kl-beta 0.001 \
    --output-model-id deepmath-qwen3-4b-$(date +%s)
