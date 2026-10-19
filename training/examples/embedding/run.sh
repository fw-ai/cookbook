#!/usr/bin/env bash
# Dedicated embedding fine-tuning on a trainer job. Extra flags are passed through and
# override the defaults below, for example a training shape:
#   ./run.sh --training-shape <training-shape> --epochs 1
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "$HERE"

: "${FIREWORKS_API_KEY:?missing required env var FIREWORKS_API_KEY}"

python train_embedding.py \
    --base-model accounts/fireworks/models/qwen3-embedding-8b \
    --tokenizer-model Qwen/Qwen3-Embedding-8B \
    --dataset-path retrieval_pairs.jsonl \
    --epochs 3 \
    --batch-size 8 \
    --temperature 0.02 \
    --learning-rate 1e-5 \
    "$@"
