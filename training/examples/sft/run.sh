#!/usr/bin/env bash
# Dedicated SFT on a trainer job. Extra flags are passed through and override the
# defaults below, for example a training shape or a smaller smoke run:
#   ./run.sh --training-shape <training-shape> --max-examples 24 --epochs 1
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "$HERE"

: "${FIREWORKS_API_KEY:?missing required env var FIREWORKS_API_KEY}"

python train_sft.py \
    --base-model accounts/fireworks/models/qwen3-8b \
    --tokenizer-model Qwen/Qwen3-8B \
    --dataset-path text2sql_dataset.jsonl \
    --max-examples 100 \
    --epochs 3 \
    --batch-size 32 \
    --learning-rate 1e-5 \
    --output-model-id sft-text-qwen3-8b-$(date +%Y%m%d%H%M) \
    "$@"
