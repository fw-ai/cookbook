#!/usr/bin/env bash
# Dedicated ORPO on a trainer job. Extra flags are passed through and override the
# defaults below, for example a training shape or a smaller smoke run:
#   ./run.sh --training-shape <training-shape> --max-pairs 12
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "$HERE"

: "${FIREWORKS_API_KEY:?missing required env var FIREWORKS_API_KEY}"

python train_ifeval_orpo.py \
    --base-model accounts/fireworks/models/qwen3-8b \
    --tokenizer-model Qwen/Qwen3-8B \
    --dataset-path dataset.jsonl \
    --max-pairs 200 \
    --epochs 1 \
    --learning-rate 1e-5 \
    --orpo-lambda 1.0 \
    --output-model-id orpo-finetuned-qwen3-8b-$(date +%Y%m%d%H%M) \
    "$@"
