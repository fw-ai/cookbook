#!/usr/bin/env bash
set -euo pipefail

HERE=$(dirname "$(realpath "$0")")
echo "$HERE"

export PYTHONPATH="${PYTHONPATH:-}:$HERE/../../../"
echo "$PYTHONPATH"

missing_vars=""
[ -z "${FIREWORKS_API_KEY:-}" ] && missing_vars="$missing_vars FIREWORKS_API_KEY"

if [ -n "$missing_vars" ]; then
    echo "Error: missing required env var(s):$missing_vars" >&2
    exit 1
fi

export FIREWORKS_BASE_URL="${FIREWORKS_BASE_URL:-https://api.fireworks.ai}"

python3.12 train_frozen_lake.py \
    --base-model accounts/fireworks/models/qwen3-vl-8b-instruct \
    --tokenizer-model Qwen/Qwen3-VL-8B-Instruct \
    --training-shape accounts/fireworks/trainingShapes/qwen3-vl-8b-65k \
    --observation-mode image \
    --allow-plaintext-action-fallback \
    --kl-beta 0 \
    --epochs 1 \
    --max-seeds 8 \
    --max-steps 8 \
    --completions-per-prompt 4 \
    --prompt-groups-per-step 1 \
    --max-concurrent 4 \
    --output-model-id frozen-lake-vl-qwen3-8b-$(date +%Y%m%d%H%M) \
    "$@"
