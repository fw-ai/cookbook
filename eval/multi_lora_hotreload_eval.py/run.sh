#!/usr/bin/env bash
set -euo pipefail

export FIREWORKS_ACCOUNT_NAME="${FIREWORKS_ACCOUNT_NAME:?set FIREWORKS_ACCOUNT_NAME}"
export FIREWORKS_API_KEY="${FIREWORKS_API_KEY:?set FIREWORKS_API_KEY}"

mkdir -p ./logs
LOG="./logs/hotreload-eval-$(date +%Y%m%d-%H%M%S).log"

python hotreload_eval.py \
  --base-model accounts/fireworks/models/qwen3p5-9b \
  --deployment-shape accounts/fireworks/deploymentShapes/rft-qwen3p5-9b-v2 \
  --deployment-id multi-lora-eval-$(date +%s) \
  --lora-models accounts/$FIREWORKS_ACCOUNT_NAME/models/lora-a,accounts/$FIREWORKS_ACCOUNT_NAME/models/lora-b,accounts/$FIREWORKS_ACCOUNT_NAME/models/lora-c \
  --eval-dataset ./eval.jsonl \
  --results-dir ./results \
  2>&1 | tee "$LOG"

echo "[run] log written to $LOG"
