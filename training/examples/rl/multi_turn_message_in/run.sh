#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ ! -f "$HERE/train.jsonl" ]]; then
    echo "train.jsonl not found; downloading openai/gsm8k from HuggingFace..."
    python "$HERE/prepare_data.py"
fi

python "$HERE/train.py" \
    --base-model accounts/fireworks/models/qwen3-1p5b-instruct \
    --tokenizer-model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset-path "$HERE/train.jsonl" \
    --max-rows 512 \
    --epochs 1 \
    --completions-per-prompt 4 \
    --prompt-groups-per-step 8 \
    --max-completion-tokens 1024 \
    --max-turns 2 \
    --learning-rate 1.7e-5 \
    --kl-beta 0.0 \
    --output-model-id "${OUTPUT_MODEL_ID:-accounts/fireworks/models/gsm8k-mt-$(date +%s)}"
