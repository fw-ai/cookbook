#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ ! -f "$HERE/train.jsonl" ]]; then
    echo "train.jsonl not found; writing tiny example dataset..."
    python "$HERE/prepare_data.py"
fi

: "${REMOTE_ROLLOUT_BASE_URL:=http://127.0.0.1:3000}"

python "$HERE/train.py" \
    --base-model accounts/fireworks/models/qwen3-1p5b-instruct \
    --tokenizer-model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset-path "$HERE/train.jsonl" \
    --remote-rollout-base-url "$REMOTE_ROLLOUT_BASE_URL" \
    --max-rows 4 \
    --epochs 1 \
    --completions-per-prompt 2 \
    --prompt-groups-per-step 1 \
    --max-completion-tokens 512 \
    --max-turns 2 \
    --learning-rate 1e-5 \
    --kl-beta 0.0 \
    "$@"
