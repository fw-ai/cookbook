#!/usr/bin/env bash
# Dedicated GRPO on a trainer plus deployment. Extra flags are passed through and
# override the defaults below, for example a training shape or a smaller smoke run:
#   ./run.sh --training-shape <training-shape> --max-rows 12 --epochs 1
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

TRAINER_REPLICAS_ARG=()
if [[ -n "${TRAINER_REPLICAS:-}" ]]; then
    TRAINER_REPLICAS_ARG=(--trainer-replicas "${TRAINER_REPLICAS}")
fi

python "$HERE/train_deepmath.py" \
    --base-model accounts/fireworks/models/qwen3-4b \
    --tokenizer-model Qwen/Qwen3-4B \
    --dataset-path "$HERE/dataset.jsonl" \
    --deployment-id deepmath-qwen3-4b-$(date +%s) \
    --max-rows 500 \
    --epochs 3 \
    --completions-per-prompt 8 \
    --learning-rate 1e-5 \
    --output-model-id deepmath-qwen3-4b-$(date +%s) \
    ${TRAINER_REPLICAS_ARG[@]+"${TRAINER_REPLICAS_ARG[@]}"} \
    "$@"
