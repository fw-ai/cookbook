#!/usr/bin/env bash
set -euo pipefail

# Async GRPO training for qwen3-30b-a3b on B200 (Ohio) with pyroworks account.
#
# Uses AsyncRolloutScheduler: rollouts overlap with training, 1:1:1 cadence.
#
# Requires:
#   FIREWORKS_API_KEY      - Fireworks API key (prod, pyroworks account)
#
# Optional env overrides:
#   FIREWORKS_BASE_URL     - API base URL (default: https://api.fireworks.ai)
#   WANDB_ENTITY           - WandB entity (default: unset, WandB disabled)
#   WANDB_PROJECT          - WandB project (default: grpo-tinker)
#
# Usage:
#   ./run_qwen3_30b_a3b_async.sh                        # create new deployment
#   ./run_qwen3_30b_a3b_async.sh <deployment-id>        # reuse existing deployment

export FIREWORKS_API_KEY="${FIREWORKS_API_KEY:?Set FIREWORKS_API_KEY env var}"
export FIREWORKS_BASE_URL="${FIREWORKS_BASE_URL:-https://api.fireworks.ai}"
export TRAINING_SHAPE="${TRAINING_SHAPE:-accounts/fireworks/trainingShapes/ts-qwen3-30b-a3b-128k}"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/fireworks-ai-python/src:${REPO_ROOT}:${PYTHONPATH:-}"

BASE_MODEL="accounts/fireworks/models/qwen3-30b-a3b"
REF_TRAINING_SHAPE="${REF_TRAINING_SHAPE:-accounts/fireworks/trainingShapes/ts-qwen3-30b-a3b-128k-ref}"
REGION="US_OHIO_1"
MAX_ROWS=200
EPOCHS=1
MAX_COMPLETION_TOKENS=122880
COMPLETIONS_PER_PROMPT=8
PROMPT_GROUPS_PER_STEP=32
MAX_HEAD_OFFPOLICY_VERSIONS=2
SAMPLE_MAX_CONCURRENCY="${SAMPLE_MAX_CONCURRENCY:-}"

DEPLOYMENT_ID="${1:-}"

cd "$HERE"

ARGS=(
    --base-model "$BASE_MODEL"
    --ref-training-shape "$REF_TRAINING_SHAPE"
    --region "$REGION"
    --deployment-region "$REGION"
    --max-rows "$MAX_ROWS"
    --epochs "$EPOCHS"
    --max-completion-tokens "$MAX_COMPLETION_TOKENS"
    --completions-per-prompt "$COMPLETIONS_PER_PROMPT"
    --prompt-groups-per-step "$PROMPT_GROUPS_PER_STEP"
    --max-head-offpolicy-versions "$MAX_HEAD_OFFPOLICY_VERSIONS"
    --skip-cleanup
    --output-model-id deepmath-async-$(date +%Y%m%d%H%M)
)

if [ -n "${SAMPLE_MAX_CONCURRENCY:-}" ]; then
    ARGS+=(--sample-max-concurrency "$SAMPLE_MAX_CONCURRENCY")
fi

if [ -n "${WANDB_ENTITY:-}" ]; then
    ARGS+=(--wandb-entity "$WANDB_ENTITY" --wandb-project "${WANDB_PROJECT:-grpo-tinker}")
fi

if [ -n "$DEPLOYMENT_ID" ]; then
    ARGS+=(--deployment-id "$DEPLOYMENT_ID")
    echo "Reusing deployment: $DEPLOYMENT_ID"
fi

echo "=== Async DeepMath qwen3-30b-a3b B200 Training (pyroworks) ==="
echo "  Base model:     $BASE_MODEL"
echo "  Training shape: $TRAINING_SHAPE"
echo "  Ref shape:      ${REF_TRAINING_SHAPE:-none (no reference model)}"
echo "  Region:         $REGION"
echo "  Max rows:       $MAX_ROWS"
echo "  Completions:    $COMPLETIONS_PER_PROMPT"
echo "  Groups/step:    $PROMPT_GROUPS_PER_STEP"
echo "  Max offpolicy:  $MAX_HEAD_OFFPOLICY_VERSIONS"
echo "  Mode:           ASYNC"
echo ""

exec python train_deepmath_async.py "${ARGS[@]}"
