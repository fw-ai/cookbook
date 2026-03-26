#!/usr/bin/env bash
set -euo pipefail

# GRPO training for qwen3-30b-a3b-instruct-2507 on B200 (Ohio).
#
# Requires:
#   FIREWORKS_API_KEY      - Fireworks API key
#
# Optional env overrides:
#   FIREWORKS_BASE_URL     - API base URL (default: https://api.fireworks.ai)
#   WANDB_ENTITY           - WandB entity (default: unset, WandB disabled)
#   WANDB_PROJECT          - WandB project (default: grpo-tinker)
#
# Usage:
#   ./run_qwen3_30b_a3b.sh                        # create new deployment
#   ./run_qwen3_30b_a3b.sh <deployment-id>        # reuse existing deployment

export FIREWORKS_API_KEY="${FIREWORKS_API_KEY:?Set FIREWORKS_API_KEY env var}"
export FIREWORKS_BASE_URL="${FIREWORKS_BASE_URL:-https://api.fireworks.ai}"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

REGION="US_OHIO_1"
MAX_ROWS=200
EPOCHS=1
MAX_COMPLETION_TOKENS=122880
COMPLETIONS_PER_PROMPT=8
PROMPT_GROUPS_PER_STEP=32

DEPLOYMENT_ID="${1:-}"

cd "$HERE"

ARGS=(
    --region "$REGION"
    --deployment-region "$REGION"
    --max-rows "$MAX_ROWS"
    --epochs "$EPOCHS"
    --max-completion-tokens "$MAX_COMPLETION_TOKENS"
    --completions-per-prompt "$COMPLETIONS_PER_PROMPT"
    --prompt-groups-per-step "$PROMPT_GROUPS_PER_STEP"
    --skip-cleanup
    --output-model-id deepmath-rl-$(date +%Y%m%d%H%M)
)

if [ -n "${WANDB_ENTITY:-}" ]; then
    ARGS+=(--wandb-entity "$WANDB_ENTITY" --wandb-project "${WANDB_PROJECT:-grpo-tinker}")
fi

if [ -n "$DEPLOYMENT_ID" ]; then
    ARGS+=(--deployment-id "$DEPLOYMENT_ID")
    echo "Reusing deployment: $DEPLOYMENT_ID"
fi

echo "=== DeepMath qwen3-30b-a3b B200 Training ==="
echo "  Training shape: auto (documented model default)"
echo "  Ref shape:      auto (documented model default)"
echo "  Region:         $REGION"
echo "  Max rows:       $MAX_ROWS"
echo "  Completions:    $COMPLETIONS_PER_PROMPT"
echo "  Groups/step:    $PROMPT_GROUPS_PER_STEP"
echo ""

exec python train_deepmath.py "${ARGS[@]}"
