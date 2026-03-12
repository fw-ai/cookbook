#!/usr/bin/env bash
set -euo pipefail

# ---- Fixed config (do not change) ----
export FIREWORKS_API_KEY="fw_CFSRKgtwX24YounxB4BwtJ"
export FIREWORKS_ACCOUNT_ID="pyroworks"
export FIREWORKS_BASE_URL="https://api.fireworks.ai"
export TRAINING_SHAPE="qwen3-30b-a3b-instruct-2507-128k-b200"
export PYTHONPATH="/Users/chengxili/workspace_ts/cookbook:${PYTHONPATH:-}"

REF_TRAINING_SHAPE="qwen3-30b-a3b-instruct-2507-128k-b200-ref"
REGION="US_OHIO_1"
MAX_ROWS=200
EPOCHS=1
MAX_COMPLETION_TOKENS=122880
COMPLETIONS_PER_PROMPT=8
PROMPT_GROUPS_PER_STEP=32
WANDB_ENTITY="myh97"
WANDB_PROJECT="grpo-tinker"

# ---- Optional: reuse existing deployment ----
DEPLOYMENT_ID="${1:-}"

cd "$(dirname "$0")"

ARGS=(
    --ref-training-shape "$REF_TRAINING_SHAPE"
    --region "$REGION"
    --deployment-region "$REGION"
    --max-rows "$MAX_ROWS"
    --epochs "$EPOCHS"
    --max-completion-tokens "$MAX_COMPLETION_TOKENS"
    --completions-per-prompt "$COMPLETIONS_PER_PROMPT"
    --prompt-groups-per-step "$PROMPT_GROUPS_PER_STEP"
    --skip-cleanup
    --wandb-entity "$WANDB_ENTITY"
    --wandb-project "$WANDB_PROJECT"
)

if [ -n "$DEPLOYMENT_ID" ]; then
    ARGS+=(--deployment-id "$DEPLOYMENT_ID")
    echo "Reusing deployment: $DEPLOYMENT_ID"
fi

echo "=== DeepMath B200 Training ==="
echo "  Training shape: $TRAINING_SHAPE"
echo "  Ref shape:      $REF_TRAINING_SHAPE"
echo "  Region:         $REGION"
echo "  Max rows:       $MAX_ROWS"
echo "  Max completion:  $MAX_COMPLETION_TOKENS"
echo "  Completions/prompt: $COMPLETIONS_PER_PROMPT"
echo "  Groups/step:    $PROMPT_GROUPS_PER_STEP"
echo ""

exec python train_deepmath.py "${ARGS[@]}"
