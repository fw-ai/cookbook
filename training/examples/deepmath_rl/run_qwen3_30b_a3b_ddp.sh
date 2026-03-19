#!/usr/bin/env bash
set -euo pipefail

# GRPO training for qwen3-30b-a3b-instruct-2507 on B200 (Ohio) with DDP (dp_replicate=2).
#
# This variant tests DDP (DistributedDataParallel) replication by passing
# --dp-shard 1 --dp-replicate 2 via extra_args to the trainer.  This means
# each data-parallel replica holds a full copy of the model (no FSDP sharding)
# and gradients are all-reduced across the 2 replicas.
#
# NOTE: Pure DDP requires each GPU group to hold the full model.  Make sure
# the training shape has enough GPUs so that after accounting for TP/PP/EP,
# dp_replicate replicas fit in the remaining world_size.
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
#   ./run_qwen3_30b_a3b_ddp.sh                        # create new deployment
#   ./run_qwen3_30b_a3b_ddp.sh <deployment-id>        # reuse existing deployment

export FIREWORKS_API_KEY="${FIREWORKS_API_KEY:?Set FIREWORKS_API_KEY env var}"
export FIREWORKS_BASE_URL="${FIREWORKS_BASE_URL:-https://api.fireworks.ai}"
export TRAINING_SHAPE="qwen3-30b-a3b-instruct-2507-128k-b200"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

REF_TRAINING_SHAPE="qwen3-30b-a3b-instruct-2507-128k-b200-ref"
REGION="US_OHIO_1"
MAX_ROWS=200
EPOCHS=1
MAX_COMPLETION_TOKENS=122880
COMPLETIONS_PER_PROMPT=8
PROMPT_GROUPS_PER_STEP=32

DEPLOYMENT_ID="${1:-}"

cd "$HERE"

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
    --output-model-id deepmath-rl-ddp-$(date +%Y%m%d%H%M)
)

if [ -n "${WANDB_ENTITY:-}" ]; then
    ARGS+=(--wandb-entity "$WANDB_ENTITY" --wandb-project "${WANDB_PROJECT:-grpo-tinker}")
fi

if [ -n "$DEPLOYMENT_ID" ]; then
    ARGS+=(--deployment-id "$DEPLOYMENT_ID")
    echo "Reusing deployment: $DEPLOYMENT_ID"
fi

echo "=== DeepMath qwen3-30b-a3b B200 Training (DDP dp_replicate=2) ==="
echo "  Training shape: $TRAINING_SHAPE"
echo "  Ref shape:      $REF_TRAINING_SHAPE"
echo "  Region:         $REGION"
echo "  Max rows:       $MAX_ROWS"
echo "  Completions:    $COMPLETIONS_PER_PROMPT"
echo "  Groups/step:    $PROMPT_GROUPS_PER_STEP"
echo "  DDP:            dp_shard=1, dp_replicate=2"
echo ""

exec python train_deepmath_ddp.py "${ARGS[@]}"
