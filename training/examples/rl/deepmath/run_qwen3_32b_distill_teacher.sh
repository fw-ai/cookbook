#!/usr/bin/env bash
# Stage-1 of the distillation pipeline.
#
# Trains a Qwen3-32B teacher via GRPO on DeepMath. --skip-cleanup keeps the
# trainer job alive after the run so the next stage (distillation) can
# attach to it as the teacher reference.
#
# After this completes, grep the logs for "policy_job_id" -- that's the ID
# you feed into run_qwen3_32b_to_8b.sh as TEACHER_JOB_ID.
#
# Requires:
#   FIREWORKS_API_KEY    Fireworks API key (export before running)
#
# Optional env:
#   FIREWORKS_BASE_URL   default https://api.fireworks.ai
#   WANDB_ENTITY         enables wandb logging
#   WANDB_PROJECT        default grpo-tinker
#   REGION               default US_VIRGINIA_1
#
# Usage:
#   ./run_qwen3_32b_distill_teacher.sh

set -euo pipefail

export FIREWORKS_API_KEY="${FIREWORKS_API_KEY:?Set FIREWORKS_API_KEY}"
export FIREWORKS_BASE_URL="${FIREWORKS_BASE_URL:-https://api.fireworks.ai}"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

REGION="${REGION:-US_VIRGINIA_1}"
RUN_TAG="distill-teacher-$(date +%Y%m%d%H%M)"

ARGS=(
    --base-model accounts/fireworks/models/qwen3-32b
    --tokenizer-model Qwen/Qwen3-32B
    --dataset-path "$HERE/dataset.jsonl"
    --region "$REGION"
    --deployment-region "$REGION"
    --max-rows 100
    --epochs 1
    --completions-per-prompt 4
    --learning-rate 1e-5
    --kl-beta 0.001
    --max-completion-tokens 1024
    --prompt-groups-per-step 1
    --skip-cleanup
    --deployment-id "${RUN_TAG}"
    --output-model-id "${RUN_TAG}"
)

if [ -n "${WANDB_ENTITY:-}" ]; then
    ARGS+=(--wandb-entity "$WANDB_ENTITY" --wandb-project "${WANDB_PROJECT:-grpo-tinker}")
fi

echo "=== DeepMath qwen3-32b teacher training (stage 1/2) ==="
echo "  Region:     $REGION"
echo "  Max rows:   100"
echo "  Run tag:    $RUN_TAG"
echo ""
echo "When this finishes, grep the log for 'policy_job_id' and pass it as"
echo "TEACHER_JOB_ID to run_qwen3_32b_to_8b.sh."
echo ""

cd "$HERE"
exec python train_deepmath.py "${ARGS[@]}"
