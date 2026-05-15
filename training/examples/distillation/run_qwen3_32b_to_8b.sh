#!/usr/bin/env bash
# Stage 2/2 of the distillation pipeline.
#
# On-policy reverse-KL distillation of the fine-tuned Qwen3-32B teacher
# (from stage 1: run_qwen3_32b_distill_teacher.sh) into Qwen3-8B student.
#
# Requires:
#   FIREWORKS_API_KEY  Fireworks API key
#   TEACHER_JOB_ID     policy_job_id from stage 1 (grep the stage-1 log)
#
# Optional env:
#   FIREWORKS_BASE_URL   default https://api.fireworks.ai
#   REGION               default US_VIRGINIA_1
#   WANDB_ENTITY         enables wandb logging
#   WANDB_PROJECT        default distillation-tinker
#
# Usage:
#   export TEACHER_JOB_ID=<from stage 1>
#   ./run_qwen3_32b_to_8b.sh

set -euo pipefail

export FIREWORKS_API_KEY="${FIREWORKS_API_KEY:?Set FIREWORKS_API_KEY}"
: "${TEACHER_JOB_ID:?Set TEACHER_JOB_ID (policy_job_id from stage 1)}"
export FIREWORKS_BASE_URL="${FIREWORKS_BASE_URL:-https://api.fireworks.ai}"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

RUN_TAG="distilled-qwen3-8b-$(date +%Y%m%d%H%M)"

ARGS=(
    --teacher-job-id "$TEACHER_JOB_ID"
    --student-base-model accounts/fireworks/models/qwen3-8b
    --teacher-base-model accounts/fireworks/models/qwen3-32b
    --student-tokenizer-model Qwen/Qwen3-8B
    --teacher-tokenizer-model Qwen/Qwen3-32B
    --max-rows 100
    --epochs 1
    --learning-rate 1e-5
    --kl-penalty-coef 1.0
    --completions-per-prompt 4
    --prompt-groups-per-step 1
    --max-completion-tokens 1024
    --region "${REGION:-US_VIRGINIA_1}"
    --output-model-id "$RUN_TAG"
)

if [ -n "${WANDB_ENTITY:-}" ]; then
    ARGS+=(--wandb-entity "$WANDB_ENTITY" --wandb-project "${WANDB_PROJECT:-distillation-tinker}")
fi

echo "=== Distillation qwen3-32b -> qwen3-8b (stage 2/2) ==="
echo "  Teacher job: $TEACHER_JOB_ID"
echo "  Max rows:    100"
echo "  Run tag:     $RUN_TAG"
echo ""

exec python "$HERE/train_distill_qwen3.py" "${ARGS[@]}"
