#!/usr/bin/env bash
# Reproduce the reviewed mixed partial-success SFT configuration.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_ROOT="$(cd "$HERE/../../.." && pwd)"
REPO_ROOT="$(cd "$TRAINING_ROOT/.." && pwd)"
PYTHON="${PYTHON:-$TRAINING_ROOT/.venv/bin/python}"

: "${RUN_DIR:?Set RUN_DIR to a validated upheaval PR #148 curation output}"
DRY_RUN="${DRY_RUN:-1}"
RUN_NAME="${RUN_NAME:-ultra-mixed-partial80}"
TRAINING_SHAPE="${TRAINING_SHAPE:-accounts/fireworks/trainingShapes/nemotron-3-ultra-550b-a55b-bf16-lora}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-$RUN_NAME}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment not found at $PYTHON; install training/.[dev] first" >&2
  exit 1
fi
if [[ "$DRY_RUN" != "1" && -z "${FIREWORKS_API_KEY:-}" ]]; then
  echo "FIREWORKS_API_KEY is required when DRY_RUN is not 1" >&2
  exit 1
fi
if [[ -n "$WANDB_ENTITY" && -z "$WANDB_PROJECT" ]] || \
   [[ -z "$WANDB_ENTITY" && -n "$WANDB_PROJECT" ]]; then
  echo "WANDB_ENTITY and WANDB_PROJECT must be set together" >&2
  exit 1
fi

args=(
  --run-dir "$RUN_DIR"
  --run-name "$RUN_NAME"
  --training-shape "$TRAINING_SHAPE"
  --epochs 1
  --batch-size 1
  --learning-rate 3e-7
  --lora-rank 16
  --lora-alpha 32
  --trainer-replicas 1
  --no-use-reservation
  --seed 20260828
  --pipeline-depth 4
  --checkpoint-interval 200
  --grad-clip-norm 1.0
  --warmup-ratio 0.03
  --min-lr-ratio 0.1
  --weight-decay 0
  --max-eval-seqs 200
)
if [[ -n "$WANDB_ENTITY" ]]; then
  args+=(
    --wandb-entity "$WANDB_ENTITY"
    --wandb-project "$WANDB_PROJECT"
    --wandb-run-name "$WANDB_RUN_NAME"
  )
fi
if [[ "$DRY_RUN" == "1" ]]; then
  args+=(--dry-run)
fi

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m training.examples.sft.nl2repo_mixed.train "${args[@]}"
