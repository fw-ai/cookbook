#!/usr/bin/env bash
# Multi-turn GSM8K async RL on GLM-5.1 (LoRA).
#
# GLM-5.1 has no auto-selectable training shape, and a normal (non-superuser)
# account cannot send skipValidations=true, so we MUST pin an explicit shape:
#   accounts/fireworks/trainingShapes/glm-5p1-200k-lora  (LORA_TRAINER, 8xB300,
#   200K ctx; paired inference deployment shape glm-5p1-rft-b300-mxfp8-w8-p1).
#
# NOTE: the GLM-5.1 RFT deployment (glm-5p1-rft-b300-mxfp8-w8-p1) has been
# failing backend-side at model-download ("Internal error"). If this script
# fails at deployment readiness, that is the serving issue, not the config --
# the identical config works on qwen3-8b (see run.sh / the qwen smoke run).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Load secrets from training/.env (FIREWORKS_API_KEY + WandB vars). Unlike the
# notebook (which calls load_dotenv()), a shell doesn't read .env on its own.
if [[ -f "$REPO_ROOT/training/.env" ]]; then
    set -a; source "$REPO_ROOT/training/.env"; set +a
fi

# WandB metrics (tokens/sec, train/inference_kld, entropy, grad_norm, reward).
# WANDB_ENTITY is REQUIRED for any logging (the recipe disables wandb if it's
# empty); WANDB_API_KEY is needed to sync to the dashboard (else offline-only).
if [[ -z "${WANDB_ENTITY:-}" ]]; then
    echo "WARNING: WANDB_ENTITY not set -> metrics will NOT be logged. Set it in training/.env." >&2
fi
export WANDB_PROJECT="${WANDB_PROJECT:-gsm8k-mt-glm5p1}"

if [[ ! -f "$HERE/train.jsonl" ]]; then
    echo "train.jsonl not found; downloading openai/gsm8k from HuggingFace..."
    python "$HERE/prepare_data.py"
fi

python "$HERE/train.py" \
    --base-model accounts/fireworks/models/glm-5p1 \
    --tokenizer-model zai-org/GLM-5.1 \
    --training-shape-id accounts/fireworks/trainingShapes/glm-5p1-200k-lora \
    --lora-rank 16 \
    --dataset-path "$HERE/train.jsonl" \
    --max-rows 16 \
    --epochs 1 \
    --completions-per-prompt 4 \
    --prompt-groups-per-step 2 \
    --max-completion-tokens 1024 \
    --max-turns 2 \
    --learning-rate 1.7e-5 \
    --kl-beta 0.0 \
    ${OUTPUT_MODEL_ID:+--output-model-id "$OUTPUT_MODEL_ID"}
