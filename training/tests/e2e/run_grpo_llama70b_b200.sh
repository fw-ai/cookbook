#!/usr/bin/env bash
# GRPO RL E2E test — LLaMA 3.3 70B with hot-loading on B200 GPUs.
#
# Full pipeline: policy + reference trainers, deployment with hotloading,
# and GSM8K reward evaluation. Requires a pre-created deployment shape.
#
# Prerequisites:
#   FIREWORKS_API_KEY              API key with training/deployment access
#   FIREWORKS_ACCOUNT_ID           Target account (default: fireworks)
#   FIREWORKS_E2E_DEPLOYMENT_SHAPE Deployment shape (default: rft-llama-v3p3-70b-b200)
#
# Shape setup (one-time):
#   bash training/tests/e2e/setup_llama70b_b200_shapes.sh
#
# Usage:
#   export FIREWORKS_API_KEY=fw_...
#   bash training/tests/e2e/run_grpo_llama70b_b200.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ── Configuration ────────────────────────────────────────────────────────────
export FIREWORKS_ACCOUNT_ID="${FIREWORKS_ACCOUNT_ID:-fireworks}"
export FIREWORKS_BASE_URL="${FIREWORKS_BASE_URL:-https://api.fireworks.ai}"

export FIREWORKS_E2E_MODEL="${FIREWORKS_E2E_MODEL:-accounts/fireworks/models/llama-v3p3-70b-instruct}"
export FIREWORKS_E2E_TOKENIZER_MODEL="${FIREWORKS_E2E_TOKENIZER_MODEL:-meta-llama/Llama-3.3-70B-Instruct}"
export FIREWORKS_E2E_DEPLOYMENT_SHAPE="${FIREWORKS_E2E_DEPLOYMENT_SHAPE:-rft-llama-v3p3-70b-b200}"
export FIREWORKS_E2E_REGION="${FIREWORKS_E2E_REGION:-US_OHIO_1}"
export FIREWORKS_E2E_TRAINING_SHAPE="${FIREWORKS_E2E_TRAINING_SHAPE:-ts-llama70b-b200-policy}"
export FIREWORKS_E2E_REF_TRAINING_SHAPE="${FIREWORKS_E2E_REF_TRAINING_SHAPE:-ts-llama70b-b200-ref}"
export FIREWORKS_E2E_DEPLOYMENT_ACCELERATOR="${FIREWORKS_E2E_DEPLOYMENT_ACCELERATOR:-NVIDIA_B200_180GB}"
export WANDB_MODE=disabled

START_TIME=$(date +%s)
log() { echo "$(date '+%H:%M:%S') [grpo-70b] $*"; }
elapsed() { echo $(( $(date +%s) - START_TIME )); }

# ── Validate ─────────────────────────────────────────────────────────────────
missing_vars=""
[ -z "${FIREWORKS_API_KEY:-}" ] && missing_vars="$missing_vars FIREWORKS_API_KEY"
[ -z "${FIREWORKS_ACCOUNT_ID:-}" ] && missing_vars="$missing_vars FIREWORKS_ACCOUNT_ID"

if [ -n "$missing_vars" ]; then
    echo "Error: missing required env var(s):$missing_vars" >&2
    exit 1
fi

log "Model:              $FIREWORKS_E2E_MODEL"
log "Tokenizer:          $FIREWORKS_E2E_TOKENIZER_MODEL"
log "Deployment shape:   $FIREWORKS_E2E_DEPLOYMENT_SHAPE"
log "Region:             $FIREWORKS_E2E_REGION"
log "Training shape:     $FIREWORKS_E2E_TRAINING_SHAPE"
log "Ref training shape: $FIREWORKS_E2E_REF_TRAINING_SHAPE"
log "Deploy accelerator: $FIREWORKS_E2E_DEPLOYMENT_ACCELERATOR"

# ── Run pytest ───────────────────────────────────────────────────────────────
log "Running GRPO E2E test (with hot-load) ..."

cd "$REPO_ROOT"
python -m pytest training/tests/e2e/test_grpo_e2e.py \
    -m e2e -v -s --log-cli-level=INFO --timeout=7200 -x

log "GRPO E2E test PASSED ($(elapsed)s total)"
