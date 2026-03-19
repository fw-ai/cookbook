#!/usr/bin/env bash
# SFT E2E test — LLaMA 3.3 70B full fine-tune on B200 GPUs.
#
# Trains SFT on a small synthetic dataset and verifies multiple optimizer
# steps complete successfully. No deployment or hotloading.
#
# Prerequisites:
#   FIREWORKS_API_KEY          API key with training access
#   FIREWORKS_ACCOUNT_ID       Target account (default: fireworks)
#
# Usage:
#   export FIREWORKS_API_KEY=fw_...
#   bash training/tests/e2e/run_sft_llama70b_b200.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ── Configuration ────────────────────────────────────────────────────────────
export FIREWORKS_ACCOUNT_ID="${FIREWORKS_ACCOUNT_ID:-fireworks}"
export FIREWORKS_BASE_URL="${FIREWORKS_BASE_URL:-https://api.fireworks.ai}"

export FIREWORKS_E2E_MODEL="${FIREWORKS_E2E_MODEL:-accounts/fireworks/models/llama-v3p3-70b-instruct}"
export FIREWORKS_E2E_TOKENIZER_MODEL="${FIREWORKS_E2E_TOKENIZER_MODEL:-meta-llama/Llama-3.3-70B-Instruct}"
export FIREWORKS_E2E_REGION="${FIREWORKS_E2E_REGION:-US_OHIO_1}"
export FIREWORKS_E2E_TRAINING_SHAPE="${FIREWORKS_E2E_TRAINING_SHAPE:-ts-llama70b-b200-policy}"
export WANDB_MODE=disabled

START_TIME=$(date +%s)
log() { echo "$(date '+%H:%M:%S') [sft-70b] $*"; }
elapsed() { echo $(( $(date +%s) - START_TIME )); }

# ── Validate ─────────────────────────────────────────────────────────────────
missing_vars=""
[ -z "${FIREWORKS_API_KEY:-}" ] && missing_vars="$missing_vars FIREWORKS_API_KEY"
[ -z "${FIREWORKS_ACCOUNT_ID:-}" ] && missing_vars="$missing_vars FIREWORKS_ACCOUNT_ID"

if [ -n "$missing_vars" ]; then
    echo "Error: missing required env var(s):$missing_vars" >&2
    exit 1
fi

log "Model:          $FIREWORKS_E2E_MODEL"
log "Tokenizer:      $FIREWORKS_E2E_TOKENIZER_MODEL"
log "Region:         $FIREWORKS_E2E_REGION"
log "Training shape: $FIREWORKS_E2E_TRAINING_SHAPE"

# ── Run pytest ───────────────────────────────────────────────────────────────
log "Running SFT E2E test ..."

cd "$REPO_ROOT"
python -m pytest training/tests/e2e/test_sft_e2e.py \
    -m e2e -v -s --log-cli-level=INFO --timeout=7200 -x

log "SFT E2E test PASSED ($(elapsed)s total)"
