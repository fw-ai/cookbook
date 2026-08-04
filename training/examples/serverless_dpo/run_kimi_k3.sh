#!/usr/bin/env bash
#
# Serverless DPO on Kimi K3 — train, checkpoint, resume, promote.
#
# Runs from anywhere; paths are resolved relative to this script. Any flags you
# pass are appended to the command, so you can override anything below, e.g.
#
#   ./run_kimi_k3.sh --steps 16 --batch-size 8
#   ./run_kimi_k3.sh --resume-from <your-account>/run-<32 hex>/dpo-0004
#
set -euo pipefail

HERE="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
# training/ — so `python -m examples.serverless_dpo...` resolves.
TRAINING_ROOT="$(cd "${HERE}/../.." && pwd)"

if [ -z "${FIREWORKS_API_KEY:-}" ]; then
    echo "Error: FIREWORKS_API_KEY is required (export it or put it in training/.env)" >&2
    exit 1
fi

# Serverless training rejects keys scoped to more than one account — use an
# account-scoped key.
BASE_MODEL="${BASE_MODEL:-accounts/fireworks/models/kimi-k3}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-moonshotai/Kimi-K3}"

# K3 ships a custom image processor behind trust_remote_code.
export HF_TRUST_REMOTE_CODE="${HF_TRUST_REMOTE_CODE:-1}"

cd "${TRAINING_ROOT}"

python -m examples.serverless_dpo.support_tone_dpo \
    --base-model "${BASE_MODEL}" \
    --tokenizer-model "${TOKENIZER_MODEL}" \
    --lora-rank 8 \
    --max-seq-len 32768 \
    --steps 8 \
    --batch-size 4 \
    --learning-rate 1e-5 \
    --dpo-beta 0.1 \
    --dcp-save-interval 4 \
    --output-model-id "serverless-dpo-tone-k3-$(date +%Y%m%d%H%M)" \
    "$@"
