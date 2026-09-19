#!/usr/bin/env bash
#
# Serverless SFT on Kimi K3 — train, checkpoint, resume, promote.
#
# Runs from anywhere; paths are resolved relative to this script. Any flags you
# pass are appended to the command, so you can override anything below, e.g.
#
#   ./run_kimi_k3.sh --steps 20 --batch-size 8
#   ./run_kimi_k3.sh --resume-from fireworks/run-<32 hex>/triage-0006
#
set -euo pipefail

HERE="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
# training/ — so `python -m examples.serverless_sft...` resolves.
TRAINING_ROOT="$(cd "${HERE}/../.." && pwd)"

if [ -z "${FIREWORKS_API_KEY:-}" ]; then
    echo "Error: FIREWORKS_API_KEY is required (export it or put it in training/.env)" >&2
    exit 1
fi

# Kimi K3 is a gated model: the key must belong to an account that can read it,
# and serverless training rejects a key with access to more than one account.
BASE_MODEL="${BASE_MODEL:-accounts/fireworks/models/kimi-k3}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-moonshotai/Kimi-K3}"

# K3 ships a custom image processor behind trust_remote_code.
export HF_TRUST_REMOTE_CODE="${HF_TRUST_REMOTE_CODE:-1}"

cd "${TRAINING_ROOT}"

python -m examples.serverless_sft.support_triage_sft \
    --base-model "${BASE_MODEL}" \
    --tokenizer-model "${TOKENIZER_MODEL}" \
    --lora-rank 8 \
    --max-seq-len 32768 \
    --steps 6 \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --dcp-save-interval 3 \
    --output-model-id "serverless-sft-triage-k3-$(date +%Y%m%d%H%M)" \
    "$@"
