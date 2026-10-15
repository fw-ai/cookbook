#!/usr/bin/env bash
#
# Serverless DPO on Kimi K3 — train, checkpoint, resume, promote.
#
# Runs from anywhere; paths are resolved relative to this script. Any flags you
# pass are appended to the command, so you can override anything below, e.g.
#
#   ./run_serverless.sh --max-pairs 128 --batch-size 8
#   ./run_serverless.sh --log-path <first run's log path> \
#       --init-from-checkpoint <your-account>/run-<32 hex>/step-4
#
set -euo pipefail

HERE="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
# training/ — so `python -m examples.dpo...` resolves.
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

# Generate the UltraFeedback training data on first run.
if [ ! -f "${HERE}/preference_train.jsonl" ]; then
    python -m examples.dpo.prepare_data
fi

python -m examples.dpo.train_dpo_serverless \
    --base-model "${BASE_MODEL}" \
    --tokenizer-model "${TOKENIZER_MODEL}" \
    --lora-rank 8 \
    --max-seq-len 32768 \
    --max-pairs 32 \
    --batch-size 4 \
    --learning-rate 1e-5 \
    --beta 0.1 \
    --dcp-save-interval 4 \
    --output-model-id "serverless-dpo-uf-k3-$(date +%Y%m%d%H%M)" \
    "$@"
