#!/usr/bin/env bash
# End-to-end GRPO training for Qwen3-4B on OCI B300 (EU_NETHERLANDS_1).
#
# Prerequisites:
#   - Deployment shape: accounts/pyroworks-dev/deploymentShapes/rl-qwen3-4b-b300
#     Image: 4.60.0-cursor-b300-config-and-capability-14c6.2 (B300 sm_103a compatible)
#   - Training shape: accounts/pyroworks-dev/trainingShapes/qwen3-4b-b300
#   - Deployment: rl-qwen3-4b-b300-v5 (READY in EU_NETHERLANDS_1)
#
# Required env vars:
#   FIREWORKS_API_KEY        - API key for pyroworks-dev account
#   FIREWORKS_ACCOUNT_ID     - pyroworks-dev
#   FIREWORKS_BASE_URL       - https://gateway-dev.fireworks.ai (dev tier)
#
# === Usage ===
#   cd cookbook/training
#   bash examples/b300_grpo/run.sh

set -euo pipefail

cd "$(dirname "$0")/../.."

python -m examples.b300_grpo.train_b300 \
    --training-shape qwen3-4b-b300 \
    --region EU_NETHERLANDS_1 \
    --deployment-id rl-qwen3-4b-b300-v5 \
    --deployment-region EU_NETHERLANDS_1 \
    "$@"
