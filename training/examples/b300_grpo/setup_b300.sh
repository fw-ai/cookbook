#!/usr/bin/env bash
# Setup script: creates deployment shape, deployment, and training shape for
# Qwen3-4B GRPO on OCI B300 (EU_NETHERLANDS_1).
#
# Run from any directory:
#   bash ~/home/cookbook/training/examples/b300_grpo/setup_b300.sh
#
# Then run GRPO:
#   cd ~/home/cookbook/training
#   python -m examples.b300_grpo.train_b300 \
#     --training-shape qwen3-4b-b300 \
#     --deployment-id rl-qwen3-4b-b300 \
#     --deployment-region EU_NETHERLANDS_1

set -euo pipefail

FIRECTL="${FIRECTL:-$(dirname "$0")/../../../../fireworks/firectl/bin/firectl-admin}"
if [ ! -x "$FIRECTL" ]; then
    FIRECTL="firectl-admin"
fi
ACCT="${FIRECTL_ACCOUNT:-pyroworks-dev}"
export FIRECTL_AGENT_SAFE_ACCOUNTS="${FIRECTL_AGENT_SAFE_ACCOUNTS:-$ACCT}"

echo "=== Using firectl: $FIRECTL ==="
echo ""

# --- Step 1: Create deployment shape ---
echo "=== Step 1: Creating deployment shape rl-qwen3-4b-b300 ==="
$FIRECTL -a "$ACCT" deployment-shape create \
  --base-model accounts/fireworks/models/qwen3-4b \
  --deployment-shape-id rl-qwen3-4b-b300 \
  --display-name "Qwen3-4B B300 RFT" \
  --accelerator-type NVIDIA_B300_288GB \
  --accelerator-count 8 \
  --world-size 8 \
  --precision BF16 \
  --image-tag 4.2.44 \
  --image-tag-reason "same as validated rft-qwen3-4b-smoke" \
  --enable-session-affinity \
  --extra-values 'enableLoraHotLoad=true,priorityClass=deployment,toleration_key=fireworks.ai/rftj' \
  --extra-args='--end-thinking-token=</think>' \
  --extra-args='--conversation-style=qwen3' \
  --extra-args='--moe-sharding=ep'
echo ""

# --- Step 2: Create deployment and wait for READY ---
echo "=== Step 2: Creating deployment rl-qwen3-4b-b300 (waiting for READY) ==="
$FIRECTL -a "$ACCT" deployment create \
  accounts/fireworks/models/qwen3-4b \
  --deployment-id rl-qwen3-4b-b300 \
  --deployment-shape rl-qwen3-4b-b300 \
  --skip-shape-validation \
  --region EU_NETHERLANDS_1 \
  --min-replica-count 1 \
  --max-replica-count 1 \
  --wait
echo ""

echo "=== Step 2b: Verifying deployment is serving ==="
$FIRECTL -a "$ACCT" deployment get rl-qwen3-4b-b300
echo ""

# --- Step 3: Get deployment shape version and create training shape ---
echo "=== Step 3: Getting deployment shape version ==="
VERSION_LINE=$($FIRECTL -a "$ACCT" deployment-shape-version list \
  "accounts/$ACCT/deploymentShapes/rl-qwen3-4b-b300" 2>&1 | grep "accounts/$ACCT/deploymentShapes" | head -1)
VERSION_NAME=$(echo "$VERSION_LINE" | awk '{print $1}')

if [ -z "$VERSION_NAME" ]; then
    echo "ERROR: Could not find deployment shape version. Listing all versions:"
    $FIRECTL -a "$ACCT" deployment-shape-version list "accounts/$ACCT/deploymentShapes/rl-qwen3-4b-b300"
    exit 1
fi
echo "Found version: $VERSION_NAME"
echo ""

echo "=== Step 4: Creating training shape qwen3-4b-b300 ==="
$FIRECTL -a "$ACCT" training-shape create \
  --base-model qwen3-4b \
  --deployment-shape-version "$VERSION_NAME" \
  --deployment-image-tag 4.2.44 \
  --trainer-image-tag 0.55.1 \
  --accelerator-type NVIDIA_B300_288GB \
  --accelerator-count 8 \
  --max-context-length 8192 \
  --node-count 1 \
  --trainer-mode policy_trainer \
  --training-shape-id qwen3-4b-b300
echo ""

echo "=== Done! ==="
echo ""
echo "Run GRPO with:"
echo "  cd ~/home/cookbook/training"
echo "  python -m examples.b300_grpo.train_b300 \\"
echo "    --training-shape qwen3-4b-b300 \\"
echo "    --deployment-id rl-qwen3-4b-b300 \\"
echo "    --deployment-region EU_NETHERLANDS_1"
