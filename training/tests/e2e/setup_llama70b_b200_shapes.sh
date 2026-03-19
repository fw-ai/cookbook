#!/usr/bin/env bash
# One-time setup: create deployment + training shapes for LLaMA 3.3 70B on B200.
#
# Creates:
#   1. Deployment shape: rft-llama-v3p3-70b-b200 (8x B200, BF16, FireAttention)
#   2. Training shape:   ts-llama70b-b200-policy  (8x B200, 1 node, policy trainer)
#   3. Training shape:   ts-llama70b-b200-ref     (4x B200, 1 node, forward only)
#
# Prerequisites:
#   firectl-admin binary accessible
#
# Usage:
#   bash training/tests/e2e/setup_llama70b_b200_shapes.sh

set -euo pipefail

FIRECTL_BIN="${FIRECTL_BIN:-firectl-admin}"
ACCOUNT="${ACCOUNT:-fireworks}"
IMAGE_TAG="${DEPLOY_IMAGE_TAG:-4.31.2}"
TRAINER_IMAGE_TAG="${TRAINER_IMAGE_TAG:-0.74.0}"

log() { echo "$(date '+%H:%M:%S') [setup] $*"; }

validate_ts() {
    local ts_name=$1
    log "Fetching training shape version for $ts_name ..."
    local version
    version=$("$FIRECTL_BIN" -a "$ACCOUNT" training-shape-version get \
      "accounts/fireworks/trainingShapes/$ts_name/versions/latest" 2>&1 \
      | awk '/^Name:/{print $2}')

    if [ -z "$version" ]; then
        echo "ERROR: Could not get training shape version for $ts_name" >&2
        exit 1
    fi
    log "Training shape version: $version"

    log "Validating $ts_name ..."
    "$FIRECTL_BIN" -a "$ACCOUNT" training-shape-version update "$version" --validated
}

# ── Phase 1: Deployment Shape ────────────────────────────────────────────────
log "Creating deployment shape: rft-llama-v3p3-70b-b200 ..."

"$FIRECTL_BIN" -a "$ACCOUNT" deployment-shape create \
  --deployment-shape-id rft-llama-v3p3-70b-b200 \
  --base-model accounts/fireworks/models/llama-v3p3-70b-instruct \
  --accelerator-type NVIDIA_B200_180GB \
  --engine FIREATTENTION \
  --precision BF16 \
  --world-size 8 \
  --max-batch-size 256 \
  --kv-cache-memory-pct 80 \
  --max-lora-batch-size 16 \
  --disable-deployment-size-validation \
  --image-tag "$IMAGE_TAG" \
  --image-tag-reason "initial shape creation for 70B B200 e2e tests" \
  --extra-args='--stream-buffer=0' \
  --extra-args='--attention-sharding=tensor_parallel' \
  --extra-args='--prefill-chunk-size=2048' \
  --extra-args='--prefill-token-threshold=4096' \
  --extra-args='--prefill-batching-delay-ms=0' \
  --extra-args='--prefill-pacing-max-util=0.3' \
  --extra-args='--adaptive-memory-scheduling' \
  --extra-values 'enableLoraHotLoad=true,sidecarReconcileFrequency=5m' \
  --display-name "RFT LLaMA-3.3-70B B200"

log "Fetching deployment shape version ..."
DS_VERSION=$("$FIRECTL_BIN" -a "$ACCOUNT" deployment-shape-version get \
  accounts/fireworks/deploymentShapes/rft-llama-v3p3-70b-b200/versions/latest 2>&1 \
  | awk '/^Name:/{print $2}')

if [ -z "$DS_VERSION" ]; then
    echo "ERROR: Could not get deployment shape version" >&2
    exit 1
fi
log "Deployment shape version: $DS_VERSION"

# ── Phase 2a: Policy Training Shape (8 GPUs) ────────────────────────────────
log "Creating training shape: ts-llama70b-b200-policy ..."

"$FIRECTL_BIN" -a "$ACCOUNT" training-shape create \
  --training-shape-id ts-llama70b-b200-policy \
  --base-model accounts/fireworks/models/llama-v3p3-70b-instruct \
  --deployment-shape-version "$DS_VERSION" \
  --deployment-image-tag "$IMAGE_TAG" \
  --accelerator-type NVIDIA_B200_180GB \
  --accelerator-count 8 \
  --node-count 1 \
  --trainer-mode policy_trainer \
  --trainer-image-tag "$TRAINER_IMAGE_TAG" \
  --max-context-length 128000 \
  --display-name "LLaMA-3.3-70B B200 Policy Trainer"

validate_ts ts-llama70b-b200-policy

# ── Phase 2b: Reference Training Shape (4 GPUs, forward-only) ───────────────
log "Creating training shape: ts-llama70b-b200-ref ..."

"$FIRECTL_BIN" -a "$ACCOUNT" training-shape create \
  --training-shape-id ts-llama70b-b200-ref \
  --base-model accounts/fireworks/models/llama-v3p3-70b-instruct \
  --deployment-shape-version "$DS_VERSION" \
  --deployment-image-tag "$IMAGE_TAG" \
  --accelerator-type NVIDIA_B200_180GB \
  --accelerator-count 4 \
  --node-count 1 \
  --trainer-mode forward_only \
  --trainer-image-tag "$TRAINER_IMAGE_TAG" \
  --max-context-length 128000 \
  --display-name "LLaMA-3.3-70B B200 Reference (forward-only)"

validate_ts ts-llama70b-b200-ref

log ""
log "Done! All shapes created and validated."
log "  Deployment shape: rft-llama-v3p3-70b-b200"
log "  Policy trainer:   ts-llama70b-b200-policy  (8x B200)"
log "  Reference trainer: ts-llama70b-b200-ref    (4x B200, forward-only)"
