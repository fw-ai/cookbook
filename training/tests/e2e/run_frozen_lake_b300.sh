#!/usr/bin/env bash
# CI runner for FrozenLake GRPO E2E on B300 GPUs.
#
# This script manages the full lifecycle:
#   1. Checks that the deployment is READY
#   2. Creates policy + reference trainer jobs via firectl-admin
#   3. Waits for jobs to become RUNNING
#   4. Runs the pytest E2E test
#   5. Cleans up trainer jobs (deployment is kept for reuse)
#
# Prerequisites:
#   FIREWORKS_API_KEY          Valid API key for dev.api.fireworks.ai
#   FIREWORKS_ACCOUNT_ID       Default: pyroworks-dev
#   FIRECTL_BIN                Path to firectl-admin binary
#   FIRECTL_PROFILE            firectl profile for the B300 gateway (default: dev-bennychen)
#   DEPLOYMENT_ID              Pre-created deployment with hotload (default: rl-qwen3-4b-b300-v8)
#
# Usage:
#   export FIREWORKS_API_KEY=fw_...
#   bash training/tests/e2e/run_frozen_lake_b300.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ── Configuration ────────────────────────────────────────────────────────────
export FIREWORKS_ACCOUNT_ID="${FIREWORKS_ACCOUNT_ID:-pyroworks-dev}"
export FIREWORKS_BASE_URL="${FIREWORKS_BASE_URL:-https://dev.api.fireworks.ai}"

FIRECTL_BIN="${FIRECTL_BIN:-$REPO_ROOT/../fireworks/firectl/bin/firectl-admin}"
FIRECTL_PROFILE="${FIRECTL_PROFILE:-dev-bennychen}"
TRAINING_SHAPE="${TRAINING_SHAPE:-qwen3-4b-b300}"
DEPLOYMENT_ID="${DEPLOYMENT_ID:-rl-qwen3-4b-b300-v8}"
REGION="${REGION:-EU_NETHERLANDS_1}"
ACCELERATOR="${ACCELERATOR:-NVIDIA_B300_288GB}"
JOB_WAIT_TIMEOUT="${JOB_WAIT_TIMEOUT:-80}"
JOB_CREATE_RETRIES="${JOB_CREATE_RETRIES:-3}"

export FIRECTL_AGENT_SAFE_ACCOUNTS="$FIREWORKS_ACCOUNT_ID"

POLICY_JOB_ID=""
REFERENCE_JOB_ID=""

log() { echo "$(date '+%H:%M:%S') [CI] $*"; }

# ── Cleanup on exit ─────────────────────────────────────────────────────────
cleanup_jobs() {
    for jid in "$POLICY_JOB_ID" "$REFERENCE_JOB_ID"; do
        [[ -z "$jid" ]] && continue
        log "Cleanup: deleting job $jid"
        "$FIRECTL_BIN" -a "$FIREWORKS_ACCOUNT_ID" rlor delete "$jid" \
            -p "$FIRECTL_PROFILE" 2>/dev/null || true
    done
}
trap cleanup_jobs EXIT

# ── 1. Verify deployment ────────────────────────────────────────────────────
log "Checking deployment $DEPLOYMENT_ID ..."
DEP_STATE=$("$FIRECTL_BIN" -a "$FIREWORKS_ACCOUNT_ID" deployment get "$DEPLOYMENT_ID" \
    -p dev 2>&1 | awk '/^State:/{print $2}' || true)

if [[ "$DEP_STATE" != "READY" ]]; then
    log "FATAL: deployment $DEPLOYMENT_ID not READY (state=$DEP_STATE)"
    exit 1
fi
log "Deployment READY"

# ── 2. Create trainer jobs ──────────────────────────────────────────────────
create_job() {
    local label=$1; shift
    local output
    output=$("$FIRECTL_BIN" -a "$FIREWORKS_ACCOUNT_ID" rlor create \
        --base-model "accounts/fireworks/models/qwen3-4b" \
        --training-shape "$TRAINING_SHAPE" \
        --accelerator-type "$ACCELERATOR" --accelerator-count 8 \
        --region "$REGION" \
        --display-name "frozen-lake-ci-$label" \
        --service-mode \
        "$@" \
        -p "$FIRECTL_PROFILE" 2>&1)
    echo "$output" | awk -F/ '/^Name:/{print $NF}'
}

log "Creating policy trainer job ..."
POLICY_JOB_ID=$(create_job policy --deployment-id "$DEPLOYMENT_ID")
log "  policy=$POLICY_JOB_ID"

log "Creating reference trainer job ..."
REFERENCE_JOB_ID=$(create_job reference --trainer-extra-args='--forward-only')
log "  reference=$REFERENCE_JOB_ID"

# ── 3. Wait for RUNNING ─────────────────────────────────────────────────────
get_state() {
    "$FIRECTL_BIN" -a "$FIREWORKS_ACCOUNT_ID" rlor get "$1" \
        -p "$FIRECTL_PROFILE" 2>&1 | awk '/^State:/{print $2}'
}

log "Waiting for trainer jobs ..."
for i in $(seq 1 80); do
    P=$(get_state "$POLICY_JOB_ID")
    R=$(get_state "$REFERENCE_JOB_ID")
    [[ "$P" == "JOB_STATE_FAILED" || "$R" == "JOB_STATE_FAILED" ]] && {
        log "FATAL: job failed (policy=$P, reference=$R)"; exit 1; }
    [[ "$P" == "JOB_STATE_RUNNING" && "$R" == "JOB_STATE_RUNNING" ]] && {
        log "Both jobs RUNNING"; break; }
    log "  [$i/80] policy=$P  reference=$R"
    sleep 15
done

P=$(get_state "$POLICY_JOB_ID")
R=$(get_state "$REFERENCE_JOB_ID")
[[ "$P" != "JOB_STATE_RUNNING" || "$R" != "JOB_STATE_RUNNING" ]] && {
    log "FATAL: jobs did not start in time"; exit 1; }

# ── 4. Run pytest ───────────────────────────────────────────────────────────
log "Running FrozenLake B300 E2E test ..."
export FROZEN_LAKE_POLICY_JOB_ID="$POLICY_JOB_ID"
export FROZEN_LAKE_REFERENCE_JOB_ID="$REFERENCE_JOB_ID"
export FROZEN_LAKE_DEPLOYMENT_ID="$DEPLOYMENT_ID"
export FROZEN_LAKE_REGION="$REGION"
export FROZEN_LAKE_TRAINING_SHAPE="$TRAINING_SHAPE"
export WANDB_MODE=disabled
export KEEP_DEPLOYMENT=1

cd "$REPO_ROOT"
python -m pytest training/tests/e2e/test_frozen_lake_b300_e2e.py \
    -v -s --log-cli-level=INFO --timeout=3600 -x

log "Test passed"
