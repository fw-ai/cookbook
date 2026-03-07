#!/usr/bin/env bash
# CI runner for FrozenLake GRPO E2E test on B300 GPUs.
#
# Prerequisites:
#   - firectl-admin binary at FIRECTL_BIN (default: ../../firectl/bin/firectl-admin)
#   - firectl profile "dev-bennychen" configured (points to dev-bennychen gateway)
#   - FIREWORKS_API_KEY set (valid for dev.api.fireworks.ai)
#   - FIREWORKS_ACCOUNT_ID set (default: pyroworks-dev)
#
# Usage:
#   export FIREWORKS_API_KEY=fw_...
#   bash training/tests/e2e/run_frozen_lake_b300.sh
#
# The script:
#   1. Creates or reuses a deployment with hotloading (rl-qwen3-4b-b300-ci)
#   2. Creates policy and reference trainer jobs via firectl-admin
#   3. Runs the pytest E2E test
#   4. Cleans up trainer jobs (deployment is kept for reuse)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export FIREWORKS_ACCOUNT_ID="${FIREWORKS_ACCOUNT_ID:-pyroworks-dev}"
export FIREWORKS_BASE_URL="${FIREWORKS_BASE_URL:-https://dev.api.fireworks.ai}"

FIRECTL_BIN="${FIRECTL_BIN:-$REPO_ROOT/../fireworks/firectl/bin/firectl-admin}"
FIRECTL_PROFILE="${FIRECTL_PROFILE:-dev-bennychen}"
TRAINING_SHAPE="${TRAINING_SHAPE:-qwen3-4b-b300}"
DEPLOYMENT_ID="${DEPLOYMENT_ID:-rl-qwen3-4b-b300-ci}"
REGION="${REGION:-EU_NETHERLANDS_1}"
ACCELERATOR="${ACCELERATOR:-NVIDIA_B300_288GB}"

export FIRECTL_AGENT_SAFE_ACCOUNTS="$FIREWORKS_ACCOUNT_ID"

log() { echo "$(date '+%H:%M:%S') [CI] $*"; }

cleanup_jobs() {
    if [[ -n "${POLICY_JOB_ID:-}" ]]; then
        log "Cleaning up policy job $POLICY_JOB_ID"
        "$FIRECTL_BIN" -a "$FIREWORKS_ACCOUNT_ID" rlor delete "$POLICY_JOB_ID" -p "$FIRECTL_PROFILE" 2>/dev/null || true
    fi
    if [[ -n "${REFERENCE_JOB_ID:-}" ]]; then
        log "Cleaning up reference job $REFERENCE_JOB_ID"
        "$FIRECTL_BIN" -a "$FIREWORKS_ACCOUNT_ID" rlor delete "$REFERENCE_JOB_ID" -p "$FIRECTL_PROFILE" 2>/dev/null || true
    fi
}

trap cleanup_jobs EXIT

# -- 1. Check deployment status -----------------------------------------------

log "Checking deployment $DEPLOYMENT_ID..."
DEP_STATE=$("$FIRECTL_BIN" -a "$FIREWORKS_ACCOUNT_ID" deployment get "$DEPLOYMENT_ID" -p dev 2>&1 | grep "^State:" | awk '{print $2}' || true)

if [[ "$DEP_STATE" != "READY" ]]; then
    log "Deployment $DEPLOYMENT_ID is not READY (state=$DEP_STATE). Please create it first."
    log "Example: grpcurl ... gateway.Gateway/CreateDeployment with enable_hot_load=true"
    exit 1
fi
log "Deployment $DEPLOYMENT_ID is READY"

# -- 2. Create trainer jobs ---------------------------------------------------

log "Creating policy trainer job..."
POLICY_OUTPUT=$("$FIRECTL_BIN" -a "$FIREWORKS_ACCOUNT_ID" rlor create \
    --base-model "accounts/fireworks/models/qwen3-4b" \
    --training-shape "$TRAINING_SHAPE" \
    --accelerator-type "$ACCELERATOR" \
    --accelerator-count 8 \
    --region "$REGION" \
    --display-name "frozen-lake-ci-policy" \
    --service-mode \
    --deployment-id "$DEPLOYMENT_ID" \
    -p "$FIRECTL_PROFILE" 2>&1)

POLICY_JOB_ID=$(echo "$POLICY_OUTPUT" | grep "^Name:" | sed 's|.*rlorTrainerJobs/||')
log "Policy job: $POLICY_JOB_ID"

log "Creating reference trainer job..."
REFERENCE_OUTPUT=$("$FIRECTL_BIN" -a "$FIREWORKS_ACCOUNT_ID" rlor create \
    --base-model "accounts/fireworks/models/qwen3-4b" \
    --training-shape "$TRAINING_SHAPE" \
    --accelerator-type "$ACCELERATOR" \
    --accelerator-count 8 \
    --region "$REGION" \
    --display-name "frozen-lake-ci-reference" \
    --service-mode \
    --trainer-extra-args='--forward-only' \
    -p "$FIRECTL_PROFILE" 2>&1)

REFERENCE_JOB_ID=$(echo "$REFERENCE_OUTPUT" | grep "^Name:" | sed 's|.*rlorTrainerJobs/||')
log "Reference job: $REFERENCE_JOB_ID"

# -- 3. Wait for jobs to be running -------------------------------------------

log "Waiting for trainer jobs to start..."
for i in $(seq 1 80); do
    P_STATE=$("$FIRECTL_BIN" -a "$FIREWORKS_ACCOUNT_ID" rlor get "$POLICY_JOB_ID" -p "$FIRECTL_PROFILE" 2>&1 | grep "^State:" | awk '{print $2}')
    R_STATE=$("$FIRECTL_BIN" -a "$FIREWORKS_ACCOUNT_ID" rlor get "$REFERENCE_JOB_ID" -p "$FIRECTL_PROFILE" 2>&1 | grep "^State:" | awk '{print $2}')

    if [[ "$P_STATE" == "JOB_STATE_FAILED" || "$R_STATE" == "JOB_STATE_FAILED" ]]; then
        log "FATAL: A trainer job failed (policy=$P_STATE, reference=$R_STATE)"
        exit 1
    fi
    if [[ "$P_STATE" == "JOB_STATE_RUNNING" && "$R_STATE" == "JOB_STATE_RUNNING" ]]; then
        log "Both trainer jobs are RUNNING"
        break
    fi

    log "[$i/80] Waiting (policy=$P_STATE, reference=$R_STATE)..."
    sleep 15
done

if [[ "$P_STATE" != "JOB_STATE_RUNNING" || "$R_STATE" != "JOB_STATE_RUNNING" ]]; then
    log "FATAL: Trainer jobs did not start in time"
    exit 1
fi

# -- 4. Run the test ----------------------------------------------------------

log "Running FrozenLake B300 E2E test..."
export FROZEN_LAKE_POLICY_JOB_ID="$POLICY_JOB_ID"
export FROZEN_LAKE_REFERENCE_JOB_ID="$REFERENCE_JOB_ID"
export FROZEN_LAKE_DEPLOYMENT_ID="$DEPLOYMENT_ID"
export FROZEN_LAKE_REGION="$REGION"
export FROZEN_LAKE_TRAINING_SHAPE="$TRAINING_SHAPE"
export WANDB_MODE=disabled
export KEEP_DEPLOYMENT=1

cd "$REPO_ROOT"
python -m pytest training/tests/e2e/test_frozen_lake_b300_e2e.py \
    -v -s \
    --log-cli-level=INFO \
    --timeout=3600 \
    -x

TEST_EXIT=$?

log "Test exit code: $TEST_EXIT"
exit $TEST_EXIT
