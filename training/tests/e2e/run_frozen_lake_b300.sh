#!/usr/bin/env bash
# Firetitan RLOR smoke test — FrozenLake GRPO on B300 GPUs.
#
# Manages the full lifecycle:
#   1. Cleans up stale CI jobs from previous runs
#   2. Verifies the deployment is READY
#   3. Creates policy + reference trainer jobs
#   4. Waits for jobs to reach RUNNING
#   5. Runs the pytest smoke test
#   6. Cleans up trainer jobs on exit
#
# Prerequisites:
#   FIREWORKS_API_KEY          API key for dev.api.fireworks.ai
#   FIREWORKS_ACCOUNT_ID       Default: pyroworks-dev
#   FIRECTL_BIN                Path to firectl-admin binary
#   FIRECTL_PROFILE            firectl profile for B300 gateway (default: dev-bennychen)
#   DEPLOYMENT_ID              Pre-created deployment with hotload
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
DEPLOYMENT_ID="${DEPLOYMENT_ID:-rl-qwen3-4b-b300-v10}"
REGION="${REGION:-EU_NETHERLANDS_1}"
ACCELERATOR="${ACCELERATOR:-NVIDIA_B300_288GB}"
JOB_WAIT_TIMEOUT="${JOB_WAIT_TIMEOUT:-80}"
JOB_CREATE_RETRIES="${JOB_CREATE_RETRIES:-3}"

export FIRECTL_AGENT_SAFE_ACCOUNTS="$FIREWORKS_ACCOUNT_ID"

POLICY_JOB_ID=""
REFERENCE_JOB_ID=""
START_TIME=$(date +%s)

log() { echo "$(date '+%H:%M:%S') [smoke] $*"; }

elapsed() {
    local now; now=$(date +%s)
    echo $(( now - START_TIME ))
}

# ── Cleanup on exit ─────────────────────────────────────────────────────────
cleanup() {
    local exit_code=$?
    [[ -n "${PF_DEPLOY_PID:-}" ]] && kill "$PF_DEPLOY_PID" 2>/dev/null || true
    for jid in "$POLICY_JOB_ID" "$REFERENCE_JOB_ID"; do
        [[ -z "$jid" ]] && continue
        log "Cleanup: deleting job $jid"
        "$FIRECTL_BIN" -a "$FIREWORKS_ACCOUNT_ID" rlor delete "$jid" \
            -p "$FIRECTL_PROFILE" 2>/dev/null || true
    done
    log "Finished in $(elapsed)s (exit=$exit_code)"
}
trap cleanup EXIT

# ── 0. Clean stale CI jobs ──────────────────────────────────────────────────
log "Cleaning stale CI jobs ..."
stale_jobs=$("$FIRECTL_BIN" -a "$FIREWORKS_ACCOUNT_ID" rlor list -p "$FIRECTL_PROFILE" 2>&1 \
    | grep -E "frozen-lake-ci-(policy|reference)" \
    | awk '{print $1}' || true)
for jid in $stale_jobs; do
    log "  deleting stale job: $jid"
    "$FIRECTL_BIN" -a "$FIREWORKS_ACCOUNT_ID" rlor delete "$jid" \
        -p "$FIRECTL_PROFILE" 2>/dev/null || true
done

# ── 1. Verify deployment ────────────────────────────────────────────────────
log "Checking deployment $DEPLOYMENT_ID ..."
DEP_STATE=$("$FIRECTL_BIN" -a "$FIREWORKS_ACCOUNT_ID" deployment get "$DEPLOYMENT_ID" \
    -p dev 2>&1 | awk '/^State:/{print $2}' || true)

if [[ "$DEP_STATE" != "READY" ]]; then
    log "FATAL: deployment $DEPLOYMENT_ID not READY (state=${DEP_STATE:-unknown})"
    log "Create a deployment with hotload enabled before running this test."
    exit 1
fi
log "Deployment READY"

# ── 2. Create trainer jobs (with retry) ─────────────────────────────────────
create_job() {
    local label=$1; shift
    local attempt output jid
    for attempt in $(seq 1 "$JOB_CREATE_RETRIES"); do
        output=$("$FIRECTL_BIN" -a "$FIREWORKS_ACCOUNT_ID" rlor create \
            --base-model "accounts/fireworks/models/qwen3-4b" \
            --training-shape "$TRAINING_SHAPE" \
            --accelerator-type "$ACCELERATOR" --accelerator-count 8 \
            --region "$REGION" \
            --display-name "frozen-lake-ci-$label" \
            --service-mode \
            "$@" \
            -p "$FIRECTL_PROFILE" 2>&1) && break
        log "  attempt $attempt/$JOB_CREATE_RETRIES for $label failed, retrying in 10s ..."
        sleep 10
    done
    jid=$(echo "$output" | awk -F/ '/^Name:/{print $NF}')
    if [[ -z "$jid" ]]; then
        log "FATAL: failed to create $label job after $JOB_CREATE_RETRIES attempts"
        log "Output: $output"
        exit 1
    fi
    echo "$jid"
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

log "Waiting for trainer jobs (timeout=${JOB_WAIT_TIMEOUT} polls) ..."
for i in $(seq 1 "$JOB_WAIT_TIMEOUT"); do
    P=$(get_state "$POLICY_JOB_ID")
    R=$(get_state "$REFERENCE_JOB_ID")

    if [[ "$P" == "JOB_STATE_FAILED" || "$R" == "JOB_STATE_FAILED" ]]; then
        log "FATAL: trainer job failed (policy=$P, reference=$R)"
        # Print failure details
        "$FIRECTL_BIN" -a "$FIREWORKS_ACCOUNT_ID" rlor get "$POLICY_JOB_ID" -p "$FIRECTL_PROFILE" 2>&1 | grep -E "^(State|Status):" || true
        "$FIRECTL_BIN" -a "$FIREWORKS_ACCOUNT_ID" rlor get "$REFERENCE_JOB_ID" -p "$FIRECTL_PROFILE" 2>&1 | grep -E "^(State|Status):" || true
        exit 1
    fi
    if [[ "$P" == "JOB_STATE_RUNNING" && "$R" == "JOB_STATE_RUNNING" ]]; then
        log "Both jobs RUNNING ($(elapsed)s elapsed)"
        break
    fi
    log "  [$i/$JOB_WAIT_TIMEOUT] policy=$P  reference=$R"
    sleep 15
done

P=$(get_state "$POLICY_JOB_ID")
R=$(get_state "$REFERENCE_JOB_ID")
if [[ "$P" != "JOB_STATE_RUNNING" || "$R" != "JOB_STATE_RUNNING" ]]; then
    log "FATAL: jobs did not start in time (policy=$P, reference=$R)"
    exit 1
fi

# ── 4. (Optional) Port-forward deployment for inference ────────────────────
# Trainer traffic now routes through the API gateway automatically via
# /training/v1/rlorTrainerJobs/{accountId}/{jobId}/* so no port-forward
# is needed for trainers. Inference deployment may still need port-forward
# for OCI clusters if the deployment isn't accessible via gateway.
K8S_CONTEXT="${K8S_CONTEXT:-}"
PF_DEPLOY_PID=""

if [[ -n "$K8S_CONTEXT" ]]; then
    DEPLOY_POD=$(kubectl --context="$K8S_CONTEXT" get pods -n default \
        -l "app.kubernetes.io/instance=${FIREWORKS_ACCOUNT_ID}-${DEPLOYMENT_ID}" \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)

    if [[ -n "$DEPLOY_POD" ]]; then
        log "Setting up deployment port-forward (context=$K8S_CONTEXT, pod=$DEPLOY_POD)..."
        _pf_loop() {
            local pod=$1 port=$2 target_port=$3
            while true; do
                kubectl --context="$K8S_CONTEXT" port-forward -n default "$pod" "${port}:${target_port}" 2>/dev/null || true
                sleep 2
            done
        }
        _pf_loop "$DEPLOY_POD" 18082 80 &
        PF_DEPLOY_PID=$!

        sleep 3
        log "Verifying deployment port-forward..."
        curl -s http://localhost:18082/v1/completions \
            -d '{"model":"accounts/fireworks/models/qwen3-4b","prompt":"test","max_tokens":1}' \
            -H "Content-Type: application/json" > /dev/null && log "  deploy port-forward OK" || log "  WARN: deploy port-forward not ready"
        export FROZEN_LAKE_INFERENCE_BASE_URL="http://localhost:18082"
    fi
fi

# ── 5. Run pytest ───────────────────────────────────────────────────────────
log "Running FrozenLake B300 smoke test ..."
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

log "Smoke test PASSED ($(elapsed)s total)"
