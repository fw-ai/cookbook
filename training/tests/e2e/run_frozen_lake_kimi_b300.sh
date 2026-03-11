#!/usr/bin/env bash
# Firetitan RLOR smoke test — FrozenLake GRPO on Kimi K2.5 (B300 LoRA).
#
# Manages the full lifecycle on B300 in EU_NETHERLANDS_1 (AMS):
#   1. Cleans up stale CI jobs from previous runs
#   2. Creates inference deployment (shapeless, with Kimi-specific args)
#   3. Creates policy + reference trainer jobs (8x B300, LoRA rank 8)
#   4. Waits for all resources to reach RUNNING/serving
#   5. Runs the pytest smoke test (1 epoch, hotload disabled)
#   6. Cleans up all resources on exit
#
# Prerequisites:
#   FIREWORKS_API_KEY          API key for dev.api.fireworks.ai
#   FIREWORKS_ACCOUNT_ID       Default: pyroworks-dev
#   FIRECTL_BIN                Path to firectl-admin binary
#   FIRECTL_PROFILE            firectl profile (default: dev-bennychen)
#
# Usage:
#   export FIREWORKS_API_KEY=fw_...
#   bash training/tests/e2e/run_frozen_lake_kimi_b300.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ── Configuration ────────────────────────────────────────────────────────────
export FIREWORKS_ACCOUNT_ID="${FIREWORKS_ACCOUNT_ID:-pyroworks-dev}"
export FIREWORKS_BASE_URL="${FIREWORKS_BASE_URL:-https://dev.api.fireworks.ai}"

FIRECTL_BIN="${FIRECTL_BIN:-$REPO_ROOT/../fireworks/firectl/bin/firectl-admin}"
FIRECTL_PROFILE="${FIRECTL_PROFILE:-dev-bennychen}"
BASE_MODEL="accounts/fireworks/models/kimi-k2p5"
REGION="${REGION:-EU_NETHERLANDS_1}"
ACCELERATOR="${ACCELERATOR:-NVIDIA_B300_288GB}"
LORA_RANK="${LORA_RANK:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-65536}"
JOB_WAIT_TIMEOUT="${JOB_WAIT_TIMEOUT:-80}"
JOB_CREATE_RETRIES="${JOB_CREATE_RETRIES:-3}"

# B300-compatible image tag (must match what's deployed in AMS)
IMAGE_TAG="${IMAGE_TAG:-4.69.0-backwards-compatible-b300.4}"

export FIRECTL_AGENT_SAFE_ACCOUNTS="$FIREWORKS_ACCOUNT_ID"

DEPLOYMENT_ID=""
POLICY_JOB_ID=""
REFERENCE_JOB_ID=""
START_TIME=$(date +%s)

log() { echo "$(date '+%H:%M:%S') [kimi-b300] $*"; }

elapsed() {
    local now; now=$(date +%s)
    echo $(( now - START_TIME ))
}

# ── Cleanup on exit ─────────────────────────────────────────────────────────
cleanup() {
    local exit_code=$?
    for jid in "$POLICY_JOB_ID" "$REFERENCE_JOB_ID"; do
        [[ -z "$jid" ]] && continue
        log "Cleanup: deleting job $jid"
        "$FIRECTL_BIN" -a "$FIREWORKS_ACCOUNT_ID" rlor delete "$jid" \
            -p "$FIRECTL_PROFILE" 2>/dev/null || true
    done
    if [[ -n "$DEPLOYMENT_ID" ]]; then
        log "Cleanup: deleting deployment $DEPLOYMENT_ID"
        "$FIRECTL_BIN" -a "$FIREWORKS_ACCOUNT_ID" deployment delete "$DEPLOYMENT_ID" \
            --ignore-checks -p "$FIRECTL_PROFILE" 2>/dev/null || true
    fi
    log "Finished in $(elapsed)s (exit=$exit_code)"
}
trap cleanup EXIT

# ── 0. Clean stale CI jobs ──────────────────────────────────────────────────
log "Cleaning stale CI jobs ..."
stale_jobs=$("$FIRECTL_BIN" -a "$FIREWORKS_ACCOUNT_ID" rlor list -p "$FIRECTL_PROFILE" 2>&1 \
    | grep -E "kimi-fl-b300-ci-(policy|reference)" \
    | awk '{print $1}' || true)
for jid in $stale_jobs; do
    log "  deleting stale job: $jid"
    "$FIRECTL_BIN" -a "$FIREWORKS_ACCOUNT_ID" rlor delete "$jid" \
        -p "$FIRECTL_PROFILE" 2>/dev/null || true
done

# ── 1. Create B300 deployment ───────────────────────────────────────────────
DEPLOYMENT_ID="kimi-k2p5-b300-ci-$(date +%m%d%H%M)"
log "Creating deployment $DEPLOYMENT_ID ..."
"$FIRECTL_BIN" -a "$FIREWORKS_ACCOUNT_ID" deployment create \
    "$BASE_MODEL" \
    --deployment-id "$DEPLOYMENT_ID" \
    --deployment-shape rft-kimi-k2p5-b300 \
    --region "$(echo "$REGION" | tr '[:upper:]' '[:lower:]' | tr '_' '-')" \
    --min-replica-count 1 \
    --max-replica-count 1 \
    --disable-accounting \
    --skip-shape-validation \
    -p "$FIRECTL_PROFILE" 2>&1 | grep -E "^Name:|^State:" || true
log "Deployment $DEPLOYMENT_ID created"

# ── 2. Create trainer jobs (with retry) ─────────────────────────────────────
create_job() {
    local label=$1; shift
    local attempt output jid
    for attempt in $(seq 1 "$JOB_CREATE_RETRIES"); do
        output=$("$FIRECTL_BIN" -a "$FIREWORKS_ACCOUNT_ID" rlor create \
            --base-model "$BASE_MODEL" \
            --accelerator-type "$ACCELERATOR" --accelerator-count 8 \
            --region "$REGION" \
            --lora-rank "$LORA_RANK" \
            --max-context-length "$MAX_SEQ_LEN" \
            --display-name "kimi-fl-b300-ci-$label" \
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

# ── 3. Wait for all resources ───────────────────────────────────────────────
get_state() {
    "$FIRECTL_BIN" -a "$FIREWORKS_ACCOUNT_ID" rlor get "$1" \
        -p "$FIRECTL_PROFILE" 2>&1 | awk '/^State:/{print $2}'
}

get_deploy_ready_replicas() {
    "$FIRECTL_BIN" -a "$FIREWORKS_ACCOUNT_ID" deployment get "$1" \
        -p "$FIRECTL_PROFILE" 2>&1 | awk '/Ready Replica Count:/{print $4}'
}

log "Waiting for trainer jobs + deployment (timeout=${JOB_WAIT_TIMEOUT} polls) ..."
for i in $(seq 1 "$JOB_WAIT_TIMEOUT"); do
    P=$(get_state "$POLICY_JOB_ID")
    R=$(get_state "$REFERENCE_JOB_ID")
    DR=$(get_deploy_ready_replicas "$DEPLOYMENT_ID")

    if [[ "$P" == "JOB_STATE_FAILED" || "$R" == "JOB_STATE_FAILED" ]]; then
        log "FATAL: trainer job failed (policy=$P, reference=$R)"
        exit 1
    fi
    if [[ "$P" == "JOB_STATE_RUNNING" && "$R" == "JOB_STATE_RUNNING" && "${DR:-0}" -ge 1 ]]; then
        log "All resources ready ($(elapsed)s elapsed)"
        break
    fi
    log "  [$i/$JOB_WAIT_TIMEOUT] policy=$P  reference=$R  deploy_replicas=${DR:-0}"
    sleep 15
done

P=$(get_state "$POLICY_JOB_ID")
R=$(get_state "$REFERENCE_JOB_ID")
DR=$(get_deploy_ready_replicas "$DEPLOYMENT_ID")
if [[ "$P" != "JOB_STATE_RUNNING" || "$R" != "JOB_STATE_RUNNING" || "${DR:-0}" -lt 1 ]]; then
    log "FATAL: resources did not start in time (policy=$P, reference=$R, deploy_replicas=${DR:-0})"
    exit 1
fi

# ── 4. Run pytest ───────────────────────────────────────────────────────────
log "Running FrozenLake Kimi B300 smoke test ..."
export KIMI_FROZEN_LAKE_POLICY_JOB_ID="$POLICY_JOB_ID"
export KIMI_FROZEN_LAKE_REFERENCE_JOB_ID="$REFERENCE_JOB_ID"
export KIMI_FROZEN_LAKE_DEPLOYMENT_ID="$DEPLOYMENT_ID"
export KIMI_FROZEN_LAKE_REGION="$REGION"
export KIMI_FROZEN_LAKE_LORA_RANK="$LORA_RANK"
export KIMI_FROZEN_LAKE_MAX_SEQ_LEN="$MAX_SEQ_LEN"
export KIMI_FROZEN_LAKE_OBSERVATION_MODE=text
export KIMI_FROZEN_LAKE_DISABLE_HOTLOAD=true
export KIMI_FROZEN_LAKE_EPOCHS=1
export WANDB_MODE=disabled
export KEEP_DEPLOYMENT=1

cd "$REPO_ROOT"
python -m pytest training/tests/e2e/test_frozen_lake_kimi_b300_e2e.py \
    -v -s --log-cli-level=INFO --timeout=5400 -x

log "Kimi B300 smoke test PASSED ($(elapsed)s total)"
