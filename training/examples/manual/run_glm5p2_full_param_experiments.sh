#!/usr/bin/env bash
set -euo pipefail

# Manual AP_MALAYSIA_2 / pyroworks / RFTJ full-param experiment runner.
#
# This script intentionally uses the normal firectl-admin service-mode trainer
# path. It does not use reservations and does not bypass scheduler admission.

ACCOUNT_ID="${ACCOUNT_ID:-pyroworks}"
REGION="${REGION:-AP_MALAYSIA_2}"
KUBE_CONTEXT="${KUBE_CONTEXT:-$(kubectl config current-context 2>/dev/null || true)}"
BASE_MODEL="${BASE_MODEL:-accounts/fireworks/models/glm-5p2-fp8}"
CUSTOM_IMAGE_TAG="${CUSTOM_IMAGE_TAG:-0.495.0}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-NVIDIA_B300_288GB}"
ACCELERATOR_COUNT="${ACCELERATOR_COUNT:-8}"
NODES="${NODES:-4}"
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-204785}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/glm5p2_full_param_experiments_$(date -u +%Y%m%dT%H%M%SZ)}"
LOAD_SCRIPT="${LOAD_SCRIPT:-$(dirname "$0")/full_param_load_test.py}"
LOAD_BASE_MODEL="${LOAD_BASE_MODEL:-glm-5p2-fp8}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-7200}"
DELETE_TIMEOUT_S="${DELETE_TIMEOUT_S:-1800}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-30}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-1}"
CASE_FILTER="${CASE_FILTER:-}"

if [[ "$ACCOUNT_ID" != "pyroworks" ]]; then
  echo "Refusing to run: ACCOUNT_ID must be pyroworks for this experiment." >&2
  exit 2
fi

if [[ "$REGION" != "AP_MALAYSIA_2" ]]; then
  echo "Refusing to run: REGION must be AP_MALAYSIA_2 for this experiment." >&2
  exit 2
fi

mkdir -p "$RESULTS_DIR"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

run_nvidia_smi_preflight() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "nvidia-smi not found locally; continuing only because the trainer is remote."
    return
  fi

  nvidia-smi | tee "$RESULTS_DIR/nvidia-smi-preflight.txt"
  local gpu_pids
  gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d' || true)"
  if [[ -n "$gpu_pids" ]]; then
    echo "Local GPU processes are running; refusing to launch GPU-consuming work." >&2
    echo "$gpu_pids" >&2
    exit 3
  fi
}

job_resource_name() {
  local job_id="$1"
  printf 'accounts/%s/rlorTrainerJobs/%s' "$ACCOUNT_ID" "$job_id"
}

kube_prefix() {
  local job_id="$1"
  printf 'trainer-%s-rlor-%s' "$ACCOUNT_ID" "$job_id"
}

snapshot_job() {
  local job_id="$1"
  local label="$2"
  local out="$RESULTS_DIR/${job_id}.${label}.firectl.txt"
  firectl-admin -a "$ACCOUNT_ID" rlor-trainer-job get "$job_id" >"$out" 2>&1 || true
  cat "$out"
}

snapshot_kube() {
  local job_id="$1"
  local label="$2"
  local prefix
  prefix="$(kube_prefix "$job_id")"
  local out="$RESULTS_DIR/${job_id}.${label}.kube.txt"
  {
    echo "context=$KUBE_CONTEXT"
    kubectl --context "$KUBE_CONTEXT" -n default get sts,svc,pod | rg "$prefix|NAME" || true
    echo
    kubectl --context "$KUBE_CONTEXT" -n default describe pod "${prefix}-0" || true
  } >"$out" 2>&1
  cat "$out"
}

snapshot_logs() {
  local job_id="$1"
  local label="$2"
  local prefix
  prefix="$(kube_prefix "$job_id")"
  local out="$RESULTS_DIR/${job_id}.${label}.trainer.log"
  kubectl --context "$KUBE_CONTEXT" -n default logs "${prefix}-0" -c trainer --tail=240 >"$out" 2>&1 || true
  cat "$out"
}

wait_for_ready() {
  local job_id="$1"
  local deadline=$((SECONDS + READY_TIMEOUT_S))
  while (( SECONDS < deadline )); do
    snapshot_job "$job_id" "poll" > /dev/null
    local poll_file="$RESULTS_DIR/${job_id}.poll.firectl.txt"
    if rg -q "State: JOB_STATE_RUNNING" "$poll_file"; then
      if rg -q "Service is running and ready|Direct Route Handle" "$poll_file"; then
        return 0
      fi
    fi
    if rg -q "State: JOB_STATE_FAILED|State: JOB_STATE_CANCELLED|State: JOB_STATE_DELETED" "$poll_file"; then
      cat "$poll_file"
      return 1
    fi
    snapshot_kube "$job_id" "poll" > /dev/null
    snapshot_logs "$job_id" "poll" > /dev/null
    sleep "$POLL_INTERVAL_S"
  done
  echo "Timed out waiting for $job_id to become ready." >&2
  return 1
}

delete_and_wait() {
  local job_id="$1"
  local prefix
  prefix="$(kube_prefix "$job_id")"

  log "deleting $job_id"
  firectl-admin -a "$ACCOUNT_ID" rlor-trainer-job delete "$job_id" --wait --wait-timeout "${DELETE_TIMEOUT_S}s" || true

  local deadline=$((SECONDS + DELETE_TIMEOUT_S))
  while (( SECONDS < deadline )); do
    if ! kubectl --context "$KUBE_CONTEXT" -n default get sts,svc,pod | rg -q "$prefix"; then
      snapshot_job "$job_id" "deleted" > /dev/null
      snapshot_kube "$job_id" "deleted" > /dev/null
      log "$job_id deleted; Kubernetes resources are gone"
      return 0
    fi
    snapshot_kube "$job_id" "deleting" > /dev/null
    sleep 10
  done

  echo "Timed out waiting for Kubernetes resources for $job_id to disappear." >&2
  snapshot_kube "$job_id" "delete-timeout"
  return 1
}

launch_case() {
  local case_name="$1"
  shift
  local job_id="glm5p2full-${case_name}-$(date -u +%m%d%H%M)"
  log "launching $case_name as $job_id"

  set +e
  firectl-admin -a "$ACCOUNT_ID" rlor-trainer-job create \
    --job-id "$job_id" \
    --service-mode \
    --skip-validations \
    --base-model "$BASE_MODEL" \
    --dataset "" \
    --lora-rank 0 \
    --max-context-length "$MAX_CONTEXT_LENGTH" \
    --accelerator-type "$ACCELERATOR_TYPE" \
    --accelerator-count "$ACCELERATOR_COUNT" \
    --nodes "$NODES" \
    --region "$REGION" \
    --custom-image-tag "$CUSTOM_IMAGE_TAG" \
    --extra-values=tolerationKey=fireworks.ai/rftj \
    --trainer-extra-args='--full-oom-check' \
    --trainer-extra-args='--tp=1' \
    --trainer-extra-args='--pp=1' \
    "$@" | tee "$RESULTS_DIR/${job_id}.create.txt"
  local create_status=${PIPESTATUS[0]}
  set -e
  if (( create_status != 0 )); then
    return "$create_status"
  fi

  snapshot_job "$job_id" "created"
  snapshot_kube "$job_id" "created"

  if ! wait_for_ready "$job_id"; then
    snapshot_job "$job_id" "not-ready"
    snapshot_kube "$job_id" "not-ready"
    snapshot_logs "$job_id" "not-ready"
    delete_and_wait "$job_id" || true
    return 1
  fi

  snapshot_job "$job_id" "ready"
  snapshot_kube "$job_id" "ready"
  snapshot_logs "$job_id" "ready"

  set +e
  python "$LOAD_SCRIPT" \
    --job-id "$job_id" \
    --base-model "$LOAD_BASE_MODEL" \
    --output-json "$RESULTS_DIR/${job_id}.load.json" \
    2>&1 | tee "$RESULTS_DIR/${job_id}.load.log"
  local load_status=${PIPESTATUS[0]}
  set -e

  snapshot_job "$job_id" "after-load"
  snapshot_kube "$job_id" "after-load"
  snapshot_logs "$job_id" "after-load"
  delete_and_wait "$job_id"
  return "$load_status"
}

should_run_case() {
  local case_name="$1"
  [[ -z "$CASE_FILTER" || "$case_name" == "$CASE_FILTER" ]]
}

run_case_or_record_failure() {
  local case_name="$1"
  shift
  if ! should_run_case "$case_name"; then
    return 0
  fi
  if launch_case "$case_name" "$@"; then
    log "$case_name succeeded"
  else
    log "$case_name failed"
    if [[ "$CONTINUE_ON_FAILURE" != "1" ]]; then
      exit 1
    fi
  fi
}

main() {
  log "results: $RESULTS_DIR"
  run_nvidia_smi_preflight

  run_case_or_record_failure cp32-ep8-offload \
    --trainer-extra-args='--cp=32' \
    --trainer-extra-args='--ep=8' \
    --trainer-extra-args='--training-quant' \
    --trainer-extra-args='moe=fp8_block128,storage_format=fp8_block128'

  run_case_or_record_failure cp32-ep8-nooffload \
    --trainer-extra-args='--cp=32' \
    --trainer-extra-args='--ep=8' \
    --trainer-extra-args='--training-quant' \
    --trainer-extra-args='moe=fp8_block128,storage_format=fp8_block128' \
    --trainer-extra-args='--no-enable-optimizer-offload'

  run_case_or_record_failure cp16-ep8-dps2 \
    --trainer-extra-args='--cp=16' \
    --trainer-extra-args='--ep=8' \
    --trainer-extra-args='--dp-shard=2' \
    --trainer-extra-args='--training-quant' \
    --trainer-extra-args='moe=fp8_block128,storage_format=fp8_block128'

  run_case_or_record_failure cp16-ep8-dps2-nooffload \
    --trainer-extra-args='--cp=16' \
    --trainer-extra-args='--ep=8' \
    --trainer-extra-args='--dp-shard=2' \
    --trainer-extra-args='--training-quant' \
    --trainer-extra-args='moe=fp8_block128,storage_format=fp8_block128' \
    --trainer-extra-args='--no-enable-optimizer-offload'

  run_case_or_record_failure cp16-ep8-dps1-dpr2 \
    --trainer-extra-args='--cp=16' \
    --trainer-extra-args='--ep=8' \
    --trainer-extra-args='--dp-shard=1' \
    --trainer-extra-args='--dp-replicate=2' \
    --trainer-extra-args='--training-quant' \
    --trainer-extra-args='moe=fp8_block128,storage_format=fp8_block128'

  log "done; results: $RESULTS_DIR"
}

main "$@"
