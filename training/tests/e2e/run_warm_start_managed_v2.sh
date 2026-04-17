#!/usr/bin/env bash
# E2E: managed V2 SFT warm-start from an HF PEFT adapter.
#
# Exercises the full CP path:
#   firectl -> gateway -> CreateSupervisedFineTuningJob -> Temporal workflow
#   -> CreateTrainingJob activity (resolveWarmStartAdapter) -> orchestrator
#   -> cookbook sft_loop (warm_start_from_adapter) -> Tinker load_adapter.
#
# Uses qwen3-4b-minimal-lora (pyroworks account) as the warm-start source.
#
# Assertions (post-submit, while polling and after completion):
#   1. firectl accepts the submit (gateway validations pass).
#   2. Trainer pod reaches RUNNING.
#   3. Orchestrator stdout contains "Fresh start with HF adapter" AND
#      "Adapter loaded" — proves cookbook routed to load_adapter.
#   4. Job completes with state SUCCEEDED.
#   5. Output Model resource is produced, Kind=HF_PEFT_ADDON, State=READY.
#
# Environment:
#   FIREWORKS_API_KEY        -- pyroworks API key (required)
#   FIRECTL_PROFILE          -- default "pyroworks"
#   WARM_START_MODEL         -- default accounts/fireworks/models/qwen3-4b-minimal-lora
#   BASE_MODEL               -- default accounts/fireworks/models/qwen3-4b
#   DATASET_URI              -- default a small jsonl uploaded to pyroworks bucket
#   JOB_TIMEOUT_SECS         -- default 3600

set -euo pipefail

: "${FIREWORKS_API_KEY:?FIREWORKS_API_KEY must be set for pyroworks}"

FIRECTL_PROFILE="${FIRECTL_PROFILE:-pyroworks}"
WARM_START_MODEL="${WARM_START_MODEL:-accounts/fireworks/models/qwen3-4b-minimal-lora}"
BASE_MODEL="${BASE_MODEL:-accounts/fireworks/models/qwen3-4b}"
DATASET_URI="${DATASET_URI:-}"
JOB_TIMEOUT_SECS="${JOB_TIMEOUT_SECS:-3600}"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
fail() { log "FAIL: $*"; exit 1; }

log "Profile=${FIRECTL_PROFILE} warm_start=${WARM_START_MODEL} base=${BASE_MODEL}"

# -- Preflight ---------------------------------------------------------------

log "Verifying warm-start Model exists and is an adapter"
model_json=$(firectl --profile "${FIRECTL_PROFILE}" get model "${WARM_START_MODEL}" --output json)
kind=$(echo "${model_json}" | jq -r '.kind // empty')
state=$(echo "${model_json}" | jq -r '.state // empty')
[[ "${kind}" == "HF_PEFT_ADDON" ]] || fail "Expected Kind=HF_PEFT_ADDON, got ${kind}"
[[ "${state}" == "READY" ]] || fail "Expected State=READY, got ${state}"

# -- Dataset -----------------------------------------------------------------

if [[ -z "${DATASET_URI}" ]]; then
    tmp=$(mktemp -t warm_start_dataset.XXXXX.jsonl)
    trap 'rm -f "${tmp}"' EXIT
    python3 -c "
import json
for i in range(16):
    row = {'messages': [
        {'role': 'user',      'content': f'What is {i} + {i}?'},
        {'role': 'assistant', 'content': f'The answer is {i + i}.'},
    ]}
    print(json.dumps(row))
" > "${tmp}"
    DATASET_URI="${tmp}"
    log "Generated tiny dataset at ${DATASET_URI}"
fi

# -- Submit ------------------------------------------------------------------

log "Submitting SFT job (managed V2 + warm_start_from=${WARM_START_MODEL})"
submit_out=$(firectl --profile "${FIRECTL_PROFILE}" create sft \
    --use-v2 \
    --base-model "${BASE_MODEL}" \
    --warm-start-from "${WARM_START_MODEL}" \
    --dataset "${DATASET_URI}" \
    --lora-rank 16 \
    --epochs 1 \
    --batch-size 4 \
    --max-context-length 512 \
    --learning-rate 1e-4 \
    --output json)

job_id=$(echo "${submit_out}" | jq -r '.name // empty')
[[ -n "${job_id}" ]] || fail "submit returned empty job name: ${submit_out}"
log "Submitted: ${job_id}"

# -- Poll for completion -----------------------------------------------------

deadline=$(( $(date +%s) + JOB_TIMEOUT_SECS ))
state="UNKNOWN"
while (( $(date +%s) < deadline )); do
    state=$(firectl --profile "${FIRECTL_PROFILE}" get sft "${job_id}" --output json | \
            jq -r '.state // "UNKNOWN"')
    log "state=${state}"
    case "${state}" in
        JOB_STATE_SUCCEEDED|SUCCEEDED) break ;;
        JOB_STATE_FAILED|FAILED|JOB_STATE_CANCELLED|CANCELLED)
            fail "Job terminated in state=${state}" ;;
    esac
    sleep 30
done

[[ "${state}" == "JOB_STATE_SUCCEEDED" || "${state}" == "SUCCEEDED" ]] || \
    fail "Timed out after ${JOB_TIMEOUT_SECS}s; last state=${state}"

# -- Verify orchestrator logs show adapter load ------------------------------

log "Fetching orchestrator logs"
logs=$(firectl --profile "${FIRECTL_PROFILE}" get sft "${job_id}" --logs 2>/dev/null || true)
echo "${logs}" | grep -q "Fresh start with HF adapter" || \
    fail "Orchestrator did not log 'Fresh start with HF adapter' — cookbook routing may be wrong"
echo "${logs}" | grep -q "Adapter loaded" || \
    fail "Orchestrator did not log 'Adapter loaded' — SDK load_adapter call may have failed"

# -- Verify output model -----------------------------------------------------

output_model=$(firectl --profile "${FIRECTL_PROFILE}" get sft "${job_id}" --output json | \
               jq -r '.outputModel // empty')
[[ -n "${output_model}" ]] || fail "Job produced no outputModel"

out_json=$(firectl --profile "${FIRECTL_PROFILE}" get model "${output_model}" --output json)
out_kind=$(echo "${out_json}" | jq -r '.kind // empty')
out_state=$(echo "${out_json}" | jq -r '.state // empty')
[[ "${out_kind}" == "HF_PEFT_ADDON" ]] || fail "Output kind=${out_kind}, expected HF_PEFT_ADDON"
[[ "${out_state}" == "READY" ]] || fail "Output state=${out_state}, expected READY"

log "PASS: managed V2 warm-start E2E succeeded"
log "  job=${job_id}"
log "  output=${output_model} (${out_kind}/${out_state})"
