#!/usr/bin/env bash
#
# Customer-facing Fireworks launcher for the bundled Terminal-Bench curriculum.
#
# Required inputs:
#   FIREWORKS_API_KEY
#   WANDB_API_KEY
#   FIREWORKS_MODEL_ID
#   FIREWORKS_TRAINING_SHAPE_ID
#   TOKENIZER_MODEL
#   LEARNING_RATE
#   MAX_PROMPT_LENGTH
#   MAX_RESPONSE_LENGTH
#
# This launcher uses only the public Fireworks API. The authenticated customer
# must be allowed to use the requested model and training shape.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

fireworks_model_id="${FIREWORKS_MODEL_ID:?Set FIREWORKS_MODEL_ID to a model resource you can use}"
training_shape_id="${FIREWORKS_TRAINING_SHAPE_ID:?Set FIREWORKS_TRAINING_SHAPE_ID to a compatible training shape}"
tokenizer_model="${TOKENIZER_MODEL:?Set TOKENIZER_MODEL to the Hugging Face tokenizer for the model}"
learning_rate="${LEARNING_RATE:?Set LEARNING_RATE for the selected model and tuning mode}"
max_prompt_length="${MAX_PROMPT_LENGTH:?Set MAX_PROMPT_LENGTH within the training shape context limit}"
max_response_length="${MAX_RESPONSE_LENGTH:?Set MAX_RESPONSE_LENGTH within the training shape context limit}"

if [ -z "${FIREWORKS_API_KEY:-}" ]; then
    echo "FIREWORKS_API_KEY is required" >&2
    exit 1
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WANDB_API_KEY is required" >&2
    exit 1
fi

require_positive_integer() {
    local name="$1"
    local value="$2"
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$name must be a positive integer; got '$value'" >&2
        exit 2
    fi
}

require_nonnegative_integer() {
    local name="$1"
    local value="$2"
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "$name must be a non-negative integer; got '$value'" >&2
        exit 2
    fi
}

harness="${TB_HARNESS:-opencode}"
case "$harness" in
    opencode|terminus-2) ;;
    *)
        echo "TB_HARNESS must be 'opencode' or 'terminus-2'; got '$harness'" >&2
        exit 2
        ;;
esac

lora_rank="${LORA_RANK:-0}"
trainer_replicas="${TRAINER_REPLICAS:-1}"
rollout_replicas="${ROLLOUT_REPLICAS:-1}"
group_size="${GROUP_SIZE:-8}"
prompt_groups_per_step="${PROMPT_GROUPS_PER_STEP:-16}"
total_epochs="${TOTAL_EPOCHS:-4}"
n_parallel_tasks="${N_PARALLEL_TASKS:-16}"
save_frequency="${SAVE_FREQUENCY:-20}"

require_nonnegative_integer LORA_RANK "$lora_rank"
require_positive_integer TRAINER_REPLICAS "$trainer_replicas"
require_positive_integer ROLLOUT_REPLICAS "$rollout_replicas"
require_positive_integer GROUP_SIZE "$group_size"
require_positive_integer PROMPT_GROUPS_PER_STEP "$prompt_groups_per_step"
require_positive_integer TOTAL_EPOCHS "$total_epochs"
require_positive_integer N_PARALLEL_TASKS "$n_parallel_tasks"
require_positive_integer SAVE_FREQUENCY "$save_frequency"
require_positive_integer MAX_PROMPT_LENGTH "$max_prompt_length"
require_positive_integer MAX_RESPONSE_LENGTH "$max_response_length"

train_task_count=48
eval_task_count=89
steps_per_epoch=$(((train_task_count + prompt_groups_per_step - 1) / prompt_groups_per_step))
test_frequency="${EVAL_FREQUENCY:-$steps_per_epoch}"
require_positive_integer EVAL_FREQUENCY "$test_frequency"

temperature="${TEMPERATURE:-1.0}"
top_p="${TOP_P:-1.0}"
beta2="${BETA2:-0.999}"
python_bin="${RLLM_PYTHON:-python}"
state_root="${TB_STATE_ROOT:-${HOME}/.rllm/terminal-bench-rl}"
run_stamp="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_name="${WANDB_RUN_NAME:-terminal-bench-${harness}-${run_stamp}}"
wandb_project="${WANDB_PROJECT:-terminal-rl}"
gateway_port="${RLLM_GATEWAY_PORT:-9200}"

export RLLM_HOME="${RLLM_HOME:-${state_root}/state}"
export HF_HOME="${HF_HOME:-${state_root}/hf-home}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${state_root}/uv-cache}"
export WANDB_MODE=online
export WANDB_DIR="${WANDB_DIR:-${state_root}/wandb}"
export WANDB_TAGS="${WANDB_TAGS:-terminal-bench,fixed-48,${harness}}"

export TERMINAL_SANDBOX_BACKEND="${TERMINAL_SANDBOX_BACKEND:-docker}"
export TB_HARNESS="$harness"
export TB_TRAIN_DATASET="terminal-bench-fixed-48"
export TB_TRAIN_SPLIT="train"
export TB_TRAIN_EXPECTED_TASKS="$train_task_count"
export TB_VAL_DATASET="terminal-bench@2.1"
export TB_VAL_SPLIT="default"
export TB_VAL_EXPECTED_TASKS="$eval_task_count"
export TB_VAL_MAX=0
export TB_BENCHMARK_DATASET=""
export TERMINUS_MAX_TURNS="${TERMINUS_MAX_TURNS:-100}"
export TERMINUS_ENABLE_SUMMARIZE="${TERMINUS_ENABLE_SUMMARIZE:-1}"
export TERMINUS_MAX_INPUT_TOKENS="${TERMINUS_MAX_INPUT_TOKENS:-$max_prompt_length}"
export RLLM_HARNESS_INSTALL_TIMEOUT_S="${RLLM_HARNESS_INSTALL_TIMEOUT_S:-900}"
export RLLM_HARNESS_RUN_TIMEOUT_S="${RLLM_HARNESS_RUN_TIMEOUT_S:-1800}"
export RLLM_HARNESS_VERIFIER_TIMEOUT_S="${RLLM_HARNESS_VERIFIER_TIMEOUT_S:-300}"
export RLLM_SANDBOX_TIMEOUT_S="${RLLM_SANDBOX_TIMEOUT_S:-3000}"

mkdir -p "$RLLM_HOME" "$WANDB_DIR" "${state_root}/logs"

# Register the fixed 48-task archive and the pinned public TB2.1 evaluation
# suite in the same RLLM_HOME that train.py reads below.
"$python_bin" prepare_data.py \
    --train-dataset "$TB_TRAIN_DATASET" \
    --train-split "$TB_TRAIN_SPLIT" \
    --eval-dataset "$TB_VAL_DATASET" \
    --eval-split "$TB_VAL_SPLIT"

global_trajectories_per_step=$((group_size * prompt_groups_per_step))
printf 'run=%s model=%s shape=%s harness=%s lora_rank=%s train_tasks=%s eval_tasks=%s epochs=%s steps_per_epoch=%s group_size=%s prompt_groups_per_step=%s trajectories_per_step=%s trainer_replicas=%s rollout_replicas=%s region=%s\n' \
    "$run_name" "$fireworks_model_id" "$training_shape_id" "$harness" \
    "$lora_rank" "$train_task_count" "$eval_task_count" "$total_epochs" \
    "$steps_per_epoch" "$group_size" "$prompt_groups_per_step" \
    "$global_trajectories_per_step" "$trainer_replicas" "$rollout_replicas" \
    "${FIREWORKS_REGION:-control-plane-selected}"

region_override=()
if [ -n "${FIREWORKS_REGION:-}" ]; then
    region_override+=("fireworks_infra.trainers.policy.region=$FIREWORKS_REGION")
fi

exec "$python_bin" -u train.py \
    rllm/backend=fireworks \
    model.name="$fireworks_model_id" \
    model.tokenizer_model="$tokenizer_model" \
    model.lora_rank="$lora_rank" \
    fireworks_config.policy_trainer_shape_id="$training_shape_id" \
    fireworks_config.policy_trainer_replica_count="$trainer_replicas" \
    fireworks_config.rollout_deployment_replica_count="$rollout_replicas" \
    "${region_override[@]}" \
    training.group_size="$group_size" \
    training.learning_rate="$learning_rate" \
    training.beta2="$beta2" \
    training.max_length=null \
    rllm.rollout.train.temperature="$temperature" \
    rllm.rollout.train.top_p="$top_p" \
    rllm.rollout.val.temperature="$temperature" \
    rllm.rollout.val.top_p="$top_p" \
    rllm.data.max_prompt_length="$max_prompt_length" \
    rllm.data.max_response_length="$max_response_length" \
    rllm.data.train_batch_size="$prompt_groups_per_step" \
    rllm.data.val_batch_size=-1 \
    rllm.compact_filtering.enable=true \
    rllm.compact_filtering.mask_max_prompt_length_exceeded=false \
    rllm.compact_filtering.mask_max_response_length_exceeded=false \
    rllm.compact_filtering.mask_max_turns_exceeded=false \
    rllm.compact_filtering.mask_timeout=false \
    rllm.compact_filtering.mask_error=true \
    rllm.compact_filtering.mask_verifier_timeout=true \
    rllm.compact_filtering.mask_grading_error=true \
    rllm.compact_filtering.mask_sandbox_error=true \
    rllm.compact_filtering.mask_agent_setup_timeout=true \
    rllm.compact_filtering.mask_env_start_timeout=true \
    rllm.compact_filtering.mask_model_error=true \
    rllm.algorithm.adv_estimator=grpo \
    rllm.algorithm.norm_adv_by_std_in_grpo=true \
    rllm.algorithm.router_replay=R3 \
    rllm.algorithm.loss_fn=ppo_clip \
    rllm.algorithm.eps_clip=0.2 \
    rllm.algorithm.loss_agg_mode=token-mean \
    rllm.algorithm.rollout_correction.bypass_mode=true \
    rllm.async_training.enable=false \
    rllm.async_training.staleness_threshold=0.0 \
    rllm.async_training.trigger_parameter_sync_step=1 \
    rllm.async_training.partial_rollout=false \
    rllm.workflow.n_parallel_tasks="$n_parallel_tasks" \
    rllm.workflow.raise_on_error=false \
    rllm.rejection_sample.filter_uniform_groups=false \
    rllm.gateway.port="$gateway_port" \
    rllm.gateway.num_workers=4 \
    rllm.gateway.cumulative_token_mode=true \
    rllm.gateway.renderer_family=auto \
    rllm.trainer.total_epochs="$total_epochs" \
    rllm.trainer.total_batches=-1 \
    rllm.trainer.logger='[console,wandb]' \
    rllm.trainer.project_name="$wandb_project" \
    rllm.trainer.experiment_name="$run_name" \
    rllm.trainer.skip_zero_advantage_batches=true \
    rllm.trainer.val_before_train=false \
    rllm.trainer.benchmark_before_train=false \
    rllm.trainer.benchmark_after_train=false \
    rllm.trainer.test_freq="$test_frequency" \
    rllm.trainer.save_freq="$save_frequency" \
    "$@"
