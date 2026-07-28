# Terminal-Bench RL on Fireworks

This example packages the fixed 48-task OpenCode curriculum used by the
[public GLM-5.2 reference report](https://wandb.ai/myh97/terminal-rl/reports/GLM-5.2-Terminal-Bench-2.1-RL-reference-run--VmlldzoxNzYwMjE3Nw==?accessToken=14s9t0puwxy4ejzk8fzk007pmltr1slscy8b2xvcxv1k0q846hnfa1rv2rjx62sx),
while keeping the Fireworks launcher model- and account-agnostic.

`train.py` is a byte-for-byte copy of
[rllm-org/rllm#779](https://github.com/rllm-org/rllm/pull/779) at
`5f2fed491589790457c6bb292734415ea108463a`. The customer-facing
`train_fireworks.sh` supplies the dataset and model configuration without
changing that training loop. Run the example through this launcher; it
overrides the upstream file's standalone dataset defaults.

The task environments, solutions, and verifiers are stored in
`data/tasks.tar.gz`; `data/task_ids.txt` records the original 48-task order.
Before training, `train_fireworks.sh` calls `prepare_data.py`, which:

1. verifies the task archive SHA-256;
2. extracts it into a content-addressed user cache;
3. registers all 48 tasks as `terminal-bench-fixed-48/train`;
4. downloads and registers the pinned 89-task
   `terminal-bench/terminal-bench-2-1@6` evaluation suite; and
5. rejects any train/evaluation task overlap.

## Training protocol

- Harness: OpenCode by default
- Reward: each task's binary Harbor verifier
- Batch: 16 prompt groups × 8 rollouts = 128 trajectories per optimizer step
- Schedule: 48 fixed tasks × 4 epochs = 12 optimizer steps
- Evaluation: all 89 Terminal-Bench 2.1 tasks after steps 3, 6, 9, and 12
- Algorithm: synchronous on-policy GRPO, R3 router replay, PPO clipping,
  compact error filtering, and no uniform-group rejection
- Resources: one trainer and one rollout replica by default; configurable

## Run as a customer

Python 3.12, Docker, a standard Fireworks API key, and a W&B API key are
required. Install the pinned public dependencies:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install \
  --overrides dependency-overrides.txt \
  -e "../../.." \
  -r requirements.txt
```

Provide a model and compatible training shape that your account can use:

```bash
export FIREWORKS_API_KEY="<your Fireworks API key>"
export WANDB_API_KEY="<your W&B API key>"

export FIREWORKS_MODEL_ID="accounts/<your-account>/models/<your-model>"
export FIREWORKS_TRAINING_SHAPE_ID="accounts/<your-account>/trainingShapes/<your-shape>"
export TOKENIZER_MODEL="<Hugging-Face tokenizer>"

export LEARNING_RATE="<learning rate for your model and tuning mode>"
export MAX_PROMPT_LENGTH="<prompt-token limit>"
export MAX_RESPONSE_LENGTH="<response-token limit>"

./train_fireworks.sh
```

The prompt and response limits must fit within the selected training shape's
context length. The trainer's total maximum length is otherwise resolved from
the shape.

Optional public settings include:

```bash
export LORA_RANK=0
export TRAINER_REPLICAS=1
export ROLLOUT_REPLICAS=1
export TB_HARNESS=opencode
export TOP_P=1.0
export TEMPERATURE=1.0
export WANDB_PROJECT=terminal-rl

# Leave unset to let the control plane select placement.
export FIREWORKS_REGION="<optional supported region>"
```

The launcher authenticates with `FIREWORKS_API_KEY` and provisions paid
resources under the account associated with that key.
