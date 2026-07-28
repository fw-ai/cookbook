# GLM-5.2 OpenCode RL on Terminal-Bench

Reproduces the full-parameter, synchronous GRPO experiment in
[W&B run `sovf8iiv`](https://wandb.ai/myh97/terminal-rl/runs/sovf8iiv).
The recipe comes from
[rllm-org/rllm#779](https://github.com/rllm-org/rllm/pull/779) at
`5f2fed491589790457c6bb292734415ea108463a`.

`train.py` and `train_fireworks_glm5p2.sh` are byte-for-byte copies of that
revision. Keep them unchanged when reproducing the run. The 48 approved
training tasks, their fixed order, solutions, and verifiers are bundled under
`data/`; `prepare_data.py` only converts them into rLLM's local registry
format. Evaluation pulls the immutable 89-task
`terminal-bench/terminal-bench-2-1@6` package.

## Experiment

- Model: `accounts/fireworks/models/glm-5p2-fp8`, full-parameter
- Shape: `accounts/fireworks/trainingShapes/glm-5p2-200k`
- Harness: OpenCode; reward: each task's binary Harbor verifier
- Batch: 16 prompt groups × 8 rollouts = 128 trajectories per optimizer step
- Schedule: 48 fixed tasks × 4 epochs = 12 optimizer steps
- Evaluation: all 89 Terminal-Bench 2.1 tasks after steps 3, 6, 9, and 12
- Resources: 2 trainer replicas + 6 rollout replicas (10 nodes)
- Numerics: synchronous on-policy sampling, R3 router replay, PPO clipping,
  compact error filtering, and no uniform-group rejection

## Run

Python 3.12, Docker, a Fireworks API key, and a W&B API key are required.
From this directory:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install \
  --overrides dependency-overrides.txt \
  -e "../../..[terminal-bench]"

python prepare_data.py

export FIREWORKS_API_KEY="<your-key>"
export WANDB_API_KEY="<your-key>"
export TB_TRAINER_REGION="AP_MALAYSIA_2"
./train_fireworks_glm5p2.sh full opencode curriculum
```

The region is explicit for this reproduction and is intentionally not a
launcher default. The command provisions paid trainer and rollout resources;
do not run it as a unit test.

To use another approved Harbor-format curriculum, provide a selection file
with the same `{"selected": [{"task_id": ...}]}` schema and matching task
directories, then register and launch it under the same dataset name:

```bash
python prepare_data.py \
  --tasks-dir /path/to/tasks \
  --selection /path/to/selection.json \
  --train-dataset my-terminal-curriculum
TB_CURRICULUM_DATASET=my-terminal-curriculum \
  ./train_fireworks_glm5p2.sh full opencode curriculum
```
