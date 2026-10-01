# Qwen3-4B SAO actor and value-head critic

This example is experimental. Projection-head SAO is not validated for every
model. It uses the public `accounts/fireworks/models/qwen3-4b` model for both
actor and critic. The actor samples from a managed inference deployment; the
separate critic trains a one-dimensional projection head on the GPU. The four
built-in arithmetic prompts are a small end-to-end check, not a math benchmark.

## Prerequisites

- A Fireworks API key with dedicated Training API access.
- The example uses the Qwen3-4B LoRA training shape
  `accounts/fireworks/trainingShapes/qwen3-4b-minimum-lora` for both the actor
  and the critic. Override it with `--training-shape` or
  `FIREWORKS_TRAINING_SHAPE`.

Install and run the standalone cookbook:

```bash
git clone https://github.com/fw-ai/cookbook.git
cd cookbook/training
uv venv --python 3.12
uv pip install --python .venv/bin/python -e .
export FIREWORKS_API_KEY='your-training-api-key'
.venv/bin/python -m training.examples.rl.SAO.qwen3_4b_sao \
  --output-dir ./qwen3_4b_sao_run
```

The run prints the actor and critic trainer IDs, sampler deployment ID, and
optimizer-step counts. Inspect `qwen3_4b_sao_run/metrics.jsonl` for
`train/critic-value_loss` and actor metrics; `summary.json` records the final
IDs. The recipe saves resumable actor and critic checkpoints before closing
the managed resources. A four-prompt smoke run takes two rollout batches, four
critic updates, and two actor updates.

Use `--preserve-jobs` when you need to inspect the live trainers or resume from
their checkpoints after the run. Otherwise, the recipe cleans up the trainer
and sampler resources on exit.

The smoke run uses SAO's DIS actor objective, adaptive GAE, two critic
updates per rollout batch, and MLP-only critic LoRA. It disables offline
value pretraining and critic-only warmup so both trainers and the sampler are
exercised immediately. The tiny dataset and 256-token completions do not
measure model improvement.

## Use your own math rows

Create a JSONL file with one `messages` array and `ground_truth` per row.
The default arithmetic task expects an integer inside `<answer>...</answer>`:

```json
{"messages":[{"role":"user","content":"What is 13 + 8? End with <answer>number</answer>."}],"ground_truth":"<answer>21</answer>"}
```

Run the same entrypoint with a longer generation limit and your desired batch:
Use a training shape whose validated context limit is at least the requested
`--max-seq-len`; a short-context smoke shape cannot run the 32K example.

```bash
.venv/bin/python -m training.examples.rl.SAO.qwen3_4b_sao \
  --dataset ./math_train.jsonl --max-rows 128 \
  --prompt-groups-per-batch 16 --max-completion-tokens 16384 \
  --max-seq-len 32768 --output-dir ./qwen3_4b_math_run
```

The terminal reward is exact matching of the first integer inside
`<answer>...</answer>`. For another task, replace `reward_fn` in
`training.recipes.experiment.ppo_value_head_loop` or pass a reward callback to `main()`.
Add held-out evaluation and `ValuePretrainData` before using this as a full SAO
benchmark.

## Prepare DeepMath-103K data

Run the converter on the machine that runs the cookbook. It downloads
`zwhe99/DeepMath-103K` from Hugging Face and writes one JSONL row per training
example. The base cookbook install above includes `datasets`; install
`math-verify` as well for the symbolic answer checker used by `--task deepmath`:

```bash
uv pip install --python .venv/bin/python math-verify
.venv/bin/python -m training.examples.rl.deepmath.prepare_data \
  --output ./deepmath_103k.jsonl
```

Each row has a `messages` array with a system instruction to put the final
answer in `\\boxed{}`, a user question, and a `ground_truth` answer copied
from the source dataset. Keep the JSONL file on the client machine; the
training script reads and tokenizes it locally. Start the run with the
matching symbolic reward:

```bash
.venv/bin/python -m training.examples.rl.SAO.qwen3_4b_sao \
  --task deepmath \
  --dataset ./deepmath_103k.jsonl \
  --max-rows 512 --prompt-groups-per-batch 16 \
  --max-completion-tokens 16384 --max-seq-len 32768 \
  --output-dir ./qwen3_4b_deepmath_run
```

`--max-rows 512` uses the first 512 prepared rows for this example; set it to
the printed row count to use the full dataset. This longer configuration is a starting
point for task validation, not a reproduction of the paper's full training or
evaluation protocol.

Confirm that the training shape supports the requested 32K context before
starting this run.

### Value pretraining

This entrypoint keeps value pretraining disabled so the example reaches an
actor update quickly. Full runs can call `main()` with disjoint training and
validation trajectories in `ValuePretrainData`, then tune
`value_pretrain_steps`. Setting the step count without supplying those
trajectories raises a validation error.
