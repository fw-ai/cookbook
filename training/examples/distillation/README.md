# Qwen3 32B → 8B distillation pipeline

End-to-end pipeline: GRPO-fine-tune a Qwen3-32B teacher on DeepMath, then
distill that teacher into Qwen3-8B via on-policy reverse-KL. Evaluations
on the held-out slice bracket the distillation gain.

## Pre-flight

Run the recipe smoke first (cheap, ~10–20 min) and confirm it's green:

```sh
export FIREWORKS_API_KEY=<your key>
pytest training/tests/smoke_test/test_distillation_smoke.py -v -m e2e
```

Do **not** start the real pipeline until the smoke passes — the distillation
recipe is new and the smoke covers the plumbing the real run depends on.

## Stage 0: baseline eval (Qwen3-8B, no fine-tuning)

```sh
cd training/examples/distillation
python eval_deepmath.py \
    --model accounts/fireworks/models/qwen3-8b \
    --dataset ../rl/deepmath/dataset.jsonl \
    --skip-rows 100 --max-rows 100 \
    --output ./eval_base_qwen3_8b.jsonl
```

`--skip-rows 100 --max-rows 100` carves an eval slice disjoint from the
first 100 rows that the training stages use. Final accuracy is logged and
also stored as the first line of the output JSONL.

## Stage 1: fine-tune the 32B teacher

```sh
export FIREWORKS_API_KEY=<your key>
./training/examples/rl/deepmath/run_qwen3_32b_distill_teacher.sh
```

When it completes, grep the log for `policy_job_id`:

```sh
# example log line:
# Training complete. Final metrics: {'steps': 100, 'policy_job_id': 'rlor-...', ...}
```

Copy that ID for stage 2. The job is preserved (`--skip-cleanup`) so the
distillation step can attach to it.

## Stage 2: distill into 8B

```sh
export TEACHER_JOB_ID=<policy_job_id from stage 1>
./training/examples/distillation/run_qwen3_32b_to_8b.sh
```

Note the `output-model-id` echoed in the log — that's the model resource you
evaluate next.

## Stage 3: post-distillation eval

```sh
python eval_deepmath.py \
    --model accounts/fireworks/models/<output-model-id from stage 2> \
    --dataset ../rl/deepmath/dataset.jsonl \
    --skip-rows 100 --max-rows 100 \
    --output ./eval_distilled_qwen3_8b.jsonl
```

Compare the `accuracy` field in the two `eval_*.jsonl` headers to read off
the lift.

## Cleanup

The training jobs from stages 1 and 2 remain alive (so re-runs are cheap).
When you're done with the experiment, delete them through the SDK
(`TrainerJobManager.delete(job_id)`) or the console.

## Known constraints

- Teacher and student must share a tokenizer. The recipe asserts this; the
  Qwen3 family is consistent.
- The `teacher_base_model` config arg must match the `base_model` the stage-1
  RL job was created with (`qwen3-32b`). The recipe attaches to the existing
  trainer by `teacher_job_id`; the `base_model` field documents what's loaded
  there.
- `max_rows=100` on every stage is the cookbook default. Scale by editing
  the shell scripts; budget accordingly.
