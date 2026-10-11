# VisualToolBench RL

Trains a vision-language policy on VisualToolBench-style tasks. The rollout is
multimodal and multi-turn: the policy can crop, zoom, rotate, or adjust an
image, inspect the result, and continue. Each assistant turn is a trainable
segment; prompt, user, and tool tokens stay loss-masked. A rubric LLM judge
scores the final answer, and that one trajectory reward is broadcast to every
segment before GRPO computes group-relative advantages.

## Layout

| File | What it is |
| --- | --- |
| `rollout.py` | `make_rollout_fn(setup, *, task)`: the tool loop, sampler/trainer prompt-parity checks, Router Replay validation. |
| `image_tools.py` | The four PIL image tools and their schemas. |
| `reward.py` | `RubricJudge`: rubric grading, official score, critical-rubric pass. |
| `task.py` | `TaskConfig` (renderer, system prompt, rollout / judge / eval-sampling settings), data loading, held-out evaluation metrics. |
| `train_visual_toolbench.py` | Dedicated trainer + deployment: task + `recipes/async_rl_loop.py`. |
| `train_visual_toolbench_serverless.py` | Pooled serverless training: task + `recipes/experiment/async_rl_loop_serverless.py`. |

Both entry points pass the recipe the same pieces:
`functools.partial(make_rollout_fn, task=TaskConfig(...))`, a filter that skips
groups whose rollouts all score the same, and a held-out `evaluation_fn`. The
recipe owns scheduling, GRPO, weight sync or snapshot publication, metrics,
W&B and checkpoints.

## Input data

Pass separate JSONL files for training and evaluation (`--dataset`,
`--eval-dataset`; their row ids must not overlap). Each row must contain:

```json
{
  "id": "task-001",
  "prompt": "What value is shown in the highlighted region?",
  "golden_answer": "42",
  "images": ["data:image/jpeg;base64,..."],
  "rubrics": [
    {
      "id": "answer",
      "description": "The answer states 42.",
      "weight": 1,
      "critical": true
    }
  ],
  "tool_alignment": {"eligible": true}
}
```

For multi-turn tasks, add a non-empty `turns` list whose entries each carry
`prompt`, `golden_answer`, `images`, and `rubrics`. Optional `category` and
`eval_focus` fields break down the held-out official score. Set
`tool_alignment.eligible=true` only after verifying the task can be solved with
the four image tools; `--no-require-tool-aligned-data` is for intentional
experiments only.

## Sampling and evaluation

- Training rollouts sample at temperature `1`, top-p `1`, top-k `0`, so the
  sampler and trainer score the same distribution.
- Held-out evaluation runs at the start, every `--evaluation-interval`
  optimizer steps, and at the end, with `--eval-group-size` samples per row and
  the benchmark's sampling (`--eval-temperature 1`, `--eval-top-p 0.95`,
  `--eval-top-k 20`, `--eval-max-completion-tokens 26666`).
- Eval metrics divide by the expected episode count, so a dropped episode
  scores zero: `eval/reward`, `eval/official_score`, `eval/critical_fraction`,
  `eval/judge_pass`, `eval/return_ratio`, and per-`category` / `eval_focus`
  official scores.

The judge defaults to Kimi K3 (`--judge-model`) with a 65,536-token budget and
uses the same `FIREWORKS_API_KEY` as the run.

## Serverless

Provisions nothing; the 214-row training split and disjoint 50-row evaluation
split are passed explicitly:

```bash
export FIREWORKS_API_KEY=fw_...
python -m training.examples.rl.visual_toolbench.train_visual_toolbench_serverless \
  --dataset /abs/train.jsonl --eval-dataset /abs/eval.jsonl \
  --base-model accounts/fireworks/models/qwen3p6-27b --tokenizer-model Qwen/Qwen3.6-27B \
  --renderer-name qwen3_6_disable_thinking_interleaved
```

Settings used for the reference runs (the per-model launchers are archived in
[`../../archive/launchers/`](../../archive/launchers/)):

| Policy | Renderer | Prompt groups × rollouts | Learning rate |
| --- | --- | --- | --- |
| Qwen3.6-27B | `qwen3_6_disable_thinking_interleaved` | 8 × 8 | `3e-5` |
| Kimi K3 | `kimi_k3_disable_thinking` | 16 × 8 | `1e-4` |
| Muse Glimmer 30B | `muse_glimmer` | 16 × 8 | `1e-4` |

Each ran two epochs with token-mean loss normalization, evaluation every five
steps, and resumable training-state saves every two steps.

## Dedicated trainer + deployment

```bash
export FIREWORKS_API_KEY=fw_...
python -m training.examples.rl.visual_toolbench.train_visual_toolbench \
  --dataset /abs/train.jsonl --eval-dataset /abs/eval.jsonl \
  --base-model accounts/fireworks/models/qwen3p6-27b \
  --tokenizer-model Qwen/Qwen3.6-27B \
  --training-shape-id <training-shape> --log-path ./vtb_dedicated
```

`--lora-rank 0` (the default) trains all parameters. Reuse a running trainer
and hot-load deployment with `--job-id <trainer-job-id> --deployment-id
<deployment-id> --no-cleanup-on-exit`.

`--help` on either entry point lists every flag; install the cookbook with the
`eval` extra (`uv run --project training --extra eval ...`) for Pillow.
