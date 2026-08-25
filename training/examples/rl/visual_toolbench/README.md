# VisualToolBench serverless RL

This example trains a vision-language policy on VisualToolBench-style tasks
through a pooled Fireworks serverless trainer. The default policy is
`accounts/fireworks/models/qwen3p6-27b` with the
`qwen3_6_disable_thinking_interleaved` renderer.

The rollout is genuinely multimodal and multi-turn: the policy can crop, zoom,
rotate, or adjust an image, inspect the transformed result, and continue. Each
assistant turn is a trainable segment; prompt, user, and tool tokens remain
loss-masked. A rubric judge assigns one dense trajectory reward, which is
broadcast to all segments before GRPO advantage calculation.

## Input data

Pass separate JSONL files for training and evaluation. Each row must contain:

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
`prompt`, `golden_answer`, `images`, and `rubrics`. Top-level fields remain the
single-turn/backward-compatible shape. Set `tool_alignment.eligible=true` only
after verifying the task can be solved with the four supported image tools.
Use `--no-require-tool-aligned-data` only for intentional experiments.

## Launch scripts

All launchers expect the same 214-row training split and disjoint 50-row
evaluation split:

```bash
cd cookbook
export FIREWORKS_API_KEY=fw_...
export VTB_DATASET=/absolute/path/to/train.jsonl
export VTB_EVAL_DATASET=/absolute/path/to/eval.jsonl
```

Run Qwen3.6-27B:

```bash
bash training/examples/serverless_rl/runs/run_qwen3p6_serverless_2epoch.sh
```

Run Kimi K3 (override `KIMI_K3_TOKENIZER` with a pinned local snapshot when
needed):

```bash
bash training/examples/serverless_rl/runs/run_kimi_k3_serverless_2epoch.sh
```

Run Muse Glimmer 30B (override `MUSE_GLIMMER_TOKENIZER` with a pinned local
snapshot when needed):

```bash
bash training/examples/serverless_rl/runs/run_muse_glimmer_serverless_2epoch.sh
```

No launcher provisions a trainer, deployment, shape, or region; each connects
to Fireworks pooled serverless training.

All launchers preserve sampling/training distribution alignment:

- training sampling: temperature `1`, top-p `1`, top-k `0`
- evaluation: temperature `1`, top-p `0.95`, top-k `20`
- training/eval generation caps: `32768` / `26666` tokens per assistant turn
- token-count loss normalization, two epochs, upfront and every-5-step eval
- resumable model and optimizer DCP checkpoints every two optimizer steps

The model-specific settings are:

| Policy | Renderer | Prompt groups × rollouts | Learning rate |
| --- | --- | --- | --- |
| Qwen3.6-27B | `qwen3_6_disable_thinking_interleaved` | 8 × 8 | `3e-5` |
| Kimi K3 | `kimi_k3_disable_thinking` | 16 × 8 | `1e-4` |
| Muse Glimmer 30B | `muse_glimmer` | 16 × 8 | `1e-4` |

The run directory contains the exact command, input SHA-256 hashes, logs,
metrics, eval completions, and checkpoint metadata. Override it with
`VTB_RUN_DIR`; otherwise the script creates a timestamped directory in `/tmp`.

The judge defaults to Kimi K3 with a 65,536-token completion budget and uses
the same `FIREWORKS_API_KEY` as the training run.

## Direct invocation

```bash
uv run --project training --extra eval python \
  -m training.examples.serverless_rl.visual_toolbench_rl --help
```
