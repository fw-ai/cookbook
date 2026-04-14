# Tools (standalone operational scripts)

Utilities that complement the main recipes. **Not training loops** -- each does one specific operation.

All live in `training/tools/`. For deeper coverage of the two most common operations, see the dedicated references:

- **Promote a checkpoint** → [`promote.md`](promote.md)
- **Re-attach a deployment to a new trainer** → [`reattach.md`](reattach.md)

---

## promote_checkpoint.py

Reads `checkpoints.jsonl` (produced by cookbook recipes), finds a sampler checkpoint ID, and promotes it to a deployable Fireworks model via the promotion API.

```bash
python -m training.tools.promote_checkpoint \
    --log-path ./my_training_logs \
    --step 100 \
    --output-model-id "my-finetuned-qwen3-v1"
```

See [`promote.md`](promote.md) for the full workflow and flow-detection decision tree.

---

## reconnect_and_adjust_lr.py

Reconnects to an already-running trainer job and adjusts the learning rate mid-run.

```bash
python -m training.tools.reconnect_and_adjust_lr \
    --job-id my-trainer-v1 \
    --lr 5e-6 \
    --steps 100
```

Use this as a reference template for other mid-run adjustments (e.g. changing batch size, swapping loss weights).

---

## verify_logprobs.py

Creates a trainer (optionally a reference too), samples completions from a deployment, and verifies per-token logprob alignment between training and inference.

### When to use

- Validating a new deployment shape (numerics check)
- Debugging train-inference gaps (why does eval score differ from training reward?)
- Quick smoke test that train ↔ inference alignment is acceptable

### What it measures

For each sampled completion, compares per-token logprobs from:

- **Training trainer** (`tc.forward()` with `echo=True`)
- **Inference deployment** (via `DeploymentSampler` with `echo=True, logprobs=True`)

Reports:

- Mean absolute logprob diff per token
- KL divergence (completion-only, P+C-1 tokens)
- Max diff (for outlier detection)

### Usage

```bash
python -m training.tools.verify_logprobs \
    --base-model accounts/fireworks/models/qwen3-8b \
    --deployment-id my-deploy \
    --trainer-id my-trainer-v1 \
    --tokenizer-path Qwen/Qwen3-8B \
    --num-prompts 20 \
    --max-tokens 512
```

### Interpreting results

| Mean abs diff | Interpretation |
|---------------|----------------|
| < 0.01 | Excellent alignment |
| 0.01 - 0.05 | Acceptable (typical bf16 + MoE routing noise) |
| 0.05 - 0.2 | Investigate (check router replay, dtype, speculative decoding flags) |
| > 0.2 | Broken (wrong tokenizer, mismatched base model, wrong dtype) |

---

## Writing a new tool

Tools are standalone scripts, not cookbook recipes. Convention:

1. Put it in `training/tools/`
2. Include a module docstring with usage
3. `if __name__ == "__main__":` block with `argparse`
4. Use `FiretitanServiceClient` / `TrainerJobManager` / `DeploymentManager` directly
5. No need to extend `Config` dataclasses from the recipes

List new tools in `training/tools/README.md` for discoverability.
