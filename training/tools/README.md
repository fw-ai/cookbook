# Tools

Standalone operational tools that complement the training recipes. Each one does a single specific task -- not a training loop.

| Tool | Description |
|------|-------------|
| `promote_checkpoint.py` | Promote a sampler checkpoint from `checkpoints.jsonl` to a deployable Fireworks model. |
| `reconnect_and_adjust_lr.py` | Reconnect to an existing trainer job and adjust the learning rate mid-run. |
| `verify_logprobs.py` | Verify per-token logprob alignment between a training trainer and an inference deployment. |
