# Snippets

Standalone utility scripts that complement the main cookbook recipes.

| Script | Description |
|--------|-------------|
| `promote_checkpoint.py` | Promote a sampler checkpoint from `checkpoints.jsonl` to a deployable Fireworks model. |
| `reconnect_and_adjust_lr.py` | Reconnect to an existing trainer job and adjust the learning rate mid-run. |
| `list_checkpoints.py` | List checkpoints the server knows about for an RLOR trainer job. |
| `verify_logprobs.py` | Verify train-inference logprob alignment for a tokenizer/model pair. |
| `check_glm5_sft_dataset.py` | Diagnose GLM-5.1 SFT datasets that produce thinking-loop behaviour at inference (`<think>`/`</think>` balance, EOS coverage, reasoning-content distribution). |
