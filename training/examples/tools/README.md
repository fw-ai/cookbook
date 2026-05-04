# Tools

Standalone utility scripts that complement the main cookbook recipes.

| Script | Description |
|--------|-------------|
| `promote_checkpoint.py` | Promote a sampler checkpoint from the control plane checkpoint list to a deployable Fireworks model. |
| `list_checkpoints.py` | List checkpoint rows known to the control plane for a trainer job. |
| `reconnect_and_adjust_lr.py` | Reconnect to an existing trainer job and adjust the learning rate mid-run. |
| `verify_logprobs.py` | Compare inference-time and training-time logprobs for a checkpoint. |
