# Tools

Standalone utility scripts that complement the main cookbook recipes.

| Script | Description |
|--------|-------------|
| `promote_checkpoint.py` | Promote a sampler checkpoint from the control plane checkpoint list to a deployable Fireworks model. |
| `merge_lora_and_promote.py` | Merge a LoRA/PEFT adapter into its base (`checkpoint_type="merged_base"`) and promote the result as a full `HF_BASE_MODEL`. Provisions a short-lived LoRA trainer, loads the adapter explicitly, saves the merged base, and promotes. |
| `list_checkpoints.py` | List checkpoint rows known to the control plane for a trainer job. |
| `reconnect_and_adjust_lr.py` | Reconnect to an existing trainer job and adjust the learning rate mid-run. |
| `verify_logprobs.py` | Compare inference-time and training-time logprobs for a checkpoint. |
