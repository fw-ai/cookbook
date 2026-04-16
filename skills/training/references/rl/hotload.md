# RL: weight sync (hotload) during training

RL is the main consumer of hotload: the recipe saves sampler checkpoints mid-training and pushes them to the serving deployment so new rollouts come from the updated policy. SFT / DPO / ORPO don't typically hotload — they save once at the end and call it a day.

All hotload behaviour in `rl_loop.py` is controlled by `cfg.weight_sync: WeightSyncConfig`.

## The knobs

| Field | Default | Meaning |
|---|---|---|
| `weight_sync_interval` | `1` | Sync every N optimizer steps. `1` = after every step (on-policy). `0` = no weight sync at all (rollouts always come from the initial policy — you almost never want this for RL). |
| `first_checkpoint_type` | `"base"` | First sampler save is a full snapshot; subsequent saves can be deltas. Do not change. |
| `weight_sync_timeout` | `600` | Per-hotload timeout in seconds. Bump if you see `Hotload did not complete within 600s` on large models. |
| `weight_sync_before_training` | `False` | Push an initial base snapshot before step 0. Useful when the deployment starts from a different snapshot than the trainer's base. |
| `dcp_save_interval` | `0` | DCP (optimizer + weights) save cadence for **resume**. Orthogonal to sampler hotload. `0` = off; no intermediate resume points. |
| `dcp_timeout` | `2700` | 45 min default for `save_state` / `load_state_with_optimizer`. |

## On-policy vs off-policy (weight-sync timing)

- `weight_sync_interval = 1` + strict 1:1 per step (the recipe default) → **on-policy**. Rollouts for step K+1 are sampled from the policy that step K produced.
- `weight_sync_interval > 1` → **off-policy** between syncs. Rollouts continue to come from an older snapshot until the next sync. CISPO / DRO / IS tolerate this better than vanilla GRPO.

## Base vs delta chain

For full-parameter training, the first sampler save is `base` (full weights, ~16 GB for 8B). Subsequent saves are `delta` (XOR diff, ~10× smaller). `WeightSyncer` manages this automatically — users don't pick per-step.

- LoRA always saves the full adapter regardless of `checkpoint_type` — every LoRA sampler checkpoint is promotable.
- Full-param `delta` saves are **not** promotable. Only `base` saves are. See the concept doc at <https://docs.fireworks.ai/fine-tuning/training-api/cookbook/checkpoints#checkpoint-kinds>.

## `dcp_save_interval` for resume

Separate from hotload: DCP saves persist the full train state (weights + optimizer) so you can resume training if the job dies. `0` (default) = off — if your run crashes mid-training, there is no intermediate resume point. Set this if your run is long enough that a crash is painful.

## When hotload fails

Symptom: `Hotload did not complete within <N>s` or `Hotload failed for snapshot <id>`.

1. **First, check the SDK version matches the cookbook's pin** (see `../../SKILL.md#first-debug-step--always`).
2. If the SDK is current, the most common cause is a trainer-first vs deployment-first flow-mix. Run through the self-check at [`../trainer-first-vs-deployment-first.md#self-check-when-something-looks-wrong`](../trainer-first-vs-deployment-first.md#self-check-when-something-looks-wrong).
3. If neither applies, reach out to Fireworks support. The recovery path (re-pointing a deployment's bucket, settling a stuck rolling restart) requires server-side access.

## Two deployments, one trainer

On-policy sampler + held-out eval deployment is a common pattern. Both copy the trainer's `hotLoadBucketUrl` at creation, and both can be hotloaded from the same `WeightSyncer`. See [`../trainer-first-vs-deployment-first.md#two-deployments-per-trainer-sampler--eval`](../trainer-first-vs-deployment-first.md#two-deployments-per-trainer-sampler--eval).

## See also

- Concept / API reference for the manager + its lifecycle: [`WeightSyncer`](https://docs.fireworks.ai/fine-tuning/training-api/reference/weight-syncer).
- `save_weights_for_sampler_ext`, `save_state`, and the raw checkpoint surface: [Saving and Loading](https://docs.fireworks.ai/fine-tuning/training-api/saving-and-loading).
