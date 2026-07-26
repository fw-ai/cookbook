# 6. The Fireworks stack map

Where every concept from parts 1–5 lives in real code, plus the vocabulary you
need to read the rest of the repo. Operational guidance lives in
[`skills/fireworks-training/SKILL.md`](../skills/fireworks-training/SKILL.md) and
its `references/`; this page is a translation table.

## 6.1 Layers

```
your Python                 recipe (a training loop you fork and own)
                            training/recipes/*.py
        │
        ▼
cookbook utilities          config, client wrapper, data rendering, losses,
                            checkpoints, async coordinator
                            training/utils/**
        │
        ▼
Fireworks Python SDK        FiretitanServiceClient, FiretitanTrainingClient,
(fireworks.training.sdk)    DeploymentSampler, AdaptiveConcurrencyController
        │
        ▼
Training API (Tinker-       forward / forward_backward / optim_step /
compatible wire protocol)   save_state / save_weights_for_sampler
        │
        ▼
Fireworks backend           RLOR trainer GPUs · inference deployment ·
                            hot-load storage
```

The design decision worth noticing: **the loop stays on your machine.** You are
not submitting a YAML job and waiting; you are making RPCs from a Python process
you control, which is why arbitrary losses, custom reward functions, curricula,
and multi-model schemes are ordinary code rather than platform features.

## 6.2 Concept → code

| Concept (part) | Where it lives |
|---|---|
| Cross-entropy SFT loss (§1.1, §3.6) | `loss_fn="cross_entropy"` in `training/recipes/sft_loop.py` |
| Chat templates and tokenization parity (§1.10) | `training/renderer/`, verifiers in `training/renderer/verifier/` |
| Loss masking / `weights` (§3.6) | `training/utils/supervised.py`, `train_on_what` config field |
| MoE routing (§1.7) and R3 (§4.9) | `training/utils/rl/router_replay.py`, `router_replay=True` |
| Speculative decoding (§2.4) | `DeployConfig.disable_speculative_decoding` |
| Prefill queue / adaptive concurrency (§2.3) | `ConcurrencyConfig.prefill_queue_target`, SDK `AdaptiveConcurrencyController` |
| KV reuse across turns (§2.2) | Session-affinity routing on dedicated deployments (`models-shapes-and-cost.md`) |
| Forward / backward / optimizer (§3.1) | `training/utils/client.py` (`ReconnectableClient`) |
| Gradient-accumulation normalization (§3.3) | `GradAccNormalization.NUM_SEQUENCES` / `NUM_LOSS_TOKENS` on `optim_step` |
| LR schedule (§3.2) | `normalize_lr_scheduler_spec`, `compute_lr`, `LRSchedulerSpec` |
| LoRA (§3.7) | `lora_rank` on every recipe config; `load_adapter` for warm start |
| DPO (§3.8) | `training/recipes/dpo_loop.py` |
| ORPO (§3.9) | `training/recipes/orpo_loop.py` |
| Distillation, fwd vs reverse KL (§3.10) | `training/recipes/distillation_loop.py`, `DistillMode` |
| GRPO / PPO clip / k3 KL (§4.4–4.6) | `training/utils/rl/grpo.py` |
| DAPO / GSPO / CISPO / REINFORCE / DRO (§4.6) | `training/utils/rl/{dapo,gspo,cispo,reinforce,dro}.py` |
| TIS, IcePop (§4.8) | `training/utils/rl/tis.py` |
| Train–inference gap metrics (§4.8) | `training/utils/rl/observability.py` |
| Sync RL loop (§4.10) | `training/recipes/rl_loop.py` |
| Async RL, admission gate (§4.11) | `training/recipes/async_rl_loop.py`, `training/utils/rl/async_rl/` |
| Weight sync / hot-load (§4.12) | `save_weights_for_sampler` + `service.hotload_sampler_snapshot`, `WeightSyncScope` |
| HSDP replicas (§5.6) | `TrainerConfig.replica_count` |
| Training shapes (§5.8) | `TrainerConfig.training_shape_id` |

## 6.3 The Tinker protocol, one more time

| Call | Compute on server | Gradient? | Used by |
|---|---|---|---|
| `forward(data, loss_fn)` | forward only | no | reference logprobs, old-policy anchor |
| `forward_backward(data, "cross_entropy")` | built-in loss + backward | yes | SFT, SDFT distillation |
| `forward_backward(data, "importance_sampling")` | built-in IS loss + backward | yes | serverless RL, sampled reverse-KL distillation |
| `forward_backward_custom(data, loss_fn)` | forward → **your loss in Python** → backward | yes | DPO, ORPO, GRPO and every RL variant |
| `forward_backward_contrastive(...)` | InfoNCE + backward | yes | embedding fine-tuning |
| `optim_step(AdamParams, ...)` | AdamW update | mutates | all |
| `save_state` / `load_state_with_optimizer` | DCP checkpoint | — | resume |
| `save_weights_for_sampler` | inference-format snapshot | — | serving, hot-load |
| `load_adapter(path)` | load LoRA weights | — | warm start |

Calls return futures; the cookbook's `ReconnectableClient` resolves them with
explicit timeouts and reconnect handling. Gradient accumulation is $k$
`forward_backward` calls followed by one `optim_step` — client-side control flow,
not a server setting.

## 6.4 Serverless vs dedicated

|  | Serverless | Dedicated |
|---|---|---|
| Provisioning | None — pooled trainer at `/training/v1/serverless` | Explicit trainer job (+ deployment for RL) |
| Billing | Per training token | Per GPU-hour, metered by runtime |
| LoRA | Required (`lora_rank > 0`) | LoRA or full-parameter |
| Sampling for RL | `service.create_sampling_client(model_path=snapshot)` — no separate deployment | Rollout deployment + `hotload_sampler_snapshot` |
| Good for | Fast LoRA SFT/RL iteration | DPO, sustained RL, full-parameter, explicit checkpoints |

Entry points: `training/utils/serverless.py` (`create_lora_training_client`) vs
`training/utils/service.py` (`build_service_client` →
`FiretitanServiceClient.from_firetitan_config`).

The dominant cost mistake is leaving a dedicated deployment up. GPU-hour billing
does not care whether you are sending traffic.

## 6.5 Two checkpoint axes, do not confuse them

| | `save_state` (DCP) | `save_weights_for_sampler` |
|---|---|---|
| Contains | weights **+ optimizer moments** | inference-format weights only |
| Purpose | resume training exactly | serve, hot-load, promote to a model ID |
| Size | large | LoRA: MB. Full-param: `"base"` then `"delta"` (~10× smaller) |
| Cannot | be served directly | be resumed from |

Managed by `TrainingCheckpoints` (`training/utils/checkpoints.py`). The local
dataloader cursor (`{log_path}/dataloader.json`) maps a checkpoint name to rows
consumed, so a resume does not replay data it already trained on.

## 6.6 A dedicated async RL step, end to end

Every concept in this primer, in order, in one loop iteration:

1. The **producer** admits a dataset row if the staleness and concurrency budgets
   allow all $G$ of its samples (§4.11).
2. `DeploymentSampler` sends the prompt to the **inference deployment**, which
   prefills it (reusing cached prefix blocks where possible), decodes with
   continuous batching and possibly speculation, and returns tokens plus
   per-token logprobs — and routing matrices if the model is MoE (§2, §4.9).
3. Your `reward_fn` scores each completion. Group $G$ samples per prompt,
   z-score the rewards into advantages (§4.3).
4. The **trainer** runs `forward` passes for the reference model (if
   `kl_beta > 0`) and the old policy (if `anchor_logp="old_policy"`) (§4.5,
   §4.8).
5. `forward_backward_custom` runs GRPO in your Python: PPO clipped ratio × TIS
   weight, plus k3 KL, masked to completion tokens (§4.6).
6. Optionally repeated over $K$ chunks, accumulating gradients (§3.3).
7. One `optim_step` — AdamW with the current scheduled LR and gradient clipping
   (§3.2).
8. `save_weights_for_sampler` + `hotload_sampler_snapshot` push the new weights
   to the deployment in place (§4.12).
9. `coordinator.publish(batch)` advances the policy version, which mints fresh
   admission credit and wakes the producer (§4.11).

Exactly one optimizer mutation, one hot-load, and one version publication per
batch. That invariant is what keeps trainer and sampler versions from diverging.

## 6.7 Metrics worth watching

| Metric | Healthy | What it means when it is not |
|---|---|---|
| `mean_loss` / reward mean | trending the right way | flat → check reward variance per group |
| `ppo_clip_frac` | 0.02–0.15 | high: LR too large or data too stale; ~0: steps too timid |
| `ppo_ratio_mean` | ≈ 1.0 | drifting: policy moving fast relative to the anchor |
| `tis/clip_frac`, `tis/weight_mean` | low, ≈ 1.0 | rising: trainer and inference engines diverging (§4.8) |
| train–inference KL (`observability.py`) | small and stable | growing: quantization/routing/speculation mismatch |
| `mask_ratio` | as designed | unexpected: loss masking or renderer bug |
| `async/version_offset_*` | ≤ your $O$ | at the cap constantly: staleness-bound, not concurrency-bound |
| grad norm | stable | spikes: clipping is saving you; investigate the data |

## 6.8 Glossary

**Adapter** — LoRA weights ($A$, $B$), servable on top of a base model.
**Advantage** — reward minus baseline; in GRPO, the group z-score.
**Base / delta checkpoint** — full vs incremental sampler snapshot for
full-parameter runs.
**DCP** — distributed checkpoint, weights + optimizer, for resume.
**Datum** — one Tinker training example: `model_input` plus `loss_fn_inputs`.
**Firetitan** — internal name of the training service; hence
`FiretitanServiceClient`.
**Hot-load** — swapping new weights into a running deployment without restart.
**MFU** — model FLOPs utilization, achieved ÷ peak.
**On-policy** — training data generated by the current policy.
**Promote** — publish a sampler checkpoint as a durable model ID.
**R3 / router replay** — replaying inference MoE routing in the trainer.
**RFT** — reinforcement fine-tuning; the managed product surface for RL.
**RLOR** — the backend trainer service RL jobs run on.
**Rollout** — one sampled trajectory from the policy.
**Serverless / dedicated** — pooled per-token vs provisioned per-GPU-hour.
**Shape** — a pinned GPU/topology profile for training or serving.
**TIS** — train–inference importance sampling (§4.8).
**Tinker** — the training API protocol this SDK is compatible with.

## 6.9 Where to go next

Read in this order, with the primer sections as context:

1. [`skills/fireworks-training/SKILL.md`](../skills/fireworks-training/SKILL.md) — routing between methods, lifecycle, confirmation gates.
2. `training/recipes/sft_loop.py` — the simplest complete loop (parts 1 and 3).
3. `training/utils/client.py` — the exact protocol surface (§3.11).
4. `training/utils/config.py` — every infrastructure knob in one file (parts 2 and 5).
5. `training/utils/rl/grpo.py` and `tis.py` — the RL math as code (part 4).
6. `training/recipes/async_rl_loop.py` plus [`references/rl-async.md`](../skills/fireworks-training/references/rl-async.md) — scheduling (§4.11).
7. [`references/rl-hotload.md`](../skills/fireworks-training/references/rl-hotload.md) — weight-sync semantics (§4.12).

The best way to make it concrete is to run a small LoRA SFT job on the serverless
path, then a small GRPO run, and watch the metrics in §6.7 move.
