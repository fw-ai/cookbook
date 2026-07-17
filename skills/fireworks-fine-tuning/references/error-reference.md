# Error reference: running jobs, deploying, and debugging

*One place for "my training job failed / is stuck / errored" and "how do I debug it." Source of truth for volatile behavior is the live docs; this is durable field-observed triage. For deep Training-SDK debugging (weight sync, checkpoint promotion, renderer, hotload) use the separately installed `fireworks-training` companion skill instead.*

## First question, always: platform-side or user-side?

A bare error string is usually **not** diagnostic on its own. Classify by symptom before acting.

- **Platform-side** = retry or escalate; it is not your config. (~69% of failed jobs.)
- **User-side** = fix your dataset/config/quota; the skill can resolve it outright. (~14%.)
- **Generic/unknown** = de-mask first (pull the real signal below), then re-classify. (~17%.)

When a job is stuck or failing, use the resource family that created it:

| Method | Status | Strongest available progress signal |
|---|---|---|
| SFT | `firectl sftj get <JOB_ID> -o json` | Job fields and linked W&B metrics when enabled. The current CLI has no `sftj export-metrics`. |
| DPO / ORPO | `firectl dpo-job get <JOB_ID> -o json` | `firectl dpo-job export-metrics <JOB_ID>` and linked W&B metrics when enabled. |
| Managed RFT | `firectl rftj get <JOB_ID> -o json` | Job, evaluator, and rollout fields plus linked W&B metrics when enabled. The current CLI has no `rftj export-metrics`. |

Run the selected resource and `get` commands with `--help` before relying on flags. Do not substitute `sftj` commands for a DPO or RFT resource. `State` alone lies: a job can read `RUNNING` before the trainer starts, and a silent crash can leave `RUNNING` with no error. Trust a real step, rollout, checkpoint, or linked W&B signal when one is available.

## Common issues (field-observed)

| Symptom | What it usually is | What to do |
|---|---|---|
| **Stuck in `RUNNING` at 0%, no loss** | Startup, not training: marked `RUNNING` before the trainer actually starts (model download / GPU placement). **Platform-side.** | Don't wait indefinitely. No loss after a few minutes → retry once; if it recurs on the same model/shape, escalate. Not your config. |
| **Fails after a readiness/startup timeout (~3600s)** | Trainer/rollout couldn't become ready in the window (large download, tight capacity). **Platform-side**, often transient. | Retry; if it repeats, capacity or a known infra issue → escalate with the job id. |
| **`RESOURCE_EXHAUSTED` / "no capacity" / unschedulable** | Scheduler can't place the job on the requested GPU class/region. **Platform-side (infra)**, not quota, not config. | Retry, try a smaller shape / different region, or request capacity. Distinct from a 429 quota ceiling. |
| **Bare "Internal error", no detail** | Masking gap: trainer crashed before writing a status file. Cause is **either** platform (base-model download failed) **or** user (dataset rejected). | Not diagnostic alone. Re-validate the dataset first (cheapest); if clean, treat as a platform startup crash → retry + escalate. |
| **Dataset rejected / validation error** | Malformed JSONL — rows missing `messages`, bad roles. **User-side.** Can surface as a plain "Internal error". | Fix the JSONL (every line needs `messages` with `role`+`content`), re-upload (`dataset create` validates on upload). See `references/choose-method.md`. |
| **400 on a warm-start / reused config** | Jobs created with validations skipped can 400 when warm-starting/reusing a shape. **Platform-side**, fix rolling out. | Create against a **validated shape** (don't skip validations); retry. If blocked, escalate for an ETA. |
| **429 on job create** | Quota — a ceiling on concurrent GPUs, not billing. Retry storms make it worse. **User-side (limits).** | Back off; raise quota or run fewer concurrent jobs / a smaller model. |
| **Account suspended / "spending limit reached"** | **Billing-side**, distinct from quota — budget cap (even with credits), no payment method, or risk review. | Fix in [Billing](https://fireworks.ai/billing). Not a platform bug. |
| **412 / shape unavailable** | The requested training shape isn't enabled on your account. | Pick a listed shape or request enablement. |
| **Context length rejected (`must be divisible by 16`)** | Requested max context isn't a multiple of the shape's tiling (advertised max may not itself be divisible). **User-side.** | Round the context length **down** to a multiple of 16. |
| **Deployed LoRA serves base behavior** | Adapter not promoted/loaded, **or** a deployment image/chart version skew. | Confirm the job finished + model exists; for multi-LoRA confirm `load-lora` and route `model#deployment`. Still base-only on a fresh deploy → likely image/chart skew, escalate. |
| **Multi-LoRA / addon load rejected** | FP8/FP4 shapes reject addons; some families don't support multi-LoRA. | Use a **BF16** shape for addons, or live-merge a single adapter. |
| **Reused / "leaked" deployment fails on redeploy** | A reused/swept deployment leaves a stale ledger entry. **Platform-side.** | Delete the old deployment and create fresh; if an RFT deploy keeps failing right after a resource sweep, escalate with the deployment id. |
| **SFT much slower than expected** | Sequences not packed → many padding tokens, throughput drops. | Enable sequence packing / batch tokenization; check token utilization. |
| **RL/GRPO reward collapses or KLD explodes** | Almost always **numerics drift** between trainer and rollout inference (esp. MoE routing), not the algorithm. | Align precision + measure logprob divergence; for MoE enable **Router Replay (R3)**. See `references/deploy-and-troubleshoot.md` (Numerics alignment). |
| **RFT "no-op": every rollout scores identically** | Evaluator saturation — the reward gives the same score to all outputs, so there's no gradient. **User setup.** | Fix the reward/evaluator so it discriminates; the RFT preflight guards against all-identical rewards before a full run. |

## Debugging a run (get the real signal)

1. **De-mask.** Pull the method's strongest available progress signal from the table above. A generic message plus no step or rollout progress = suspected stall, not health.
2. **Isolate user vs platform.** Re-validate the dataset locally (instant, cheap). If clean, the failure is almost certainly platform-side.
3. **Reproduce minimally.** For a config-shaped failure, retry once with defaults on a validated shape before assuming a bug.
4. **Escalate with context**, not just "stuck": include the job id, base model + shape, elapsed time, last step, and the exact status message. Do not paste account or customer identifiers into shared or customer-facing places.

## Escalation checklist

- Job id + method (`sftj` / `dpo-job` / `reinforcement-fine-tuning-job`)
- Base model + training shape
- Elapsed time and last observed step, rollout, checkpoint, or W&B metric
- Exact status message
- What you already ruled out (dataset validated? quota checked? billing checked?)
