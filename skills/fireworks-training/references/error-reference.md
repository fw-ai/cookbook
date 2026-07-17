# Error reference: running jobs, deploying, and debugging

*One place for "my training job failed / is stuck / errored" and "how do I debug it." Source of truth for volatile behavior is the live docs; this is durable field-observed triage. Deep Training API debugging lives under `references/sdk/` in this skill.*

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
| **Stuck in `RUNNING` at 0%, no loss** | Possible startup or placement stall; state alone does not identify the cause. | Use the numeric no-progress timeout approved for the run (10 minutes for a small smoke by default). Gather method-specific status and progress evidence. Before replacement, cancel the old job, confirm it is terminal, and get approval for new spend. Escalate recurrence with both job records. |
| **Fails after a readiness/startup timeout (~3600s)** | Trainer/rollout couldn't become ready in the window (large download, tight capacity). **Platform-side**, often transient. | Confirm the failed resource is terminal. Propose one bounded retry and get approval before replacement spend; escalate recurrence with both job IDs. |
| **`RESOURCE_EXHAUSTED` / "no capacity" / unschedulable** | Scheduler can't place the job on the requested GPU class/region. **Platform-side (infra)**, not quota, not config. | Offer a smaller supported shape, another supported region, or a capacity request. Confirm terminal state and get approval before a replacement run. Distinct from a 429 quota ceiling. |
| **Bare "Internal error", no detail** | Unknown. It can mask user data/config errors or platform startup failures. | Re-validate locally, then collect method-specific status, progress, request, and correlation evidence. Keep the classification unknown until evidence supports user or platform attribution. |
| **Dataset rejected / validation error** | Malformed JSONL — rows missing `messages`, bad roles. **User-side.** Can surface as a plain "Internal error". | Fix the JSONL (every line needs `messages` with `role`+`content`), re-upload (`dataset create` validates on upload). See `references/choose-method.md`. |
| **400 on a warm-start / reused config** | Jobs created with validations skipped can 400 when warm-starting/reusing a shape. **Platform-side**, fix rolling out. | Create against a **validated shape** (don't skip validations); retry. If blocked, escalate for an ETA. |
| **429 on job create** | Quota — a ceiling on concurrent GPUs, not billing. Retry storms make it worse. **User-side (limits).** | Back off; raise quota or run fewer concurrent jobs / a smaller model. |
| **Account suspended / "spending limit reached"** | **Billing-side**, distinct from quota — budget cap (even with credits), no payment method, or risk review. | Fix in [Billing](https://fireworks.ai/billing). Not a platform bug. |
| **412 / shape unavailable** | The requested training shape isn't enabled on your account. | Pick a listed shape or request enablement. |
| **Context length rejected (`must be divisible by 16`)** | Requested max context isn't a multiple of the shape's tiling (advertised max may not itself be divisible). **User-side.** | Round the context length **down** to a multiple of 16. |
| **Deployed LoRA serves base behavior** | Adapter promotion, loading, routing, or runtime skew are all possible. | Read deployment state and loaded-addon state, verify the exact route, then compare paired requests against the base and adapter targets. If a fresh correctly routed deployment remains base-only, image/chart version skew becomes a leading platform hypothesis; escalate it as a hypothesis, not a conclusion. |
| **Multi-LoRA / addon load rejected** | FP8/FP4 shapes reject addons; some families don't support multi-LoRA. | Use a **BF16** shape for addons, or live-merge a single adapter. |
| **Reused / "leaked" deployment fails on redeploy** | The visible deployment and backend state may disagree. | Read and reconcile the deployment identity, state, model, and shape first. Show the consequences and get approval before deletion or replacement. Create at most one replacement; escalate ambiguity or recurrence with both IDs. |
| **SFT much slower than expected** | Length skew or padding can reduce useful tokens per batch. | Inspect rendered lengths and loss masks, enable the recipe's `group_by_length` option when appropriate, and compare batch utilization before changing GPU shape. |
| **RL/GRPO reward collapses or KLD explodes** | Numerics drift is one hypothesis; reward, data, rollout, and algorithm issues remain possible. | Measure same-token trainer/inference logprob divergence before attribution. For MoE, enable Router Replay only after routing divergence is confirmed. See `references/deploy-and-troubleshoot.md` (Numerics alignment). |
| **RFT "no-op": every rollout scores identically** | Evaluator saturation — the reward gives the same score to all outputs, so there's no gradient. **User setup.** | Fix the reward/evaluator so it discriminates; the RFT preflight guards against all-identical rewards before a full run. |
| **Serverless run quota is 8/8 with stale sessions** | It may be legitimate concurrency or a session-reclamation failure. | Close every client through the supported SDK path and wait the documented session expiry. If old `READY` or `UNSPECIFIED` sessions still consume slots, stop creating runs and escalate session IDs, states, create/heartbeat times, and the account quota. Use a current SDK or account-UI deletion action only when it is documented and visible. |
| **Quota error masks an invalid model or rank** | Quota can be checked before request validation. | Preserve both hypotheses. Free one slot through supported close/expiry, rerun the negative validation once, and do not mark the model/rank test passed until the underlying validation response is observed. |
| **Failed creates leave `TRAINING_SESSION_STATE_UNSPECIFIED` rows** | Possible reserve-before-bind rollback failure. | Stop further create attempts, record UTC timestamps and session IDs for every leaked row, and escalate. Do not treat them as completed sessions. |
| **Same serverless workload returns both 429 and 503** | Different components or inconsistent status mapping may be involved. | Record the emitting component, endpoint, status, request/correlation ID, retry history, and session state separately for each response. Escalate inconsistent mapping instead of collapsing both into trainer instability. |

## Debugging a run (get the real signal)

1. **De-mask.** Pull the method's strongest available progress signal from the table above. A generic message plus no step or rollout progress = suspected stall, not health.
2. **Isolate user vs platform.** Re-validate the dataset locally, then combine that result with method-specific status, progress, and request evidence. A clean dataset alone does not prove a platform failure.
3. **Reproduce minimally.** For a config-shaped failure, propose one bounded retry with defaults on a validated shape. Cancel and confirm the old resource is terminal, then get user approval before creating the replacement.
4. **Escalate with context**, not just "stuck": include UTC timestamps, exact installed CLI command family, job/trainer/deployment/checkpoint IDs, model + shape, elapsed time, last progress, exact status, request/correlation IDs, CLI and SDK versions, cookbook commit, retry history, and what was redacted. Do not paste account or customer identifiers into shared or customer-facing places.

## Escalation checklist

- Job or trainer ID + method. Use the command family shown by installed `firectl --help` consistently (`sftj`, `dpo-job`, or `rftj` aliases are acceptable when supported).
- Base model + training shape
- UTC start/failure timestamps, elapsed time, and last observed step, rollout, checkpoint, or W&B metric
- Exact status message
- Request/correlation IDs plus deployment and checkpoint IDs when relevant
- `firectl` version, SDK version, and cookbook commit
- Retry/cancel history and final state of prior resources
- What you ruled out and which customer identifiers or secrets were redacted
