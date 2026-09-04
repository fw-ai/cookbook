# Managed RFT — launch, monitor, and validate

*Source of truth: live [RFT overview](https://docs.fireworks.ai/fine-tuning/reinforcement-fine-tuning-models.md), [Models matrix](https://docs.fireworks.ai/fine-tuning/models.md), [RFT parameters](https://docs.fireworks.ai/fine-tuning/rft-parameters-reference.md), [eval-protocol](https://evalprotocol.io/introduction). Defer flags and defaults to installed CLI `--help`.*

Use this reference for **managed** RFT: eval-protocol launches, `firectl rftj`, preflight validation, job states, and monitoring. Training API / cookbook RL → `references/training-api.md`, `references/rl-async.md`.

## Preflight

Before any upload or create:

- Base model is RFT-compatible in the live [Models](https://docs.fireworks.ai/fine-tuning/models.md) matrix.
- Dataset JSONL: one object per line with a `messages` array (system + user prompts; assistant turns optional).
- Evaluator registered; a 5-row probe returns **non-identical** scores (flat reward = no gradient).
- `export FIREWORKS_API_KEY=...` or `firectl signin`; prefer scoped service-account keys.
- `firectl quota list` shows headroom for the resolved shape.
- Full base-model path: `accounts/fireworks/models/<id>` (never a bare model name).

## Launch surfaces

| Surface | When | Entry |
|---|---|---|
| eval-protocol | Scriptable, reproducible configs | `eval-protocol create rft ...` |
| firectl | Direct managed API, advanced flags | `firectl rftj create ...` |
| Dashboard | Exploratory first job | app.fireworks.ai → Fine-Tuning → Reinforcement |

### eval-protocol workflow

```bash
pip install eval-protocol
export FIREWORKS_API_KEY=...

cd evaluator_directory
ep local-test                    # pytest @evaluation_test; Docker rules apply for containerized evaluators

eval-protocol create rft \
  --base-model accounts/fireworks/models/qwen3-4b \
  --output-model my-rft-output
```

The CLI uploads changed evaluator/dataset artifacts, creates the job, and prints dashboard links.

**Docker evaluators:** Debian-based images only; single-stage; supported instructions: `FROM`, `RUN`, `COPY`, `ADD`, `WORKDIR`, `USER`, `ENV`, `CMD`, `ENTRYPOINT`, `ARG`.

### firectl alternative

```bash
firectl rftj create \
  --base-model accounts/fireworks/models/qwen3-4b \
  --dataset accounts/<acct>/datasets/<id> \
  --evaluator accounts/<acct>/evaluators/<id> \
  --output-model accounts/<acct>/models/<out>
```

Run `firectl rftj create --help` for checkpoint frequency, rollout timeout, W&B, and other flags not exposed on eval-protocol.

### Warm start (SFT → RFT)

Continue from a promoted or uploaded LoRA — **do not** pass `--base-model` with `--warm-start-from`:

```bash
eval-protocol create rft \
  --warm-start-from accounts/<acct>/models/<SFT_MODEL_ID> \
  --output-model <RFT_MODEL_ID>
```

Error `not of kind base_model, but HF_PEFT_ADDON` → remove `--base-model`.

### Common flags (eval-protocol / firectl)

Run `--help` for the authoritative list. Typical knobs:

| Flag | Default (typical) | Notes |
|---|---|---|
| `--epochs` | 1 | Whole numbers; watch reward curve before adding |
| `--learning-rate` | 1e-4 | Decrease if reward spikes then crashes |
| `--lora-rank` | 8 | 4–32, powers of 2 |
| `--batch-size` | 32k tokens | RFT V1 packed-token budget (not `batch_size_samples`) |
| `--chunk-size` | 200 | Prompts per GRPO step; `-1` disables chunking |
| `--rl-loss-method` | grpo | `grpo`, `dapo`, `gspo-token` |
| `--rl-kl-beta` | 0.001 | GRPO only; rejected for dapo/gspo-token |
| `--temperature` | 0.7 | Rollout sampling |
| `--n` / `--response-candidates-count` | 4 / 8 | Rollouts per prompt (surface-dependent) |
| `--max-tokens` | 32768 | Cap per rollout response |
| `--max-concurrent-rollouts` | 96 | Throughput only |
| `--remote-server-url` | — | Remote evaluator / environment |
| `--warm-start-from` | — | LoRA adapter resource; omit `--base-model` |
| `--wandb-project` | — | Requires `WANDB_API_KEY` |

Parameter semantics and GRPO metrics: live [RFT parameters reference](https://docs.fireworks.ai/fine-tuning/rft-parameters-reference.md) and `references/choose-method.md`.

## Job validation

Fireworks validates before `RUNNING`:

- **Dataset:** valid JSONL; each line has `messages` with `role` + `content`.
- **Evaluator:** syntax, dependencies, entry point; local `ep local-test` or pytest upload path succeeded.
- **Resources:** GPU quota, RFT permissions, model tunable/RFT-compatible.

On failure, fix the cited line or resource — see `references/error-reference.md`.

## Job states

Poll: `firectl rftj get <job-id> -o json`

| State | Meaning | Agent action |
|---|---|---|
| PENDING | Queued for GPU | Wait |
| VALIDATING | Dataset/evaluator checks | Wait ~1–2 min |
| RUNNING | Active rollouts | Require progress signal, not state alone |
| COMPLETED | Success | Evaluate → promote → deploy (separate approvals) |
| FAILED | Hard error | Triage in `error-reference.md` |
| CANCELLED | User stopped | Review partial results if needed |
| EARLY_STOPPED | Flat reward / no improvement | Fix evaluator or data |

**Do not trust `RUNNING` alone.** Use step/rollout/W&B movement within the approved no-progress timeout (default 10 min for smoke runs). See `references/run-state-and-reporting.md`.

Dashboard URL pattern (also printed at create):

`https://app.fireworks.ai/dashboard/fine-tuning/reinforcement/<job-id>`

Use the dashboard for human rollout inspection; keep polling with `firectl rftj get` for automation.

## Secrets in evaluators

Create secrets in [Dashboard → Secrets](https://app.fireworks.ai). They inject as environment variables into the evaluator container. Reference by name in evaluator code. Do not paste secret values into run manifests or chat.

Evaluator authoring: `references/preference-data-and-evaluators.md`.

## Common errors

| Symptom | Fix |
|---|---|
| `Dataset validation failed: invalid JSON on line N` | Fix JSONL syntax on that line |
| `Missing required field 'messages'` | Each row needs `{"messages":[...]}` |
| `Evaluator 'X' not found` | Upload/register evaluator first |
| All rollouts identical score | Evaluator saturation — discriminate rewards |
| 429 on create | Quota ceiling — backoff, smaller concurrency |
| `not of kind base_model, but HF_PEFT_ADDON` | Remove `--base-model` when using `--warm-start-from` |

## Related

- Evaluator authoring: `references/preference-data-and-evaluators.md`
- Failure triage: `references/error-reference.md`
- Run manifest + reporting: `references/run-state-and-reporting.md`
- Cost formulas (live pricing): docs [multi-turn cost comparison](https://docs.fireworks.ai/fine-tuning/multi-turn-cost-comparison.md)
