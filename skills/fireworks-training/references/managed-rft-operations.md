# Managed RFT: launch, monitor, and validate

*Source of truth: live [RFT overview](https://docs.fireworks.ai/fine-tuning/reinforcement-fine-tuning-models.md), [Models matrix](https://docs.fireworks.ai/fine-tuning/models.md), [RFT parameters](https://docs.fireworks.ai/fine-tuning/rft-parameters-reference.md), and [Eval Protocol](https://evalprotocol.io/introduction). Defer flags and defaults to installed CLI `--help`.*

Use this reference for managed RFT preflight, launch, job states, monitoring, and recovery. Training API or cookbook RL belongs in `references/training-api.md` and `references/rl-async.md`.

## Preflight

Before any upload or create:

- Confirm the base model is RFT-compatible in the live Models matrix.
- Validate JSONL locally. Each row needs a `messages` array; evaluator-specific fields must match the reviewed evaluator contract.
- Probe the evaluator on at least five rows and require non-identical scores.
- Check authentication, account, billing readiness, and `firectl quota list`.
- Use full resource names such as `accounts/fireworks/models/<id>`.
- Run `firectl rftj create --help` and resolve every user-set or defaulted value before the confirmation gate in `SKILL.md`.

## Launch surfaces

| Surface | When | Entry |
|---|---|---|
| Eval Protocol | Reproducible evaluator and dataset workflow | `eval-protocol create rft ...` |
| `firectl` | Direct managed API and advanced flags | `firectl rftj create ...` |
| Dashboard | Human-guided exploratory launch | Fine-Tuning, then Reinforcement |

### Eval Protocol workflow

```bash
pip install eval-protocol
export FIREWORKS_API_KEY=...

cd evaluator_directory
ep local-test

eval-protocol create rft \
  --base-model accounts/fireworks/models/qwen3-4b \
  --output-model my-rft-output
```

The create command uploads changed evaluator and dataset artifacts, creates the job, and prints dashboard links. Treat upload and create as protected work under the confirmation gate.

### `firectl` alternative

```bash
firectl rftj create \
  --base-model accounts/fireworks/models/qwen3-4b \
  --dataset accounts/<acct>/datasets/<id> \
  --evaluator accounts/<acct>/evaluators/<id> \
  --output-model accounts/<acct>/models/<out>
```

Run `firectl rftj create --help` for the authoritative checkpoint, rollout, W&B, and sampling flags.

### Warm start

Continue from a promoted or uploaded LoRA without also passing `--base-model`:

```bash
eval-protocol create rft \
  --warm-start-from accounts/<acct>/models/<sft-model-id> \
  --output-model <rft-model-id>
```

If the API reports that an `HF_PEFT_ADDON` is not a base model, remove `--base-model`.

## Parameter policy

Do not freeze volatile defaults in this skill. Resolve them from `--help`, the dry-run output, and live docs. The important relationships are:

- `epochs`, learning rate, and LoRA rank control optimization.
- Batch size is a token budget, while chunk size controls prompts per GRPO step.
- GRPO accepts a KL coefficient; DAPO and GSPO-token use different clipping and reject GRPO-only settings.
- Candidate count and temperature control rollout diversity.
- Maximum concurrent rollouts controls throughput, not the objective.
- Remote server URL selects a remote evaluator or environment.

Method semantics and starting points live in `references/choose-method.md`; exact parameter fields live in the RFT parameter reference.

## Job states and progress

Poll with:

```bash
firectl rftj get <job-id> -o json
```

| State | Meaning | Agent action |
|---|---|---|
| `PENDING` | Waiting for resources | Continue bounded monitoring |
| `VALIDATING` | Dataset and evaluator checks | Wait for validation result |
| `RUNNING` | Active work | Require rollout, step, reward, or W&B movement |
| `COMPLETED` | Successful run | Evaluate, then separately confirm promotion and deployment |
| `FAILED` | Hard failure | Classify with `references/error-reference.md` |
| `CANCELLED` | User stopped the run | Report final state and partial artifacts |
| `EARLY_STOPPED` | No useful improvement | Inspect evaluator and data signal |

State alone is not progress. Use the approved no-progress timeout and the evidence contract in `references/run-state-and-reporting.md`.

Use the dashboard for rollout inspection, reward curves, logs, job comparison, and human diagnosis. Keep automation on stable resource IDs and `firectl ... get`.

## Secrets in evaluators

Create secrets through the Fireworks secret-management surface and reference them by environment-variable name in evaluator code. Never paste secret values into a manifest, dataset, or chat. For remote environments, also read `references/rft-agent-tracing.md`.

## Common failures

| Symptom | Next action |
|---|---|
| Invalid JSON on line N | Fix that JSONL row locally |
| Missing `messages` | Add the required conversation array |
| Evaluator not found | Register or select the reviewed evaluator |
| Flat reward across probes | Fix evaluator saturation before launch |
| HTTP 429 | Check quota and back off; do not create replacement IDs |
| `HF_PEFT_ADDON` rejected as base model | Remove `--base-model` when warm-starting |

## Related

- Evaluator authoring: `references/preference-data-and-evaluators.md`
- Remote tracing: `references/rft-agent-tracing.md`
- Failure triage: `references/error-reference.md`
- Run manifest and reporting: `references/run-state-and-reporting.md`
- Cost formulas: [multi-turn cost comparison](https://docs.fireworks.ai/fine-tuning/multi-turn-cost-comparison.md)
