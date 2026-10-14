# Configure path and method intake

Run **before** the final plan when the user wants to train. Reuse the
discriminating QA from
`../../fireworks-training/references/choose-method.md` and add an explicit
workflow path choice.

Read `output-template.md` for how configure turns should look.

## Completion gate (before final plan)

Do **not** present a spend plan until:

1. **Method** is known (SFT / DPO / ORPO / RFT / IGPO / distillation /
   embedding) from research handoff or
   `../../fireworks-training/references/choose-method.md`.
2. **Workflow path** is chosen: the easiest compatible default from the table
   below, the user's stated preference, or a Q-path answer. The final plan names
   it, and plan approval confirms it.
3. Local dataset / model facts are resolved or labeled **unknown**.

If the user says "just use defaults" or "whatever's easiest," pick the easiest
compatible path and name it in the plan (never silent defaults).

## Easiest compatible path

| Signal | Default `workflow_path` |
|---|---|
| Plain SFT on chat JSONL, text or vision, no custom code | `managed_sdk` (agent runs SDK `supervised_fine_tuning_jobs`) |
| User wants to write a loop, sample in-session, or compare base vs tuned without a deployment | `serverless` |
| RL, serverless DPO, distillation, custom loss | `serverless` |
| User will run commands themselves in a terminal | `managed_firectl` |
| Full-parameter, model not on the serverless pool, reserved capacity, custom infra | `dedicated` |

Choose Dedicated only when an easier path cannot do the job. When the user asks
for Dedicated without such a reason, recommend the easier path in one sentence
(no provisioning, no idle cost) and follow their answer.

## Inherit from research

When `run.md` includes research handoff fields (legacy: discover handoff), reuse them:

| Field | Use |
|---|---|
| `case_study` | Case-study slug when present |
| `cookbook_entry_tier` | Case study, example, or recipe |
| `cookbook_entry_path` | Exact starting path for every tier |
| `notebook`, `readme` | Supporting entry files when present |
| `implied_method` | Method (SFT, DPO, …) |
| `suggested_path` | `managed` \| `serverless` \| `training_api_dedicated` |
| `dataset_plan` | Approved source and dataset candidates |
| `eval_plan` | Approved metric, baseline requirement, and cookbook hook |

Research does not set `workflow_path` or `execution_surface`. Configure picks
the exact path from the easiest-path table and names it in the final plan. A
`research-q3` answer never satisfies this gate by itself, even when its option
ID has the same spelling; the path is confirmed by approval of the plan that
names it, or by a recorded `question_id: configure-q-path` answer.

## Q-path — Which workflow? (only when needed)

Ask Q-path only when two paths fit equally well and the user has not stated a
preference. Otherwise use the easiest compatible default. When asked, fire
**one** AskQuestion and STOP.

Title: `Configure`

Prompt: `How do you want to run this training job?`

| Option ID | Label | `workflow_path` | `execution_surface` |
|---|---|---|---|
| `managed_sdk` | Managed — I run it for you through the Python SDK (easiest) | managed_sdk | sdk |
| `serverless` | Training API — serverless (your own loop, in-session eval, RL) | serverless | training_api |
| `managed_firectl` | Managed — you run the `firectl` commands yourself | managed_firectl | firectl |
| `dedicated` | Training API — dedicated trainer (full-parameter, reserved capacity) | dedicated | training_api |

List the recommended option first.

**Training API access:** if the account cannot reach the serverless pool, say so
and offer Managed SFT. See
`../../fireworks-training/references/training-api.md`.

## Q-method — Supervision signal (when method unclear)

If method is not set by research or the user message, use one AskQuestion from
`../../fireworks-training/references/choose-method.md`:

Prompt: `What supervision do you have?`

Telemetry: `question_id: configure-q-method`.

| Option ID | Option | Method |
|---|---|---|
| `labeled` | Labeled input → correct output | SFT |
| `preference_pairs` | Pairs where one answer is better | DPO by default; ORPO only when the user explicitly wants no reference model |
| `scored_prompts` | Prompts + scorer 0–1 | RFT |
| `unsure` | Not sure yet | stay in intake; do not plan spend |

The completion gate requires one final method. `preference_pairs` resolves to
DPO unless the user explicitly requests ORPO before the final plan.

For vague goals ("make my chatbot better"), run the full discriminating list in
`../../fireworks-training/references/choose-method.md` one question per turn.

`managed_firectl` and `managed_sdk` use the same managed backend. The agent
uses the SDK because `firectl` blocks agent mutations; choose `managed_firectl`
only when the user will run the commands.

## Record in run manifest

After the path is confirmed, update the private run manifest and record
`question_id: configure-q-path` with the selected option ID (response source
`plan_approval` when no question was asked) through
`firectl skill-journey record` when available. Use
`../../fireworks-training/references/telemetry.md`.

```yaml
entry_skill: configure
workflow_path: managed_firectl | managed_sdk | serverless | dedicated
execution_surface: firectl | sdk | training_api
implied_method: sft | dpo | orpo | rft | igpo | distillation | embedding
inherited_from_research: true | false
research_handoff:  # when present (legacy: discover_handoff)
  starter_job_id: <built-in starter ID>  # when present
  case_study: <slug>
  cookbook_entry_tier: case_study | example | recipe
  cookbook_entry_path: training/...
  notebook: training/...  # when present
  readme: training/...    # when present
  suggested_path: <from discover>
  dataset_plan:
    source: local | account_preloaded | cookbook_bundled | huggingface | labeling_required
    candidates: []
  eval_plan:
    metric:
    baseline_required: true
    cookbook_eval_hook: eval-protocol | recipe_inline | gap
    cookbook_eval_entry: training/...  # optional adaptation source
```

Set `inherited_from_research: true` only when the user had already selected one
exact path before the handoff. A recommendation alone is not inherited.

## Final plan must state

Every confirmed plan includes a **Path** section:

| Field | Example |
|---|---|
| Workflow | Managed / Training API serverless / dedicated |
| Surface | `firectl sftj` / SDK `supervised_fine_tuning_jobs` / `sft_loop.py` |
| Why | One sentence tied to user choice |
| Availability | Managed and Serverless Training are GA; Dedicated Training needs account access |

Never show only `firectl` commands when the user chose SDK or Training API.

### DPO recipe on serverless trainers

`training.recipes.dpo_loop.Config(serverless=True)` uses one positive-rank LoRA
policy run and requires an explicit `max_seq_len`. It saves `dpo-reference`
before the first policy update and scores that fixed snapshot through a sampling
client. The reference is inference-only; no second training run is created.
Reference token logprobs are aligned to the trainer targets before applying the
existing DPO loss. Warm starts and resume are rejected on this initial path so
the reference cannot accidentally be reanchored to an already-trained policy.
The snapshot is internal reference state, not the final trained output. Policy
training and sampler scoring have separate usage paths; do not quote the
reference as free or included in training-token usage. ORPO remains unchanged.

### Capacity-409 retries for serverless SFT/DPO

The shared serverless SFT/DPO recipe setup and the standalone serverless SFT/DPO
examples opt in to SDK capacity/recovery-409 retries before creating or resuming
a training client. Only opted-in serverless future-retrieve requests receive
this policy; terminal and unrecognized explicit conflicts still fail fast. It
does not change trainer placement or opt RFT/RL clients in. Older supported SDK
releases without the optional hook retain their existing retry behavior; the
cookbook does not require an unpublished SDK release just to start training.
