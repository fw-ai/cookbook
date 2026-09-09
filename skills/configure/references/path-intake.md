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
2. **Workflow path** is confirmed with Q-path. A coarse research
   `suggested_path` never skips this question.
3. Local dataset / model facts are resolved or labeled **unknown**.

If the user says "just use defaults," record that explicitly and still name the
chosen path in the plan (never silent defaults).

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

Research does not set `workflow_path` or `execution_surface`. Reuse its
recommendation as the suggested option, then run Q-path. Skip Q-path only when
the manifest contains an answer with `question_id: configure-q-path`. A
`research-q3` answer never satisfies this gate, even when its option ID has the
same spelling.

## Q-path — Which workflow? (required)

Fire **one** AskQuestion. STOP and wait.

Title: `Configure`

Prompt: `How do you want to run this training job?`

| Option ID | Label | `workflow_path` | `execution_surface` |
|---|---|---|---|
| `managed_firectl` | Managed — `firectl` CLI (simplest production path) | managed_firectl | firectl |
| `managed_sdk` | Managed — Python SDK / case-study notebook | managed_sdk | sdk |
| `serverless` | Training API — serverless (fast LoRA experiments) | serverless | training_api |
| `dedicated` | Training API — dedicated trainer (custom loop, full control) | dedicated | training_api |

**Routing hints (do not skip the question):**

| Signal | Suggest, don't assume |
|---|---|
| Plain SFT on JSONL, no custom code | `managed_firectl` |
| User references a case-study notebook | `managed_sdk` |
| Custom loss, rollouts, distillation, research loop | `dedicated` or `serverless` |
| Discover `suggested_path: serverless` | `serverless` |
| Discover `suggested_path: training_api_dedicated` | `dedicated` |

**Training API access:** serverless and dedicated are private preview. If
entitlement is unverified, say so and offer managed paths. See
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

## Q-surface — firectl vs SDK (optional follow-up)

Only when user picked managed but case study implies SDK (e.g. CORD notebook)
and they chose `managed_firectl`, confirm:

> Same managed backend — SDK notebook vs `firectl` CLI. Prefer CLI unless you
> need the notebook eval harness.

## Record in run manifest

After Q-path, update the private run manifest and record
`question_id: configure-q-path` with the selected option ID through
`firectl skill-journey record` when available. Use
`../../fireworks-training/references/telemetry.md`.

```yaml
entry_skill: configure
workflow_path: managed_firectl | managed_sdk | serverless | dedicated
execution_surface: firectl | sdk | training_api
implied_method: sft | dpo | orpo | rft | igpo | distillation | embedding
inherited_from_research: true | false
research_handoff:  # when present (legacy: discover_handoff)
  case_study: <slug>
  cookbook_entry_tier: case_study | example | recipe
  cookbook_entry_path: training/...
  notebook: training/...  # when present
  readme: training/...    # when present
  suggested_path: <from discover>
  dataset_plan:
    source: local | cookbook_bundled | huggingface | labeling_required
    candidates: []
  eval_plan:
    metric:
    baseline_required: true
    cookbook_eval_hook: eval-protocol | recipe_inline | gap
```

Set `inherited_from_research: true` only when the user had already selected one
exact Q-path option before the handoff. A recommendation alone is not inherited.

## Final plan must state

Every confirmed plan includes a **Path** section:

| Field | Example |
|---|---|
| Workflow | Managed / Training API serverless / dedicated |
| Surface | `firectl sftj` / SDK `supervised_fine_tuning_jobs` / `sft_loop.py` |
| Why | One sentence tied to user choice |
| GA vs preview | Managed GA; Training API preview if applicable |

Never show only `firectl` commands when the user chose SDK or Training API.
