# Configure path and method intake

Run **before** the final plan when the user wants to train. Reuses the
discriminating QA from `choose-method.md` and adds an explicit **workflow path**
choice (managed vs Training API, CLI vs SDK).

Read `references/output-template.md` for how configure turns should look.

## Completion gate (before final plan)

Do **not** present a spend plan until:

1. **Method** is known (SFT / DPO / ORPO / RFT / embedding) — from research
   handoff or `choose-method.md` intake.
2. **Workflow path** is confirmed — AskQuestion below unless research recorded
   `suggested_path` **and** `execution_surface`.
3. Local dataset / model facts are resolved or labeled **unknown**.

If the user says "just use defaults," record that explicitly and still name the
chosen path in the plan (never silent defaults).

## Inherit from research

When `run.md` includes research handoff fields (legacy: discover handoff), reuse them:

| Field | Use |
|---|---|
| `case_study` | Notebook / recipe starting point |
| `implied_method` | Method (SFT, DPO, …) |
| `suggested_path` | `managed` \| `serverless` \| `training_api_dedicated` |

If `execution_surface` is missing, still run **Q-path** (firectl vs SDK vs
recipe).

## Q-path — Which workflow? (required)

Fire **one** AskQuestion. STOP and wait.

Title: `Configure`

Prompt: `How do you want to run this training job?`

| Option ID | Label | `workflow_path` | `execution_surface` |
|---|---|---|---|
| `managed_firectl` | Managed — `firectl` CLI (simplest production path) | managed | firectl |
| `managed_sdk` | Managed — Python SDK / case-study notebook | managed | sdk |
| `serverless` | Training API — serverless (fast LoRA experiments) | serverless | training_api |
| `dedicated` | Training API — dedicated trainer (custom loop, full control) | training_api_dedicated | training_api |

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
`references/training-api.md`.

## Q-method — Supervision signal (when method unclear)

If method is not set by discover or the user message, use **one** AskQuestion
from `choose-method.md` intake (question 2):

Prompt: `What supervision do you have?`

| Option | Method |
|---|---|
| Labeled input → correct output | SFT |
| Pairs where one answer is better | DPO / ORPO |
| Prompts + scorer 0–1 | RFT |
| Not sure yet | stay in intake; do not plan spend |

For vague goals ("make my chatbot better"), run the full discriminating list in
`choose-method.md` § Intake for a vague request — one question per turn.

## Q-surface — firectl vs SDK (optional follow-up)

Only when user picked managed but case study implies SDK (e.g. CORD notebook)
and they chose `managed_firectl`, confirm:

> Same managed backend — SDK notebook vs `firectl` CLI. Prefer CLI unless you
> need the notebook eval harness.

## Record in run manifest

After Q-path (and Q-method when run), update the journey block and emit
`configure_path_answered` when `FIREWORKS_API_KEY` is set.

```yaml
entry_skill: configure
workflow_path: managed_firectl | managed_sdk | serverless | dedicated
execution_surface: firectl | sdk | training_api
implied_method: sft | dpo | orpo | rft | embedding
inherited_from_research: true | false
research_handoff:  # when present (legacy: discover_handoff)
  case_study: <slug>
  suggested_path: <from discover>
```

Set `inherited_from_research: true` when `workflow_path` / `execution_surface`
came from research handoff without re-asking Q-path.

## Final plan must state

Every confirmed plan includes a **Path** section:

| Field | Example |
|---|---|
| Workflow | Managed / Training API serverless / dedicated |
| Surface | `firectl sftj` / SDK `supervised_fine_tuning_jobs` / `sft_loop.py` |
| Why | One sentence tied to user choice |
| GA vs preview | Managed GA; Training API preview if applicable |

Never show only `firectl` commands when the user chose SDK or Training API.
