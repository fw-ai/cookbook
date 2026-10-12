# Research methodology

Interview-driven planning before any GPU spend. Modeled on Tinker
[`research`](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/skills/research/SKILL.md)
and LangChain
[eval-engineering](https://github.com/langchain-ai/langchain-skills/blob/main/config/skills/eval-engineering/SKILL.md)
(inspect → propose → user approves → hand off).

## Principles

1. **Inspect before you ask.** Read `cookbook-catalog.md`, `starter-jobs.md`,
   and the closest README. Do not ask questions the catalog already answers.
2. **Offer starters to greenfield users.** When there is no dataset yet, ask
   `research-starter` before the long interview.
3. **One question per turn.** Use AskQuestion; STOP and wait. Interview, not
   interrogation.
4. **Ground proposals in evidence.** Cite cookbook paths, bundled datasets, or HF
   hits — not generic advice.
5. **Eval before train.** No handoff to configure without an eval plan (metric
   class + baseline yes/no + cookbook hook or explicit gap).
6. **User approves the package.** Present one recommendation; offer alternatives
   only when two entries are genuinely close.

## Arc

```text
Inspect cookbook (+ starter-jobs.md)
  → Starter menu (greenfield) or Interview (task, data, eval, path)
  → Readiness package (method, entry, dataset_plan, eval_plan)
  → User approves handoff
  → configure
```

## Inspect phase

| Input | Action |
|---|---|
| User message | Tentative match from catalog |
| Greenfield / "what can I try?" | Offer `starter-jobs.md` before Q1 |
| Repo context | Note local JSONL, notebooks, prior runs |
| `training/case-studies/<slug>/README.md` | "Is this you?" paragraph |
| `training/examples/*/README.md` | Minimal fork when no case study fits |
| `training/recipes/*_loop.py` | Custom Training API path |
| HF Hub (gated) | Search public metadata only after explicit approval; send keywords, never customer data |

## Interview phase

See `interview-questions.md`. Completion gate requires:

- Task shape (Q1)
- Data / supervision (Q2)
- Eval direction (Q-eval)
- Applicable follow-up (Q1b, domain, Q3)

## Readiness package

Prepare this readiness package, but do not persist the handoff until the user
selects `plan_configure`:

```yaml
entry_skill: research
starter_job_id:             # when a built-in starter was selected
case_study:               # omit for example or recipe
cookbook_entry_tier: case_study | example | recipe
cookbook_entry_path: training/...
implied_method: sft | dpo | orpo | rft | igpo | distillation | embedding
suggested_path: managed | serverless | training_api_dedicated
supervision_signal:
notebook:                # when applicable
readme:
dataset_plan:
  source: local | account_preloaded | cookbook_bundled | huggingface | labeling_required
  candidates: []
eval_plan:
  metric:
  baseline_required: true
  cookbook_eval_hook: eval-protocol | recipe_inline | gap
  cookbook_eval_entry: training/...  # optional adaptation source
match_confidence: high | medium | low
```

## What research does not do in this phase

- Download HF datasets
- Upload to Fireworks
- Run `firectl` create commands
- Pick final base model or quote price (configure)
