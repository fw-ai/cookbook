# Research methodology

Interview-driven planning before any GPU spend. Modeled on Tinker
[`research`](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/skills/research/SKILL.md)
and LangChain
[eval-engineering](https://github.com/langchain-ai/langchain-skills/blob/main/config/skills/eval-engineering/SKILL.md)
(inspect → propose → user approves → hand off).

## Principles

1. **Inspect before you ask.** Read `cookbook-catalog.md` and the closest
   README. Do not ask questions the catalog already answers.
2. **One question per turn.** Use AskQuestion; STOP and wait. Interview, not
   interrogation.
3. **Ground proposals in evidence.** Cite cookbook paths, bundled datasets, or HF
   hits — not generic advice.
4. **Eval before train.** No handoff to configure without an eval plan (metric
   class + baseline yes/no + cookbook hook or explicit gap).
5. **User approves the package.** Present one recommendation; offer alternatives
   only when two entries are genuinely close.

## Arc

```text
Inspect cookbook (+ optional HF search)
  → Interview (task, data, eval, path)
  → Readiness package (method, entry, dataset_plan, eval_plan)
  → User approves handoff
  → configure
```

## Inspect phase

| Input | Action |
|---|---|
| User message | Tentative match from catalog |
| Repo context | Note local JSONL, notebooks, prior runs |
| `training/case-studies/<slug>/README.md` | "Is this you?" paragraph |
| `training/examples/*/README.md` | Minimal fork when no case study fits |
| `training/recipes/*_loop.py` | Custom Training API path |
| HF Hub (gated) | When Q2 = exploring / raw_only — see external-dataset draft |

## Interview phase

See `interview-questions.md`. Completion gate requires:

- Task shape (Q1)
- Data / supervision (Q2)
- Eval direction (Q-eval)
- Applicable follow-up (Q1b, domain, Q3)

## Readiness package

Write to `run.md` before handoff:

```yaml
entry_skill: research
implied_method:
supervision_signal:
cookbook_entry:
  tier: case_study | example | recipe
  path:
  slug:          # when tier = case_study
dataset_plan:
  source: local | cookbook_bundled | huggingface | labeling_required
  candidates: []
eval_plan:
  metric:
  baseline_required: true
  cookbook_eval_hook: eval-protocol | recipe_inline | gap
match_confidence: high | medium | low
```

## What research does not do in this phase

- Download HF datasets
- Upload to Fireworks
- Run `firectl` create commands
- Pick final base model or quote price (configure)
