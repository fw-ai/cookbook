# Research methodology

Recommendation-first planning before any GPU spend. Modeled on Tinker
[`research`](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/skills/research/SKILL.md)
and LangChain
[eval-engineering](https://github.com/langchain-ai/langchain-skills/blob/main/config/skills/eval-engineering/SKILL.md)
(inspect → propose → user approves → hand off).

## Principles

1. **Inspect before you ask.** Read `cookbook-catalog.md`, `starter-jobs.md`,
   and the closest README. Do not ask questions the catalog already answers.
2. **Offer starters to users without a task.** A `research-starter` selection
   goes straight to Configure's final plan. When the user has a real task but
   no labels, do not replace their goal with a generic starter. Give a concrete
   data plan in the same reply: a credible label source, a pilot sized to
   expose the main uncertainty, a representative uncontaminated holdout, and
   a task-appropriate success metric. Make the pilot actionable with a
   starting count or coverage target and the evidence that would justify
   expanding it; treat that as a task-specific hypothesis, not a universal
   threshold. Offer an immediately runnable baseline when it will teach the
   user something relevant while labels are built.
3. **Recommend first; ask only what blocks.** Infer from what the user said,
   state defaults in one line, and ask one decision-critical question by
   default only when no reasonable default exists. Group tightly coupled
   blocking facts when separating them would add churn. State defaults for
   non-blocking gaps. No formal interview.
4. **Ground proposals in evidence.** Cite cookbook paths, bundled datasets, or HF
   hits — not generic advice.
5. **Eval before train.** No handoff to configure without an eval plan (metric
   class + baseline yes/no + cookbook hook or explicit gap).
6. **User approves the package.** Present one recommendation; offer alternatives
   only when two entries are genuinely close.
7. **Exploring a domain.** When the user names a domain but no task ("I'm at
   a fintech, where would you start?"), give two to four ideas from that
   domain, each with its method and a cookbook hook, e.g. transaction or
   ticket categorization → SFT; document field extraction → SFT (vision for
   scans); policy or KB retrieval → embedding fine-tune; tool-using agent with
   a checkable outcome → RL. Then offer the free Muse Glimmer starter as the
   hands-on first step, and avoid unrelated intake questions.
8. **Diagnose before prescribing.** For "RAG answers are wrong", first separate
   the stages and use controlled comparisons to locate the bottleneck before
   choosing a method. For retrieval pipelines, check whether relevant context
   reaches the generator before deciding between retrieval work and generator
   tuning, and propose a retrieval metric on representative held-out queries
   in the same reply. When that diagnosis is already clear, stop after the
   recommendation, metric, and one specific blocker; do not append a generic
   intake menu. For model comparisons, define comparable untuned, tuned,
   and reference-model baselines on the same holdout and metric. Honor the
   requested size class: verify a candidate from the live catalog or leave it
   open rather than substituting a promotional mid-size model. Confirm the
   exact model identity before launch. Mention a promotion only when its model
   and method fit the goal.
9. **Use controls before learning.** Separate constraints the serving layer can
   enforce from behavior the model must learn. For structured output, first
   offer to run JSON/schema-constrained generation on a small representative
   sample; use training for field semantics, classification, extraction, or
   quality that constraints cannot supply. Do not append an unrelated training
   starter merely to make the answer executable.
10. **Make missing rewards concrete.** For RL without a verifier, propose the
    smallest task-shaped executable check before discussing a run. For coding,
    this is usually sandboxed execution plus the relevant unit tests, with
    explicit timeout and failure behavior; smoke-test that reward on examples
    before collecting rollouts.

## Arc

```text
Inspect cookbook (+ starter-jobs.md)
  → Starter (greenfield) or a direct recommendation with stated defaults
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

## Filling gaps

Use defaults first; `interview-questions.md` is the fallback. Its completion
gate is met when each item is stated, inferable, or defaulted:

- Task shape (Q1), answered or clearly inferable
- Data / supervision (Q2), answered or clearly inferable
- Eval direction (Q-eval), answered or a proposed default metric
- Applicable follow-up (Q1b, domain)

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
- Create jobs or other resources
- Pick final base model or quote price (configure)
