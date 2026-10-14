# Research interview questions

These questions are a fallback, not a script. Default to recommending from
what the user said, with stated defaults. Ask one of these only when the answer
is blocking — no reasonable default exists and it changes what you would do.
**Inspect** `cookbook-catalog.md` first.

Read `output-template.md` for turn shapes. Record bounded telemetry after each
answer using `../../fireworks-training/references/telemetry.md`. Show
`../../fireworks-training/references/telemetry-notice.md` before the first
interview question.

## Interview rules

- Use **AskQuestion** — **one question per turn**, then **STOP** and wait.
- Plain language. Q1, Q2, and Q-eval use the six registered contract options;
  all other questions use at most four.
- Start every turn with the **Research** skill banner.
- Propose options grounded in catalog evidence when helpful.
- Do not ask what the opening message or repo already answers.
- After each answer: update journey block, append `intake_responses[]`, set
  `task_summary` once (≤200 chars, redacted).

## Completion gate (before readiness package)

Present the recommendation once each item below is answered, stated, clearly
inferable, or covered by a stated default:

1. **Q1** (task shape) is answered, stated, or clearly inferable, **and**
2. **Q2** (data / supervision) is answered, stated, or clearly inferable, **and**
3. **Q-eval** (how to measure success) is answered or has a proposed default
   metric the user can accept, **and**
4. Any applicable follow-up (Q1b, domain) is answered or inferable.

"Clearly inferable" means the user's words leave one sensible reading (for
example "always output a valid JSON ticket" is `structured_output` /
`extraction`; "we have no labeled data" is `raw_only`). State the inference in
one line instead of asking. Ask only the questions that remain open, one per
turn, and share a provisional recommendation while asking.

**Exception:** user says "skip questions", "just pick one", names a cookbook
slug, or selects a starter from `starter-jobs.md` (any option except
`explore_custom`). Set `completion_gate_bypassed: true`.

**Novel domains** (finance, legal, healthcare): state the inferred Q1, Q2,
and Q-eval answers in the recommendation so the user can correct them.

## Question sequence

### Starter — What should we try first?

When the user is greenfield or asks what they can try, run this before Q1.
Telemetry and options: `starter-jobs.md` (`question_id: research-starter`).

If the answer is a concrete starter ID, skip Q1 / Q2 / Q-eval and go straight
to Configure's fast path in the same turn (read-only preflight and the complete
final plan). Do not fire **Handoff** for a starter. If the answer is
`explore_custom`, continue with Q1.

### Q1 — What are you trying to improve?

Telemetry: `question_id: research-q1` (legacy: `discover-q1`), field `intake_q1_task_shape`

| Option ID | Option | Routes toward |
|---|---|---|
| `structured_output` | Get structured output right every time (JSON, labels, extraction) | Q1b |
| `tone` | Make answers sound right (tone, brand, helpfulness) | `dpo_style` |
| `reasoning` | Improve reasoning on verifiable tasks (math, code, checks) | `reasoning_rl`, `rl/deepmath` |
| `rag` | Fix search / RAG retrieving the wrong document | `embedding_support_search` |
| `agentic_rl` | Tool-calling agent with RL | SQL / API navigation, multi-turn GRPO |
| `unsure` | Not sure yet | Q2 |

### Q1b — Structured output: which shape?

Telemetry: `question_id: research-q1b` (legacy: `discover-q1b`), field `intake_q1b_shape`

| Option ID | Option | Routes toward |
|---|---|---|
| `extraction` | Fixed JSON / fields from each input | `sft_cord_receipts` |
| `classification` | Pick one label or route from a small set | `sft_prompt_router` |
| `template` | Long-form answer following a template | `sft_cord_receipts` (text) |
| `unsure` | Not sure | Q2 |

### Q2 — What data do you have today?

Telemetry: `question_id: research-q2` (legacy: `discover-q2`), field `intake_q2_data`

| Option ID | Option | Routes toward |
|---|---|---|
| `labeled` | Labeled input → correct output (JSONL, etc.) | catalog match → Q3 or handoff |
| `raw_only` | Raw inputs only — no gold outputs yet | HF search gate + labeling plan |
| `preference_pairs` | Pairs where one answer is better | `dpo_style` |
| `scored_prompts` | Prompts + scorer 0–1 | `reasoning_rl` |
| `query_doc` | Query + document that should rank first | `embedding_support_search` |
| `exploring` | Not sure / still exploring | Q4 + HF gate |

### Q-eval — How will you know it worked?

Telemetry: `question_id: research-q-eval`, field `intake_q_eval`

| Option ID | Option | Eval plan |
|---|---|---|
| `exact_match` | Exact label or JSON match | per-field / exact-match (prompt router pattern) |
| `retrieval_metric` | Right document or passage ranks first | MRR / recall@k (embedding case study) |
| `win_rate` | Preferred answer beats baseline | DPO win-rate judge |
| `verifier_score` | Programmatic check (math, code, schema) | RFT grader / verifier |
| `human_rubric` | Human rubric or review | flag `eval_plan.gap`; outline rubric in research |
| `unsure` | Not sure yet | propose metric from cookbook entry |

### Q3 — How do you want to run training?

Telemetry: `question_id: research-q3` (legacy: `discover-q3`), field `intake_q3_path`

| Option ID | Option | Suggested path |
|---|---|---|
| `managed` | Simplest managed job | managed |
| `serverless` | My own loop, in-session eval, or RL on shared compute | serverless |
| `dedicated` | Full-parameter, embedding recipe, or reserved capacity | training_api_dedicated |
| `no_preference` | No preference | easiest compatible path |

Ask Q3 only when the user's goal makes two paths equally good. Otherwise omit it
and let Configure pick the easiest compatible path.

### Q4 — What have you tried so far?

Telemetry: `question_id: research-q4` (legacy: `discover-q4`), field `intake_q4_tried`

| Option ID | Option | Implication |
|---|---|---|
| `prompting` | Prompting only | SFT or DPO likely |
| `rag` | RAG over documents | embedding + maybe generation SFT |
| `other_vendor` | Another vendor notebook | map to closest catalog entry |
| `greenfield` | Nothing yet | starter entry + labeling path |

### HF gate — Search public Hugging Face Hub?

Before first Hub API call. Telemetry: `question_id: research-hf-gate` (legacy: `discover-hf-gate`)

| Option ID | Option |
|---|---|
| `hf_search_ok` | Yes — search public Hub (keywords only) |
| `hf_search_skip` | No — cookbook sources only |

### Handoff — Ready to plan training?

Telemetry: `question_id: research-handoff` (legacy: `discover-handoff`), field `handoff_choice`

| Option ID | Option | Action |
|---|---|---|
| `plan_configure` | Yes — show me the exact plan to approve | write handoff YAML → **configure** final plan in the same turn |
| `readme_first` | Show the README first | summarize; stay in research |
| `labeling_help` | Help building labeled data | stay in research; labeling schema |
| `defer` | Not now | `session_outcome: research_only` |

## Domain follow-ups

Telemetry: `question_id: research-domain` (legacy: `discover-domain`), field `domain_followup`

| Option ID | Domain signal | Ask |
|---|---|---|
| `sec_finance` | SEC, finance | Retrieval vs JSON signal? |
| `invoices_vision` | Invoices, forms | Images or text? |
| `tone_prefs` | Brand / tone | Preference pairs available? |
| `agentic_rl` | Tool-calling agent | SQL/API tools, multi-turn RL? |

## After the completion gate

1. Present **readiness package** (`output-template.md`).
2. Set `matched_case_study` or `cookbook_entry_path`, `match_confidence`.
3. Emit `research_recommendation_presented`.
4. Unless the user already said to proceed:
   Fire the **Handoff** AskQuestion (a plain "want me to plan this run?" is
   fine) and STOP.
5. Write the handoff block to `run.md` only after the user selects
   `plan_configure`. Other answers stay in Research and do not create a
   Configure handoff.

## Attribution

```bash
export FIREWORKS_SESSION_ID="$(python -c 'import uuid; print(uuid.uuid4())')"
export FIREWORKS_CLIENT_SOURCE="fireworks-training-skill/2.3.0"
```

Record `entry_skill: research` in the run manifest.
