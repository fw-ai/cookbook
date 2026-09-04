# Research interview questions

Interview the user when the goal is vague, multiple cookbook entries could fit,
or the domain is novel. **Inspect** `cookbook-catalog.md` first, then ask only
what you still need.

Read `output-template.md` for turn shapes. Record telemetry after each answer —
see `../../references/telemetry-schema.md`. Show `../../references/telemetry-notice.md`
before the **first** interview question if welcome was skipped.

## Interview rules

- Use **AskQuestion** — **one question per turn**, then **STOP** and wait.
- Plain language; max **4 options** per question.
- Start every turn with the **Research** skill banner.
- Propose options grounded in catalog evidence when helpful.
- Do not ask what the opening message or repo already answers.
- After each answer: update journey block, append `intake_responses[]`, set
  `task_summary` once (≤200 chars, redacted).

## Completion gate (before readiness package)

Do **not** present the final recommendation until:

1. **Q1** (task shape) is answered or stated in the opening message, **and**
2. **Q2** (data / supervision) is answered, **and**
3. **Q-eval** (how to measure success) is answered, **and**
4. Any applicable follow-up (Q1b, domain, Q3) is answered.

**Exception:** user says "skip questions", "just pick one", or names a cookbook
slug. Set `completion_gate_bypassed: true`.

**Novel domains** (finance, legal, healthcare): always run **Q1 + Q2 + Q-eval +
one follow-up**.

## Question sequence

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
| `serverless` | Fast LoRA on shared compute | serverless |
| `dedicated` | Custom loop or embedding recipe | training_api_dedicated |
| `no_preference` | No preference | omit `suggested_path` |

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
| `plan_configure` | Yes — plan a run (don't start yet) | write handoff YAML → **configure** |
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
4. Write handoff block to `run.md`.
5. Fire **Handoff** AskQuestion.

## Attribution

```bash
export FIREWORKS_SESSION_ID="$(python -c 'import uuid; print(uuid.uuid4())')"
export FIREWORKS_CLIENT_SOURCE="fireworks-training-skill/2.2.0"
```

Record `entry_skill: research` in the run manifest.
