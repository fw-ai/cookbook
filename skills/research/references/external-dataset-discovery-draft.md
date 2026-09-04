# Draft: external dataset discovery (Hugging Face)

**Status:** draft for review — not wired into `SKILL.md` yet.

## Problem

Discover today matches **six** curated case studies. That is enough to route
**technique** (SFT vs DPO vs RFT vs embedding) and **path** (managed vs
serverless vs recipe), but it is thin for **domain** discovery:

| Layer | What it answers | Count today |
|---|---|---|
| Technique + path | "How should I train?" | 6 case studies |
| Runnable pattern | "What notebook do I fork?" | 6 notebooks |
| Domain + data | "What dataset fits my task?" | **mostly manual** |

Users who say "I want to fine-tune on medical Q&A" or "receipt parsing" often
need **two** things:

1. **Closest cookbook pattern** (still one of six slugs).
2. **Candidate public datasets** to bootstrap labeling or a first run.

Hugging Face Hub is the obvious external corpus. The Hub search API is public,
read-only, and needs no customer data upload.

## Proposed model: layered discover

```text
User task
   │
   ├─► Layer A — case study match (existing)
   │      technique, notebook, suggested_path
   │
   ├─► Layer B — cookbook examples index (new, optional)
   │      training/examples/* when slug is close but not exact
   │
   └─► Layer C — HF dataset search (new, gated)
          3–5 public datasets; schema fit notes; no auto-download
```

**Discover still does not train.** It may recommend `openai/gsm8k` as a
bootstrap dataset while handing off `reasoning_rl` as the pattern.

## When to run HF search

| Trigger | Example |
|---|---|
| Q2 = `exploring` or `raw_only` | User has no JSONL yet |
| Q2 = `labeled` but no local path | "I haven't picked a dataset" |
| Novel domain after completion gate | Finance, legal, niche vertical |
| User asks explicitly | "Find a dataset on Hugging Face" |
| Low case-study confidence | `match_confidence: low` |

**Do not** run HF search when:

- User already named a local file or private dataset.
- User opted out of external search (see privacy gate).
- Match is high-confidence and user has data (`labeled` + path).

## Privacy gate (required)

Before the first Hub API call, show a one-line notice (reuse
`telemetry-notice.md` tone):

> I can search the **public** Hugging Face Hub for datasets that match your
> task. I'll send **search keywords only** (not your files or examples). Skip?

AskQuestion:

| Option ID | Label |
|---|---|
| `hf_search_ok` | Yes — search public Hub |
| `hf_search_skip` | No — stay with cookbook examples only |

Telemetry: `question_id: discover-hf-gate`, `hf_search_consent`.

Search query construction:

- Build from `task_summary` + Q1/Q2 option IDs — **never** paste user file
  contents or row samples into the query.
- Cap query at ~80 chars; strip paths, emails, account IDs.

## HF search mechanics

**API (no auth for public datasets):**

```http
GET https://huggingface.co/api/datasets?search=<query>&limit=8
```

Optional enrichment per hit:

```http
GET https://huggingface.co/api/datasets/<repo_id>
```

**Helper script** (for agents): `scripts/hf_dataset_search.py`

```bash
python skills/discover/scripts/hf_dataset_search.py \
  --query "receipt ocr json" \
  --method sft \
  --limit 5
```

Output: JSON with `id`, `downloads`, `tags`, `card_summary`, `schema_hint`,
`fit_notes`, `hub_url`.

### Ranking heuristics (v0)

1. Prefer datasets with **downloads > 100** (signal, not quality guarantee).
2. Prefer tags matching implied method: `task_categories:text-generation`,
   `task_categories:question-answering`, `size_categories:…`.
3. Deprioritize gated repos unless user has `HF_TOKEN` (mention in fit note).
4. Flag **license** field when present (agent surfaces, does not legal-review).
5. Never auto-download; link `https://huggingface.co/datasets/<id>` only.

### Schema fit notes (agent-authored, template)

After API results, agent checks dataset card / `features` when available:

| Implied method | Look for |
|---|---|
| SFT | `messages`, `instruction`/`output`, `text` pairs |
| DPO | `chosen`/`rejected` or preference columns |
| RFT | prompt + verifiable answer field |
| Embedding | `query`/`passage` or `anchor`/`positive` |

If schema does not match, say so and suggest a **conversion step** in discover
(stay read-only) or hand off to configure for `prepare_data` scripts.

## UX: recommendation shape (extended)

After Layer A match, optionally append:

```markdown
### Public datasets to bootstrap (Hugging Face)

These are **starting points**, not endorsements. Check license and schema before training.

| Dataset | Downloads | Why it might fit | Caveat |
|---|---|---|---|
| `openai/gsm8k` | 1.1M | Grade-school math Q→A; matches verifiable reasoning | Needs RFT grader, not plain SFT |
| … | | | |

**Pattern to fork:** `reasoning_rl` → `training/case-studies/reasoning_rl/`
**Next:** configure will map columns → Fireworks JSONL and pick a base model.
```

## Handoff block extension

```yaml
entry_skill: discover
case_study: reasoning_rl
implied_method: rft
suggested_path: managed
hf_search_consent: true
hf_search_query: "grade school math word problems"
hf_dataset_candidates:
  - id: openai/gsm8k
    rank: 1
    fit: high
    caveat: "verifiable numeric answers; use RFT not SFT"
external_data_source: huggingface_hub   # vs local | unknown
```

Configure owns: download, column mapping, train/val split, upload to Fireworks.

## Is six case studies enough?

**For routing technique:** six is a reasonable v1 **menu** — each slug is a
distinct training shape (vision SFT, DPO, RFT, embedding, multi-LoRA, router).

**For domain discovery:** six is **not** enough. Users will ask about domains
outside the menu every session. The fix is not "ship 50 case studies" first;
it is **splitting the problem**:

| User question | Right surface |
|---|---|
| "How do I train for X?" | Case study slug (6 → grow slowly) |
| "What public data exists for X?" | HF search (unbounded) |
| "Show me a minimal script for X?" | `training/examples/` index (15+ recipes) |

**Near-term catalog growth (curated):**

- Add **example index** to discover (`references/examples-index.md`) — maps task
  signals to `training/examples/rl/coding_agent`, `deepmath`, `multihop_qa`, etc.
- Add 2–3 **vertical micro-studies** only when we have a validated notebook
  (e.g. `sft_text2sql` from `examples/sft/text2sql_dataset.jsonl`).
- HF search covers long-tail domain without maintaining 50 notebooks.

## Repurpose map

| Existing piece | Reuse for HF |
|---|---|
| `intake_q2_data = exploring` | Triggers HF branch |
| `task_summary` (≤200 chars, redacted) | HF query input |
| `telemetry-notice.md` | Same opt-out pattern as journey telemetry |
| `match_confidence: low` | Triggers HF + examples index |
| `swegym_data.py`, `prepare_data.py` in examples | Configure references for HF→JSONL |
| Case study `README` "Is this you?" | Still primary; HF supplements data |

## Test scenarios (manual)

Run the helper and confirm sensible hits:

| User task | Expected slug | HF query | Top hit (Sep 2026) |
|---|---|---|---|
| Receipt JSON extraction | `sft_cord_receipts` | `receipt ocr json` | `kucingcoder/raw-ocr-receipt-to-json` |
| Math reasoning | `reasoning_rl` | `gsm8k` | `openai/gsm8k` |
| Tone / preferences | `dpo_style` | `helpsteer preference` | `RLHFlow/Helpsteer-preference-standard` |
| RAG policy retrieval | `embedding_support_search` | `query passage retrieval` | (noisy — agent must filter) |

## Implementation phases

| Phase | Scope |
|---|---|
| **0 (this draft)** | Design + `hf_dataset_search.py` + manual agent test |
| **1** | Add `references/examples-index.md`; extend output template |
| **2** | Wire HF gate + branch into `SKILL.md` + intake Q2 follow-up |
| **3** | Telemetry: `hf_search_*` fields in `telemetry-schema.md` |
| **4** | Configure: optional `hf_dataset_id` → download helper |

## Open questions

1. Should Fireworks-first datasets (`fireworks-ai/*`) rank above Hub noise?
2. Gated datasets: discover links only, or check for `HF_TOKEN`?
3. Vision datasets: Hub tag filtering is weaker — need modality in Q1b?
4. Rate limits: cache last query per session in manifest to avoid repeat calls?
