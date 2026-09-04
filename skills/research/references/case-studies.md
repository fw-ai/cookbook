# Case study catalog

Runnable end-to-end notebooks in `training/case-studies/`. Each README has an
**"Is this you?"** block — match customer intent against that text first.

| Slug | Technique | Is this you? (summary) | Notebook(s) | Path variants |
|---|---|---|---|---|
| `sft_prompt_router` | SFT / classification | End-to-end fine-tuning on a gradeable classification task; compare dedicated vs serverless | `prompt_router_dedicated.ipynb`, `prompt_router_serverless.ipynb` | dedicated + serverless |
| `sft_cord_receipts` | Vision SFT | One right output shape (JSON, tags, codes) from examples; invoice/OCR/form extraction | `cord_receipt_sft_sdk.ipynb` | managed SDK |
| `dpo_style` | DPO | Accurate but wrong tone; easier to rank two answers than write the ideal one | `dpo_helpsteer3_sdk.ipynb` | managed SDK |
| `reasoning_rl` | GRPO / managed RFT | Objectively checkable answers; grader exists but no gold worked solutions | `rft_grpo_math.ipynb` | managed RFT |
| `embedding_support_search` | Contrastive embedding | RAG returns adjacent but wrong article; policy structure not in base model | `airbnb_policy_embedding.ipynb` | Training API `embedding_loop` |
| `agentic_rl_text2sql` | GRPO / serverless RL | Tool-calling agent (SQL, APIs); multi-turn rollouts with verifiable rewards | `sql_agent_rl_loop.ipynb` | serverless Training API |

Cookbook table: [`training/README.md`](https://github.com/fw-ai/cookbook/blob/main/training/README.md#case-studies).

## Implied method per slug

| Slug | Implied method | Typical path |
|---|---|---|
| `sft_prompt_router` | SFT | managed or serverless (both notebooks) |
| `sft_cord_receipts` | SFT | managed SDK |
| `dpo_style` | DPO | managed SDK |
| `reasoning_rl` | RFT (GRPO) | managed RFT |
| `embedding_support_search` | embedding fine-tune | Training API dedicated |
| `agentic_rl_text2sql` | RL (GRPO) | serverless Training API |

## Match rules

1. Read the case study README **"Is this you?"** paragraph before recommending.
2. Prefer the closest **task shape**, not the buzzword (e.g. "routing" →
   `sft_prompt_router`, not embedding).
3. When two studies fit, pick the one with the closer data modality (vision vs
   text vs preferences vs verifiable reward).
4. When confidence is low or the domain is novel (finance, legal, etc.), use
   `references/intake-questions.md` — run the **completion gate** (Q1 + Q2 +
   follow-up), one question per turn.

## Handoff block for configure

After a match, record in the run manifest and tell the user:

```yaml
entry_skill: research
case_study: <slug>
implied_method: sft | dpo | orpo | rft | embedding
suggested_path: managed | serverless | training_api_dedicated
execution_surface: firectl | sdk | training_api   # set when known; else configure runs Q-path
notebook: training/case-studies/<slug>/<notebook>.ipynb
readme: training/case-studies/<slug>/README.md
```

Then hand off to the **configure** skill. Do not create jobs from research.
