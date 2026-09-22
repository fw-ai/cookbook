# Case study catalog

Runnable end-to-end notebooks in `training/case-studies/`. Each README has an
**"Is this you?"** block — match customer intent against that text first.

| Slug | Technique | Is this you? (summary) | Notebook(s) | Path variants |
|---|---|---|---|---|
| `sft_prompt_router` | SFT / classification | End-to-end fine-tuning on a gradeable classification task | `prompt_router_dedicated.ipynb`, `prompt_router_serverless.ipynb` | managed SDK, serverless |
| `sft_cord_receipts` | Vision SFT | One right output shape (JSON, tags, codes) from examples; invoice/OCR/form extraction | `cord_receipt_sft_sdk.ipynb` | managed SDK |
| `dpo_style` | DPO | Accurate but wrong tone; easier to rank two answers than write the ideal one | `dpo_helpsteer3_sdk.ipynb` | managed SDK |
| `reasoning_rl` | GRPO | Objectively checkable answers; grader exists but no gold worked solutions | `rft_grpo_math.ipynb` (legacy managed RFT) | New runs: Training API `training/examples/rl/deepmath/` |
| `embedding_support_search` | Contrastive embedding | RAG returns adjacent but wrong article; policy structure not in base model | `airbnb_policy_embedding.ipynb` | Training API `embedding_loop` |
| `agentic_rl_text2sql` | GRPO / serverless RL | Tool-calling agent (SQL, APIs); multi-turn rollouts with verifiable rewards | `sql_agent_rl_loop.ipynb` | serverless Training API |
| `multilora_fleet` | LoRA SFT / multi-LoRA serving | Many tenants or locales sharing one base model; per-tenant adapters served from a single deployment | `multilora_fleet.ipynb` | managed SDK |

Cookbook table: [`training/README.md`](https://github.com/fw-ai/cookbook/blob/main/training/README.md#case-studies).

## Implied method per slug

| Slug | Implied method | Typical path |
|---|---|---|
| `sft_prompt_router` | SFT | managed SDK |
| `sft_cord_receipts` | SFT | managed SDK |
| `dpo_style` | DPO | managed SDK |
| `reasoning_rl` | RL (GRPO) | Training API `training/examples/rl/deepmath/`; `rft_grpo_math.ipynb` is legacy managed RFT |
| `embedding_support_search` | embedding fine-tune | Training API dedicated |
| `agentic_rl_text2sql` | RL (GRPO) | serverless Training API |
| `multilora_fleet` | SFT (LoRA) | managed SDK |

## Match rules

1. Read the case study README **"Is this you?"** paragraph before recommending.
2. Prefer the closest **task shape**, not the buzzword (e.g. "routing" →
   `sft_prompt_router`, not embedding).
3. When two studies fit, pick the one with the closer data modality (vision vs
   text vs preferences vs verifiable reward).
4. When confidence is low or the domain is novel (finance, legal, etc.), use
   `interview-questions.md` and run the **completion gate** (Q1 + Q2 +
   follow-up), one question per turn.

## Handoff block for configure

After a match, record in the run manifest and tell the user:

```yaml
entry_skill: research
case_study: <slug>
cookbook_entry_tier: case_study
cookbook_entry_path: training/case-studies/<slug>
implied_method: sft | dpo | orpo | rft | igpo | distillation | embedding
suggested_path: managed | serverless | training_api_dedicated
notebook: training/case-studies/<slug>/<notebook>.ipynb
readme: training/case-studies/<slug>/README.md
dataset_plan: ...
eval_plan: ...
```

Then hand off to **configure**. `suggested_path` is a recommendation, not the
final `workflow_path`; Configure always confirms its exact Q-path option. Do not
create jobs from research.
