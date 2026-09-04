# Cookbook catalog

Full index for **research** — scan all tiers before recommending. Case studies
are the best end-to-end stories; examples and recipes are valid matches when
the task shape fits but no case study does.

Cookbook root: `training/`. Live table: [`training/README.md`](https://github.com/fw-ai/cookbook/blob/main/training/README.md).

## Tier 1 — Case studies (`training/case-studies/`)

End-to-end notebooks with eval. See `case-studies.md` for slugs and match rules.

| Slug | Method | Task signal |
|---|---|---|
| `sft_prompt_router` | SFT | Classification / routing JSON |
| `sft_cord_receipts` | SFT | Structured extraction (vision) |
| `dpo_style` | DPO | Tone / preference pairs |
| `reasoning_rl` | RFT | Verifiable reasoning |
| `embedding_support_search` | embedding | RAG retrieval / policy match |
| `agentic_rl_text2sql` | GRPO / serverless RL | Tool-calling SQL agent with multi-turn rollouts |

## Tier 2 — Examples (`training/examples/`)

Minimal scripts; often the best match for novel domains.

| Path | Method | Task signal | Bundled data |
|---|---|---|---|
| `sft/train_sft.py` | SFT | Generic SFT loop | `food_reasoning.jsonl`, `text2sql_dataset.jsonl` |
| `dpo/train_dpo.py` | DPO | Preference tuning | — |
| `embedding/train_embedding.py` | embedding | Query–passage pairs | `retrieval_pairs.jsonl` |
| `orpo/ifeval/` | ORPO | Instruction prefs | `dataset.jsonl` |
| `serverless_rl/` | RFT | Serverless RL smoke | `countdown_train.jsonl` |
| `rl/deepmath/` | RFT | Math verify | `dataset.jsonl` |
| `rl/coding_agent/` | RFT | SWE / coding agent | HF via `swegym_data.py` |
| `rl/eval_protocol_chat/` | RFT | Eval-protocol chat RL | `train.jsonl` |
| `rl/frozen_lake/` | RFT | Toy RL env | `seeds.jsonl` |
| `rl/multi_turn_message_in/` | RFT | Multi-turn RL | prepare scripts |
| `multihop_qa/` | RFT (IGPO) | Search + QA | prepare scripts |
| `distillation/gsm8k_privileged/` | distillation | Teacher–student | prepare scripts |
| `distillation/routed_mopd/` | distillation | Multi-teacher | — |

## Tier 3 — Recipes (`training/recipes/`)

Training API loops when the customer needs a custom harness.

| File | Method |
|---|---|
| `sft_loop.py` | SFT |
| `dpo_loop.py` | DPO |
| `orpo_loop.py` | ORPO |
| `rl_loop.py` | GRPO (sync RL) |
| `async_rl_loop.py` | GRPO (async) |
| `igpo_loop.py` | IGPO multi-turn |
| `distillation_loop.py` | Distillation / OPD |
| `embedding_loop.py` | Embedding / contrastive |

## Match priority

1. Task shape + supervision signal (from interview).
2. Closest tier with a runnable path (prefer case study on tie).
3. Data availability (bundled > user local > HF > labeling plan).

## Handoff block for configure

After interview completion and user approval:

```yaml
entry_skill: research
case_study: <slug>              # omit if tier = example | recipe
cookbook_entry_tier: case_study | example | recipe
cookbook_entry_path: training/...
implied_method: sft | dpo | orpo | rft | embedding
suggested_path: managed | serverless | training_api_dedicated
execution_surface: firectl | sdk | training_api
notebook: ...                   # when applicable
readme: ...
dataset_plan: ...
eval_plan: ...
```

Then hand off to **configure**. Do not create jobs from research.
