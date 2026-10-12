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
| `multilora_fleet` | LoRA SFT | Many tenants/locales; per-tenant adapters on one deployment |
| `grpo_countdown` | GRPO / serverless RL | Single-turn verifiable GRPO (Countdown); dedicated as config swap |

## Tier 2 — Examples (`training/examples/`)

Minimal scripts; often the best match for novel domains.

| Path | Method | Task signal | Bundled data |
|---|---|---|---|
| `sft/train_sft.py`, `sft/train_sft_serverless.py` | SFT | Generic supervised fine-tuning (dedicated / serverless) | local JSONL |
| `dpo/train_dpo.py`, `dpo/train_dpo_serverless.py` | DPO | Preference tuning (dedicated / serverless) | local JSONL / prepared UltraFeedback |
| `embedding/train_embedding.py` | embedding | Query and passage pairs | local JSONL |
| `orpo/ifeval/train_ifeval_orpo.py` | ORPO | Instruction preferences | prepared IFEval data |
| `rl/deepmath/train_deepmath.py`, `rl/deepmath/train_deepmath_serverless.py` | RFT | Single-turn verifiable text (dedicated / serverless) | bundled DeepMath rows |
| `rl/visual_toolbench/train_visual_toolbench.py`, `..._serverless.py` | RFT | Multi-turn vision tool use (dedicated / serverless) | prepared VTB data |
| `rl/frozen_lake/train_frozen_lake.py` | RFT | Toy multi-turn environment (text or image) | prepared seeds |
| `rl/multihop_qa/train_multihop_qa_igpo.py` | IGPO | Search and multi-hop QA | prepared data |
| `rl/experiment/sao/qwen3_4b_sao.py` | SAO / PPO (experimental) | Token-level actor-critic math RL | built-in smoke rows or local JSONL |
| `harbor_rl/` | RFT | Coding agents and terminal tasks in sandboxes | prepared Harbor tasks |
| `distillation/gsm8k_privileged/train_gsm8k_privileged.py` | distillation | Privileged teacher context | prepared data |

`examples/archive/` holds deprecated examples; do not recommend them.

## Tier 3 — Recipes (`training/recipes/`)

Training API loops when the customer needs a custom harness.

| File | Method |
|---|---|
| `sft_loop.py` | SFT |
| `dpo_loop.py` | DPO |
| `orpo_loop.py` | ORPO |
| `rl_loop.py` | GRPO (sync RL) |
| `async_rl_loop.py` | GRPO (async) |
| `experiment/ppo_value_head_loop.py` | SAO / token-level PPO (experimental) |
| `distillation_loop.py` | Distillation / OPD |
| `sdft_loop.py` | SDFT on serverless (needs the `sdft` extra) |
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
implied_method: sft | dpo | orpo | rft | igpo | distillation | embedding
suggested_path: managed | serverless | training_api_dedicated
notebook: ...                   # when applicable
readme: ...
dataset_plan: ...
eval_plan: ...
```

Then hand off to **configure**. Research recommends a coarse path; Configure
always confirms the exact `workflow_path` and `execution_surface`. Do not create
jobs from research.
