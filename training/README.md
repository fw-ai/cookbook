# Fireworks Training Cookbook

Training loops for RL, preference optimization, SFT and embedding fine-tuning
on [Fireworks](https://fireworks.ai). Full docs: [Training API](https://docs.fireworks.ai/fine-tuning/training-api/introduction).

New here? The [Muse Glimmer mini cookbook](examples/muse_glimmer/README.md)
runs a free first fine-tune on preloaded example data in about 10 minutes.

## Layout

The tree is layered so each kind of change has one home:

```
recipes/              One file per training loop; fork to change the algorithm
recipes/experiment/   Loops not yet stable (serverless async RL, SAO/PPO)
utils/                Reusable pieces recipes assemble: config, data, losses,
                      rollout helpers, metrics (structured, per step)
examples/             Task code + one recipe call; no loop logic
  muse_glimmer/        Free first fine-tune walkthrough (Managed and serverless SFT)
  sft/ dpo/ orpo/ embedding/ distillation/
  rl/                 deepmath (text) · visual_toolbench (vision) ·
                      frozen_lake (multi-turn env) · multihop_qa (IGPO) ·
                      experiment/sao
  harbor_rl/          Sandboxed agentic RL: tasks/ × harnesses/ over tito/
  tools/              Standalone checkpoint, promotion and trainer utilities
  archive/            Deprecated examples, kept importable; do not extend
case-studies/         End-to-end notebooks (data -> eval -> train -> eval)
renderer/             Chat renderers and parity verifier
datasets/             Lightweight dataset validation and image geometry checks
tests/                Unit and smoke tests
```

Inside an example folder, `train_<task>.py` runs on a dedicated trainer +
deployment and `train_<task>_serverless.py` on the shared serverless pool with
the same task code. A new task belongs in `examples/`; a new objective in a
forked recipe; anything two recipes need in `utils/`.
Dataset checks shared with preflight validation belong in `datasets/`, which
stays independent of the renderer registry and training runtime.

## Recipes

| Recipe | File | What it does |
| --- | --- | --- |
| GRPO | `recipes/rl_loop.py` | Synchronous on-policy RL; trainer's built-in GRPO loss clipped against rollout logprobs. IGPO runs here via `prepare_batch_fn` |
| Async GRPO | `recipes/async_rl_loop.py` | Same update with rollout/train overlap and bounded staleness |
| Distillation / OPD | `recipes/distillation_loop.py` | On-policy distillation from a frozen teacher's token scores |
| SDFT | `recipes/sdft_loop.py` | Top-K self-distillation on serverless |
| DPO | `recipes/dpo_loop.py` | Preference optimization with cached reference logprobs |
| ORPO | `recipes/orpo_loop.py` | Odds-ratio preference optimization; no reference model |
| SFT | `recipes/sft_loop.py` | Response-only cross-entropy; `serverless=True` uses the shared pool |
| Embedding | `recipes/embedding_loop.py` | In-batch InfoNCE on `(query, positive)` pairs |
| SAO / PPO (experimental) | `recipes/experiment/ppo_value_head_loop.py` | Actor-critic with a value-head critic; see [`examples/rl/experiment/sao/`](examples/rl/experiment/sao/) |
| Serverless async RL (experimental) | `recipes/experiment/async_rl_loop_serverless.py` | Async GRPO on the shared LoRA pool |

Sync RL collects and trains one complete batch; async RL sends currently ready
groups together. Both wait for every backward before one optimizer update.
`prompt_groups_per_step` defines optimizer membership;
remove the old `pipeline_chunks_per_step` field. See the skills for
[losses](../skills/fireworks-training/references/rl-custom-loss.md) and
[metrics](../skills/fireworks-training/references/async-rl-metrics.md).

To train another RL objective, fork a recipe and replace its loss implementation;
see [`rl-custom-loss.md`](../skills/fireworks-training/references/rl-custom-loss.md).

## Case studies

Self-contained, runnable notebooks in [`case-studies/`](case-studies/) — each takes one real problem end-to-end (**build data -> evaluate the base model -> fine-tune -> evaluate again and measure the improvement**). These run on the **Fireworks Python SDK** (`fireworks-ai`) plus the [eval-protocol](https://github.com/eval-protocol/python-sdk) framework — no `firectl` required. Most need no cookbook training loop either; `embedding_support_search` is the exception, since it drives [`recipes/embedding_loop.py`](recipes/embedding_loop.py) directly.

| Case study | Technique | Dataset | "Customer problem" | Open |
| --- | --- | --- | --- | --- |
| [sft_prompt_router](case-studies/sft_prompt_router) | Text SFT / classification (Python SDK), dedicated + serverless | Prompt-Routing-Dataset | Route prompts to a small vs big model (multi-field classifier) | dedicated [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fw-ai/cookbook/blob/main/training/case-studies/sft_prompt_router/prompt_router_dedicated.ipynb) · serverless [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fw-ai/cookbook/blob/main/training/case-studies/sft_prompt_router/prompt_router_serverless.ipynb) |
| [sft_cord_receipts](case-studies/sft_cord_receipts) | Vision SFT (Python SDK) | CORD receipts | Teach the model a new structured-output task | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fw-ai/cookbook/blob/main/training/case-studies/sft_cord_receipts/cord_receipt_sft_sdk.ipynb) |
| [dpo_style](case-studies/dpo_style) | DPO (Python SDK) | HelpSteer3 | "Write the way we write" (style/quality preferences) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fw-ai/cookbook/blob/main/training/case-studies/dpo_style/dpo_helpsteer3_sdk.ipynb) |
| [reasoning_rl](case-studies/reasoning_rl) | GRPO via Training API (legacy managed RFT notebook is deprecated) | GSM8K | Improve step-by-step reasoning | [deepmath](examples/rl/deepmath/) |
| [embedding_support_search](case-studies/embedding_support_search) | Contrastive embedding fine-tuning (`embedding_loop` recipe) | Airbnb Help Center scenarios | Retrieve the help article that *governs* a situation, not the one that sounds like it | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fw-ai/cookbook/blob/main/training/case-studies/embedding_support_search/airbnb_policy_embedding.ipynb) |
| [agentic_rl_text2sql](case-studies/agentic_rl_text2sql) | Serverless tool-calling GRPO (Training API SDK) | BIRD text-to-SQL | Train a multi-turn tool-calling SQL agent; evaluate on unseen (out-of-distribution) databases | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fw-ai/cookbook/blob/main/training/case-studies/agentic_rl_text2sql/sql_agent_rl_loop.ipynb) |
| [sft_bfcl_toolcalls](case-studies/sft_bfcl_toolcalls) | SFT on the Training API: serverless, dedicated, and managed compared (Python SDK) | xLAM function-calling-60k + xlam-irrelevance-7.5k, scored on BFCL V4 non-live | Fix tool-call emission/abstention on a public benchmark, not a metric we invented | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fw-ai/cookbook/blob/main/training/case-studies/sft_bfcl_toolcalls/bfcl_toolcall_sft.ipynb) |
| [multilora_fleet](case-studies/multilora_fleet) | LoRA SFT + multi-LoRA serving (Python SDK) | MASSIVE (51 locales) | One deployment, a per-tenant specialist for every locale | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fw-ai/cookbook/blob/main/training/case-studies/multilora_fleet/multilora_fleet.ipynb) |
| [grpo_countdown](case-studies/grpo_countdown) | Serverless non-agentic GRPO (Training API SDK) | TinyZero Countdown | Own the single-turn GRPO loop on serverless; same algorithm as a dedicated config swap | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fw-ai/cookbook/blob/main/training/case-studies/grpo_countdown/countdown_grpo.ipynb) |

**Setup.** Follow the install instructions below, using `uv pip install -e '.[eval]'` to include the SDK and Eval Protocol, and create a `.env` with `FIREWORKS_API_KEY` (`FIREWORKS_ACCOUNT_ID` for the deploy/RFT cells). A couple of case studies add extras, noted in their own READMEs. Train/deploy cells provision real GPU and cost money. Each folder has its own README with the specifics.

## Getting started

```bash
git clone https://github.com/fw-ai/cookbook.git && cd cookbook/training
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .                      # pulls fireworks-ai[training]>=1.2.22,<2
export FIREWORKS_API_KEY=...             # or put it in training/.env
python -m training.examples.sft.train_sft --help
```

The `eval` and `dev` extras require Git and install Eval Protocol from a pinned
upstream commit. Its published 0.3.32 release requires an older LiteLLM than the
security fixes allow. The pinned source admits the tested LiteLLM 1.105.0 release;
replace it with a compatible PyPI release when one is available.

Each recipe's `Config` dataclass documents its fields. Leave `TrainerConfig()`
empty for SDK-selected defaults, or set `training_shape_id`; deployments resolve
their shape from the training shape. Weight sync, shapes, HSDP replicas,
reservations and distillation teachers are covered by the
[skills](../skills/GETTING-STARTED.md) and the
[configuration reference](https://docs.fireworks.ai/fine-tuning/training-api/cookbook/reference).

## Tests

```bash
uv pip install -e ".[dev]"
pytest tests/
```
