# Fireworks Training Cookbook

Training loops for RL, preference optimization, SFT and embedding fine-tuning
on [Fireworks](https://fireworks.ai). Full docs: [Training API](https://docs.fireworks.ai/fine-tuning/training-api/introduction).

New here? The [Muse Glimmer mini cookbook](examples/muse_glimmer/README.md)
runs a free first fine-tune on preloaded example data in about 10 minutes.

## Versioned installations

Applications can depend on `fireworks-training-cookbook==<version>` from
[PyPI](https://pypi.org/project/fireworks-training-cookbook/) instead of a Git
URL once the first release has been published. Imports remain `training.*`.
GitHub tag `cookbook-v2026.10.26.1` maps to version `2026.10.26.1` (an
illustrative version; select an available version from PyPI).

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
| SDFT | [Customer guide](docs/sdft.md) | Maintained in the external Fireworks Tinker-Cookbook fork |
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
Optional reference KL uses [`utils/rl/reference.py`](utils/rl/reference.py):
LoRA scores the frozen base with adapters disabled on its trainer;
full-parameter tuning scores a separate frozen base-model inference deployment.
Reference setup and scoring stay outside both stock loops.

## Case studies

Self-contained, runnable notebooks in [`case-studies/`](case-studies/) — each takes one real problem end-to-end (**build data -> evaluate the base model -> fine-tune -> evaluate again and measure the improvement**). These run on the **Fireworks Python SDK** (`fireworks-ai`) plus the [eval-protocol](https://github.com/eval-protocol/python-sdk) framework — no `firectl` required. Most need no cookbook training loop either; `embedding_support_search` is the exception, since it drives [`recipes/embedding_loop.py`](recipes/embedding_loop.py) directly.

| Case study | Technique | Dataset | "Customer problem" | Open |
| --- | --- | --- | --- | --- |
| [sft_prompt_router](case-studies/sft_prompt_router) | Text SFT / classification (Python SDK), dedicated + serverless | Prompt-Routing-Dataset | Route prompts to a small vs big model (multi-field classifier) | dedicated [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fw-ai-external/cookbook/blob/main/training/case-studies/sft_prompt_router/prompt_router_dedicated.ipynb) · serverless [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fw-ai-external/cookbook/blob/main/training/case-studies/sft_prompt_router/prompt_router_serverless.ipynb) |
| [sft_cord_receipts](case-studies/sft_cord_receipts) | Vision SFT (Python SDK) | CORD receipts | Teach the model a new structured-output task | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fw-ai-external/cookbook/blob/main/training/case-studies/sft_cord_receipts/cord_receipt_sft_sdk.ipynb) |
| [dpo_style](case-studies/dpo_style) | DPO (Python SDK) | HelpSteer3 | "Write the way we write" (style/quality preferences) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fw-ai-external/cookbook/blob/main/training/case-studies/dpo_style/dpo_helpsteer3_sdk.ipynb) |
| [reasoning_rl](case-studies/reasoning_rl) | GRPO via Training API (the legacy managed RFT notebook no longer runs) | GSM8K | Improve step-by-step reasoning | [deepmath](examples/rl/deepmath/) |
| [embedding_support_search](case-studies/embedding_support_search) | Contrastive embedding fine-tuning (`embedding_loop` recipe) | Airbnb Help Center scenarios | Retrieve the help article that *governs* a situation, not the one that sounds like it | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fw-ai-external/cookbook/blob/main/training/case-studies/embedding_support_search/airbnb_policy_embedding.ipynb) |
| [agentic_rl_text2sql](case-studies/agentic_rl_text2sql) | Serverless tool-calling GRPO (Training API SDK) | BIRD text-to-SQL | Train a multi-turn tool-calling SQL agent; evaluate on unseen (out-of-distribution) databases | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fw-ai-external/cookbook/blob/main/training/case-studies/agentic_rl_text2sql/sql_agent_rl_loop.ipynb) |
| [sft_bfcl_toolcalls](case-studies/sft_bfcl_toolcalls) | SFT on the Training API: serverless, dedicated, and managed compared (Python SDK) | xLAM function-calling-60k + xlam-irrelevance-7.5k, scored on BFCL V4 non-live | Fix tool-call emission/abstention on a public benchmark, not a metric we invented | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fw-ai-external/cookbook/blob/main/training/case-studies/sft_bfcl_toolcalls/bfcl_toolcall_sft.ipynb) |
| [multilora_fleet](case-studies/multilora_fleet) | LoRA SFT + multi-LoRA serving (Python SDK) | MASSIVE (51 locales) | One deployment, a per-tenant specialist for every locale | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fw-ai-external/cookbook/blob/main/training/case-studies/multilora_fleet/multilora_fleet.ipynb) |
| [grpo_countdown](case-studies/grpo_countdown) | Serverless non-agentic GRPO (Training API SDK) | TinyZero Countdown | Own the single-turn GRPO loop on serverless; same algorithm as a dedicated config swap | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fw-ai-external/cookbook/blob/main/training/case-studies/grpo_countdown/countdown_grpo.ipynb) |

**Setup.** Follow the install instructions below, using `uv pip install -e '.[eval]'` to include the SDK and Eval Protocol, and create a `.env` with `FIREWORKS_API_KEY` (`FIREWORKS_ACCOUNT_ID` for the deploy/RFT cells). A couple of case studies add extras, noted in their own READMEs. Train/deploy cells provision real GPU and cost money. Each folder has its own README with the specifics.

## Open Weight Room

[Open Weight Room case studies](case-studies/README.md) explore changes in model behavior through small adapters: concise answers, editing, output constraints, writing style, visual design, and clarification before acting. Each study includes recorded evidence and explains its evaluation limits. Training walkthroughs and recorded-results analyses are labeled separately.

| Study | Focus | Entry point |
| --- | --- | --- |
| Caveman | Concision with factual checks | [Read the walkthrough](https://nbviewer.org/github/fw-ai-external/cookbook/blob/68750d4980cc8798343c67a49ec5c5b92093499f/training/case-studies/sft_caveman/caveman_sft.ipynb) · [Source](case-studies/sft_caveman/caveman_sft.ipynb) |
| Humanizer | Preservation and restraint when editing | [SFT walkthrough](case-studies/sft_humanizer/humanizer_walkthrough.ipynb) |
| E-vasion | Useful answers without the letter e | [SFT + GRPO study](case-studies/lipogram_data/lipogram_demo.ipynb) |
| Anti-slop | Preference optimization and reward tradeoffs | [DPO walkthrough](case-studies/dpo_antislop/dpo_glm53_flash.ipynb) |
| Taste | Visual preference versus functionality | [Recorded comparison](case-studies/ui_design/taste_walkthrough.ipynb) |
| Ask Before Acting | Clarification in simulated tasks | [RL walkthrough](case-studies/rl_ask_before_acting/ask_before_acting_walkthrough.ipynb) |

## Getting started

Use Python 3.11 or newer. For released `training.*` recipes and utilities:

```bash
python -m pip install --upgrade fireworks-training-cookbook
```

For the runnable notebooks and examples below, clone the source and install it
in editable mode:

```bash
git clone https://github.com/fw-ai-external/cookbook.git && cd cookbook/training
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .                      # installs the SDK and recipe dependencies
export FIREWORKS_API_KEY=...             # or put it in training/.env
python -m training.examples.sft.train_sft --help
```

The `eval` and `dev` extras install `eval-protocol==0.3.33` from PyPI,
which supports the tested LiteLLM 1.105.0 release.

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


### SDFT

SDFT is maintained in the [Fireworks Tinker-Cookbook fork](https://github.com/fw-ai-external/tinker-cookbook).
See the [customer guide](docs/sdft.md) for where to start.

## Publishing releases

Releases are managed by the maintainers' private CI after the reviewed promotion
PR merges. Promotion commits the calendar version in `pyproject.toml` and the
root package entry in `uv.lock` before review. For example,
`cookbook-v2026.1.2.1` publishes package version `2026.1.2.1`.

The publisher builds the wheel from the source distribution, validates both
artifacts, publishes to PyPI, and verifies the installed version. This repository
has no publishing workflow or stored release credentials. Contact a maintainer
to retry a failed release.

The `eval` extra pins `eval-protocol==0.3.33`. Release builds reject direct URL
dependencies in all extras so Git pins cannot break PyPI publication.
