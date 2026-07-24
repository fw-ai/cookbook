# Fireworks Training Cookbook

Ready-to-run training recipes for reinforcement learning, preference optimization, supervised fine-tuning, and embedding (retrieval) fine-tuning on [Fireworks](https://fireworks.ai).
Each recipe is a single Python file you can fork and customize.

> **Full documentation**: For detailed guides on each recipe, configuration reference, and the underlying SDK, see the [Training API documentation](https://docs.fireworks.ai/fine-tuning/training-api/introduction).

## Recipes

| Recipe | File | Description |
| --- | --- | --- |
| GRPO | `recipes/rl_loop.py` | Opinionated synchronous RL using group-normalized advantages and a direct client-side GRPO loss. |
| Async GRPO | `recipes/async_rl_loop.py` | The same client-side GRPO update with rollout/train overlap and bounded off-policy staleness. |
| IGPO (multi-turn turn-level Information Gain) | `recipes/igpo_loop.py` | GRPO + per-turn IG rewards for agent trajectories (Wang et al., ICLR 2026). |
| Distillation / OPD | `recipes/distillation_loop.py` | Sampled-token on-policy distillation. The student rolls out on policy, one or more teachers score those same tokens, and training uses the server-side importance-sampling loss. |
| DPO | `recipes/dpo_loop.py` | Direct preference optimization with cached reference logprobs. |
| ORPO | `recipes/orpo_loop.py` | Odds-ratio preference optimization -- no reference model needed. |
| SFT | `recipes/sft_loop.py` | Supervised fine-tuning with response-only cross-entropy loss. |
| Embedding | `recipes/embedding_loop.py` | Retrieval/embedding fine-tuning with bidirectional in-batch InfoNCE on `(query, positive)` pairs. Three interchangeable trainer modes: `embedding`, `cos_similarity_matrix`, `contrastive_loss`. |

## Case studies

Self-contained, runnable notebooks in [`case-studies/`](case-studies/) — each takes one real problem end-to-end (**build data -> evaluate the base model -> fine-tune -> evaluate again and measure the improvement**). These run on the **Fireworks Python SDK** (`fireworks-ai`) plus the [eval-protocol](https://github.com/eval-protocol/python-sdk) framework — no `firectl` and no cookbook training loop required.

| Case study | Technique | Dataset | "Customer problem" | Open |
| --- | --- | --- | --- | --- |
| [sft_prompt_router](case-studies/sft_prompt_router) | Text SFT / classification (Python SDK), dedicated + serverless | Prompt-Routing-Dataset | Route prompts to a small vs big model (multi-field classifier) | dedicated [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fw-ai/cookbook/blob/main/training/case-studies/sft_prompt_router/prompt_router_dedicated.ipynb) · serverless [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fw-ai/cookbook/blob/main/training/case-studies/sft_prompt_router/prompt_router_serverless.ipynb) |
| [sft_cord_receipts](case-studies/sft_cord_receipts) | Vision SFT (Python SDK) | CORD receipts | Teach the model a new structured-output task | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fw-ai/cookbook/blob/main/training/case-studies/sft_cord_receipts/cord_receipt_sft_sdk.ipynb) |
| [dpo_style](case-studies/dpo_style) | DPO (Python SDK) | HelpSteer3 | "Write the way we write" (style/quality preferences) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fw-ai/cookbook/blob/main/training/case-studies/dpo_style/dpo_helpsteer3_sdk.ipynb) |
| [reasoning_rl](case-studies/reasoning_rl) | GRPO via managed RFT (Python SDK) | GSM8K | Improve step-by-step reasoning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fw-ai/cookbook/blob/main/training/case-studies/reasoning_rl/rft_grpo_math.ipynb) |

**Setup.** These are independent of the cookbook install above — just `pip install fireworks-ai eval-protocol` (a couple add extras, noted in their own READMEs) and a `.env` with `FIREWORKS_API_KEY` (`FIREWORKS_ACCOUNT_ID` for the deploy/RFT cells). Train/deploy cells provision real GPU and cost money. Each folder has its own README with the specifics.

## Getting started

### 1. Clone and install

```bash
git clone https://github.com/fw-ai/cookbook.git
cd cookbook/training

# Install uv (skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate a virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install this package in editable mode.
# Dependencies, including the Fireworks training SDK, are pulled from pyproject.toml.
uv pip install --pre -e .

# If you skip `--pre`, pip may resolve to the stable `0.x` line,
# which does not include the managed Fireworks training SDK clients.
```

### 2. Set your credentials

Create a `.env` file in the `training/` directory (picked up automatically via `python-dotenv`):

```bash
FIREWORKS_API_KEY="your-api-key"
```

Or export it directly:

```bash
export FIREWORKS_API_KEY="..."
```

### 3. Configure your recipe

Each recipe has a `Config` dataclass at the top of the file. Open the recipe you want to run and edit the `if __name__ == "__main__"` block at the bottom. Here are the fields you **must** set:

**All recipes:**

| Field | What to set |
| --- | --- |
| `dataset` | Path to your JSONL training data |
| `base_model` | Fireworks model ID (e.g. `"accounts/fireworks/models/qwen3-8b"`) |
| `max_seq_len` | Max token length for training examples |
| `trainer` | Optional trainer job and shape overrides. Leave as `TrainerConfig()` for SDK-managed defaults, or set `training_shape_id` when you need a specific shape. |

**SFT** (`recipes/sft_loop.py`) -- also requires:

| Field | What to set |
| --- | --- |
| `tokenizer_model` | HuggingFace model name matching your base model (e.g. `"Qwen/Qwen3-8B"`) |

**Embedding** (`recipes/embedding_loop.py`) -- also requires:

| Field | What to set |
| --- | --- |
| `tokenizer_model` | HuggingFace model name matching your base model (e.g. `"Qwen/Qwen3-Embedding-8B"`) |
| `output_mode` | `embedding` (default), `cos_similarity_matrix`, or `contrastive_loss` -- three points on the client/server compute split for the same bidirectional InfoNCE loss |

Dataset rows are `(query, positive)` pairs, one JSON object per line:
`{"query": "...", "positive": "..."}`. Training uses in-batch negatives, so
`batch_size` must be `>= 2` and the dataset must contain at least `batch_size`
pairs (it fails loudly otherwise). `cos_similarity_matrix` additionally requires
the whole batch to fit in a single request (single-GPU / DP=1 trainer).

This recipe truncates each side with `max_query_len` / `max_doc_len` rather than
a single `max_seq_len`; the trainer context is provisioned to fit the larger of
the two.

**RL** (`recipes/rl_loop.py`) -- also requires:

| Field | What to set |
| --- | --- |
| `deployment` | `DeployConfig(tokenizer_model="Qwen/Qwen3-8B")` for inference rollouts |
| `sampler_refresh_interval` | Refresh the sampler from trainer-saved weights every N optimizer steps. |

**Distillation / OPD** (`recipes/distillation_loop.py`) -- also requires:

| Field | What to set |
| --- | --- |
| `teacher_model` | Fireworks teacher model or deployment ID using the same tokenizer as the student |
| `deployment` | `DeployConfig(tokenizer_model="Qwen/Qwen3-8B")` for student rollouts |

If `teacher_model` is a base model resource such as
`accounts/fireworks/models/qwen3p5-9b`, the distillation recipe creates or reuses a
separate teacher inference deployment for scoring during provisioning. Set
`teacher_deployment_id` to pin that deployment name. If `teacher_model` is
already a deployment/deployed model resource, the recipe uses it directly.

Distillation rows can include `teacher_messages` for privileged teacher context. The
student samples from `messages`; the frozen teacher scores the sampled response
under `teacher_messages`. If `teacher_messages` is absent, the recipe falls
back to the student prompt. The aliases `privileged_messages` and
`teacher_prompt_messages` are also accepted.

For multi-teacher SDFT (`distill_mode="topk_forward_kl"`), every teacher scores
the same sampled response and the recipe blends teacher top-K probability mass
over the union of token ids. Set `TeacherConfig.blend_weight` to weight a
teacher in that blend; omitted weights default to `1.0`. Sampled reverse-KL OPD
still routes each row to one teacher and does not blend.

The recipes create SDK-managed training and sampling clients directly.
They do not require explicit trainer-job, deployment, or sampler setup.
When `training_shape_id` is not set, the SDK selects validated runtime
defaults. Explicit `training_shape_id`, `reference_training_shape_id`, and
deployment-shape overrides still take precedence.

To launch trainers with replicated HSDP, set the run-level replica count on
`TrainerConfig`; it is not part of the validated training shape:

```python
config = rl_loop.Config(
    base_model="accounts/fireworks/models/qwen3-8b",
    trainer=TrainerConfig(
        training_shape_id="accounts/fireworks/trainingShapes/your-shape",
        replica_count=2,
    ),
    deployment=DeployConfig(tokenizer_model="Qwen/Qwen3-8B"),
)
```

The example entrypoints expose the same setting as `--trainer-replicas 2`.
It applies to trainer jobs the recipe creates for that run; a pre-created
`trainer.job_id` value is reused as-is.

**LoRA RL** -- set `lora_rank` to train an adapter:

| Field | What to set |
| --- | --- |
| `lora_rank` | Rank of the LoRA adapter (e.g. `64`, `128`) |

The generic GRPO recipes default to `kl_beta=0.001` and provision a reference
for the direct client loss. Set `kl_beta=0` to skip it. To switch to the trainer
built-in or another research loss, follow
[`custom-loss.md`](../skills/fireworks-training/references/rl-custom-loss.md)
in the canonical training skill.

**DPO / ORPO** -- also requires:

| Field | What to set |
| --- | --- |
| `tokenizer_model` | HuggingFace model name matching your base model |

### 4. Run

```bash
cd cookbook/training
python -m recipes.sft_loop      # or whichever recipe you configured
```

## Useful examples

- `examples/tools/promote_checkpoint.py` queries the control plane (`list_checkpoints(job_id)`) for the trainer job's promotable rows and calls the promotion API. No `checkpoints.jsonl`, no temporary trainer — pass `--job-id <id>` and `--base-model <model>` and pick which checkpoint via `--checkpoint-name` / `--step` (default: newest promotable).
- `examples/tools/merge_lora_and_promote.py` merges a LoRA/PEFT adapter into its base (`checkpoint_type="merged_base"`) and promotes the result as a full `HF_BASE_MODEL`. Provisions a short-lived LoRA trainer, loads the adapter explicitly (`load_adapter`), saves the merged base, and promotes.
- `examples/tools/reconnect_and_adjust_lr.py` shows how to reconnect to an already-running trainer job and resume training with a different learning rate.

## Documentation

For detailed guides, configuration reference, and examples, see the official documentation:

- [Introduction & Quickstart](https://docs.fireworks.ai/fine-tuning/training-api/introduction)
- [Cookbook recipes (SFT, RL, DPO, ORPO)](https://docs.fireworks.ai/fine-tuning/training-api/cookbook/overview)
- [Configuration reference](https://docs.fireworks.ai/fine-tuning/training-api/cookbook/reference)

## Directory layout

```
recipes/                                Training loop scripts (fork these)
utils/                                  Shared config, data loading, loss functions, metrics
examples/sft/                           Worked example: SFT getting started
examples/embedding/                     Worked example: embedding (retrieval) fine-tuning
examples/dpo/                           Worked example: DPO
examples/orpo/ifeval/                   Worked example: IFEval with ORPO
examples/rl/deepmath/                   GRPO on DeepMath (rl_loop)
examples/rl/frozen_lake/                Frozen Lake tool-use RL (custom loop)
examples/rl/single_turn_token_in/       Async RL single-turn, token-in rollout
examples/rl/multi_turn_message_in/      Async RL multi-turn, message-in rollout
examples/serverless_rl/                 Serverless (Tinker-style) RL on Countdown -- no provisioning
examples/distillation/                  Distillation examples, including routed MOPD
examples/multihop_qa/                   Multi-hop QA async RL (+ optional IGPO turn-level scoring)
examples/manual/                        Manual hotload-scope tests (PER_TRAINER / PER_DEPLOYMENT)
examples/tools/                         Standalone utility scripts
tests/                                  Unit and end-to-end tests
```

## Tests

```bash
uv pip install -e ".[dev]"
pytest tests/
```

Coverage for the training entrypoints:

```bash
cd training
pytest -q tests/unit tests/test_smoke_imports.py examples/rl/frozen_lake/test_masking.py \
  --cov=. \
  --cov-report=term-missing \
  --cov-report=json:coverage.json
python tests/coverage_summary.py coverage.json
```

See [issues/training-script-coverage-baseline.md](./issues/training-script-coverage-baseline.md)
for the current baseline and
[issues/training-script-coverage-roadmap.md](./issues/training-script-coverage-roadmap.md)
for the expansion plan.
