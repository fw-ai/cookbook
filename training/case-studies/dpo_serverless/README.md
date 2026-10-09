# DPO from scratch on Fireworks serverless training

Build a Direct Preference Optimization loop by hand — the reference model, the loss, the training step —
on the Fireworks Training API. Nothing to provision, nothing to tear down.

## Who this is for

You have a model that's accurate but **doesn't sound right**, and it's easier to say *which of two
answers is better* than to write the ideal one. That's preference tuning: brand voice, tone,
helpfulness, less rambling.

And you want to understand the mechanism, not just call an API.

**If you only want to ship**, use managed DPO jobs instead — `[../dpo_style/](../dpo_style/)` does the
same training server-side in four SDK calls. Read that one to *use* DPO; read this one to *understand*
it, because here every number in the log has a line of code you can point at.

## What the notebook does

1. Downloads and validates preference pairs ([UltraFeedback](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences), or your own JSONL)
2. Connects to the serverless trainer and snapshots a **frozen reference model**
3. Shows — with no GPU — how DPO's objective can improve while the model gets *worse*
4. Scores every pair against the reference, once, up front
5. Runs the loop, evaluating a fixed held-out set as it goes
6. Prints base-vs-tuned answers side by side, then a judge-scored **win-rate**
7. Optionally promotes the result to a servable model

Base model `kimi-k3`; `glm-5p3-fast` judges, a different family so there's no self-preference bias.

## Run it

Python **3.11+** (a 3.10 environment fails on import).

```bash
cd cookbook/training
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .
uv pip install ipykernel matplotlib openai        # not declared in pyproject
python -m ipykernel install --user --name cookbook-training --display-name "cookbook (3.12)"

export FIREWORKS_API_KEY=fw_...
jupyter lab case-studies/dpo_serverless/dpo_ultrafeedback_serverless.ipynb
```

**Use an account-scoped API key.** Serverless training rejects keys with access to multiple
accounts — creating the training client fails with `create_session: account not found`.

On Colab, uncomment the block at the top of the setup cell. It clones the repo and installs
`training/` itself, so you get the repo's pinned `transformers` / `torch` / `tinker` versions rather
than whatever is preinstalled.

## Cost and run size

Run-All on a fresh kernel **spends nothing**. `RUN_LIVE = False` in the config cell, and every cell
that calls the paid serverless trainer starts with `%%live` and prints a skip notice instead. The data
download, validation, the section 4 math, and the rendering checks all still run.

Set `RUN_LIVE = True` to train. Serverless training bills per token on three meters (prefill, sample,
train). Check current rates on the [pricing page](https://fireworks.ai/pricing) or the
[training cost estimator](https://docs.fireworks.ai/fine-tuning/cost-estimator).

| | `STEPS` | `MAX_PAIRS` | what it is |
| --- | --- | --- | --- |
| **default (smoke)** | 10 | 200 | proves the pipeline end to end; 80 pair-visits, too short to move the win-rate |
| **full run** | 370 | 3000 | ~1 epoch over 2968 pairs, **2.5–3 hours** wall-clock including the reference pass |

Rough token volume for the full run, at the ~400-token median sequence length on this dataset:

- **train:** 370 steps × 8 pairs × 2 sequences ≈ 2.4M tokens
- **prefill:** ≈ 2.3M for the one-off reference pass, plus ≈ 0.5M for about 20 held-out evals
- **sample:** win-rate generations, 2 × `EVAL_PAIRS` responses of at most 2048 tokens each,
  plus the judge calls (billed as ordinary serverless inference)

`EARLY_STOP_CHOSEN_DROP` can end the run before `STEPS`. Promotion is off by default.

**Teardown:** nothing. No deployment, no trainer job — the sampling clients release when the notebook
exits.

## Your own data

Set `DATA_JSONL` in the config cell. Three schemas are accepted (`training/utils/data.py`):


| Format            | Shape                                                                                      |
| ----------------- | ------------------------------------------------------------------------------------------ |
| chosen / rejected | `{"chosen": {"messages": [...]}, "rejected": {"messages": [...]}}`                         |
| OpenAI preference | `{"input": {"messages": [...]}, "preferred_output": [...], "non_preferred_output": [...]}` |
| scored samples    | `{"samples": [{"messages": [...], "evals": {"score": 1.0}}, {... "score": 0.0}]}`          |


Section 1 validates the schema and catches the two mistakes that produce no error and a meaningless  
run: prompt prefixes that differ between the two sides, and identical responses.

