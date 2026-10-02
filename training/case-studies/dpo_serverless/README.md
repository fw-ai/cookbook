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

Base model `kimi-k3`; `glm-5p2-fast` judges, a different family so there's no self-preference bias.

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

Defaults are a real training run: ~1 epoch over 2968 preference pairs, **2.5–3 hours**. Lower
`MAX_PAIRS` and `STEPS` in the config cell for a smoke test. Promotion is off by default.

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

