# Caveman

Teach concise answers without dropping the facts.

**Method:** Supervised fine-tuning.

[Open the notebook](caveman_sft.ipynb) · [All case studies](../README.md)

## What this study shows

The frozen v2c release includes training, development and test examples, recorded comparisons, and a training plan. The selected run cost $4.68 for training only; data generation and evaluation are additional. The test set was reused during model selection.

## Start here

Use this folder inside the [Fireworks cookbook](https://github.com/fw-ai/cookbook). Install the dependencies declared in the cookbook’s `training/pyproject.toml`, plus Jupyter for the notebook interface. Open Jupyter from this study folder so relative data paths resolve. Shared training utilities come from the parent cookbook; this folder is not a standalone SDK.

Read the recorded results before enabling training or evaluation. Cells marked for live inference or training require your own credentials and may incur charges. Keep credentials in your environment, never in a notebook or commit. Recorded results describe the saved experiment, not a guarantee for a new run.

## Method and reproduction

[Read the detailed guide](GUIDE.md) for data preparation, training settings, evaluation and limitations.
