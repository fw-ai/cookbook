# Humanizer

Rewrite naturally while preserving details and leaving good writing alone.

**Method:** Supervised fine-tuning.

[Open the notebook](humanizer_walkthrough.ipynb) · [All case studies](../README.md)

## What this study shows

The notebook uses the served seed-29 evaluation and compares adapter behavior with prompting. Improvements in preservation and overediting do not imply an overall pass-rate improvement. Read the individual metrics and failure cases.

## Start here

Use this folder inside the [Fireworks cookbook](https://github.com/fw-ai/cookbook). Install the dependencies declared in the cookbook’s `training/pyproject.toml`, plus Jupyter for the notebook interface. Open Jupyter from this study folder so relative data paths resolve. Shared training utilities come from the parent cookbook; this folder is not a standalone SDK.

Read the recorded results before enabling training or evaluation. Cells marked for live inference or training require your own credentials and may incur charges. Keep credentials in your environment, never in a notebook or commit. Recorded results describe the saved experiment, not a guarantee for a new run.

## Method and reproduction

[Read the detailed guide](GUIDE.md) for data preparation, training settings, evaluation and limitations.
