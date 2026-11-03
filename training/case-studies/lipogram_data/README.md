# E-vasion

Answer useful questions without the letter e.

**Method:** Supervised fine-tuning followed by reinforcement learning.

[Open the notebook](lipogram_demo.ipynb) · [All case studies](../README.md)

## What this study shows

This study uses the step-100 SFT + GRPO adapter. Compare constraint satisfaction, completeness, and retention together. Recorded outputs work without a key; live inference requires explicit opt-in. Earlier lipogram experiments are separate from this study.

## Start here

Use this folder inside the [Fireworks cookbook](https://github.com/fw-ai/cookbook). Install the dependencies declared in the cookbook’s `training/pyproject.toml`, plus Jupyter for the notebook interface. Open Jupyter from this study folder so relative data paths resolve. Shared training utilities come from the parent cookbook; this folder is not a standalone SDK.

Read the recorded results before enabling training or evaluation. Cells marked for live inference or training require your own credentials and may incur charges. Keep credentials in your environment, never in a notebook or commit. Recorded results describe the saved experiment, not a guarantee for a new run.

## Method and reproduction

[Read the detailed guide](GUIDE.md) for data preparation, training settings, evaluation and limitations.
