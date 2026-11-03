# Anti-slop

See what preference optimization learns, and what a detector score misses.

**Method:** Direct preference optimization and RL follow-up.

[Open the notebook](dpo_glm53_flash.ipynb) · [All case studies](../README.md)

## What this study shows

This is the DPO/Anti-slop experiment documented in this folder. Detector optimization and output diversity are separate measurements. The results below belong to this DPO and RL experiment. The notebook defaults to the general UltraFeedback example; select the Anti-slop pack for this case study.

## Start here

Use this folder inside the [Fireworks cookbook](https://github.com/fw-ai/cookbook). Install the dependencies declared in the cookbook’s `training/pyproject.toml`, plus Jupyter for the notebook interface. Open Jupyter from this study folder so relative data paths resolve. Shared training utilities come from the parent cookbook; this folder is not a standalone SDK.

Read the recorded results before enabling training or evaluation. Cells marked for live inference or training require your own credentials and may incur charges. Keep credentials in your environment, never in a notebook or commit. Recorded results describe the saved experiment, not a guarantee for a new run.

## Method and reproduction

[Read the detailed guide](GUIDE.md) for data preparation, training settings, evaluation and limitations.
