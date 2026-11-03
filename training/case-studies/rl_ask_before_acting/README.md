# Ask Before Acting

## Get the data

Download one archive: [ask-before-acting-v1.zip](../bundles/ask-before-acting-v1.zip). A full repository checkout already includes it. The first notebook cell verifies its SHA-256 checksum and restores the data to the paths used below. For command-line scripts, run `python ../common/study_data.py ask-before-acting` from this folder first.

The archive preserves separate training, development, test and recorded-result files. It does not overwrite files you have changed. See [bundle setup](../bundles/README.md) for manual downloads and verification.

Train a simulated agent to ask when an action needs clarification.

**Method:** Reinforcement learning.

[Open the notebook](ask_before_acting_walkthrough.ipynb) · [All case studies](../README.md)

## What this study shows

The original walkthrough teaches the earlier training experiment. The later v2 evaluation is linked separately below; its results must not be attributed to the earlier checkpoint. The recorded tasks simulate tool use and do not execute real actions.

## Start here

Use this folder inside the [Fireworks cookbook](https://github.com/fw-ai/cookbook). Install the dependencies declared in the cookbook’s `training/pyproject.toml`, plus Jupyter for the notebook interface. Open Jupyter from this study folder so relative data paths resolve. Shared training utilities come from the parent cookbook; this folder is not a standalone SDK.

Read the recorded results before enabling training or evaluation. Cells marked for live inference or training require your own credentials and may incur charges. Keep credentials in your environment, never in a notebook or commit. Recorded results describe the saved experiment, not a guarantee for a new run.

## Method and reproduction

[Read the detailed guide](GUIDE.md) for data preparation, training settings, evaluation and limitations.

## Later checkpoint evaluation

[Read the v2 held-out evaluation](results/2026-10-14/README.md). It uses 16 tasks and four draws per task. Its paired uncertainty intervals include zero; interpret it as small-sample evidence.
