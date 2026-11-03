# Taste

Compare visual design and functionality across a base model, a skill prompt, SFT and RL.

**Method:** Recorded comparison with training source.

[Open the notebook](taste_walkthrough.ipynb) · [All case studies](../README.md)

## What this study shows

Start with the recorded four-arm comparison. RL improves judged visual preference over the plain base in this experiment, but functionality regresses. Retain the prompted skill and SFT baselines, and do not treat selected showcase websites as aggregate evidence.

## Start here

Use this folder inside the [Fireworks cookbook](https://github.com/fw-ai/cookbook). Install the dependencies declared in the cookbook’s `training/pyproject.toml`, plus Jupyter for the notebook interface. Open Jupyter from this study folder so relative data paths resolve. Shared training utilities come from the parent cookbook; this folder is not a standalone SDK.

Read the recorded results before enabling training or evaluation. Cells marked for live inference or training require your own credentials and may incur charges. Keep credentials in your environment, never in a notebook or commit. Recorded results describe the saved experiment, not a guarantee for a new run.

## Evidence and implementation

- [Recorded four-arm results](results/compare-rl/summary.md) and [machine-readable counts](results/compare-rl/stats.json).
- [Earlier SFT comparison](results/V2_COMPARISON.md).
- Data preparation: `build_pack.py`; SFT: `train_sft.py`; RL: `ui_rl.py`.
- Rendering and scoring: `generate.py`, `render.py`, `reward.py`.

The notebook is a recorded-results walkthrough, not an end-to-end reproduction of the historical RL20 training run. The RL implementation shares the sibling `rl_lipogram` loop.
