# Caveman

## Get the data

Download one archive: [caveman-v2.zip](../bundles/caveman-v2.zip). A full repository checkout already includes it. The first notebook cell verifies its SHA-256 checksum and restores the data to the paths used below. For command-line scripts, run `python ../common/study_data.py caveman` from this folder first.

The archive preserves separate training, development, test and recorded-result files. It does not overwrite files you have changed. See [bundle setup](../bundles/README.md) for manual downloads and verification.

Teach concise answers without dropping the facts.

**Method:** Supervised fine-tuning.

[Read the rendered notebook](https://nbviewer.org/github/fw-ai-external/cookbook/blob/68750d4980cc8798343c67a49ec5c5b92093499f/training/case-studies/sft_caveman/caveman_sft.ipynb) · [Notebook source](caveman_sft.ipynb) · [All case studies](../README.md)

The rendered link is a fixed snapshot at commit `68750d4`. For the current code and setup, use the notebook source in this checkout.

## What this study shows

The frozen v2c release includes training, development and test examples, recorded comparisons, and a training plan. The selected run cost $4.68 for training only; data generation and evaluation are additional. The test set was reused during model selection.

## Start here

Use this folder inside the [Fireworks cookbook](https://github.com/fw-ai-external/cookbook). Install the dependencies declared in the cookbook’s `training/pyproject.toml`, plus Jupyter for the notebook interface. Open Jupyter from this study folder so relative data paths resolve. Shared training utilities come from the parent cookbook; this folder is not a standalone SDK.

Read the recorded results before enabling training or evaluation. Cells marked for live inference or training require your own credentials and may incur charges. Keep credentials in your environment, never in a notebook or commit. Recorded results describe the saved experiment, not a guarantee for a new run.


## Reproduce the v2c recipe

The bundled release contains 2,348 training rows, 392 development cases and 392 test cases. The notebook inspects the data hashes and selection examples before any paid calls. It trains for two epochs with rank 64, alpha 128, batch size 64, learning rate 0.0001 and seed 29, dropping the student system prompt. Follow the notebook’s generated command to retain all of these settings; command-line defaults alone are not the full v2c recipe.

`run_pack_sft.py` and `evaluate_pack.py` default to `release/v2c/`. For a newly generated pack, pass `--pack` and `--pack-dir` explicitly. The optional `build_caveman_pack.py` pipeline generates new data and is not needed to use the frozen release. Its outputs will not reproduce the saved data byte for byte.

## Evaluation and limitations

Compare base, full-skill prompting and your adapter on development cases before freezing a new recipe. The recorded test comparison uses 392 questions and eight draws per arm. Its intervals account for repeated answers and related questions. The test set was reused across two adapter versions; treat the result as exploratory rather than a once-only confirmatory test. Word counts measure answer length, while billing uses tokens.

The selected training run cost an estimated $4.68; this excludes data preparation and evaluation. See the [release notes](release/v2c/README.md) for the broader research ledger and the limits of the cost estimate. New end-to-end learner cost has not been measured.

## Implementation

- `notebook_display.py`: recorded answer comparisons and evaluation plots.
- `run_pack_sft.py`: training and checkpoint promotion through the shared SFT recipe.
- `evaluate_pack.py`: matched serving comparisons.
- `build_caveman_pack.py`, `validate_data.py`, `calibrate_judge.py`: data generation and checks.
- `aggregate_seeds.py`: paired comparisons across recorded evaluations.
- `sources/`: pinned skill source and license notices.
- `tests/`: offline display checks.

## Notebook dependencies

In addition to the cookbook training package, install the shared display dependencies from this study directory:

```bash
python -m pip install -r ../common/requirements-notebooks.txt
```

This installs Python packages only; it does not start training or inference.
