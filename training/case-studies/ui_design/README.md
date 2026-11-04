# Taste

## Get the data

Download one archive: [taste-v2.zip](../bundles/taste-v2.zip). A full repository checkout already includes it. The first notebook cell verifies its SHA-256 checksum and restores the data to the paths used below. For command-line scripts, run `python ../common/study_data.py taste` from this folder first.

The archive preserves separate training, development, test and recorded-result files. It does not overwrite files you have changed. See [bundle setup](../bundles/README.md) for manual downloads and verification.

Compare visual design and functionality across a base model, a skill prompt, SFT and RL.

**Method:** Recorded comparison with training source.

[Open the notebook](taste_walkthrough.ipynb) · [All case studies](../README.md)

## What this study shows

Start with the recorded four-arm comparison. RL improves judged visual preference over the plain base in this experiment, but functionality regresses. Retain the prompted skill and SFT baselines, and do not treat selected showcase websites as aggregate evidence.

## Start here

Use this folder inside the [Fireworks cookbook](https://github.com/fw-ai/cookbook). Install the dependencies declared in the cookbook’s `training/pyproject.toml`, plus Jupyter for the notebook interface. Open Jupyter from this study folder so relative data paths resolve. Shared training utilities come from the parent cookbook; this folder is not a standalone SDK.

Read the recorded results before enabling training or evaluation. Cells marked for live inference or training require your own credentials and may incur charges. Keep credentials in your environment, never in a notebook or commit. Recorded results describe the saved experiment, not a guarantee for a new run.

## Evidence and implementation

- [Recorded four-arm results](results/compare-rl/summary.md) and [machine-readable counts (in data bundle)](../bundles/taste-v2.zip).
- [Earlier SFT comparison](results/V2_COMPARISON.md).
- Data preparation: `build_pack.py`; SFT: `train_sft.py`; RL: `ui_rl.py`.
- Rendering and scoring: `generate.py`, `render.py`, `reward.py`.

The notebook is a recorded-results walkthrough, not an end-to-end reproduction of the historical RL20 training run. The RL implementation shares the sibling `rl_lipogram` loop.

## Optional rendering setup

The recorded notebook uses only saved counts and does not need a browser. To render generated sites, restore the Taste bundle first, then install the pinned frontend dependencies and a Playwright browser:

```bash
cd scaffold
npm ci
cd ..
python -m pip install "playwright>=1.48"
python -m playwright install chromium
```

These setup steps download dependencies. `render.py` then builds against the local scaffold and blocks external page requests. The bundle includes the briefs and their image assets; source notes are restored under `assets/`. The saved experiment’s frontend lockfile is retained for reproducibility.

Generated pages run without dedicated, shared or service workers. The local
server enforces `worker-src 'none'` in its CSP response header because sockets
created inside workers bypass Playwright's page/context WebSocket routing.
Worker policy violations are recorded as console errors and fail the page's
functional checks.

## Notebook dependencies

In addition to the cookbook training package, install the shared display dependencies from this study directory:

```bash
python -m pip install -r ../common/requirements-notebooks.txt
```

This installs Python packages only; it does not start training or inference.
