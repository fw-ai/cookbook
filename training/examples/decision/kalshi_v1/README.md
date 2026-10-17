# Kalshi decision-tuning v1

Reproduce the **Decision Model Tuning** Muse Glimmer setup on public Kalshi
event rows. The loss loop is `recipes/decision_loop.py`; this example only
pins the dataset split and the training knobs.

## Datasets (public Fireworks)

| Split | Resource | Rows | Role |
|---|---|---|---|
| train | `accounts/fireworks/datasets/kalshi-decision-v1-train` | 6,461 | Category-balanced training |
| dev | `accounts/fireworks/datasets/kalshi-decision-v1-dev` | 2,000 | Calibration / fit-T |
| test | `accounts/fireworks/datasets/kalshi-decision-v1-test` | 8,765 | Full held-out eval |
| unseen_family | `accounts/fireworks/datasets/kalshi-decision-v1-unseen-family` | 765 | Generalization slice (also inside `test`) |

`test` is the concatenation, in order, of the original chronological test
(2,000) + the 6,000-row expand (`test_extra`) + `unseen_family` (765).
`unseen_family` is published again on its own so reports can score that slice
without filtering the big file.

Each download is sha256-checked by `prepare_data.py`.

## Row contract

JSONL decision rows: `{"state": {...}, "questions": {id: {type, instructions, criteria, label, ...}}}`.
See `training/utils/decision.py`. Sources are public Kalshi HTTP + the public
Hugging Face market dump; no keys and no order-book history.

## Quick start

```bash
cd cookbook/training
python3 -m venv .venv && source .venv/bin/activate && pip install -e .
export FIREWORKS_API_KEY=fw_...   # never paste into chat

python -m examples.decision.kalshi_v1.prepare_data --splits train,dev
python -m examples.decision.kalshi_v1.train_kalshi_decision \
  --output-model-id kalshi-decision-v1-glimmer
```

Defaults match the published E2E: Muse Glimmer 30B, LoRA rank 64, 1 epoch,
batch 64, lr `1e-4`, seed 0, serverless pool. The promoted checkpoint is the
calibrated `step-N-cal` artifact.

For offline accuracy / ECE on the full 8,765-row test set (and the 765-row
unseen slice alone):

```bash
python -m examples.decision.kalshi_v1.prepare_data --splits test,unseen_family
```

## Managed SFT (no local download)

Once `tuningMode=DECISION` is public on supervised fine-tuning jobs:

```bash
firectl supervised-fine-tuning-job create \
  --base-model accounts/fireworks/models/muse-glimmer-30b \
  --dataset accounts/fireworks/datasets/kalshi-decision-v1-train \
  --evaluation-dataset accounts/fireworks/datasets/kalshi-decision-v1-dev \
  --output-model kalshi-decision-v1-glimmer \
  --tuning-mode decision
```

Until that flag ships, the Training API path above is the supported repro.

## Provenance

- Builder: Decision Model Tuning project store (`kalshi_build.py`,
  `kalshi_build_extra.py`), seed `20261012`, build clock
  `2026-10-12T22:19:46Z` (+ expand fetch `2026-10-15T01:04:05Z`).
- Windows: train settled before 2026-01-04; 7-day gaps before dev and test
  open times; see the source `manifest.json` in the project store.
