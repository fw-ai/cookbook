# Nemotron Ultra mixed NL2Repo SFT ablation

This example reproduces a one-epoch LoRA ablation that mixes fully successful
PRD-to-Repo trajectories with trajectories whose verifier tests pass at least
80%. It consumes prepared artifacts from the provenance-bound workflow in
[fw-ai/upheaval#148](https://github.com/fw-ai/upheaval/pull/148); it does not
duplicate raw-trajectory curation or semantic rubric judging.

This is a recorded **negative ablation**, not a recommended checkpoint. The
completed run was numerically stable but regressed on DeepSWE, so this launcher
always saves an unpromoted checkpoint and does not expose an output-model
promotion option.

## Data contract

Set `RUN_DIR` to one curation output directory containing:

- `report.json` with schema `upheaval-prd-to-repo-curation-v1`;
- `renderer-report.json` with schema
  `upheaval-prd-to-repo-render-validation-v1`; and
- `train.jsonl`, `val.jsonl`, `test.jsonl`, and the row-level
  `manifest.jsonl`.

Before a run, `train.py` rechecks the curation and renderer-report hashes,
target model and tokenizer revision, exact `nemotron3_ultra` renderer, 262,144
token limit, 80% partial-success threshold, mixed quality labels, split hashes,
physical JSONL row counts, reviewed curator/validator source hashes, exact
renderer source hashes, and every fail-closed renderer check. In particular,
the report must match the reviewed aggregate think-boundary counts, including
zero trainable prefilled `<think>` tokens and a trainable generated `</think>`
token in every rendered datum. The four training/provenance files must also
match the canonical byte hashes recorded in `RESULTS.json`; a self-consistent
but drifted or synthetic report is rejected.

The frozen Wentao-v1 input contained 692 trajectories from 268 tasks:

- 260 full successes and 432 partial successes;
- Claude Code 2.1.231 and 2.1.237 source captures;
- 10,528 windows: 10,215 train, 168 validation, and 145 test; and
- 6,524 partial-success train windows (63.9% of train windows).

Raw Harbor trials, capture bundles, repositories, prepared JSONL, credentials,
and evaluation outputs are intentionally not committed here.

## Environment

From the repository root:

```bash
cd training
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --pre -e ".[dev]"
```

The API key used for a paid run must belong to an account that can use the
configured Nemotron Ultra LoRA training shape. Do not commit `.env` files or
keys.

## Resolve without creating resources

Dry-run is the default:

```bash
RUN_DIR=/path/to/wentao-v1-curation-output \
  training/examples/sft/nl2repo_mixed/run.sh
```

Review
`$RUN_DIR/training/ultra-mixed-partial80/resolved_config.json`. It binds the
dataset reports and split hashes, renderer implementation hashes, model
settings, optimization settings, and Fireworks session ID.

## Launch the paid ablation

Only after reviewing the resolved config:

```bash
RUN_DIR=/path/to/wentao-v1-curation-output \
DRY_RUN=0 \
CONFIRM_FIREWORKS_TRAINING=YES \
FIREWORKS_API_KEY=... \
WANDB_ENTITY=... \
WANDB_PROJECT=nl2repo-ultra-sft \
  training/examples/sft/nl2repo_mixed/run.sh
```

The reviewed defaults are Nemotron 3 Ultra BF16, `nemotron3_ultra`, 262K
context, one epoch, batch size 1, LoRA rank/alpha 16/32, peak learning rate
`3e-7`, 3% warmup, cosine decay to `3e-8`, seed `20260828`, and no reservation.
This launcher saves a final checkpoint but does not promote or deploy it. The
historical checkpoint was later promoted and deployed through a separate,
explicit evaluation-only workflow; those resource IDs in `RESULTS.json` are
evidence of the measured ablation, not an endorsement for production use.

## Completed result

[`RESULTS.json`](RESULTS.json) records the training job, evaluated model and
deployment, benchmark run IDs, exact metrics, renderer hashes, and SHA-256
commitments for the retained source artifacts.

The freshly materialized #148 output matched the historically validated
dataset's semantic row-hash multiset exactly in all three splits (10,528 of
10,528 rows, with no rows present on only one side). This binds the canonical
byte hashes accepted by the launcher to the full renderer report used by the
completed run.

The completed 10,215-step run had finite, contiguous metrics:

- first/last 100-step mean cross entropy: 0.4497 / 0.3775;
- final validation cross entropy: 0.3507;
- no overlength or zero-loss datums; and
- corrected think-boundary masks throughout.

Lower training and validation loss did not transfer to agent performance:

- DeepSWE Pass@1: Base 5/113 (4.42%), Mixed SFT 2/113 (1.77%);
- infrastructure errors: Base 4, Mixed SFT 20;
- on 91 paired-valid tasks: 2 gains, 5 regressions, and 84 unchanged failures;
- fail-to-pass coverage fell 6.67 percentage points; and
- partial reward fell 2.87 percentage points.

The most likely causes are objective and task-distribution mismatch, a
validation split covering only four canonical tasks, and correlated
partial-success windows receiving the same token loss weight as full
successes. The corrected renderer means the previous think-boundary masking
bug does not explain this run's remaining regression.
