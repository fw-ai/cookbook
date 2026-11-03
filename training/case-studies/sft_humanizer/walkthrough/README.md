# Humanizer guided experience

A no-key companion to the Humanizer cookbook. Open it from the case-study directory:

```bash
# From training/case-studies (the shared brand stylesheet lives in common/):
python3 -m http.server 8765 --bind 127.0.0.1
# Open http://localhost:8765/sft_humanizer/walkthrough/
```

The page uses only local static assets. It contains:

- All 99 held-out passages and 396 recorded outputs, with selectable arms and word-level differences.
- A success, an unchanged passage, and a semantic failure as guided starting points.
- Per-passage checks and aggregate rates computed from the same evidence file.
- A small authored meaning-preservation exercise with immediate feedback.
- One real training row, a JSONL download, and the guided notebook.

`../humanizer_walkthrough.ipynb` uses standard-library Python for its free sections.
Its paid training and evaluation cells are disabled by default. `../train_walkthrough.py`
prints a local plan without importing the training SDK; `--execute` explicitly runs
the current shared serverless SFT recipe. A smoke is eight rows / one epoch; the
tutorial uses the full pack / two epochs. The latter is about 102 steps with 203 rows,
not a replay of the historical 100-step run. Neither preset promises a quality improvement.

## Evidence and regeneration

`recorded.json` is the single evidence source for the page and notebook. It freezes
the Oct 8, 2026 seed-29 evaluation (one greedy draw, four arms). Sources are fictional
synthetic passages. Per-output scores are historical model judgments, not human labels,
and predate the later calibration gate. Account/deployment names are omitted.
The page's interpretation of the example is labeled separately from recorded scores.

To reproduce the exact historical snapshot (the original evaluation must be available locally):

```bash
python walkthrough/export_recorded.py --samples results/2026-10-08/pack-eval-s29/samples.jsonl
```

The exporter rejects changed evaluation or train/test files by their fixed SHA-256 hashes,
then validates fixture/arm completeness. It cannot import a new experiment under the old
date, decoding settings, or editorial conclusions. New experiments need a separate lesson
with reviewed metadata, examples, and claims.

## Authoring conventions for the next field guide

Keep the sequence: see a real result → make one judgment → follow one data row → run
and adapt it. Always include a failure case. Keep historical observations, editorial
interpretations, and authored exercises visibly distinct. Hide detailed code behind
disclosures; keep the decision and its evidence visible. Serve the site from the
case-study root so notebook and README links resolve.

This is buildless HTML/CSS/JavaScript so it can be previewed without installing a
publishing tool. It can be linked or embedded in a later Quarto site. There are no
remote fonts, telemetry, model calls, or credentials. Network is used only to load
the local evidence JSON. No deployment is included.

## Checks

From the case-study directory:

```bash
python3 -B test_walkthrough.py
node --check walkthrough/app.js
```

Also open the page at desktop and phone widths. Check all four arms, filter switching,
the diff toggle, the three quiz responses, JSONL download, and notebook links.

## Reader pilot

Ask a Python developer unfamiliar with LoRA to use the page without coaching. Record
where they hesitate. They should be able to explain why a literal check can miss a
meaning change, identify the student versus teacher prompt, validate one JSONL row,
and distinguish a training smoke from a quality result. Do not require a paid run
to test the learning experience.

For notebook setup and the training-to-evaluation handoff, follow [Getting started](../GETTING_STARTED.md).
