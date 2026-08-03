# Coding eval with Braintrust + Fireworks

Benchmark **vanilla** (base) Fireworks models on Python function completion before you fine-tune or build agents. Uses [Braintrust `Eval`](https://www.braintrust.dev/docs/evaluate/run-evaluations) for experiment tracking and the [Fireworks integration](https://www.braintrust.dev/docs/integrations/ai-providers/fireworks) for inference.

**Notebook:** [`coding_eval.ipynb`](coding_eval.ipynb)

## Tiers

| Tier | Dataset | Rows | Purpose |
| --- | --- | ---: | --- |
| **1 — Smoke** | Bundled `smoke_data.jsonl` | 20 | Verify wiring in minutes |
| **2 — Standard** | HuggingFace `openai/openai_humaneval` | 164 | Baseline pass@1 number |

Scoring is **deterministic**: each row has unit tests; we execute the model's completion in a subprocess (HumanEval-style). No LLM judge required.

## Prerequisites

```bash
pip install braintrust openai datasets python-dotenv
```

Environment (repo-root `.env` is loaded automatically):

```bash
FIREWORKS_API_KEY=...
BRAINTRUST_API_KEY=...
# Optional: add Fireworks as a provider in Braintrust org settings when using the gateway
```

## What you'll do

1. Run tier-1 smoke on one Fireworks model — confirm Braintrust logs an experiment.
2. Scale to tier-2 HumanEval (subset or full split).
3. Optionally loop over multiple models and compare in the Braintrust UI.

> This is **eval-only** — no Fireworks training jobs. For agentic repo-repair benchmarks (SWE-bench), see `training/examples/rl/coding_agent/`.
