# Lipogram, answer-only: no letter e at low effort, with no instruction

Companion to [`rl_lipogram/`](../rl_lipogram/) ([walkthrough](../rl_lipogram/lipogram_walkthrough.ipynb)). Evaluation targets:
on held-out prompts, with **no system instruction** and **`reasoning_effort=low`**, answers must be strictly e-free,
useful and complete (greedy ≥ 95%, sampled ≥ 90%), with retention on ordinary tasks within 2 points. Trace SFT reaches
the rule by planning at max effort; at low effort there is no trace, so the rule has to live in the weights. This study
builds **answer-only** data (answer normally, then rewrite e-free), trains on it, then runs the strict GRPO loop from that
SFT state.

**Start here: [`lipogram_demo.ipynb`](lipogram_demo.ipynb).** The notebook shows recorded answers, evaluation results, a disguised-character audit, and the data pipeline. It starts in recorded mode even if an API key is configured. To submit new prompts, explicitly enable live inference and configure an adapter accessible to your account.

## Results (60 held-out prompts, effort low, no instruction)

| model | greedy | sampled | sampled e-free | completeness | retention |
|---|---:|---:|---:|---:|---:|
| base GLM 5.3 Flash | 0.0% | 0.0% | 0.0% | 0.98 | 98.3% |
| base + lipogram system prompt | 1.7% | 4.6% | 4.6% | 0.92 | 98.3% |
| Trace-SFT adapter `glm53-lipogram-sft-v2-300` | 0.0% | 0.8% | 0.8% | 0.97 | 98.3% |
| SFT on 10.5k rewritten answers (`arxm-lipo-p2-lr1e4-e1`) | 35.0% | 29.2% | 36.2% | 0.82 | 97.8% |
| + strict GRPO, step 60 (`arxm-lipo-p2-rl-s60`) | 86.7% | 82.1% | 91.7% | 0.80 | 96.6% |
| **+ strict GRPO, step 100 (`arxm-lipo-p2-rl-s100`)** | **85.0%** | **84.2%** | **91.7%** | **0.87** | **97.8%** |

The bar is not met: s100 is 10 points short on greedy and 5.8 points short on sampled success. Gains flattened after
step 60. Paired difference against the prompted base at s100: +79.2 points [+71.7, +85.8]. Every checkpoint (s20–s100,
2-epoch and lr 3e-5 SFT variants, the 1.1k-row pilot) is in `recorded/`.

**Disguised e's:** the eval answers of every SFT and RL checkpoint contain no accented or look-alike e, no escapes and no
non-ASCII letters (the prompted base used Cyrillic е in 222 of 3,600). On 60 prompts built to tempt *café*, *crème
brûlée*, *René*, s100 abbreviates instead; it wrote é or Cyrillic е only when explicitly asked to spell the word that
way, and the checker rejected those. Leet in code literals (`CR8`, `@l8t`) is the one channel the checker misses: 5 of
620 answers. Details in section 4 of the notebook.

**Caveats:** answers are short (median about 52–55 words vs about 145 for the base) and names with an e collapse to
initials ("D.C." for Destiny's Child), so the judge's completeness is generous on some answers.

## Try your adapter

The recorded results use `accounts/research-train/models/arxm-lipo-p2-rl-s100`. That identifier records provenance; it does not grant access. For live inference, use a compatible adapter you can access and load it through your account's supported serving workflow.

Before starting Jupyter, set `FIREWORKS_API_KEY` and `LIPO_ADAPTER` in your environment. Set `LIPO_SFT` as well if you want to include your SFT checkpoint in the comparison. In the notebook, set `RUN_LIVE=True`. Use low reasoning effort and no lipogram system instruction to match the recorded protocol. Calls to your models produce new results; the recorded tables remain unchanged.

## Files

| file | role |
|---|---|
| `lipogram_demo.ipynb` | the demo: live playground, results, cases, audit, data, reproduce |
| `seeds.py` | real user questions from Dolly-15k, oasst1 and ELI5, deduped (exact and 8-gram) against every `rl_lipogram` prompt; samples a request type, length band and reader persona per seed |
| `pipeline.py` | per seed: GLM 5.3 Flash at effort low answers normally; the rewriter (GLM at max effort, or `--rewriter kimi`) rewrites with no e using e-free WordNet hints (`lexicon.py`); the strict checker drives up to 3 fix rounds; e-free names and digits must survive (`checks.py`); a judge rejects misleading changes and unrecognisable names (one repair round) and scores fidelity and naturalness. Logs every stage and its cost; resumes |
| `build_rows.py`, `stats.py` | accepted attempts → instruction-free answer-only rows (validated with `rl_lipogram/train_sft.validate_rows`); yield, cost and failure modes per stage |
| `run_sft.py`, `promote.py` | serverless LoRA SFT through `recipes/sft_loop.py` with a low-effort GLM renderer, one promotable checkpoint per epoch; promote per-epoch or named (RL) checkpoints |
| `lipo_eval.py` | Evaluation of a served adapter: `evaluate_served.py`, `evaluate_retention.py`, `check_bar.py`, unmodified |
| `audit_unicode.py`, `adversarial_probe.py` | independent disguised-e audit of recorded answers; trap prompts against a served adapter |
| `recorded/` | every evaluation summary, seed manifests, a 20-row data sample, the notebook's cases, audit and recorded playground answers |
| `walkthrough/` | `demo_helpers.py` (playground, checker highlighting, tables, figure) and `build_notebook.py` |

## Reproduce (paid calls)

From this folder, with `PYTHONPATH` set to the cookbook root and your key in `FIREWORKS_API_KEY`. The recorded experiment estimated about $425 for 10.5k data rows, $54 per two SFT epochs, and $1–2 per RL step. These are historical estimates, not current price quotes.

```bash
python seeds.py --n 13000 --seed 11 --eli5-sample 60000 --out data/seeds_p2.jsonl
python pipeline.py --seeds data/seeds_p2.jsonl --rewriter glm --workers 64 --out runs/p2
python build_rows.py runs/p2/attempts.jsonl --out data/sft_p2.jsonl
python run_sft.py --pack data/sft_p2.jsonl --run-dir runs/sft-p2
python promote.py --run-dir runs/sft-p2 --prefix lipo-sft
cd ../rl_lipogram && python lipogram_rl.py --dataset data/lipogram_train.jsonl \
  --init-from-state <account>/<sft-run-id>/step-<N> --reward-phase strict --drop-system-prompt-share 1 \
  --train-efforts low --eval-effort low --eval-prompt-groups 16 --eval-seed 1 --max-sample-tokens 4096 \
  --steps 100 --eval-interval 20 --dcp-save-interval 20 --output-model-id lipo-rl && cd ../lipogram_data
firectl -a <account> model load-lora lipo-rl --serverless --wait
python lipo_eval.py --model accounts/<account>/models/lipo-rl --tag rl
```

Raw generation and training outputs are not included. The `recorded/` directory contains the saved results used by the notebook. Keep newly generated outputs separate so the published comparison remains reproducible.
