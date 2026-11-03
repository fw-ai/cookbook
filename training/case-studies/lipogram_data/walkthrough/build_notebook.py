#!/usr/bin/env python3
"""Build ../lipogram_demo.ipynb. Execute it afterwards so GitHub shows the outputs:

    python walkthrough/build_notebook.py
    LIPO_RECORD=1 jupyter nbconvert --to notebook --execute --inplace lipogram_demo.ipynb   # live, records playground answers
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent.parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("""# Lipogram, answer-only: no letter e at low effort, with no instruction

A LoRA adapter on GLM 5.3 Flash that answers ordinary questions in useful English **without the letter e**, with **no
system prompt** and **`reasoning_effort="low"`** (no reasoning trace at all). It is the answer-only companion to
Rob's [`rl_lipogram`](../rl_lipogram/lipogram_walkthrough.ipynb), which reaches the rule by planning in a long
reasoning trace at max effort. Here the rule has to live in the weights.

**Recipe:** ~10.5k instruction-free rows built by *answer normally, then rewrite e-free* (checker-guided rewrites, names
and numbers kept, a judge rejects meaning changes) → LoRA SFT → Rob's strict GRPO for 100 steps.

**Headline (60 held-out prompts, Rob's `check_bar` protocol):** the RL adapter `arxm-lipo-p2-rl-s100` gives a complete,
useful, e-free answer on **85.0% of greedy** and **84.2% of sampled** draws, against 1.7% / 4.6% for the base model
with the rule in its system prompt and 0% / 0.8% for Rob's trace adapter `sft-v2-300` at this effort. Retention on ordinary tasks: 97.8% vs
98.3% for the base. **The bar (95% greedy, 90% sampled) is not met yet.**

1. [Play with the model live](#play) · 2. [Results in detail](#results) · 3. [Six prompts, six models](#cases) ·
4. [Does it cheat with é or Cyrillic е?](#audit) · 5. [The data](#data) · 6. [Where it fails](#fails) · 7. [Reproduce](#reproduce)

Every cell runs with no API key: live cells fall back to the answers recorded when this notebook was executed."""),
    code("""from pathlib import Path
import sys

HERE = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "walkthrough/demo_helpers.py").exists()), None)
if HERE is None:
    raise RuntimeError("Open the notebook from the lipogram_data folder (walkthrough/demo_helpers.py not found)")
sys.path.insert(0, str(HERE / "walkthrough"))
import demo_helpers as D
from demo_helpers import ask_lipogram

demo, ckpt, audit = D.load()
D.results_table(demo)"""),
    code("""fig = D.progression_figure(ckpt, HERE / "figures/lipogram_data_progression.png")"""),
    md("""**What to see:** SFT on rewritten answers moves the model from 0% to about a third; strict RL from that state does most
of the work in the first 60 steps (sampled 29% → 82%), then flattens (84% at step 100).

<a id="play"></a>
## 1. Play with the model live

`ask_lipogram(prompt, effort="low", temperature=0.0, compare_all=True)` sends one prompt, with no system prompt, to the
RL adapter and, with `compare_all=True`, to the base model with and without the rule in its system prompt and to the
SFT adapter if it is loaded. Every answer goes through rl_lipogram's strict checker: Unicode NFKD, look-alikes
(Cyrillic е, accented é è ê ë, fullwidth ｅ), escapes such as `\\u0065`, and non-Latin letters all count as e. Each
offending character is shown as <mark>[e]</mark>, with a PASS/FAIL badge, the strict e count and the word count.

**To run it live** you need a key for the account that serves the adapter:

- The adapter is `accounts/research-train/models/arxm-lipo-p2-rl-s100`, loaded on **serverless** in the `research-train`
  account (ask Wenyun for access), so `export FIREWORKS_API_KEY=<research-train key>` before starting Jupyter, or
  set `FIREWORKS_API_KEY_FILE=<path to a file holding the key>` if your notebook kernel does not see your shell's environment.
- If it has been unloaded: `firectl -a research-train model load-lora arxm-lipo-p2-rl-s100 --serverless --wait`.
  The SFT comparison arm is `arxm-lipo-p2-lr1e4-e1` (same command). Unload with `firectl ... model unload-lora`.
- On a dedicated deployment instead: create one with `--enable-addons` on `accounts/fireworks/models/glm-5p3-flash`,
  then `firectl model load-lora <model> --deployment <deployment-id>`.
- To point the helpers at your own copy, set `LIPO_ADAPTER=accounts/<account>/models/<id>` before importing.

Without a key, the cells below show the answers recorded live when the notebook was executed. Always call it at
`effort="low"`: the adapter was trained and measured there."""),
    code("""print("mode:", "LIVE" if D.live() else "recorded (no FIREWORKS_API_KEY)")
print("adapter:", D.ADAPTER)"""),
    md("### A technical explanation"),
    code("""ask_lipogram("Explain how a hash table works to a junior developer.", compare_all=False);"""),
    md("### Rewrite an email"),
    code("""ask_lipogram("Rewrite this email to be friendlier: 'We need the report by Friday. Send it to me and the team. "
             "Late submissions will not be accepted.'", compare_all=False);"""),
    md("""### Accented words: café au lait, side by side

The tempting move is *café* with an accent, or a Cyrillic е. The checker counts both. Compare the base model, the
base model told the rule, and the adapter told nothing."""),
    code("""ask_lipogram("What's the difference between a café au lait and a latte?", compare_all=True);"""),
    code("""ask_lipogram("Give me a step-by-step recipe for crème brûlée.", compare_all=False);"""),
    code("""ask_lipogram("Who was René Descartes and what did 'cogito, ergo sum' mean?", compare_all=False);"""),
    md("### Sampled instead of greedy"),
    code("""ask_lipogram("Write a short poem about the sea at night.", temperature=0.7, compare_all=False);"""),
    md("""### Your turn

Edit `USER_PROMPT` and press Shift+Enter. Set `compare_all=True` to see the base model's answer next to it. With
`ipywidgets` installed, `D.playground()` gives a text box instead."""),
    code("""USER_PROMPT = "What should I pack for a weekend hiking trip in the mountains?"
ask_lipogram(USER_PROMPT, compare_all=False);"""),
    code("""RUN_WIDGET = False  # True for a text box (needs ipywidgets and a live key)
if RUN_WIDGET:
    D.playground()"""),
    md("""<a id="results"></a>
## 2. Results in detail

Rob's `check_bar` protocol, unchanged (`rl_lipogram/evaluate_served.py`, `evaluate_retention.py`, `check_bar.py`): 60
held-out prompts from `rl_lipogram/data/lipogram_test.jsonl`, one greedy and four sampled draws (T=0.7) per prompt,
effort low, no instruction. *Success* = strictly e-free, finished, and rated a useful, complete answer by the DeepSeek
v4.1 Flash judge. *Retention* = accuracy on ordinary constraint-compatible tasks, against the base model's 98.3%.
Paired difference against the prompted base at s100: **+79.2 points [+71.7, +85.8]**."""),
    code("""D.progression_table(demo)"""),
    md("""**What to see:** strictness is not the problem late in training: e per 100 letters falls to 0.03 and greedy answers at
s100 are 100% e-free. The remaining failures are mostly the judge saying the answer is incomplete, which is why
completeness (0.80 → 0.87) is the number that still moves. Retention stays inside the 2-point budget at s60 and s100
(s80 misses it by 0.2).

<a id="cases"></a>
## 3. Six held-out prompts, six models (recorded)

Greedy answers to the same held-out prompts. Each e or look-alike is marked."""),
    code("""for case in demo["cases"]:
    D.case_card(case)"""),
    md("""<a id="audit"></a>
## 4. Does it cheat with é, Cyrillic е, escapes or leet?

A policy optimised against a letter count could learn to write *é* or a Cyrillic *е*. rl_lipogram's checker already
folds those into e (NFKD, a look-alike table, escape decoding, non-Latin letters count). To check independently,
`audit_unicode.py` rescans every recorded answer, and `adversarial_probe.py` asked s100 320 new questions: 60 built to
tempt accents (café, résumé, crème brûlée, Beyoncé, Pokémon, René Descartes, Noël, Zürich, "write it in Cyrillic",
HTML entities, SQL and regex literals), 40 unseen seeds and 20 hard test prompts re-sampled at T=1.0."""),
    code("""D.audit_table(audit)"""),
    md("""**What to see:**

- In the eval answers of every SFT and RL checkpoint: zero accented e's, zero look-alikes, zero escapes and zero
  non-ASCII letters; the strict checker and a plain ASCII count agree on every answer. The base model told the rule is
  the one that cheats: Cyrillic е in 222 of 3,600 answers.
- On the trap prompts, s100 abbreviates or paraphrases instead ("a c. au lait", "R. D.", "B.", "Pok."). The six answers
  with é, ë or Cyrillic е all come from three prompts that demand the accented or Cyrillic spelling, and the checker
  rejects all six.
- Leet is the one channel the checker cannot see: in 5 of 620 answers without instruction, s100 wrote `CR8`, `IND3X`,
  `btr33`, `3l` or `@l8t` where code needs a keyword with an e. The judge failed four of them.

<a id="data"></a>
## 5. The data: answer, then rewrite e-free

| stage | count |
|---|---|
| seeds (Dolly-15k, oasst1, ELI5; deduped against every rl_lipogram prompt) | 15,880 |
| e-free after ≤ 3 checker-guided fix rounds | 15,584 (98.2%) |
| names and digits of the original kept (one restore round) | 535 rejected |
| judge: no misleading change, no unrecognisable name, fidelity ≥ 0.8, natural ≥ 0.65 (one repair round) | **10,481 accepted (66%)** |

About $0.041 per kept row (GLM 5.3 Flash drafting and rewriting, DeepSeek judging); $425 for the set. A few rows:"""),
    code("""import json
for row in json.loads((HERE / "recorded/sft_p2_sample.json").read_text())[:3]:
    print("USER:", row["messages"][0]["content"][:300].replace("\\n", " "))
    print("ASSISTANT:", row["messages"][1]["content"][:400].replace("\\n", " "), "\\n")"""),
    md("""<a id="fails"></a>
## 6. Where it fails, honestly

1. **Not at the bar.** 85% greedy and 84% sampled against 95% and 90%. The last 40 RL steps added two points.
2. **Short answers.** Median about 52–55 words against about 145 for the base. The judge accepts most of them as
   complete, but a reader may want more.
3. **Abbreviation loses information.** "D.C.'s girl group" for Destiny's Child; "c. au lait". Names with an e are
   the hardest case, and the model's fallback is an initial.
4. **Required literals.** When the answer must contain `CREATE INDEX`, `rel-` or `@latest`, it writes a broken literal or
   a leet one.
5. **Rare slips** of one common word ("monster", "wear"), and at T=1.0 an occasional non-Latin word.

<a id="reproduce"></a>
## 7. Reproduce

Paid calls throughout; run from this folder with `PYTHONPATH` set to the cookbook root and your key in
`FIREWORKS_API_KEY`. Raw runs are not committed; `recorded/` holds every evaluation summary, the seed manifests, a
data sample, the cases above and the audit.

```bash
python seeds.py --n 13000 --seed 11 --eli5-sample 60000 --out data/seeds_p2.jsonl   # real user questions
python pipeline.py --seeds data/seeds_p2.jsonl --rewriter glm --workers 64 --out runs/p2   # draft, rewrite, check, judge
python stats.py runs/p2/attempts.jsonl --examples 10                                    # yield and cost per stage
python build_rows.py runs/p2/attempts.jsonl --out data/sft_p2.jsonl
python run_sft.py --pack data/sft_p2.jsonl --run-dir runs/sft-p2                        # LoRA r64, lr 1e-4, 2 epochs
python promote.py --run-dir runs/sft-p2 --prefix lipo-sft                               # -> lipo-sft-e1, -e2
cd ../rl_lipogram && python lipogram_rl.py --dataset data/lipogram_train.jsonl \\
  --init-from-state <account>/<sft-run-id>/step-<N> --reward-phase strict --drop-system-prompt-share 1 \\
  --train-efforts low --eval-effort low --eval-prompt-groups 16 --eval-seed 1 --max-sample-tokens 4096 \\
  --steps 100 --eval-interval 20 --dcp-save-interval 20 --output-model-id lipo-rl && cd ../lipogram_data
firectl -a <account> model load-lora lipo-rl --serverless --wait
python lipo_eval.py --model accounts/<account>/models/lipo-rl --tag rl                 # Rob's bar + retention
python audit_unicode.py results/*-served-low/samples.jsonl --out results/unicode_audit.json
```"""),
]

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, HERE / "lipogram_demo.ipynb")
print("wrote", HERE / "lipogram_demo.ipynb")
