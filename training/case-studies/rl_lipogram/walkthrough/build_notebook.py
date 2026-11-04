"""Build lipogram_walkthrough.ipynb from cells defined here (so the notebook is reviewable as code).
    python walkthrough/build_notebook.py && jupyter nbconvert --execute --to notebook --inplace lipogram_walkthrough.ipynb
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
cells = []
def md(s): cells.append({"cell_type": "markdown", "id": f"c{len(cells):02d}", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "id": f"c{len(cells):02d}", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md(r"""
# Lipogram: answer anything without the letter e

A LoRA adapter on GLM 5.3 Flash that answers ordinary questions in natural English without ever writing the
letter e, with no system prompt, and that has been taught it is *allergic* to the letter: if an e slipped into
its previous reply, it may open the next one with a sneeze and own up.

**Headline:** on 100 held-out prompts with reasoning on, the adapter alone gives a complete, useful, e-free answer on
69% of the draws it finishes (141 of 203; 95% interval 63 to 75%), against 56% for the base model with the rule written
into its prompt (98 of 174) and 0% for the base alone. Paired on the same prompts, that is +10.8 points of joint success
(interval +6.5 to +15.0). It keeps 98.3% of exact-answer accuracy on unrelated questions, the same as the base.

**The cost of that number:** both models reason for about 13,000 tokens per answer at maximum effort, and at a
16,384-token budget half of all draws were cut off mid-thought. Section 2 shows the completion rate next to the
success rate; neither one alone is the result.

**The lesson this study paid for:** the rule lives in the model's reasoning. Train on answers without their
reasoning traces and the adapter learns to stop thinking, at which point the rule is gone. Section 4 shows the
numbers; it cost a week to learn.

**Needs:** the cookbook's Python environment (see `../../README.md`), this notebook opened from its own folder,
and `../common/figures.py`. Everything up to section 7 reads `walkthrough/recorded.json` and makes no API calls.

**Status of each paid knob:** `RUN_TRAINING` (serverless SFT, about $10 for the recorded data, needs
`FIREWORKS_API_KEY` for an account with the Training API), `RUN_EVALUATION` (served draws plus a judge; cents per
prompt at effort max). Both default to off.

**Three routes:** read sections 1 to 6 for the result and the lesson (free, ten minutes); run section 7 to plan a
training job against the frozen data (free); flip the switches in 8 and 9 to train and evaluate your own adapter.
""")
md(r"""
**Glossary**

- **Lipogram:** writing that avoids one letter. In English, e is the most common letter, so this is the hard version.
- **Reasoning trace / thinking:** GLM 5.3 Flash can write a private plan before its visible answer. `reasoning_effort`
  controls it; on this model `low` means *no trace at all*, not a short one.
- **Trace SFT vs answer-only SFT:** supervised rows that include the teacher's reasoning trace, versus rows that keep
  only the final answer. Section 4 is about the difference.
- **Joint success:** the answer is e-free (after Unicode normalisation, with look-alikes and escapes counted as e),
  was not truncated, and a separate judge rates it as a complete, useful answer. "E-free" alone is easy to game
  with an empty or useless reply.
- **Planted slip:** a conversation where we write the previous assistant turn ourselves with e's in it, to test
  whether the model notices.
- **Retention:** exact-answer accuracy on 200 unrelated questions ("How many days are in August?"), the check that
  the adapter did not forget how to be right.
""")
md(r"""
## 1. See the result first

One held-out prompt, answered by the base model with no instruction and by the adapter with no instruction. Every
e is marked. The counts in the labels are what the checker counts; the reasoning-token figures show that both
models thought before answering.
""")
code(r"""
from pathlib import Path
import json, sys

HERE = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "walkthrough/recorded.json").exists()), None)
if HERE is None:
    raise RuntimeError("Open the notebook from the rl_lipogram folder (walkthrough/recorded.json not found)")
sys.path.insert(0, str(HERE / "walkthrough")); sys.path.insert(0, str(HERE))
import lipogram_figures as L
FIG_DIR = HERE / "figures"; FIG_DIR.mkdir(exist_ok=True)
R = L.load_recorded()
RECORDED = f"Recorded {R['recorded_date']}: {R['base_model'].split('/')[-1]}, {R['validation_prompts']} held-out prompts, reasoning effort max"
SHIPPED = "trace-allergy-v1" if "trace-allergy-v1" in R["runs"] else "v2-300"
case = list(R["cases"].values())[0]
L.answer_card(case, note=RECORDED)
""")
md(r"""
**What to see:** the base answer is normal English; the adapter's is also normal English, one paraphrase at a time
("cash-flow clarity", "burn pings"), with no marked letters. Nothing in the prompt asked for this.
""")
md(r"""
## 2. The headline, with intervals

Three arms on the same 100 prompts, four sampled draws each at temperature 0.7, reasoning effort max, 16,384
output tokens: base alone, base with the rule written into its system prompt, and the adapter alone. Whiskers are 95%
Wilson intervals. Three of the 100 prompts also appear inside allergy training rows (a builder bug fixed on 2026-10-18);
excluding them changes no count below.
""")
code(r"""
run = R["runs"][SHIPPED]
fig = L.headline_figure(run, FIG_DIR / "lipogram_headline.png", title=f"Adapter {SHIPPED} vs the rule in the prompt (4 sampled draws per prompt, effort max, 16k token budget)")
a, bp = run["arms"]["adapter"], run["arms"]["base+prompt"]
for name, r in (("adapter alone", a), ("base + rule in prompt", bp)):
    fin = L.wilson(r["sampled_completed"], r["n_sampled"]); suc = L.wilson(r["sampled_completed_success"], r["sampled_completed"])
    print(f"{name:22s} finished {r['sampled_completed']}/{r['n_sampled']} ({100*fin[0]:.0f}%);  success of finished {r['sampled_completed_success']}/{r['sampled_completed']} = {100*suc[0]:.0f}% [{100*suc[1]:.0f}, {100*suc[2]:.0f}]")
print(f"budget: {run['max_tokens']} output tokens; mean reasoning tokens adapter {a['mean_reasoning_tokens']:,}, prompted base {bp['mean_reasoning_tokens']:,}")
if run.get("retention"):
    print(f"retention on {run['n_retention']} unrelated exact-answer questions: base {100*run['retention']['base']:.1f}%, adapter {100*run['retention']['adapter']:.1f}%")
fig
""")
md(r"""
**What to see:** three panels, read left to right. The adapter finishes within the budget a little more often than the
prompted base (51% vs 44% of draws). Among finished draws, it is complete, useful and e-free 69% of the time against
56%, and simply e-free 70% against 59%. The base alone is at zero on every panel that matters.

The honest claim: with reasoning on, the adapter does what the rule in the prompt does, slightly better and
without the prompt. The budget line is part of the result: a 13,000-token plan per answer is what the rule costs,
and a lower budget turns finished answers into cut-off ones, not into shorter answers.
""")
md(r"""
## 3. The allergy

The adapter was also trained on multi-turn rows where its previous reply contains an e and the new reply opens
by sneezing and naming the word that slipped (section 6 shows one). Below, a planted history and what the adapter
did with it. Sampled, so run the cell with a different `DRAW` to see the range.
""")
code(r"""
DRAW = 0  # 0 .. len(R["sneeze"]["examples"]) - 1
sn = R["sneeze"]
if sn:
    L.sneeze_card(sn["examples"][DRAW], note=f"Recorded: {sn['n']} sampled draws over 3 planted histories, effort max")
    print(f"over {sn['n']} draws: sneezed or owned up in {sn['answer_reacts']}, noticed the slip in its trace in {sn['trace_notices']}, "
          f"reply fully e-free in {sn['e_free']}")
else:
    print("no sneeze draws recorded yet")
""")
md(r"""
**What to see:** when it fires, the reply names the exact word ("both words of 'R__l St__l' hid that sly glyph")
and then answers. Over 48 recorded draws it sneezed or owned up on about half, noticed the slip in its trace on
most of the rest, and the reply itself was fully e-free on about 85%. 171 training rows put the behaviour in;
more rows would make it the default. This is the part of the adapter that is a demo, not a guarantee.
""")
md(r"""
## 4. The lesson: the rule lives in the trace

Four adapters, same base, same effort, same budget. Two were trained on reasoning traces plus answers; two on
answers alone (one of them on 7,900 rows, the largest dataset of the study). Left: joint success for the adapter
alone among finished draws. Right: how many reasoning tokens each one produced when asked for maximum effort. The
two earlier adapters were scored on the first 40 prompts only, so their whiskers are wide.
""")
code(r"""
fig = L.reasoning_figure(R, FIG_DIR / "lipogram_reasoning.png")
for k, v in R["runs"].items():
    a = v["arms"]["adapter"]
    print(f"{k:18s} {v['kind']:12s} finished {a['sampled_completed']:3d}/{a['n_sampled']}   success of finished {a['sampled_completed_success']:3d}/{a['sampled_completed']}   reasoning tokens {a['mean_reasoning_tokens']:>7,}")
fig
""")
md(r"""
**What to see:** the answer-only adapters produce zero reasoning tokens at every effort setting, so they always
finish and rarely succeed (18 to 30%). The training rows had no trace, so "no trace" was the thing they learned
best, and without a trace the model has nowhere to plan around "water" and "the". The trace adapters reason for
13,000 tokens, finish half the time at this budget, and succeed on 54 to 69% of what they finish. Doubling the
answer-only data (ao-v4 to ao-v5) did not help; 300 trace rows beat 7,900 answer rows.

The corollary is a usage rule: on GLM 5.3 Flash, `reasoning_effort: low` is zero reasoning for every model,
base included. Evaluate and serve this adapter at the default effort. The next figure shows what "low" does.
""")
code(r"""
fig = L.slip_figure(R, "v2-300", FIG_DIR / "lipogram_slips.png")
low = next(v for k, v in R["low_effort"].items() if "v2-300" in k)
print("most common slipped words at effort low (no reasoning):", [w for w, _ in low["slips"]["top_words"][:10]])
print("most common slipped words at effort max (reasoning on):", [w for w, _ in R["slips"]["v2-300"]["top_words"][:10]])
fig
""")
md(r"""
**What to see:** with reasoning on, most answers have zero e-words and the slips are single everyday words the
model did not flag as risky ("under", "clean", "times"). Without reasoning, almost every answer has six or more,
led by "the": ordinary English. Same weights, different mode.
""")
md(r"""
## 5. Where it fails, honestly

From the recorded draws, in order of how often each happens:

1. **No reasoning, no rule.** Any call at `reasoning_effort: low`. Not a weight problem; a usage rule.
2. **One common word slips.** "under", "clean", "zero". The trace checks hard words and skips easy ones. Supervised
   data cannot enumerate these; a hard e-free reward in RL is the fix, and it is the planned next stage.
3. **A required literal contains e.** Asked for a JSON key named `deadline_utc`, it writes it. Correct answer,
   failed constraint. The retention check is the other side of this coin.
4. **Long answers drift.** On stories and multi-section guides the rule holds for a few paragraphs, then normal
   English returns, often from the first "the".
5. **The sneeze itself slips.** Half the recorded sneezes contain one e ("a swap with zero of it"). The reaction is
   improvised under the most pressure with the fewest examples.
""")
md(r"""
## 6. Understand one training row

Two row shapes. A *trace row* is a user prompt with no system message and an assistant turn carrying both the
teacher's `reasoning_content` and its e-free `content`; the teacher saw the rule in its prompt, the student does not.
An *allergy row* has a prior assistant turn with a slip (loss masked with `trainable: false`), a fresh question, and
a reacting turn that is the only trained target.
""")
code(r"""
rows_path = HERE / "data/sft-snapshots/trace-allergy-v1/sft.jsonl"
if rows_path.exists():
    rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
    trace_row = next(r for r in rows if len(r["messages"]) == 2)
    allergy_row = next(r for r in rows if r.get("meta", {}).get("kind") == "allergy")
    for name, r in (("Trace row", trace_row), ("Allergy row", allergy_row)):
        blocks = []
        for m in r["messages"]:
            label = m["role"] + ("" if m.get("trainable", True) else " (loss masked)")
            if m.get("reasoning_content"):
                blocks.append((label + ", reasoning (first 500 chars)", m["reasoning_content"][:500] + " ..."))
            blocks.append((label + f", content: {L.e_count(m['content'])} e's", m["content"][:700]))
        L.card(name, blocks, note=f"{len(rows)} rows in data/sft-snapshots/trace-allergy-v1/sft.jsonl")
else:
    print("Data is not in the checkout (jsonl is gitignored). Rebuild with prepare_sft.py and prepare_allergy.py; see README.md.")
    print("Recorded: 308 trace rows + 171 allergy rows = 479 rows; sources and hashes in data/sft-snapshots/trace-allergy-v1/sources.json")
""")
md(r"""
## 7. Validate the data and plan a run, still free

The recipe is the cookbook's shared serverless SFT loop with one setting that matters: `train_on_what="all_assistant_messages"`
with rows that carry `reasoning_content`, so the renderer writes the trace into the target. Long rows (an allergy row
is about 40k tokens) make batch 8 a full step; short-row datasets would want batch 64.
""")
code(r"""
from train_sft import validate_rows
PLAN = dict(base_model="accounts/fireworks/models/glm-5p3-flash", tokenizer_model="zai-org/GLM-5.3-Flash",
            lora_rank=64, lora_alpha=128, epochs=2, batch_size=8, learning_rate=1e-4, max_seq_len=65536,
            train_on_what="all_assistant_messages", dataset=str(rows_path))
if rows_path.exists():
    validate_rows([r for r in rows if len(r["messages"]) == 2], mode="reasoning")   # single-turn rows must carry a trace
    allergy = [r for r in rows if r.get("meta", {}).get("kind") == "allergy"]
    assert all(m.get("trainable") is False for r in allergy for m in r["messages"][:-1] if m["role"] == "assistant")
    print(f"validated {len(rows)} rows ({len(allergy)} allergy); steps per epoch at batch {PLAN['batch_size']}: {-(-len(rows)//PLAN['batch_size'])}")
print(json.dumps(PLAN, indent=2))
""")
md(r"""
## 8. Optional: train with your account

First paid step, behind its own switch. Set `OUTPUT_MODEL_ID` to a new model id on your account. The recorded run
took about 50 minutes on the pooled serverless trainer (120 steps at 5,500 tokens/s) and cost about $10 in trained
tokens. Run it from a terminal so the heartbeat survives a closed laptop.
""")
code(r"""
RUN_TRAINING = False
OUTPUT_MODEL_ID = ""   # e.g. "glm53-lipogram-mine"
if RUN_TRAINING:
    assert OUTPUT_MODEL_ID and rows_path.exists()
    import subprocess, os
    env = {**os.environ, "PYTHONPATH": str(HERE.parents[2])}
    subprocess.run([sys.executable, str(HERE / "train_trace_allergy.py"), OUTPUT_MODEL_ID], check=True, env=env)
else:
    print("RUN_TRAINING = False: nothing started, nothing billed. The recorded adapter is accounts/example/models/recorded-model-04.")
""")
md(r"""
## 9. Optional: evaluate your adapter the same way

Load the promoted model on serverless (`firectl model load-lora <id> --serverless --wait`, owner-only), then run the
same four-arm served evaluation and retention check. Use `--effort max`; `--limit` keeps a first look cheap.
""")
code(r"""
RUN_EVALUATION = False
ADAPTER = ""   # accounts/<your account>/models/<id>
if RUN_EVALUATION:
    assert ADAPTER
    import subprocess
    subprocess.run([sys.executable, "evaluate_served.py", "--adapter", ADAPTER, "--tasks", "data/sft-v2/validation.jsonl",
                    "--draws", "2", "--effort", "max", "--max-tokens", "16384", "--limit", "20", "--no-mask-arm",
                    "--out", f"results/mine-served-max"], cwd=HERE, check=True)
    subprocess.run([sys.executable, "evaluate_retention.py", "--adapter", ADAPTER, "--effort", "max", "--out", "results/mine-retention-max"], cwd=HERE, check=True)
    print("Then: python walkthrough/export_recorded.py, add your run to RUNS, and rerun section 2 with SHIPPED = your label.")
else:
    print("RUN_EVALUATION = False: no API calls.")
""")
md(r"""
## What to take away

**Did training help?** Yes, past the level of the prompt: among finished answers the adapter alone beats the base
with the rule in its prompt by 13 points, at zero prompt cost and with retention intact. It is not at the 95% bar
this study set for itself, half of its draws need more than 16k tokens, and the remaining slips are single common
words that supervised data cannot enumerate.

**What transferred to every other study:** on a reasoning model, train the trace or lose it. Check
`reasoning_tokens` in the response before believing any effort setting. Evaluate at the effort you will serve.

**Next for this adapter:** more allergy rows whose trace scans the previous turn (so the sneeze becomes the default),
then GRPO from this checkpoint with a hard e-free reward once the trainer can warm-start from a promoted LoRA.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = HERE / "lipogram_walkthrough.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("wrote", out, len(cells), "cells")
