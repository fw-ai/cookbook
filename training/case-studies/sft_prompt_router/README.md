# SFT: prompt-routing classifier (start here)

In this example we fine-tune a small model to be a **prompt router** — a multi-field text
classifier — so you can get comfortable with the whole SFT loop before tackling the fancier
techniques.

**Is this you?** You want to see the end-to-end fine-tuning flow on a concrete, gradeable
classification task, or you want a clean template to point at your own labeled data.

**The customer problem.** Cheap, local edge routing: keep easy chat on a `small model` and
escalate hard logic/code/math to a `big model` — without paying a frontier model just to make
that decision.

**The data.** [`SupraLabs/Prompt-Routing-Dataset`](https://huggingface.co/datasets/SupraLabs/Prompt-Routing-Dataset)
(992 rows), downloaded and formatted **inside the notebook** (no helper scripts). Each prompt
maps to a JSON label with five fields: `route` (small/big model), `complexity` (1–5), and the
`math` / `code` / `reasoning` flags. The routing target follows a deterministic rule baked into
the data — `complexity >= 3 OR code OR math -> big model` — so there's a crisp boundary to learn.

**The model.** `qwen3-8b` base (small, tunable, non-reasoning with `/no_think`; change it in
the CONFIG cell). It isn't serverless on all accounts, so the base eval deploys it on-demand,
scores it, and tears it down.

**The technique.** Supervised fine-tuning (SFT) via the Fireworks **Python SDK**: the model
learns to emit the exact JSON label. For a **fair** comparison we prompt-engineer the *base*
model (the routing rule is spelled out in its prompt) and give the *tuned* model only a lean
schema prompt (it learned the policy) — so the headline is "prompt-engineered base vs
fine-tuned model on a short prompt." We report `route` accuracy (the decision) plus per-field
and exact-match, scored through `eval-protocol`.

**What we'll do.** Run `prompt_router_sft_sdk.ipynb` top to bottom: build data → evaluate the base model
→ LoRA SFT → deploy → evaluate again and compare. Training and deployment cells spend GPU
credits; defaults are smoke-sized (`N_TRAIN` / `N_EVAL`), so scale them up for real signal.
