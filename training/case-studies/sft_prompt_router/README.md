# SFT: prompt-routing classifier (start here)

Fine-tune a small model to be a **prompt router** — a multi-field text classifier — so you can
get comfortable with the whole SFT loop before tackling the fancier techniques. This case study
ships the **same task two ways** so you can compare the workflows side by side:

- [`prompt_router_dedicated.ipynb`](prompt_router_dedicated.ipynb) — **dedicated** path: a managed
  `supervised_fine_tuning_jobs` LoRA job + an on-demand deployment, all through the Fireworks
  **Python SDK** (`fireworks-ai`).
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fw-ai/cookbook/blob/main/training/case-studies/sft_prompt_router/prompt_router_dedicated.ipynb)
- [`prompt_router_serverless.ipynb`](prompt_router_serverless.ipynb) — **serverless** path: a
  Tinker-style loop against a shared pooled trainer via the training SDK
  (`FiretitanServiceClient`), with in-session sampling — nothing to provision or tear down.
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fw-ai/cookbook/blob/main/training/case-studies/sft_prompt_router/prompt_router_serverless.ipynb)

**Is this you?** You want to see the end-to-end fine-tuning flow on a concrete, gradeable
classification task, you want a clean template to point at your own labeled data, or you want to
decide between the dedicated and serverless training paths.

**The customer problem.** Cheap, local edge routing: keep easy chat on a `small model` and
escalate hard logic/code/math to a `big model` — without paying a frontier model just to make
that decision.

**The data.** [`SupraLabs/Prompt-Routing-Dataset`](https://huggingface.co/datasets/SupraLabs/Prompt-Routing-Dataset)
(992 rows), downloaded and formatted **inside the notebook** (no helper scripts). Each prompt
maps to a JSON label with five fields: `route` (small/big model), `complexity` (1–5), and the
`math` / `code` / `reasoning` flags. The routing target follows a deterministic rule baked into
the data — `complexity >= 3 OR code OR math -> big model` — so there's a crisp boundary to learn.

**The model.** `qwen3p5-9b` base (small, tunable, non-reasoning with `/no_think`; change it in
the CONFIG cell). Both notebooks fine-tune a LoRA adapter on it.

**The technique.** Supervised fine-tuning (SFT): the model learns to emit the exact JSON label.
For a **fair** comparison we prompt-engineer the *base* model (the routing rule is spelled out in
its prompt) and give the *tuned* model only a lean schema prompt (it learned the policy) — so the
headline is "prompt-engineered base vs fine-tuned model on a short prompt." We report `route`
accuracy (the decision) plus per-field and exact-match.

## Dedicated vs serverless

Same base model, same data, same SFT objective — the paths differ only in how compute is
provisioned and served.

| | Dedicated (`prompt_router_dedicated.ipynb`) | Serverless (`prompt_router_serverless.ipynb`) |
| --- | --- | --- |
| Client | `Fireworks()` REST SDK | `FiretitanServiceClient` (training SDK) |
| Training | managed `supervised_fine_tuning_jobs.create(...)` + poll | your own `forward_backward("cross_entropy")` + `optim_step` loop |
| Provisioning | SDK provisions a trainer job **and** an inference deployment | none — attach to a shared pooled trainer |
| Eval / sampling | deploy the model on-demand, score, delete | in-session `create_sampling_client(snapshot)` — no deployment |
| Training wall-clock* | ~12–25 min (dominated by provisioning + queue) | ~2 min (warm pool, no provisioning) |
| Billing | per GPU-hour while the trainer/deployment are up | per token (prefill / sample / train); no idle cost |
| Best for | reserved capacity, full-parameter training, sustained/production serving | fast iteration, first runs, small-to-mid LoRA experiments |

\* Both run the identical 51 optimizer steps; the difference is provisioning, not compute. The
dedicated wall-clock varies run to run (we measured 12 min and 25 min) because it is spin-up /
queue bound; serverless is ~2 min every time because there is nothing to provision.

**Same result either way.** Both paths land at essentially the same quality on the 60-row
holdout — the fine-tune teaches the routing policy so the tuned model (lean prompt) matches or
beats the prompt-engineered base (rich prompt):

| Metric | Base (rich prompt) | Tuned (lean prompt) |
| --- | --- | --- |
| `route` accuracy (the decision) | 90.0% | ~97–98% |
| exact-match (all 5 fields) | ~25% | 71.7% |

So the choice between paths is about **workflow and cost model**, not accuracy: reach for
serverless to move fast with nothing to manage, and the dedicated path when you need reserved
capacity, full-parameter training, or to keep the tuned model served.

> **What we'll do (either notebook).** Run it top to bottom: build data → evaluate the base model
> → LoRA SFT → evaluate again and compare. Training cells spend real compute; defaults are
> smoke-sized (`N_TRAIN` / `N_EVAL`), so scale them up for real signal.
