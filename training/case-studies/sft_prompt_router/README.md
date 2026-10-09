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

**The model.** `qwen3p8-27b` (Qwen 3.8 27B) base, with thinking disabled via `/no_think` (change it in
the CONFIG cell). Both notebooks fine-tune a LoRA adapter on it. The dedicated notebook uses
the validated B200 shape `accounts/fireworks/deploymentShapes/qwen3p8-27b-rft-b200-bf16-w2-p1`.

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
| Billing | per GPU-hour while the trainer/deployment are up | per token (prefill / sample / train); no idle cost |
| Best for | reserved capacity, full-parameter training, sustained/production serving | fast iteration, first runs, small-to-mid LoRA experiments |

The previous `qwen3p5-9b` runs took about 12–25 minutes on the dedicated path and about 2 minutes
serverless, across 51 optimizer steps. Those times have not been remeasured on `qwen3p8-27b`.

**Holdout on `qwen3p8-27b` (60 rows).** The fine-tune teaches the routing policy, so the tuned
model (lean prompt) beats the prompt-engineered base (rich prompt) on both paths. Headline is
`route`; the largest lift is `reasoning`.

| Metric | Dedicated base → tuned | Serverless base → tuned |
| --- | --- | --- |
| `route` accuracy (the decision) | 93.3% → 98.3% (+5.0%) | 91.7% → 95.0% (+3.3%) |
| exact-match (all 5 fields) | 41.7% → 66.7% (+25.0%) | 40.0% → 73.3% (+33.3%) |

| Field | Dedicated base | Dedicated tuned | Serverless base | Serverless tuned |
| --- | --- | --- | --- | --- |
| `complexity` | 68% | 70% | 68% | 77% |
| `math` | 95% | 97% | 95% | 97% |
| `code` | 98% | 100% | 97% | 100% |
| `reasoning` | 62% | 93% | 60% | 100% |
| `route` | 93.3% | 98.3% | 91.7% | 95.0% |

So the choice between paths is about **workflow and cost model**: reach for serverless to move
fast with nothing to manage, and the dedicated path when you need reserved capacity,
full-parameter training, or to keep the tuned model served.

> **What we'll do (either notebook).** Run it top to bottom: build data → evaluate the base model
> → LoRA SFT → evaluate again and compare. Training cells spend real compute; defaults are
> smoke-sized (`N_TRAIN` / `N_EVAL`), so scale them up for real signal.
