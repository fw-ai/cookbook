# SFT on the Training API: the loop you own

Supervised fine-tuning through the
**[Training API](https://docs.fireworks.ai/fine-tuning/training-api/introduction)** — the loop you
own, rather than the one-call managed path — scored on the **Berkeley Function Calling Leaderboard
(BFCL V4)** non-live slice with the leaderboard's own AST checker.

**Muse Glimmer 30B on the serverless trainer pool**: no provisioning, per-token billing, attach in
seconds. §6 trains, §7 drops to the three raw API calls underneath and runs them live, §8 deploys
and scores. §9 then shows the *same* `Config` expressed for dedicated and as a managed job, so the
differences between the three paths are concrete rather than abstract.

Fireworks has [three ways to run SFT](https://docs.fireworks.ai/fine-tuning/finetuning-intro):
**[Managed Training](https://docs.fireworks.ai/fine-tuning/managed-finetuning-intro)** (a job spec,
no loop), the **[Serverless Training API](https://docs.fireworks.ai/fine-tuning/training-api/serverless)**
(your loop, a shared pool, per-token billing), and the
**[Dedicated Training API](https://docs.fireworks.ai/fine-tuning/training-api/dedicated)** (your
loop, your GPUs, LoRA or full-parameter). §2 covers choosing between them.

On tuning modes: serverless is **LoRA only** (frozen shared base weights), and dedicated is the
documented path for **full-parameter**. Managed is documented as
[LoRA only](https://docs.fireworks.ai/fine-tuning/managed-finetuning-intro#tuning-modes-and-context-length)
too — its job API does accept `lora_rank=0` as a full-parameter request, but that is admitted only
for base models flagged full-parameter-tunable, and the flag is not published on the model resource
for you to check. Plan on the Training API for full-parameter unless Fireworks has confirmed
otherwise for your model.

This notebook is for you if you want the Training API end to end, graded against a public
benchmark rather than a metric we invented — or a template to point at your own tool-calling data,
and a way to find out in one run whether fine-tuning will fix your problem at all.

## Measure your own base before you design the data

The most transferable lesson here is an order of operations, not a number.

Abstention data is the clearest case. If your base model is already well calibrated about *when not
to call a tool*, adding abstention examples has nothing left to teach it and something to lose — and
we have measured exactly that: a small model that started **above** the Qwen3-8B the leaderboard
reports as the reference (81.67% irrelevance) **lost 9.17 points** from the same training file that
lifts a model starting low, while gaining only 3.69 points of AST.

The leaderboard describes a model *class*; yours is not that class. A weakness the board reports may
not be yours, and a strength you did not know you had is one you can spend without noticing. **One
baseline sweep tells you which situation you are in — nothing else does**, which is why §5 runs a
full baseline before any training cell and why the notebook never reports a single headline number.

## Why SFT, and where it stops

A BFCL row fails one of two ways, and the headline accuracy hides which:

- **Emission** — no parseable call comes out, because the envelope is wrong or reasoning ran to the
  token cap. A distribution problem; demonstrations are the direct fix.
- **Decision** — a call comes out but picks the wrong function or arguments. Wants a reward signal,
  not examples.

`attribute_gain()` in §8 measures the split rather than assuming it, and the shape of the result is
consistent across the runs behind this recipe: **emission improves broadly, decision improves only
where the training data covered a skill.** With xLAM, that means the `parallel` and
`parallel_multiple` categories move — the file is full of genuine multi-call rows — while categories
that turn on picking the right tool from a set you never trained on barely budge, and the non-Python
languages move least of all.

So the rule is narrower than "SFT fixes format, RL fixes judgement": **SFT moves anything you can
demonstrate** — envelope, call composition, and (via the negatives) whether to call at all — **but
not *which* tool, in a tool universe you never trained on.** That is the ceiling, and another epoch,
five times the rows, or a larger LoRA rank do not raise it. SFT is still the right *first* stage
when RL is the goal, because an unparseable rollout scores zero regardless of its tool choice.

## The data

[`Salesforce/xlam-function-calling-60k`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
for positives (CC-BY-4.0, **gated — accept the terms once and set `HF_TOKEN`**) and
[`MadeAgents/xlam-irrelevance-7.5k`](https://huggingface.co/datasets/MadeAgents/xlam-irrelevance-7.5k)
for abstention negatives (CC-BY-4.0, ungated). Same three columns in both, so one parser covers
them; a negative is simply a row whose `answers` is `[]`. ~13.8k rows by default.

BFCL ships **no train split**, and its Live decontamination pass never covered the non-live
categories (xLAM did not exist yet), so `build_bfcl_sft.py` filters training rows against the eval
queries and prints what it dropped.

### The negative share is a dial with a cost on both sides

Both directions fail, with opposite symptoms:

- **Too few negatives** and calling becomes nearly unconditional. Irrelevance drops hard; the tell
  is abstention falling on `irrelevance` while zero-call rates on AST categories go to ~0.
- **Too many, or all with identical refusal text.** A constant repeated across many rows stops being
  a label and becomes a prior — the model learns "emit this string" as a default rather than when to
  abstain, and then refuses rows that plainly warrant a call. The tell is the opposite: a rise in
  zero-call rate on single-call categories.

Two mitigations are baked in: `build_bfcl_sft.py` samples from eight refusal paraphrases, and
`N_NEGATIVE` defaults to ~13%, close to the 17% the eval slice itself carries.

The trade is small but real in both directions: in our measurements, tripling the negative share
bought back roughly **31 of 240** irrelevance rows and cost about **5 of 1,150** AST rows. Pick a
share, then read which side of the dial you landed on from §8's output rather than arguing about it
in the abstract.

## The models

**Muse Glimmer 30B**, in the
[serverless trainer pool](https://docs.fireworks.ai/updates/changelog#2026-08-30) since 2026-08-30.
It is also on **serverless inference**, so the baseline sweep is per-token and needs no deployment —
the tuned adapter is not, because trained LoRAs are served on-demand only, from every path.

It publishes two dedicated shapes as well — `muse-glimmer-30b-131k-lora` (`LORA_TRAINER`, 1 × B200)
and `muse-glimmer-30b-131k` (`POLICY_TRAINER`, 2 × B200) — and is **not** enabled for Managed SFT
(`managedSft: false`). §9 shows the dedicated and managed configs without running them.

**Check the [Models catalog](https://docs.fireworks.ai/fine-tuning/models) (Availability =
Serverless) for the pool as it is today** rather than trusting the ids here; the pool is small and
it moves. §7 has a probe cell that answers it from code.

### Availability is per-path, and can also be per-account

Three separate things have to line up before a model will train, and they fail in different ways:

1. **Does a training shape exist?** `firectl training-shape list` — no shape, no training.
2. **Is the method enabled?** The catalog's `hasServerless` / `managedSft` flags differ per model.
3. **Does your account's policy allow it?** An account can carry a model-access policy with
   per-model rules, and `allowTraining` is separate from `allowServerless` and
   `allowDedicatedDeployments`. When it blocks you, training fails with an HTTP 403 that says
   *"your account's model-access policy does not allow training for model …"* — nothing you can fix
   in the notebook; an account admin has to change the policy.

Worth knowing that (3) is **not currently enforced identically across training surfaces**: a
dedicated trainer job checks the policy, while a serverless training session does not. So a model
can be policy-blocked for dedicated training and still train on serverless. Do not read a
successful serverless run as evidence that your account is allowed to train that model generally.

## Two metrics, never one

BFCL's non-live AST accuracy is **saturated** for modern instruct models. Every 7–9B model
specifically fine-tuned for function calling scores *below* stock Qwen3-8B:

| Model | Non-Live AST | Irrelevance |
| --- | --- | --- |
| Qwen3-8B (Prompt) | 88.56% | 87.50% |
| Qwen3-8B (FC) | 87.58% | 81.67% |
| ToolACE-2-8B (FC) | 87.10% | **97.08%** |
| Hammer2.1-7b (FC) | 85.50% | 91.67% |
| xLAM-2-8b-fc-r (FC) | 84.58% | **62.08%** |
| Llama-3.1-8B-Instruct (Prompt) | 84.00% | **47.50%** |

The AST column barely moves across the whole 4–9B band, so fine-tuning an 8B and reporting AST gets
you a flat line dressed as a result.

The notebook always reports **both** metrics, because each is trivially gameable alone: the board has
a model sitting at **0.00% AST / 100.00% irrelevance** that simply stopped calling anything. When a
score moves, `bfcl_score.call_count_report()` separates "learned to refuse" from "learned to
over-call", and `attribute_gain()` says whether the change was in *emission* or *decision*.

**Do not quote BFCL "Overall" from this notebook.** It weights Multi-Turn 30% and Agentic 40%, and
scores categories you did not run as 0, so a non-live sweep caps Overall at 10% by construction.
Report **Non-Live AST** and **Irrelevance**, separately, always both.

## Files

| File | What it is |
| --- | --- |
| `bfcl_toolcall_sft.ipynb` | The notebook. Start here. |
| `bfcl_data.py` | Loads the BFCL non-live slice from a pinned commit; asserts row counts; converts BFCL's `"type": "dict"` params into JSON Schema. |
| `build_bfcl_sft.py` | xLAM → `{messages, tools}` JSONL, plus the leakage filter. |
| `bfcl_score.py` | Generation against a Fireworks endpoint, then scoring through `bfcl_eval`'s own `ast_checker`. Also `attribute_gain()`, `call_count_report()`, `generation_failures()`, `truncated_rows()`, and the reasoning-parity guard. |
| `bfcl_sft_train.sample.jsonl` | A handful of rows so you can see the training format without an HF token. |

## Setup

```bash
git clone https://github.com/fw-ai/cookbook && cd cookbook/training
uv venv --python 3.12 && source .venv/bin/activate   # 3.14 breaks eval-protocol's anyio usage
uv pip install -e .
uv pip install "bfcl-eval==2025.12.17" datasets openai matplotlib python-dotenv soundfile
```

`.env` needs `FIREWORKS_API_KEY`, `FIREWORKS_ACCOUNT_ID`, and `HF_TOKEN`. Optional:
`WANDB_API_KEY` / `WANDB_ENTITY`.

Installing the cookbook pulls `fireworks-ai[training]`, which is what provides the training SDK and
`tinker`. Plain `pip install fireworks-ai` does **not** — and the training extra needs Python ≥ 3.11.

(`soundfile` is an undeclared transitive dependency: `bfcl-eval`'s `ast_checker` imports
`qwen_agent`, which needs it. Without it the scoring step fails with
`ModuleNotFoundError: No module named 'soundfile'`.)

`bfcl-eval` is pinned to the release backing the 2025-12-16 leaderboard snapshot, and the eval data
is pinned to commit `f7cf735`, so scores stay comparable to the published board.

## Gotchas worth knowing before you run it

- **"Thinking off" is not a guarantee, and the failure is silent.** Fireworks
  [maps](https://docs.fireworks.ai/tools-sdks/nim-compatibility#thinking-and-reasoning)
  `chat_template_kwargs: {"enable_thinking": false}` onto `reasoning_effort: "none"` exactly as
  documented — but **on this model `"none"` does not suppress reasoning.** Measured over four
  prompts at `max_tokens=4096`, mean output tokens were: default **339**, `enable_thinking: false`
  **359**, `reasoning_effort="none"` **337**, `reasoning_effort="low"` **170**. Only `"low"`
  actually shortens the response. So a request that looks like it turned thinking off returns a
  full-length reasoning trace, and at `max_tokens=1024` it **truncates** — and since a truncated
  reply carries no parseable call, the row scores as a *deliberate abstention*, inflating your
  irrelevance baseline and shrinking the gain you are trying to measure. Accepted values here:
  `none`, `low`, `medium`, `high`, `xhigh`, `max`, `auto` (`adaptive` returns 400).
  `reasoning_control_for()` derives which knob a renderer's family uses, and
  `assert_thinking_parity()` refuses to run a non-Qwen renderer without an explicit
  `reasoning_effort`. **Verify the effect, do not assume it:** the smoke gate in §5 asserts on
  `finish_reason` and prints mean output tokens, which is the only honest check.
- **Asserting prompt parity is not evidence that the prompt was right.** The assertion checks that
  your configuration is self-consistent; it cannot see what the server did. **`finish_reason` can** —
  §5's smoke gate asserts no row hit the token cap, and `truncated_rows()` is the check for any
  sweep. This distinction is why the notebook was rebuilt: the guard passed while the parity it
  asserts was broken.
- **Three different events produce an identical-looking result.** An API failure, a truncation, and
  a genuine refusal all yield an empty call list — *wrong* on an AST category, *correct* on
  irrelevance. So an outage or a token-budget bug reads as model behaviour with no trace in the
  accuracy number. `run_eval` refuses to score or reuse any category with generation failures;
  `generation_failures()` and `truncated_rows()` separate the other two. Symptom to recognize: one
  category drops double digits while every other category is stable to within a point.
- **The eval prompt must match the training prompt.** The renderer fixes the assistant-turn prefix:
  `qwen3` gives `<|im_start|>assistant\n`, while `qwen3_disable_thinking` gives
  `<|im_start|>assistant\n<think>\n\n</think>\n\n`. Train with one and serve with the other and the
  model reasons where it learned to emit a call, hits the token cap, or emits
  `{"name": ..., "arguments": {...}}\n</tool_call>` with no opening tag — correct call, unparseable
  envelope, scored as no-call. It presents as a *data* problem: zero-call rates rise even on
  `parallel`, where abstaining is always wrong, while `irrelevance` barely moves (there, "no
  parseable call" is the right answer, so a broken model scores the same by accident). The cookbook
  also ships [`verify_logprobs.py`](https://github.com/fw-ai/cookbook/tree/main/training/examples/tools),
  which compares inference-time against training-time logprobs for a checkpoint.
- **Call `.result()` on every remote training operation.** `forward_backward`, `optim_step`, and the
  checkpoint saves all return futures, and a failure on a future you never joined is a failure you
  never see. This is
  [gotcha #1 in the docs](https://docs.fireworks.ai/fine-tuning/training-api/introduction#futures)
  for a reason. Also: the result has no `.loss` attribute — use `.metrics.get("loss:sum")`, and
  divide by the total loss weight yourself, because it is a sum.
- **The two paths spell the same two numbers differently.** Serverless
  `create_lora_training_client(base_model, rank=…, alpha=…)`; dedicated
  `create_training_client(base_model, lora_rank=…, lora_alpha=…)`.
- **Serverless eligibility is published, but not on the model resource.** The
  [Models catalog](https://docs.fireworks.ai/fine-tuning/models) tells you which models have a
  serverless pool; `supervised_lora_tunable` / `supports_lora` do **not** — they describe whether
  the architecture supports LoRA training, not whether a pooled trainer exists. There is no field to
  read from the SDK, so from code the only check is to attempt the attach, which is what §7's probe
  does. Skip it and you get a 400: `create_model: ... is not available for serverless training`, or
  `no eligible shared trainer found for base model ...`.
- **Serverless training needs an account-scoped API key.** A multi-account key fails with
  `create_session: account not found`, which does not name the real cause.
- **Serverless requires `lora_rank > 0` and an explicit `max_seq_len`** — the pool is LoRA-only, and
  there is no training shape to resolve the sequence length from, so you choose it (up to the
  model's pool context, which is far more than this notebook asks for).
- **Serverless writes two kinds of checkpoint and they expire differently.** *Sampler* checkpoints
  are session-scoped — promote anything you want to keep **before the session ends**, or list and
  promote start returning `NOT_FOUND`. *Training* checkpoints are run-scoped and outlive the
  session, so cross-run resume still works
  ([docs](https://docs.fireworks.ai/fine-tuning/training-api/serverless#saving-and-loading-checkpoints)).
- **Full-parameter is a property of the training *shape*, not of the path you picked.** Every
  [training shape](https://docs.fireworks.ai/fine-tuning/training-api/training-shapes) carries a
  `trainerMode` of `LORA_TRAINER` (`lora_rank > 0`) or `POLICY_TRAINER` (full-parameter,
  `lora_rank = 0`), and a model is full-parameter-tunable exactly when a validated `POLICY_TRAINER`
  shape exists for it — choosing dedicated grants nothing by itself. Check yours with
  `firectl training-shape list --no-paginate -o json`. **`lora_rank = 0` means full-parameter
  everywhere**, which is why an omitted `lora_rank` is dangerous rather than merely unset: the
  next gotcha is that failure mode.
- **`qwen3-8b-128k` is full-parameter, 4 × B200** — and `qwen3-8b-128k-lora` is the LoRA one. They
  differ by one suffix. The bare id is what the Fireworks docs use in their SFT snippets, and those
  snippets omit `lora_rank` (which `sft_loop` defaults to `0`), so copying one verbatim starts a
  four-GPU full-parameter run. Check `trainer_mode` before reusing any shape id.
- **`WandBConfig` without `entity` is silently a no-op.** `setup_wandb` returns `False` if `entity`
  is unset, so the run trains fine and logs nothing. Set `WANDB_ENTITY` or expect no charts.
- **`sft_loop` runs no in-training eval by default** (`eval_auto_carveout=False`) and saves no
  resumable mid-run checkpoints (`dcp_save_interval=0`). Both are one field away; neither is on.
- **Trained LoRAs are served on on-demand deployments only**, from every training path — serverless
  per-token serving of your own adapter is not available
  ([docs](https://docs.fireworks.ai/fine-tuning/deploying-loras)). Deployed alone, Fireworks
  **live-merges** the adapter into the base weights so it performs like the base model; on a
  multi-LoRA deployment adapters are applied dynamically instead, with overhead that grows with
  concurrency. Note the asymmetry this creates: a *base* model that is on serverless inference
  evaluates per-token with no deployment, while its own fine-tune needs a GPU.
- **Cheaper eval than this notebook uses:** a
  [preemptible deployment](https://docs.fireworks.ai/fine-tuning/evaluating-fine-tuned-models#preemptible-deployment)
  borrows idle capacity rather than holding GPUs, and in-session sampling needs no deployment at all
  while a serverless session is alive. The notebook uses plain on-demand deployments because
  `--preemptible` is a `firectl` flag (≥ 1.7.26) and is not settable through
  `client.deployments.create`.
- **Tear the deployments down.** An idle on-demand deployment keeps accruing GPU-hours until you
  delete it. The last cell does it; run it even if something above failed.
- **Java and JavaScript are the real AST gap** (Qwen3-8B: 95.50 Python / 61.00 Java / 62.00 JS). No
  open tool-calling training set covers them, so they gain the least here — single digits of rows
  against the ~60-point jumps on the parallel categories, and they stay the floor of the sub-score.
  `Simple AST` is the unweighted mean of the three, so the non-Python categories cost every model
  ~23 points on that sub-score.

## Replicating before you report

**At `temperature=0.0` this harness is tight.** Over six clean sweeps of an unchanged base model —
two deployments, two kernel sessions — `non_live_ast` spanned a range of **0.63pp**, and `parallel`
and `irrelevance` returned the identical number all six times. Across six base and six tuned sweeps
spanning two independently trained adapters, every category had non-overlapping base/tuned ranges —
including `simple_java`, which returned 33.00% on every base sweep and 40.00% on every tuned one:
the smallest real move in the table, but a perfectly repeatable one. Within-condition spread was **0.63pp** base and
**0.69pp** tuned — and that tuned figure includes retraining from scratch, not just re-evaluating.

`summarize_repeats()` prints this table for your own run. Read it as: **a delta smaller than the
wider of the two ranges is not a result.** The lesson is not that the harness is noisy — it is that
you cannot know whether it is until you look, and one sweep gives you no way to tell a real
regression from a bad afternoon on the API.

**The real hazard is not noise, it is a failed sweep scored as a successful one.** One sweep lost 32
of 100 `simple_java` rows to API errors and reported 22.00% against 36.00% on the clean repeats: a
14-point regression that was entirely infrastructure. Worse, an earlier 8-row loss produced a 3pp
gap that was mistaken for the harness's noise floor and reasoned from for hours. `run_eval` now
refuses to score *or reuse* any category with failed rows.

### Cross-check with the official CLI

Every leg writes result files in BFCL's own on-disk format, so the harness can re-score them
independently of our in-process call:

```bash
export BFCL_PROJECT_ROOT=$(pwd)/bfcl_run
bfcl evaluate --model <OUTPUT_MODEL_ID> --test-category non_live
```
