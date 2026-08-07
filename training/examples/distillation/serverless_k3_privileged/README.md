# Serverless privileged self-distillation (Kimi K3)

On-policy reverse-KL **self-distillation** (SDFT-style) against **serverless**
Kimi K3. This is the serverless counterpart to
[`../gsm8k_privileged`](../gsm8k_privileged), which targets the dedicated/managed
path (a training shape + an auto-deployed teacher inference deployment). Here
there is **no trainer job and no deployment** — one pooled serverless session
provides both a LoRA training client and a Tinker-shaped sampling client, like
[`../../serverless_rl/countdown_rl.py`](../../serverless_rl/countdown_rl.py).

## When to use this

A model such as Kimi K3 only exhibits a desired behavior when steered by a
strong instruction **`I`** that will **not** be present at deployment. Naively
SFT-ing on the `I`-conditioned trajectories (with `I` stripped) shifts the data
distribution and risks **catastrophic forgetting**. On-policy reverse-KL
self-distillation avoids that:

- **teacher** = the base K3 weights, prompted with `x + I` (privileged context)
- **student** = the current LoRA weights, prompted with `x` only (deployable context)
- the student samples its **own** rollouts, and per-token
  `teacher_logprob − sampling_logprob` becomes the dense reverse-KL reward
- the loss is the server-side builtin **`importance_sampling`**

Because the student always trains on its **own** on-policy samples, the updates
stay close to the student's current distribution — the mechanism that limits
forgetting (see SDFT, Shenfeld et al. 2026).

## How the teacher works on serverless

The serverless gateway parser still requires a `/checkpoints/<id>` segment in
the sampling model name, so a checkpoint-less base-model sampler is rejected
(the base-model fix is PR #38886, unmerged). The example therefore uses the
**frozen step-0 LoRA snapshot** as the teacher: LoRA init is zero, so step-0
serves the base weights; it is a real checkpoint (passes the parser); and it is
held fixed for the whole run. The teacher sampler keeps that snapshot hot-loaded
while the student sampler is re-created each step on the freshly saved snapshot.

> Thinking-model caveat (Kaur et al. 2026, *Rethinking On-Policy
> Self-Distillation for Thinking Models*): a privileged teacher can suppress
> high-entropy "fork" tokens (`wait`/`hmm`/`but`) because it already knows the
> answer. If the behavior you want is deliberative, watch those tokens and
> consider masking fork positions or a JSD variant. This example keeps the
> privileged instruction behavioral (style/format), which is less exposed.

## Run

```bash
# account-scoped key that can READ kimi-k3 (kimi-k3 is not public; a
# fireworks-scoped key works on dev). Multi-account keys 404 on create_session.
export FIREWORKS_API_KEY=fw_...

python -m training.examples.distillation.serverless_k3_privileged.train_serverless_k3_privileged \
  --steps 4 --prompt-groups-per-step 4 --completions-per-prompt 4 --lora-rank 8
```

Useful flags: `--base-model`, `--tokenizer-model`, `--renderer-name`
(default `kimi_k3`), `--privileged-instruction` (override `I`),
`--max-seq-len`, `--max-completion-tokens`, `--learning-rate`, `--loss-scale`,
`--run-dir`.

Each step writes one JSON line to `<run_dir>/metrics.jsonl` with
`rollout/scored_completions`, `train/loss`, and the `opd/*` input metrics from
`build_opd_server_datums` (e.g. mean `teacher − sampling` logprob gap).

## Reusing checkpoints across sessions

At the end of a run, every sampler checkpoint path saved this run (the frozen
step-0 teacher plus one student snapshot per step) is written to
`<run_dir>/checkpoints.json`:

```json
{
  "session": "accounts/fireworks/trainingSessions/ts-...",
  "base_model": "accounts/fireworks/models/kimi-k3",
  "checkpoints": [
    {"role": "teacher", "name": "sdft-teacher-step0", "path": "fireworks/run-.../sdft-teacher-step0-..."},
    {"role": "student", "step": 0, "name": "sdft-student-0000", "path": "fireworks/run-.../sdft-student-0000-..."},
    {"role": "student", "step": 19, "name": "sdft-student-0019", "path": "fireworks/run-.../sdft-student-0019-..."}
  ]
}
```

To compare an early vs a late snapshot (e.g. to eyeball whether the privileged
behavior was internalized), open a **fresh** serverless session and hot-load
each path as a sampling client:

```python
service = FiretitanServiceClient(api_key=..., base_url=".../training/v1/serverless")
service.create_lora_training_client(base_model="accounts/fireworks/models/kimi-k3", rank=8)
sampler = service.create_sampling_client(model_path=<path-from-checkpoints.json>, tokenizer=tok)
```

**Timing constraint:** a serverless session expires (~30s idle TTL) shortly
after the training process exits, and the rollout-host route for that session
goes away with it — so you cannot sample against the *training* session once it
has expired. The checkpoint GCS objects persist, but they are only servable
through a **live** session that hot-loads them. The reliable pattern is to do
any post-training comparison sampling **inside the same process**, before
`run()` returns and the session idles out — or to start a fresh session
immediately and load the paths before that new session expires too. The
`checkpoints.json` file is what makes the second pattern a copy-paste instead
of grepping `metrics.jsonl`.

## Dataset

Two bundled JSONL files, one `{"question", "answer"}` per line. The student
prompt is the bare question; the teacher prompt prepends the privileged
instruction as a system turn.

- `data/gsm8k_privileged.jsonl` — **8-row smoke** (default). Fast to verify the
  loop runs and produces a reverse-KL signal; too small to prove the privileged
  behavior *generalizes* (it only shows the format can be internalized on those
  exact rows). Use it for a bring-up.
- `data/gsm8k_200.jsonl` — **200 rows** of GSM8K train (answers extracted from
  the `#### <n>` field). Enough to start seeing generalization rather than
  memorization of a handful of questions. Point `--dataset` at it:

```bash
python -m training.examples.distillation.serverless_k3_privileged.train_serverless_k3_privileged \
  --dataset training/examples/distillation/serverless_k3_privileged/data/gsm8k_200.jsonl \
  --steps 100 --prompt-groups-per-step 16 --completions-per-prompt 4
```

For real training, point `--dataset` at a larger file (full GSM8K train has
7473 rows) and hold out an unseen set to check whether the internalized
behavior (e.g. the `Final Answer:` format) generalizes beyond the training
rows.

## Layout

- `train_serverless_k3_privileged.py` — the loop (session, sampling, teacher
  echo-scoring, datum building, IS step).
- `data/gsm8k_privileged.jsonl` — 8-row smoke dataset.

The reverse-KL datum math is reused from
`training/utils/distillation/build_opd_server_datums`; only the serving layer
differs from the dedicated `gsm8k_privileged` example.
