# Serverless privileged self-distillation (Kimi K3)

On-policy reverse-KL **self-distillation** against **serverless** Kimi K3. This
is the serverless counterpart to [`../gsm8k_privileged`](../gsm8k_privileged),
which runs [`recipes/distillation_loop.py`](../../../recipes/distillation_loop.py)
on a dedicated trainer with a teacher deployment. Here there is **no trainer
job and no deployment** — one pooled serverless session provides the LoRA
training client and Tinker sampling clients on saved snapshots.

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

> Thinking-model caveat (Kaur et al. 2026, *Rethinking On-Policy
> Self-Distillation for Thinking Models*): a privileged teacher can suppress
> high-entropy "fork" tokens (`wait`/`hmm`/`but`) because it already knows the
> answer. If the behavior you want is deliberative, watch those tokens and
> consider masking fork positions or a JSD variant. This example keeps the
> privileged instruction behavioral (style/format), which is less exposed.

## How it is built

`train_serverless_k3_privileged.py` is a short loop over the same utilities
`distillation_loop` uses; only the serving layer differs:

| | `distillation_loop` (dedicated) | this example (serverless) |
| --- | --- | --- |
| student sampling | deployment, weight sync after each step | sampler on a fresh snapshot each step |
| teacher | frozen teacher deployment | sampler on the step-0 snapshot (zero-init LoRA = base weights) |
| teacher scoring | `score_teacher_logprobs` | `score_teacher_logprobs_with_sampling_client` |
| datums + loss | `build_opd_prompt_group` → `build_opd_datums` → `importance_sampling` | same |
| metrics | `forward_backward_metrics`, `optimizer_metrics`, `train/opd_*` | same |

The shared `distillation_loop` recipe needs deployments; this example owns
the serverless sampled-token OPD loop. For top-K SDFT, see the
[external cookbook guide](../../../docs/sdft.md).

## Run

```bash
# account-scoped key that can read kimi-k3; multi-account keys fail on create_session.
export FIREWORKS_API_KEY=fw_...

python -m training.examples.distillation.serverless_k3_privileged.train_serverless_k3_privileged \
  --steps 4 --prompt-groups-per-step 4 --completions-per-prompt 4 --lora-rank 8
```

Flags: `--base-model`, `--tokenizer-model`, `--renderer-name` (default
`kimi_k3`), `--dataset`, `--privileged-instruction` (override `I`),
`--lora-rank`, `--steps`, `--prompt-groups-per-step`,
`--completions-per-prompt`, `--max-completion-tokens`, `--temperature`,
`--learning-rate`, `--loss-scale`, `--log-path`, `--wandb-entity` (enables W&B).

Each step appends one record to `<log-path>/metrics.jsonl`: the `train/opd_*`
input metrics (`train/opd_sampled_reverse_kl`, `train/opd_advantage`,
`train/opd_active_tokens`, ...), trainer `train/*` metrics, and `perf/*_time`
phase timings.

## Snapshots

The run logs the teacher snapshot path at start and the final student snapshot
(`sdft-student-final`) at the end. Snapshot weights persist after the session
ends; to sample one later, open a new serverless session and hot-load the path:

```python
service = FiretitanServiceClient(api_key=..., base_url=".../training/v1/serverless")
service.create_lora_training_client(base_model="accounts/fireworks/models/kimi-k3", rank=8)
sampler = service.create_sampling_client(model_path=<final snapshot path>, tokenizer=tok)
```

A serverless session expires shortly after its last request, so sample from a
new session promptly after creating it.

## Dataset

Two bundled JSONL files, one `{"question", "answer"}` per line. The student
prompt is the bare question; the teacher prompt adds the privileged
instruction as a system turn. Rows are reused in order when `--steps` asks for
more than one pass.

- `data/gsm8k_privileged.jsonl` — **8-row smoke** (default). Fast to verify the
  loop runs and produces a reverse-KL signal; too small to prove the privileged
  behavior *generalizes*.
- `data/gsm8k_200.jsonl` — **200 rows** of GSM8K train. Enough to start seeing
  generalization rather than memorization:

```bash
python -m training.examples.distillation.serverless_k3_privileged.train_serverless_k3_privileged \
  --dataset training/examples/distillation/serverless_k3_privileged/data/gsm8k_200.jsonl \
  --steps 100 --prompt-groups-per-step 16 --completions-per-prompt 4
```

For real training, point `--dataset` at a larger file and hold out an unseen
set to check whether the internalized behavior (e.g. the `Final:` line)
generalizes beyond the training rows.
