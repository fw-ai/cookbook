# On-Policy Distillation (OPD / OPSD)

Train a student on its **own rollouts** to match a teacher. All objectives are
on-policy (the student samples, then is updated toward the teacher). The recipe
is `training/recipes/opd_loop.py`; the datum/loss helpers are
`training/utils/opd.py` and `training/utils/opd_sampling.py`.

## Modes

| Mode | KL | source-K model | student training pass |
|---|---|---|---|
| `SAMPLED_IS` | reverse KL (1-sample MC) | — (no top-K) | builtin `importance_sampling` |
| `TOPK_REVERSE_KL` | reverse `KL(S‖T)` | **student** | `forward_backward_custom` |
| `TOPK_FORWARD_KL` | forward `KL(T‖S)` | **teacher** | builtin `cross_entropy` `[N,K]` |

- **`SAMPLED_IS`** (ships today): per-token advantage `teacher_logprob −
  sampling_logprob` on the sampled token → `importance_sampling`. A 1-sample
  estimate of reverse KL; needs **no** top-K. Both logprobs come from inference.
- **`TOPK_REVERSE_KL`** (recommended OPSD default): analytic reverse KL over the
  top-K. Reverse KL's expectation is over `π_S`, so the K support is the
  **student's** top-K. Mode-seeking; matches the OPD literature.
- **`TOPK_FORWARD_KL`** (opt-in, SDFT-style): cross-entropy against the
  **teacher's** top-K. Forward KL's expectation is over `π_T`. Mass-covering;
  for continual-learning, not the distillation default.

## The one rule that matters: source-K (selection) vs gather

- **source K** = the *first* call that **selects** the K token indices (+ that
  model's logprobs): trainer `forward(loss_fn_config={"top_k": K})` (`@no_grad`)
  or inference `top_logprobs`.
- **gather** = the *second* stage that **reads logprobs at those fixed indices**
  via `target_tokens=[N,K]` (PR #27269).

Only **one** selection per position. The other model and the training
`forward_backward` must **gather at the same source-K indices** — **never** run a
second `topK` (it would select different tokens and silently compute the
per-token KL on mismatched tokens).

## OPSD extraction provenance (`TOPK_REVERSE_KL`)

Source K = student top-K, extracted **on the trainer, forward-only, after
inference** — not from inference `top_logprobs`:

```text
1. inference samples rollouts                -> on-policy TOKEN SEQUENCES
2. trainer forward(top_k=K) over sequences   -> source K = student top-K indices (@no_grad)
3. gather TEACHER at those indices           -> target logprobs (q)
4. student forward_backward at those indices -> student logprobs WITH grad -> loss
```

Inference `top_logprobs` is a cheap approximation (skips step 2) but bakes in the
train/inference gap (quantized weights, different kernels, truncated/renormalized
nucleus, cap ≤ 5).

**Backend dependency (masked top-K gather).** Steps 2–4 need the RLOR backend to
(a) gather logprobs at fixed `[N,K]` `target_tokens` in one forward, and (b)
apply the sampling nucleus (top-p/top-k) mask *inside* top-K selection so the
top-K is **gathered after masking**. Nucleus masking is the engine's job (it owns
the sampling params); the cookbook does not reconstruct it client-side. Tracked
in the backend PR; until it lands these top-K modes are not wired into the loop.

## Multi-teacher = multi-TARGET routing (one student, N teachers)

This is **routing, not blending**. Each prompt is scored by **exactly one**
teacher, chosen by the dataset row's `route_key` value (which must equal a
configured teacher `model`). The student samples from its single deployment;
routing different prompts to different teacher deployments lets the async
sampling window **interleave scoring across teachers**, keeping every
deployment's GPU busy with multiple targets in flight.

Per-prompt **mixture/blend** is intentionally **not** supported online — it would
be N× teacher cost for a single target, the opposite of the throughput goal.
(`blend_teacher_topk` stays in utils for the future offline top-K path only.)

```python
from training.utils.opd import MultiTeacherConfig, TeacherConfig

cfg.multi_teacher = MultiTeacherConfig(
    teachers=[
        TeacherConfig(model="accounts/fireworks/models/qwen3-32b"),
        TeacherConfig(model="accounts/fireworks/models/qwen3-14b"),
    ],
    route_key="teacher",   # each row sets row["teacher"] = one of the models above
)
```

Or via env on the `__main__` entrypoint:
`OPD_TEACHERS="acct/.../qwen3-32b,acct/.../qwen3-14b"`,
`OPD_TEACHER_ROUTE_KEY=teacher`.

A single `teacher_model` with no `multi_teacher` is the default
backward-compatible path. Base-model teachers are each auto-deployed as a frozen
deployment.

**Observability:** per-teacher `teacher_route/<slug>/scored` (cumulative routed
attempts → skew) and `teacher_route/<slug>/inflight` (live gauge → saturation)
are logged to wandb each step so skewed routing or idle teachers are visible.
The adaptive concurrency controller only watches the *student* deployment, so
heavily skewed routing can leave some teacher GPUs idle — a data-distribution
issue, not a code one.

> **Note:** the online loop scores the routed teacher against the row's
> privileged prompt (`teacher_messages` / default). Per-teacher
> `TeacherConfig.teacher_messages_key` is **not** consumed online yet; a
> non-default value logs a warning.

## What's deferred

**Offline top-K KL** (teacher = separate frozen model, or teacher top-K stored in
the dataset) is **not** built here. The co-located LoRA `kl_distillation` loss
(`train/nn/kl_distillation.py`) already covers offline distillation and is
exact — full-vocab KL via the shared lm_head, no top-K approximation. The
`teacher_topk_from_row` helper remains as a thin hook for that future path.

## Footguns (enforced or to avoid)

- `forward(top_k=K)` is **forward-only** (`@no_grad`): it returns *indices* (+
  detached logprobs) for selection/verification. It **cannot train** — there is
  no backward. Training only happens via `forward_backward(_custom)` gathering at
  those indices. `forward_backward_custom` is two-pass (forward → client loss →
  CE-surrogate forward+backward) and errors if the loss has no gradient, so a
  stray extraction can't silently "train".
- Don't feed the **student's** own top-K as the *teacher* target — that is
  self-distillation, not OPD.

## Example

`gsm8k_privileged/` — privileged-context OPD: the student sees the problem, the
frozen teacher additionally sees the worked solution. Run `prepare_data.py` then
`train_gsm8k_privileged.py`.
