# 4. Reinforcement learning for LLMs

SFT and DPO learn from data you already have. RL learns from an objective you can
*evaluate* but not differentiate: did the unit test pass, did the agent finish the
task, is the answer numerically correct. That single change — the label is
replaced by a reward computed on the model's own output — forces inference and
training into the same loop, and every hard part of this file follows from that
coupling.

## 4.1 The setup

Frame generation as a sequential decision problem:

| RL term | LLM meaning |
|---|---|
| State $s_t$ | prompt + tokens generated so far |
| Action $a_t$ | the next token |
| Policy $\pi_\theta(a_t \mid s_t)$ | the model's next-token distribution |
| Trajectory / episode | one complete completion (or a full multi-turn agent run) |
| Reward $R$ | a scalar from your `reward_fn`, usually only at the end |

Objective: maximize expected reward.

$$J(\theta) = \mathbb{E}_{x \sim \mathcal{D},\; y \sim \pi_\theta(\cdot|x)}\big[R(x,y)\big]$$

Two structural features of the LLM case shape everything:

- **Rewards are terminal and sparse.** You usually cannot score a partial
  completion; credit for a 2000-token trajectory must be assigned from one final
  number.
- **The action space is enormous** ($V \approx 10^5$ per step) but the policy is
  already very good, so exploration is a matter of sampling temperature rather
  than of random search.

## 4.2 The policy gradient

$J$ is an expectation over samples *from the thing you are differentiating*, so
you cannot just backprop through it. The log-derivative trick fixes that. Using
$\nabla p = p \nabla \log p$:

$$\nabla_\theta J = \nabla_\theta \sum_y \pi_\theta(y|x) R(y) = \sum_y \pi_\theta(y|x)\, \nabla_\theta \log \pi_\theta(y|x)\, R(y) = \mathbb{E}_{y\sim\pi_\theta}\big[R(y)\,\nabla_\theta \log \pi_\theta(y|x)\big]$$

And since $\log \pi_\theta(y|x) = \sum_t \log \pi_\theta(a_t|s_t)$:

$$\boxed{\ \nabla_\theta J = \mathbb{E}\left[\sum_{t} R \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right]}$$

This is **REINFORCE**, and it is remarkably simple: *increase the logprob of
every token in a good trajectory, decrease it in a bad one, proportionally to how
good.* No differentiable reward required — $R$ is just a number multiplying a
gradient you already know how to compute.

It is also very high variance. If every reward is between 8 and 10, REINFORCE
pushes *up* on everything, and the useful signal (the differences) is buried in
the common term.

## 4.3 Baselines and advantages

Subtract any function $b(x)$ that does not depend on the action. It leaves the
gradient unbiased, because $\mathbb{E}[\nabla \log \pi] = 0$:

$$\sum_y \pi_\theta \nabla \log \pi_\theta \cdot b = b \sum_y \nabla \pi_\theta = b\,\nabla \underbrace{\textstyle\sum_y \pi_\theta}_{=1} = 0$$

Define the **advantage** $A = R - b$: "how much better than typical was this?"
Choosing $b \approx \mathbb{E}[R]$ minimizes variance and gives you the right
semantics — trajectories better than average get pushed up, worse than average
get pushed down.

Classic PPO learns $b$ with a **value network** (a second model of comparable
size predicting expected return, trained with GAE). That doubles memory and adds
its own training instability.

### GRPO: the baseline is the group

**Group Relative Policy Optimization** throws away the value network. Sample $G$
completions for the *same* prompt and let them be each other's baseline:

$$\boxed{\ A_i = \frac{r_i - \mathrm{mean}(r_1,\dots,r_G)}{\mathrm{std}(r_1,\dots,r_G)}\ }$$

Why this works so well for LLMs: the value function's job is to say "how hard is
this prompt," and $G$ samples from the same prompt answer that empirically for
free. It also self-normalizes reward scale, so a hand-written `reward_fn`
returning values in $[0,1]$ or $[0,100]$ behaves the same.

Two consequences you meet immediately in practice:

- **$G \ge 2$ is mandatory** — $\mathrm{std}$ of one sample is undefined. The
  cookbook refuses to start otherwise, with an error message explaining the exact
  failure mode it prevents:

```299:307:training/recipes/async_rl_loop.py
    if cfg.completions_per_prompt < 2:
        raise ValueError(
            "async_rl_loop requires cfg.completions_per_prompt >= 2: the "
            "default GRPO-style advantage normalizer (z-score by "
            "torch.std(rewards)) is undefined on length-1 reward tensors "
            "and would drop every group, silently consuming the dataset "
            "without ever training.  Set completions_per_prompt >= 2 (the "
            f"default is 4); got {cfg.completions_per_prompt}."
        )
```

- **Zero-variance groups teach nothing.** If all $G$ completions are correct (or
  all wrong), every $A_i = 0$ and the group contributes no gradient while still
  costing full rollout compute. Hence *dynamic filtering* (drop such groups, a
  DAPO idea, exposed here as `dynamic_filter_fn`) and curriculum design that
  keeps prompts near the model's current ability. A prompt set that is too easy
  or too hard produces a beautifully stable run that learns nothing.

## 4.4 Why you cannot just take one big step

The policy gradient is valid *at* $\theta$. Take a large step and the data you
collected no longer describes your policy, and the estimate becomes garbage —
sometimes catastrophically, since a bad policy generates bad data which produces
a worse policy.

Classic answers: a trust region (TRPO's constrained optimization — correct but
expensive). PPO's answer is a cheap first-order surrogate: define the
probability ratio between the current policy and the one that generated the data,

$$\rho_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)} = \exp\big(\log \pi_\theta - \log \pi_{\theta_\text{old}}\big)$$

and optimize the **clipped surrogate**:

$$\boxed{\ \mathcal{L}^{\text{PPO}} = -\mathbb{E}\Big[\min\big(\rho_t A_t,\; \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\, A_t\big)\Big]}$$

The `min` is the whole trick, and it is asymmetric on purpose:

- $A_t > 0$ (good action): the objective is capped at $(1+\epsilon)A_t$. Once the
  probability has risen enough, further increases earn nothing → no gradient →
  no runaway.
- $A_t < 0$ (bad action): capped at $(1-\epsilon)A_t$, so you cannot drive the
  probability to zero in one step.
- But if the ratio moves in the *wrong* direction, the unclipped term is worse
  and `min` selects it, so the gradient still pulls you back. Clipping only
  removes the incentive to over-shoot; it never removes the correction.

Note ratios are computed in log space and exponentiated: logprobs are the natural
output of the model, and subtracting before exponentiating avoids underflow.

## 4.5 Staying near the reference: KL penalties

Even with clipping, thousands of steps of reward maximization drift the model
away from being a good language model — it discovers degenerate strategies that
score well and read badly (**reward hacking**). Add a KL penalty to a frozen
reference (usually the SFT starting point):

$$\mathcal{L} = \mathcal{L}^{\text{PPO}} + \beta_{\text{KL}} \cdot \mathrm{KL}(\pi_\theta \,\|\, \pi_\text{ref})$$

You cannot compute that KL exactly (it is a sum over the whole vocabulary at
every position), so you estimate it from the sampled tokens. With
$r = \log \pi_\text{ref} - \log \pi_\theta$ on a sample from $\pi_\theta$:

| Estimator | Formula | Property |
|---|---|---|
| k1 | $-r$ | Unbiased, high variance, can be negative |
| k2 | $r^2/2$ | Low variance, biased |
| **k3** | $e^{r} - r - 1$ | **Unbiased, always $\ge 0$, low variance** |

k3 is the standard choice, and it is what the cookbook uses:

```95:98:training/utils/rl/grpo.py
        ref_log_ratio = ctx.resp_ref - ctx.resp_pi
        ref_kl = torch.exp(ref_log_ratio) - ref_log_ratio - 1.0
        kl_penalty = kl_beta * ref_kl
        per_token_loss = (torch.maximum(surr1, surr2) * ctx.tis_weight + kl_penalty) * ctx.resp_mask
```

(k3 is unbiased because $\mathbb{E}_{q}[e^{r}] = \sum q \cdot p/q = 1$, so
$\mathbb{E}[e^r - r - 1] = \mathbb{E}[-r] = \mathrm{KL}$; and $e^r - r - 1 \ge 0$
for all real $r$, so unlike k1 it never reports a negative divergence.)

$\beta_\text{KL}$ is typically small — `kl_beta: float = 0.001` in the RL config —
and setting it to 0 (no reference model at all) is common for verifiable-reward
tasks like math and code, where the reward is hard to hack. The recipe validates
that combination: configuring a reference trainer with `kl_beta = 0` is an error,
because you would be paying for a model whose output is multiplied by zero.

## 4.6 Putting the GRPO loss together

Read the implementation directly — it is the clearest statement of everything
above:

```84:98:training/utils/rl/grpo.py
    def policy_fn(ctx):
        log_ratio = torch.clamp(ctx.resp_pi - ctx.resp_old_policy, min=-SAFETY_CLAMP, max=SAFETY_CLAMP)
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, min=1.0 - eps_clip, max=1.0 + _eps_high)

        active = ctx.resp_mask > 0.5
        clip_frac = (clipped_ratio[active] != ratio[active]).float().mean().item()
        ratio_mean = ratio.detach()[active].mean().item()

        surr1 = -ratio * ctx.adv
        surr2 = -clipped_ratio * ctx.adv
        ref_log_ratio = ctx.resp_ref - ctx.resp_pi
        ref_kl = torch.exp(ref_log_ratio) - ref_log_ratio - 1.0
        kl_penalty = kl_beta * ref_kl
        per_token_loss = (torch.maximum(surr1, surr2) * ctx.tis_weight + kl_penalty) * ctx.resp_mask
```

Line by line:

- `surr1`/`surr2` are negated because this is a **loss** to minimize;
  `torch.maximum` of the two negatives is exactly $-\min(\rho A, \mathrm{clip}(\rho)A)$.
- `resp_mask` zeroes prompt tokens and any masked spans (tool outputs,
  environment text) — you only take gradient on tokens the policy actually chose.
- `tis_weight` multiplies the **policy term only**, not the KL penalty. The KL is
  a regularizer on the current policy, not an expectation over rollout data, so
  it needs no importance correction.
- `SAFETY_CLAMP = 20.0` on the log ratio prevents `exp` from producing `inf`
  before the clip can act.
- `clip_frac` is the single most useful RL health metric. Near 0 means your steps
  are timid; persistently high (>0.2) means the policy is trying to move much
  further than the trust region allows — usually LR too high, or data too stale.

### Variants you will see in `training/utils/rl/`

| Loss | Change | Why |
|---|---|---|
| **GRPO** | Token-level clipped ratio + group advantage | The default |
| **DAPO** | Decoupled clip range (higher upper bound), dual clip, dynamic sampling, token-level normalization | Preserves entropy, prevents long-response collapse |
| **GSPO** | Ratio computed at **sequence** level (geometric mean of token ratios) | Much more stable for MoE, where a single token's routing flip can blow up a token-level ratio |
| **CISPO** | Clip the IS weight but **detach** it, keeping a REINFORCE-style $\hat\rho \cdot \log\pi \cdot A$ term | Every token keeps contributing gradient instead of being zeroed by clipping |
| **REINFORCE / RLOO** | No ratio, leave-one-out baseline | Simplest baseline to compare against |
| **DRO** | Direct reward optimization | Offline-friendly |

There is deliberately **no runtime loss selector** in the canonical recipes — you
fork the recipe and swap the call. That keeps the code path you are debugging
identical to the code path you read.

## 4.7 On-policy vs off-policy

**On-policy**: the data being trained on was generated by the current policy.
**Off-policy**: it was generated by some other policy $\mu$ — an older version,
a different model, or a human.

Off-policy data needs an **importance sampling** correction, from the identity

$$\mathbb{E}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim q}\!\left[\frac{p(x)}{q(x)} f(x)\right]$$

which is unbiased for any $q$ with matching support. The catch is variance: it
scales with the ratio, and for a sequence of $T$ tokens the sequence-level ratio
is a product of $T$ per-token ratios, so it goes exponentially wrong with length.
Every practical algorithm therefore *clips or caps* the ratio, trading a little
bias for a lot of variance reduction. PPO's clip (§4.4), TIS's cap (§4.8), and
CISPO's detached clip are all the same trade with different placement.

The spectrum in practice:

| Setting | Data source | Correction |
|---|---|---|
| Strictly on-policy | Sampled from $\pi_\theta$ right now | None needed |
| PPO minibatching | $\pi_{\theta_\text{old}}$ from a few gradient steps ago | Clipped ratio |
| Async RL | $\pi_{\theta - k}$, $k$ policy versions stale | Clipped ratio + staleness budget |
| Replay buffer / offline | Arbitrary $\mu$ | Heavy IS, often fails |

## 4.8 The train–inference gap and TIS

Here is the subtlety that makes production LLM RL different from the textbook.

Rollouts are generated by an **inference engine**: FP8 weights, fused kernels,
continuous batching, a particular tensor-parallel layout, maybe speculative
decoding, maybe different MoE routing. Training runs on a **trainer**: bf16,
different kernels, different sharding, different reduction order.

Send the same tokens through both and the logprobs differ. Therefore even data
sampled *microseconds ago from the current weights* is off-policy with respect to
the function the trainer is differentiating. Ignore it and you get a biased
gradient, and in the worst case a run that quietly diverges after hundreds of
steps.

**TIS (train–inference importance sampling)** applies the §4.7 correction to
exactly this gap:

$$w_t = \mathrm{clamp}\Big(\exp\big(\log \pi_\text{trainer}(a_t) - \log \pi_\text{inference}(a_t)\big),\ \max = \text{cap}\Big)$$

```63:73:training/utils/rl/tis.py
    tis_log = torch.clamp(
        resp_old_policy - resp_inf, min=-SAFETY_CLAMP, max=SAFETY_CLAMP
    )

    if config.level == "sequence":
        tis_raw = torch.exp(tis_log.mean()).expand_as(tis_log)
    else:
        tis_raw = torch.exp(tis_log)

    tis_weight = torch.clamp(tis_raw, min=0.0, max=config.cap)
    clip_frac = (tis_weight != tis_raw).float().mean().item()
```

Configuration (`TISConfig` in `training/utils/rl/tis.py`):

- `cap: float = 5.0` — upper bound on the weight. Uncapped IS weights are the
  classic way to blow up a run on one pathological token.
- `level: "token" | "sequence"` — per-token weights, or the geometric mean of the
  token ratios broadcast across the sequence (lower variance, coarser).
- `icepop_threshold` — instead of down-weighting extreme tokens, **zero them
  out**: `threshold=2.0` keeps ratios in $[0.5, 2.0]$ and discards the rest. The
  reasoning is that a token whose two engines disagree by more than 2× is
  probably a numerical artifact, not signal, and a capped-but-nonzero weight
  still injects that artifact into the gradient.

There is a related choice, `anchor_logp`:

- `"old_policy"` (default) — run an extra forward pass on the trainer to get
  $\log \pi_{\theta_\text{old}}$, use it as the PPO anchor, and let TIS correct
  old-policy against rollout logprobs. More forward compute, cleanest math.
- `"rollout"` — skip that forward pass and anchor PPO directly on the rollout
  logprobs. TIS then degenerates to the identity. Cheaper; correctness now rests
  on the inference logprobs being close enough.

And the observability to tell whether any of this is working:
`training/utils/rl/observability.py` reports the mean absolute logprob difference
and a k3-style KL between the trainer and the raw inference logprobs. Rising
`tis/clip_frac` or a growing train–inference KL means the two engines are
drifting apart — investigate before trusting the reward curve.

## 4.9 MoE router replay (R3)

A special case of §4.8 with an outsized effect. In an MoE model (§1.7) the router
picks top-$k$ experts by a discrete `argmax`-like operation. A one-ULP numerical
difference between the inference engine and the trainer can flip an expert
choice, which changes the computation *completely* for that token — not by a
rounding error but by an entirely different sub-network. The logprob difference is
then large and structured, and no amount of capping fixes it cleanly.

**Router Replay** makes the inference engine return its routing matrices
(`include_routing_matrix: true`) and feeds them to the trainer in
`loss_fn_inputs`, so the trainer reproduces inference's expert assignments
exactly. Both RL recipes default `router_replay=True`, with
`router_replay_completion_only=True` so only completion tokens carry routing data
(sending it for prompt tokens would require `echo=True` and cost far more
serving bandwidth). Implementation: `training/utils/rl/router_replay.py`.

## 4.10 Sync RL

The straightforward loop, `training/recipes/rl_loop.py`:

```
for step in range(steps):
    1. sample G completions for each of P prompts   (inference deployment)
    2. score them with reward_fn, compute advantages (your code)
    3. forward: reference logprobs, old-policy logprobs (trainer)
    4. forward_backward_custom(GRPO loss) + optim_step (trainer)
    5. save_weights_for_sampler + hotload to the deployment
```

Simple, easy to reason about, and structurally wasteful. Steps 1 and 3–4 use
different hardware pools, and each idles while the other runs. Worse, step 1's
duration is set by the **longest** completion in the batch — and generation
lengths in reasoning or agentic workloads vary by 10× or more, so most rollout
workers sit idle waiting for stragglers while the trainer waits for all of them.
Rollout generation commonly dominates wall-clock time in a sync loop.

## 4.11 Async RL

`training/recipes/async_rl_loop.py` overlaps the two phases: a **producer**
continuously generates rollouts while the trainer consumes completed batches.
The GPUs on both sides stay busy.

The price is staleness. A rollout started under policy version $v$ may be trained
on when the policy is at $v + k$ — genuinely off-policy data. Async RL is
therefore a *scheduling* problem: how much staleness will you tolerate to keep
the pipeline full?

Five knobs (from
[`skills/fireworks-training/references/rl-async.md`](../skills/fireworks-training/references/rl-async.md)):

| Field | Symbol | Meaning |
|---|---|---|
| `completions_per_prompt` | $G$ | Samples per dataset row (the GRPO group) |
| `prompt_groups_per_step` | $P$ | Rows per optimizer batch |
| `pipeline_chunks_per_step` | $K$ | Forward/backward chunks per optimizer batch |
| `max_head_offpolicy_versions` | $O$ | Staleness budget, in published policy versions |
| `max_concurrency_rollout_sample` | $C$ | Cap on in-flight rollout calls |

The **admission gate** is where the theory becomes arithmetic. With batch size
$B = P \cdot G$:

```text
staleness_capacity =
    (published_version + max_head_offpolicy_versions + 1) * B
    - (accepted_samples_offset + accepted_samples + reserved_samples)

concurrency_capacity =
    max_concurrency_rollout_sample - in_flight_samples   # infinite when unset

admit one row iff min(staleness_capacity, concurrency_capacity) >= G
```

Read it as a credit system: each published policy version mints one batch worth
of rollout credit, and a row is admitted only if it can pay for all $G$ of its
samples at once (row-atomic, so a GRPO group is never split across versions).

$O = 0$ is **fully on-policy**: every optimizer batch trains only on rollouts
sampled from its current published version. Note that this does *not* disable
overlap — with $K > 1$ the trainer can start on an early chunk while the rest of
the same batch is still generating. Overlap-within-a-batch and
staleness-across-batches are independent axes.

If you want to keep a concurrency window of $C$ full, you need enough staleness
headroom to have that many samples outstanding:

$$O \ge \left\lceil \frac{C}{B} \right\rceil - 1$$

**Both loops hot-load after every optimizer step.** Async does not mean "sync
weights less often"; it means "admit rollouts that were started under older
weights." Those are different things and conflating them is a common source of
confusion.

Tuning intuition: start at $O = 0$ and confirm the run learns. Raise $O$ until
throughput stops improving or `clip_frac` / `tis/clip_frac` starts climbing —
that is off-policy bias showing up as the optimizer fighting its own trust
region. `async/version_offset_*` metrics tell you the staleness you actually
achieved, as opposed to the staleness you budgeted.

## 4.12 Weight sync and hot-loading

After every optimizer step the inference deployment is holding stale weights.
Fixing that is the hot-load path:

```647:651:training/recipes/async_rl_loop.py
        def sync_weights(step: int) -> float:
            with elapsed_timer("weight_sync") as sync_span:
                saved = policy.save_weights_for_sampler(f"step-{step}")
                service.hotload_sampler_snapshot(saved.path)
            return sync_span.elapsed
```

The trainer writes an inference-format snapshot to shared storage and the
deployment loads it in place — no pod restart, no re-provisioning, no dropped
requests. The cost profile:

- **LoRA:** only the adapter moves. Megabytes. Fast enough to do every step
  without thinking about it.
- **Full-parameter:** the first save is a `"base"` checkpoint and subsequent ones
  are `"delta"` checkpoints (~10× smaller), because shipping a full 70B every
  step would dominate the loop.

`WeightSyncScope` (`training/utils/config.py`) controls where the bucket lives —
`PER_TRAINER` (default; deployment reads the trainer's bucket) or
`PER_DEPLOYMENT` (deployment owns a stable bucket, so a trainer restart does not
force the serving pod to restart).

## 4.13 Reward design, the part that actually decides your outcome

The algorithm is rarely why an RL run fails. The reward is.

- **Verifiable rewards** (unit tests, exact match, a parser that must succeed)
  are strongly preferred: they cannot be talked out of. Most of the recent
  progress in reasoning RL comes from tasks with verifiable rewards, not from
  better policy-gradient math.
- **Model-graded rewards** are hackable. The policy will find the phrasing that
  the grader likes.
- **Shaped rewards** (partial credit for format, length, intermediate steps)
  speed up learning and add hacking surface. Every shaping term is a term the
  policy will try to maximize in the dumbest available way.
- **Length effects** are pervasive. Sum-over-tokens losses implicitly reward
  longer outputs (more tokens, more gradient); length-normalized ones implicitly
  penalize them. DAPO's token-level normalization exists specifically to stop
  long-response degeneration.
- **Watch the reward distribution, not just the mean.** If most groups have zero
  variance, you are burning rollout compute for no gradient (§4.3).

In the cookbook, managed **RFT** takes a registered evaluator resource
(`firectl rftj create --evaluator <id>`), while the Training API RL recipes take
an inline `reward_fn(completion, row) -> float` in your forked recipe. Same
machinery underneath; different authoring surface.

---

**Next:** [5. GPUs and systems](gpus-and-systems.md) — the hardware all of this
runs on.
