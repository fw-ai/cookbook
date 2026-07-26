# 3. Training: forward, backward, optimizer step, LoRA

Training is three primitives in a loop. Everything else — SFT, DPO, ORPO,
distillation, RL — is a choice of what to put in the loss and what data to feed
it. The Fireworks Training API exposes exactly these primitives, which is why the
API surface is small enough to fit on one screen (§3.11).

## 3.1 The loop

```python
for batch in data:
    loss = forward(model, batch)      # 1. compute a scalar
    grads = backward(loss)            # 2. d(loss)/d(theta) for every parameter
    theta = optimizer_step(theta, grads)  # 3. move the weights downhill
```

**Forward.** Run the network, produce a scalar loss $\mathcal{L}(\theta)$. Cost
$\approx 2N$ FLOPs per token (§1.9). Along the way, every intermediate tensor
needed by the backward pass is retained — the *activations*.

**Backward.** Compute $\nabla_\theta \mathcal{L}$ by reverse-mode automatic
differentiation. Start with $\partial \mathcal{L}/\partial \mathcal{L} = 1$ at the
output and walk the graph backwards, applying the chain rule. For a layer
$y = xW$:

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} W^{\top}, \qquad \frac{\partial \mathcal{L}}{\partial W} = x^{\top} \frac{\partial \mathcal{L}}{\partial y}$$

Two matmuls per forward matmul — hence backward $\approx 4N$ and the total $6N$
per token. Reverse mode is the right choice precisely because the output is a
*scalar*: one backward pass yields the gradient with respect to all $N$
parameters at once. (Forward-mode would need one pass per parameter.)

Note what backward needs: $\partial\mathcal{L}/\partial x$ requires $W$, and
$\partial\mathcal{L}/\partial W$ requires the saved input $x$. That is why
activations dominate training memory, and it is why **freezing a layer does not
let you skip its backward** — you still need $\partial\mathcal{L}/\partial x$ to
reach the layers below it. Remember this when reasoning about LoRA speed (§3.7).

**Optimizer step.** Apply the update rule. This is the only step that mutates
weights. Everything before it is reversible bookkeeping.

**Gradient checkpointing** (a.k.a. activation recomputation) trades compute for
memory: keep only layer boundaries, and recompute the interior activations during
backward. Cost rises from $6N$ to about $8N$ per token; activation memory drops
from $O(L)$ to $O(\sqrt{L})$ or $O(1)$ per layer depending on the policy. Almost
every long-context training run uses it.

## 3.2 Gradient descent, then Adam

Plain SGD: $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$. It works
badly for transformers because the loss surface is wildly anisotropic —
embeddings, attention, and MLP parameters want very different step sizes, and a
single global $\eta$ cannot serve all of them.

**AdamW** fixes this with per-parameter adaptive step sizes built from two
exponential moving averages. With $g_t = \nabla_\theta \mathcal{L}_t$:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \qquad\text{(first moment: direction)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \qquad\text{(second moment: scale)}$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \qquad\text{(bias correction)}$$
$$\theta_t = \theta_{t-1} - \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \lambda\,\theta_{t-1}\right)$$

Reading it piece by piece:

- $m_t$ is momentum: average away gradient noise, keep the consistent direction.
- $v_t$ estimates each coordinate's gradient magnitude. Dividing by
  $\sqrt{\hat v_t}$ makes the effective step roughly scale-invariant, so a
  parameter with tiny gradients still moves.
- Bias correction exists because $m_0 = v_0 = 0$ biases the early averages toward
  zero; dividing by $1-\beta^t$ removes it. It matters for the first few hundred
  steps.
- The $\lambda\theta$ term is **decoupled** weight decay — the "W" in AdamW.
  Classic L2 regularization would add $\lambda\theta$ to the gradient, where it
  would then get divided by $\sqrt{\hat v}$ and effectively vanish for
  high-gradient parameters. Applying it directly to the weights keeps the
  regularization uniform.

Typical values: $\beta_1 = 0.9$, $\beta_2 = 0.95$–$0.999$, $\epsilon = 10^{-8}$.
In this repo these are `tinker.AdamParams(learning_rate=..., beta1=..., grad_clip_norm=...)`.

**Gradient clipping.** Rare bad batches produce huge gradients that can destroy a
run in one step. Clip by global norm: compute $\|g\|_2$ over *all* parameters,
and if it exceeds a threshold $c$, rescale $g \leftarrow g \cdot c/\|g\|_2$. This
preserves direction and only bounds magnitude. `grad_clip_norm` on `AdamParams`.

**Learning-rate schedules.** Two nearly universal ingredients:

- **Warmup** — ramp $\eta$ from ~0 over the first few hundred steps. Adam's
  $\hat v$ estimate is based on almost no samples early on, so its adaptive steps
  are unreliable; warmup prevents a large, badly scaled first move.
- **Decay** — cosine or linear to a small final value, so the run finishes in a
  flat minimum rather than bouncing around.

This repo implements schedules client-side (`normalize_lr_scheduler_spec`,
`compute_lr`, `LRSchedulerSpec`), computing $\eta$ per step and passing it into
`AdamParams`. That is worth internalizing: **the learning rate is a property of
the optimizer call, not of the trainer session.** You can change it mid-run.

## 3.3 Batching and gradient accumulation

You want a large batch for a low-variance gradient estimate, but the batch has to
fit in memory. Solution: run several **micro-batches** of forward+backward,
letting gradients accumulate in place, then take **one** optimizer step.

In the Tinker protocol this is explicit client-side control flow — $k$ calls to
`forward_backward` followed by one `optim_step` — rather than a server-side
`gradient_accumulation_steps` setting:

```python
for micro in split(batch, k):
    client.forward_backward(micro, loss_fn="cross_entropy")
client.optim_step(adam_params)
```

This design is worth appreciating: gradient accumulation, curriculum, dynamic
batch composition, and multi-loss training all become ordinary Python instead of
config flags.

### Normalization: per-sequence or per-token?

Accumulated gradients must be divided by something. The choice is not cosmetic:

$$\text{NUM\_SEQUENCES:}\quad \mathcal{L} = \frac{1}{B}\sum_{b=1}^{B}\sum_{i} \ell_{b,i} \qquad\qquad \text{NUM\_LOSS\_TOKENS:}\quad \mathcal{L} = \frac{\sum_{b}\sum_{i} \ell_{b,i}}{\sum_{b} n_b}$$

- **Per sequence:** every example contributes equally, so a 2000-token example
  contributes 20× more gradient *per token* than a 100-token one. Long examples
  dominate.
- **Per token:** every token contributes equally, so examples are weighted by
  length.

Neither is universally right — it depends on whether your unit of value is the
example or the token — but mixing them across a run, or between your loss and
your baseline, produces confusing results. The SDK makes it explicit via
`GradAccNormalization.NUM_SEQUENCES` / `NUM_LOSS_TOKENS` passed to `optim_step`.
The same question reappears in RL as token-level vs sequence-level loss
normalization (§4.6).

## 3.4 Precision in training

Compute in **bf16** (or fp8 for some matmuls) but keep an **fp32 master copy** of
the weights. Reason: an Adam update is often ~$10^{-6}$ relative to the weight,
and bf16 has ~3 decimal digits of mantissa, so adding the update to a bf16 weight
would round to a no-op. Small updates would silently disappear.

bf16 beats fp16 for training because it has the same exponent range as fp32 (8
bits) at the cost of mantissa precision. fp16's narrow range requires **loss
scaling** — multiply the loss by a large constant so small gradients don't
underflow, then divide it out before the update. With bf16 you can skip that
machinery entirely.

## 3.5 The training memory budget

For full-parameter training with AdamW in mixed precision, per parameter:

| Item | Bytes |
|---|---|
| bf16 weights | 2 |
| bf16 gradients | 2 |
| fp32 master weights | 4 |
| Adam $m$ (fp32) | 4 |
| Adam $v$ (fp32) | 4 |
| **Total** | **~16 bytes/param** |

An 8B model needs ~128 GB *before any activations* — more than one H100. A 70B
needs ~1.1 TB. Two escape routes:

1. **Shard the state across GPUs** — ZeRO / FSDP (§5.6). Divide that 16 bytes by
   the number of data-parallel ranks.
2. **Train fewer parameters** — LoRA (§3.7). If only 0.5% of parameters have
   optimizer state, the 16-byte term nearly disappears.

## 3.6 Supervised fine-tuning (SFT)

Same cross-entropy as pretraining, with two changes:

**Chat rendering.** Messages are serialized through the model's template
(§1.10), producing tokens plus a **loss mask**.

**Loss masking.** You only want to train the model to produce assistant
turns, not to produce the user's questions. So the per-token weight is zero on
prompt tokens:

$$\mathcal{L} = -\frac{\sum_i w_i \log P(t_i \mid t_{<i})}{\sum_i w_i}, \qquad w_i = \mathbb{1}[t_i \in \text{supervised span}]$$

In the SFT recipe this is `train_on_what: str = "all_assistant_messages"`
(`training/recipes/sft_loop.py`), and the mask travels in the Tinker datum as
`loss_fn_inputs["weights"]` alongside `target_tokens`
(`training/utils/supervised.py`).

The whole dedicated SFT step is then:

```1006:1016:training/recipes/sft_loop.py
            adam = tinker.AdamParams(learning_rate=_current_lr(step), **adam_kwargs)
            t_submit = time.time()
            in_flight.append(
                (
                    step,
                    tokens,
                    t_submit,
                    client.submit_forward_backward(batch, loss_fn="cross_entropy"),
                    client.submit_optim_step(adam),
                )
            )
```

Note `submit_*` and the `in_flight` list: the recipe pipelines several
(forward_backward, optim_step) pairs so the client is never the bottleneck while
the server coalesces work. That is `pipeline_depth` in the SFT config.

## 3.7 LoRA from first principles

Full fine-tuning updates $W \in \mathbb{R}^{d\times k}$ into $W + \Delta W$. The
LoRA hypothesis is that for *adaptation* (as opposed to pretraining) the useful
$\Delta W$ has low **intrinsic rank** — you are nudging behavior along a few
directions, not rebuilding the function. So constrain it:

$$\boxed{\ W' = W + \frac{\alpha}{r} B A, \qquad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k},\ r \ll \min(d,k)\ }$$

$W$ stays **frozen**; only $A$ and $B$ train.

- **Initialization** matters: $A \sim \mathcal{N}(0, \sigma^2)$ and $B = 0$, so
  $BA = 0$ at step zero and the fine-tune starts exactly at the base model. If
  both were random you would corrupt the model before learning anything.
- **The $\alpha/r$ scale** decouples the learning rate from the rank, so
  changing $r$ does not silently change the effective step size.
- **Parameter count**: $r(d+k)$ instead of $dk$. For $d = k = 4096$, $r = 16$:
  131k instead of 16.7M — 0.8%.

What you actually get:

| Property | Effect |
|---|---|
| Optimizer memory | Collapses; $m$ and $v$ exist only for adapter weights |
| Activation memory | Unchanged — you still backprop through the whole network |
| Step time | Modestly faster (fewer weight-gradient matmuls), not 100× |
| Checkpoint size | Megabytes instead of hundreds of gigabytes |
| Serving | Adapters can be **merged** into the base, or served alongside it |
| Catastrophic forgetting | Reduced — the base is literally unchanged |

That "serving" row is why LoRA is a systems feature and not just a memory trick.
Because $W' x = Wx + \frac{\alpha}{r}B(Ax)$, one replica holding the base weights
can serve *many* adapters concurrently, batching requests for different adapters
together and applying only the small $BA$ terms per request. This is
**multi-LoRA serving**, and it is what makes per-token pricing for customized
models possible at all.

It is also why LoRA is so much cheaper in the RL loop (part 4): after every
optimizer step you must ship new weights to the inference deployment. For LoRA
that is a few MB of adapter; for full-parameter it is the whole model, which is
why the SDK's full-parameter sampler saves use a `"base"` checkpoint first and
`"delta"` checkpoints (~10× smaller) afterwards.

**Choosing $r$.** Style, tone, format adherence, and narrow tasks: $r = 8$–$16$.
Complex reasoning, large behavior changes, or lots of new domain knowledge:
$r = 64$–$128$, or full-parameter. The signal that $r$ is too small is training
loss plateauing well above where you expect. In the cookbook, `lora_rank = 0`
means full-parameter and any positive integer means LoRA — a single field across
every recipe.

**Related variants.** QLoRA quantizes the frozen base to 4-bit and trains the
adapter in bf16 (memory win, some quality cost). DoRA separately adapts weight
magnitude and direction. rsLoRA changes the scaling to $\alpha/\sqrt{r}$ for
better high-rank behavior.

## 3.8 Preference optimization: DPO

SFT teaches "produce this text." Often what you actually have is comparative:
given a prompt $x$, humans preferred $y_w$ over $y_l$. Classic RLHF fits a reward
model to those comparisons and then runs PPO against it. **DPO** shows you can
skip the reward model entirely.

Start with the Bradley–Terry model of preference:

$$P(y_w \succ y_l \mid x) = \sigma\big(r(x,y_w) - r(x,y_l)\big)$$

The RLHF objective is "maximize reward, stay close to the reference in KL," whose
closed-form optimum is
$\pi^*(y|x) \propto \pi_\text{ref}(y|x)\exp(r(x,y)/\beta)$. Invert that for $r$:

$$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)} + \beta \log Z(x)$$

Substituting into Bradley–Terry, the intractable partition function $Z(x)$ cancels
because it appears in both terms of the difference. What is left is a plain
supervised loss on the policy:

$$\boxed{\ \mathcal{L}_\text{DPO} = -\log \sigma\!\left(\beta\left[\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right]\right)}$$

The model *is* the reward model. Interpretation: push up the chosen response's
logprob and push down the rejected one's, but measured **relative to the
reference**, so the model is not rewarded for changes it was already going to
make. $\beta$ (typically 0.1) controls how far from the reference you will drift;
small $\beta$ means a loose leash.

Mechanically this needs two forward passes over each pair — one under $\pi_\theta$
(with backward) and one under $\pi_\text{ref}$ (no backward, cacheable). That is
exactly what `training/recipes/dpo_loop.py` does: a separate reference trainer
computes and caches ref logprobs, then the policy runs
`forward_backward_custom`; `release_reference_after_cache: bool = True` frees the
reference resources once caching is done.

## 3.9 ORPO: preference learning without a reference

DPO's reference model costs memory and an extra forward pass. **ORPO** removes it
by regularizing with an odds ratio on top of the ordinary SFT loss.

Define the odds of a response under length-normalized sequence probability $p$:

$$\text{odds}_\theta(y|x) = \frac{p_\theta(y|x)}{1 - p_\theta(y|x)}$$

$$\mathcal{L}_\text{ORPO} = \mathcal{L}_\text{SFT}(y_w) + \lambda \cdot \underbrace{\left[-\log \sigma\left(\log\frac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)}\right)\right]}_{\mathcal{L}_\text{OR}}$$

The SFT term anchors the model to the chosen responses (playing the role DPO's
reference plays), and the odds-ratio term separates chosen from rejected. Using
odds rather than raw probabilities gives a gentler penalty on the rejected
response: it discourages $y_l$ without driving its probability to zero, which
would degrade the model's general fluency.

One trainer, no reference, one pass — `training/recipes/orpo_loop.py`, with
`orpo_lambda: float = 1.0`.

## 3.10 Distillation

Train a small **student** to mimic a large **teacher**. Two directions, and the
choice determines the failure mode:

**Forward KL** $\mathrm{KL}(p_\text{teacher} \| p_\text{student})$ — train on the
teacher's distribution (in practice its top-$k$ logits) over a fixed dataset.
Forward KL is **mode-covering**: it is infinitely penalized wherever the teacher
has mass and the student has none, so the student spreads itself thin trying to
cover everything the teacher might say.

**Reverse KL** $\mathrm{KL}(p_\text{student} \| p_\text{teacher})$ — the student
samples, the teacher scores those samples. Reverse KL is **mode-seeking**: the
student concentrates on a subset of the teacher's behavior and does it well.
This usually produces better task performance for a small student, at the cost of
diversity. It also requires online generation, which makes it structurally an RL
loop.

`training/recipes/distillation_loop.py` implements both: `DistillMode.SAMPLED_REVERSE_KL`
(using the server-side `"importance_sampling"` loss, since the student's samples
are off-policy by the time they are trained on) and top-$k$ SDFT (`sdft_top_k: int = 5`,
using `"cross_entropy"`).

## 3.11 The Tinker protocol

Fireworks' Training API is Tinker-compatible: the client sends primitives and
gets futures back, and the training *loop* stays in your Python process. The
cookbook wraps it in `ReconnectableClient` (`training/utils/client.py`):

| Call | Purpose |
|---|---|
| `forward(data, loss_fn)` | Logprobs only, no gradient. Reference models, old-policy snapshots. |
| `forward_backward(data, "cross_entropy")` | Server computes a built-in loss and backprops. SFT, distillation. |
| `forward_backward_custom(data, loss_fn)` | **Server returns logprobs, your Python computes the loss, the server backprops through it.** DPO, ORPO, GRPO. |
| `forward_backward_contrastive(...)` | Server-side InfoNCE for embedding models. |
| `optim_step(AdamParams, grad_accumulation_normalization=...)` | The one mutating call. |
| `save_state(name)` / `load_state_with_optimizer(path)` | Full checkpoint (weights **and** optimizer moments) for resume. |
| `save_weights_for_sampler(name, checkpoint_type=...)` | Inference-shaped snapshot for serving or hot-load. |
| `load_adapter(path)` | Warm-start from an existing LoRA adapter. |

`forward_backward_custom` is the load-bearing one. It means an arbitrary loss —
anything you can express in PyTorch over returned logprobs — runs without
shipping code to the cluster or waiting for a platform feature. Every RL variant
in `training/utils/rl/` (GRPO, DAPO, GSPO, CISPO, REINFORCE, DRO) is just a
different function passed to this call.

Note also the **two distinct checkpoint types**, a distinction people get wrong:

- `save_state` → DCP checkpoint, includes optimizer state, used to *resume
  training*. Large.
- `save_weights_for_sampler` → inference-format weights, used to *serve or
  hot-load*. No optimizer state; cannot resume from it.

Both are managed by `TrainingCheckpoints` in `training/utils/checkpoints.py`.

---

**Next:** [4. Reinforcement learning](reinforcement-learning.md) — where training
and inference become one system.
