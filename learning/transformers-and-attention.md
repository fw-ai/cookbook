# 1. Transformers and attention

## 1.1 The only task: predict the next token

Everything a language model does reduces to one operation repeated: given a
prefix of tokens, output a probability distribution over which token comes next.

Text is first converted to integers by a **tokenizer**. Modern tokenizers use
byte-pair encoding (BPE): start from raw bytes, then repeatedly merge the most
frequent adjacent pair into a new symbol, until you have a vocabulary of
$V \approx 32{,}000$ to $256{,}000$ symbols. A token is roughly 3–4 characters of
English, less for code, much less for languages that do not use Latin script.
Practical consequences: token counts (not word counts) drive both cost and
context limits, and two tokenizers disagreeing about the same string is a real
source of training bugs (see §1.10).

A language model defines a joint probability over a sequence by factoring it with
the chain rule:

$$P(t_1,\dots,t_n) = \prod_{i=1}^{n} P(t_i \mid t_{<i})$$

The model produces a **logit** vector $z \in \mathbb{R}^{V}$ at every position,
and the distribution is the softmax:

$$P(t_i = v \mid t_{<i}) = \frac{e^{z_v}}{\sum_{u=1}^{V} e^{z_u}} \qquad\text{often written}\qquad \mathrm{softmax}(z)_v$$

Training maximizes the log-likelihood of real text, i.e. minimizes **cross
entropy**:

$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n} \log P(t_i \mid t_{<i})$$

That single loss — averaged negative log probability of the correct next token —
is what supervised fine-tuning still uses today. In this repo it is the
server-side loss ID `"cross_entropy"` (`training/recipes/sft_loop.py`).

Two derived quantities you will see constantly:

- **Perplexity** $= e^{\mathcal{L}}$, interpretable as "effective number of
  equally likely choices per token."
- **Logprob** of a specific token, $\log P(t_i \mid t_{<i})$. RL and preference
  methods manipulate logprobs directly, and almost every subtlety in part 4 is
  about two systems disagreeing on the same logprob.

## 1.2 From tokens to vectors

Token IDs are looked up in an **embedding matrix** $E \in \mathbb{R}^{V \times d}$:

$$x_i = E[t_i] \in \mathbb{R}^{d}$$

The width $d$ (the "residual stream" or $d_\text{model}$) is 4096 for an 8B model,
8192 for a 70B, and so on. The sequence becomes a matrix
$X \in \mathbb{R}^{n \times d}$: one row per position.

Every transformer layer reads $X$ and writes an updated $X$ of the same shape.
At the very end, an output projection (the "unembedding" or LM head,
$W_U \in \mathbb{R}^{d \times V}$, sometimes tied to $E$) turns each row into
logits: $z_i = x_i W_U$.

So the architecture question is only: *how should positions exchange
information, and how should each position think on its own?* Attention answers
the first; the MLP answers the second.

## 1.3 Attention, derived

Suppose position $i$ needs information stored at earlier positions. A hard lookup
table would need exact keys. We want a **soft, learned, content-addressed
lookup**.

Give every position three vectors:

- a **query** $q_i$ — "what am I looking for?"
- a **key** $k_j$ — "what do I have on offer?"
- a **value** $v_j$ — "what I will contribute if you pick me."

Relevance of $j$ to $i$ is the dot product $q_i \cdot k_j$ (large when the vectors
point the same way). Normalize the relevances into weights that sum to one, then
take the weighted average of the values:

$$\alpha_{ij} = \frac{\exp(q_i \cdot k_j / \sqrt{d_h})}{\sum_{j'} \exp(q_i \cdot k_{j'} / \sqrt{d_h})}, \qquad o_i = \sum_j \alpha_{ij}\, v_j$$

That is attention. Three details justify themselves:

**Why divide by $\sqrt{d_h}$.** If $q$ and $k$ have $d_h$ independent components
with unit variance, their dot product has variance $d_h$. Without rescaling, the
logits grow with head width, the softmax saturates into a near-one-hot
distribution, and gradients vanish. Dividing by $\sqrt{d_h}$ keeps the score
variance $\approx 1$ regardless of head size.

**Why softmax.** It is the smooth, differentiable relaxation of "pick the
argmax," and it guarantees weights are non-negative and sum to one, so the output
stays in the convex hull of the values (a stable scale for the residual stream).

**Why causal masking.** For next-token prediction, position $i$ must not see
$j > i$, otherwise the model trivially cheats. Implement by adding a mask
$M_{ij} = -\infty$ for $j > i$ before the softmax, which sends those weights to
zero. This is why the whole sequence can be trained in parallel: the mask makes
one forward pass compute the loss at every position simultaneously, as if you had
run $n$ separate prefixes.

### Q, K, V are linear projections — this is "KQV"

The three vectors are not separate inputs; they are three learned views of the
same residual stream:

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V, \qquad W_Q, W_K, W_V \in \mathbb{R}^{d \times d_h}$$

In matrix form, with mask $M$:

$$\boxed{\ \mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_h}} + M\right) V\ }$$

$QK^\top$ is $n \times n$: every position scored against every position. That
quadratic term is the source of both the expressive power and the entire cost
problem of long context.

The **Q/K circuit** decides *where* to read; the **V circuit** (with the output
projection $W_O$) decides *what* gets written back. They are functionally
different, which is why K/V can be shared across heads while Q is not (§1.5), and
why only K and V need caching at inference time (§2.2).

### Multi-head attention

One attention operation can only compute one weighted average. Real reasoning
needs several relations at once ("the subject of this verb", "the matching
bracket", "the last mention of this variable"). So run $h$ attention operations
in parallel with independent projections, each of width $d_h = d/h$, then
concatenate and mix:

$$\mathrm{MHA}(X) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h)\, W_O, \qquad \mathrm{head}_i = \mathrm{Attn}(XW_Q^i, XW_K^i, XW_V^i)$$

Total parameters stay $\approx 4d^2$ ($W_Q, W_K, W_V, W_O$), because the heads
split the width rather than duplicating it. Splitting is nearly free and strictly
more expressive than a single wide head, since a single head is forced to use one
softmax distribution for everything.

### Cost

- Time: $O(n^2 d)$ for the scores and the value mixing, plus $O(n d^2)$ for the
  projections. Short sequences are dominated by the $d^2$ projection term; long
  sequences by the $n^2$ term.
- Memory (naive): the $n \times n$ score matrix per head. At $n = 32{,}768$ and 32
  heads in bf16 that is $32 \times 32768^2 \times 2 \approx 68$ GB — impossible.

**FlashAttention** removes that memory cost. Instead of materializing the score
matrix, it tiles the computation into blocks that fit in on-chip SRAM and uses an
*online softmax*: keep a running max $m$ and running sum $\ell$, and rescale the
accumulated output as each new block arrives, using
$\mathrm{softmax}$'s shift-invariance ($\mathrm{softmax}(z) = \mathrm{softmax}(z-c)$).
Memory becomes $O(n)$, and the kernel becomes compute-bound instead of
bandwidth-bound. The backward pass recomputes the scores block by block rather
than storing them. Every serious implementation uses this or a descendant; it is
the reason 128k+ context is affordable at all.

## 1.4 Position information: RoPE

Attention as defined is permutation-equivariant — shuffle the tokens and the
outputs shuffle with them. Nothing in $QK^\top$ knows about order. Position has
to be injected.

Older models added a learned or sinusoidal position vector to the embedding.
Modern models use **RoPE** (rotary position embedding), which rotates $q$ and $k$
by an angle proportional to their absolute position. Treat the $d_h$ components
as $d_h/2$ 2-D pairs; pair $j$ at position $m$ is rotated by $m\theta_j$ with

$$\theta_j = b^{-2j/d_h}, \qquad b = 10{,}000 \text{ (the "RoPE base")}$$

$$R_m = \begin{pmatrix} \cos m\theta_j & -\sin m\theta_j \\ \sin m\theta_j & \cos m\theta_j \end{pmatrix} \text{ applied blockwise}$$

The point is what happens to the score:

$$\langle R_m q,\; R_n k \rangle = \langle q,\; R_{n-m} k \rangle$$

The dot product depends only on the **relative** distance $n - m$, even though
each vector was rotated by its absolute position. You get relative-position
awareness for free, with no extra parameters, no additive bias term, and — the
practical payoff — the ability to extend context after training by scaling the
frequencies (linear interpolation, NTK-aware scaling, YaRN). "Context extension"
almost always means "we changed the RoPE base or interpolation and did a bit of
long-context fine-tuning."

Note that RoPE is applied to $Q$ and $K$ only, not $V$: position should affect
*where you look*, not *what gets copied*.

## 1.5 MHA → MQA → GQA → MLA: shrinking the K/V footprint

At inference, K and V for every past token must be kept around (§2.2). Their size
is proportional to the number of **KV heads**. Since the Q/K matching circuit
tolerates fewer distinct key spaces than you might expect, architectures share
them:

| Variant | KV heads | KV cache size | Notes |
|---|---|---|---|
| MHA (multi-head) | $h$ | $1\times$ | Original; largest cache |
| MQA (multi-query) | 1 | $1/h$ | Cheapest, some quality loss |
| GQA (grouped-query) | $g$, e.g. 8 | $g/h$ | Today's default; Q heads form $g$ groups sharing one K/V head |
| MLA (multi-head latent) | — | small | DeepSeek: cache a low-rank latent, project up per head |

A concrete example: Llama-3-70B has $L = 80$, 64 query heads, **8** KV heads,
$d_h = 128$. With MHA the cache would be 8× larger. GQA is the single biggest
reason long-context, high-batch serving is economical, and it is invisible in the
math of §1.3 — only $W_K$ and $W_V$ get narrower and their outputs get broadcast
across the query heads in each group.

## 1.6 The rest of the block

A transformer layer is two sublayers, each wrapped in a residual connection and
preceded by a normalization (**pre-norm**, now universal because it makes deep
stacks trainable without careful warmup):

$$x \leftarrow x + \mathrm{Attn}(\mathrm{Norm}(x))$$
$$x \leftarrow x + \mathrm{MLP}(\mathrm{Norm}(x))$$

The residual stream is the backbone: every layer *adds* to it rather than
replacing it, so gradients flow to layer 1 through an unbroken identity path.
This is why depth works.

**RMSNorm** replaced LayerNorm because the mean-subtraction term turns out to be
unnecessary:

$$\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \odot \gamma$$

One fewer reduction, one fewer parameter vector, same quality.

**The MLP** is where most parameters live and where "facts" are commonly
localized. Modern models use **SwiGLU**, a gated variant:

$$\mathrm{MLP}(x) = \big(\mathrm{silu}(x W_\text{gate}) \odot x W_\text{up}\big) W_\text{down}, \qquad \mathrm{silu}(z) = z\,\sigma(z)$$

The elementwise product makes it multiplicative (a soft gate: one branch decides
how much of the other passes), which empirically beats a plain
$\mathrm{ReLU}(xW_1)W_2$ at equal parameter count. Because SwiGLU needs three
matrices instead of two, implementations shrink $d_\text{ff}$ from $4d$ to about
$\tfrac{8}{3}d$ to keep the parameter count the same.

## 1.7 Mixture of Experts

A dense model spends all $N$ parameters on every token. MoE decouples **total**
parameters from **active** parameters: replace the single MLP with $E$ expert
MLPs and a small **router** that picks the top-$k$ (typically 1, 2, or 8) per
token:

$$g = \mathrm{softmax}(x W_\text{router}) \in \mathbb{R}^{E}, \qquad y = \sum_{i \in \mathrm{TopK}(g)} \frac{g_i}{\sum_{j \in \mathrm{TopK}(g)} g_j} \, E_i(x)$$

A model like Qwen3-235B-A22B has 235B total parameters but ~22B active per token:
memory footprint of a 235B model, compute cost of a 22B one. MoE is how frontier
open models get capacity without proportional serving cost.

Three consequences that matter operationally:

1. **Routing is discrete.** $\mathrm{TopK}$ is not differentiable and is
   sensitive to tiny numerical differences. Two systems computing the same token
   can pick *different experts* and therefore produce meaningfully different
   logprobs. This is exactly why this repo has **Router Replay (R3)** — the
   inference engine returns its routing matrices and the trainer replays them
   (`training/utils/rl/router_replay.py`, `router_replay=True` by default in the
   RL recipes). See §4.9.
2. **Load balancing.** Left alone, the router collapses onto a few popular
   experts. Training adds an auxiliary balancing loss (or bias-based
   balancing) to spread tokens out.
3. **Expert parallelism.** Experts are sharded across GPUs, so each token's
   routing decision becomes a network all-to-all. MoE serving is a
   communication problem as much as a compute one.

## 1.8 Counting parameters

Per layer, with GQA ($g$ KV heads out of $h$):

| Component | Parameters |
|---|---|
| $W_Q$ | $d \times d$ |
| $W_K, W_V$ | $2 \times d \times (g\,d_h)$ |
| $W_O$ | $d \times d$ |
| MLP (SwiGLU) | $3 \times d \times d_\text{ff}$ |

With $d_\text{ff} \approx \tfrac{8}{3}d$ the MLP is $\approx 8d^2$ and attention
is $\approx 2\text{–}4d^2$, so a layer is roughly $10\text{–}12 d^2$ and the whole
model is

$$N \approx 12 L d^2 + \underbrace{2Vd}_{\text{embed + unembed}}$$

Sanity check for an 8B-class model: $L = 32$, $d = 4096$ →
$12 \times 32 \times 4096^2 \approx 6.4$B, plus embeddings ≈ 7–8B. The formula
works.

## 1.9 Counting FLOPs — the numbers you actually plan with

A matrix multiply of $(m \times k)$ by $(k \times n)$ costs $2mkn$ FLOPs (one
multiply and one add per term). Every parameter is used in exactly one such
multiply per token, so:

$$\text{forward} \approx 2N \ \text{FLOPs/token}$$

Backward computes two gradients per matmul (with respect to the input and with
respect to the weight), so it costs about twice the forward:

$$\text{backward} \approx 4N, \qquad \boxed{\text{training} \approx 6N \ \text{FLOPs/token}}$$

Add roughly $2 \cdot 2 \cdot L \cdot n \cdot d$ for the attention score/value
matmuls, which only matters once $n$ is comparable to $d$ — i.e. at long context.

These two numbers ($2N$ inference, $6N$ training) let you estimate anything.
Training an 8B model on 1B tokens: $6 \times 8\times10^9 \times 10^9 = 4.8\times10^{19}$
FLOPs. On one H100 at ~400 TFLOP/s effective (about 40% of peak bf16, a realistic
MFU), that is $\approx 1.2\times10^{5}$ s ≈ 33 GPU-hours. That estimate is usually
within 2× of reality, and it tells you immediately whether an idea is a coffee
break or a cluster booking.

## 1.10 Chat templates: where the math meets reality

Base models see raw text. Instruction models are trained on a **chat template**
that serializes a message list into tokens with special role markers, e.g.
`<|im_start|>user ... <|im_end|>`. Reasoning models add more structure (thinking
blocks, tool-call syntax).

If training renders a conversation differently than serving does — a stray space,
a different end-of-turn token, thinking blocks kept in one and stripped in the
other — the model is optimized for strings it will never see. Symptoms are
maddening: fine-tuning "works" (loss goes down) and the deployed model is worse
than the base.

This is why this repo has a per-model renderer layer (`training/renderer/`, with
`glm5.py`, `deepseek_v4.py`, `gemma4.py`, and friends) plus parity verifiers
under `training/renderer/verifier/`. Treat renderer parity as a correctness
requirement, not a detail.

---

**Next:** [2. Inference](inference.md) — what it costs to actually run this thing.
