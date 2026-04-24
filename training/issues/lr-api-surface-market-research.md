# LR API Surface — Market Research

**Context.** Cookbook PR [#359](https://github.com/fw-ai/cookbook/pull/359) adds a
per-step learning-rate (LR) surface to the cookbook recipes (SFT / DPO / GRPO /
ORPO). Paired with trainer PR [#23105](https://github.com/fw-ai/fireworks/pull/23105)
which lets the trainer honor a per-step LR passed through `optim_step(AdamParams)`.
The Slack debate was: *is this customer-facing API worth adding, and what should
its shape be?*

This doc answers that by looking at how nine other trainers / fine-tuning
services expose LR to their users. It's intended as the reference for deciding
what to keep in PR #359 and what to cut.

## TL;DR

1. **Almost everyone exposes exactly four knobs**: `learning_rate`, `lr_scheduler` (enum), `warmup_ratio` (or `warmup_steps`), `min_lr_ratio`. That's the 80% surface. PR #359's `lr_schedule` / `warmup_ratio` / `min_lr_ratio` fields match this consensus directly.
2. **Nobody (except Levanter, barely) exposes a "custom per-step LR list"** as first-class config — the escape hatches are always "subclass `Trainer`" or "point `_component_` at a callable". `lr_per_step: list[float] | None` is genuinely novel surface, and it's a clean one for the Tinker / Fireworks shape because the schedule is computed client-side anyway.
3. **Managed services (OpenAI, Together, Anyscale) expose 0–6 LR flags**, opaque schedule, never an optimizer knob. That's the bar to clear if "simplicity" is the goal.
4. **Muon is supported natively in 3 of 9** — Axolotl (`optimizer: muon`), LLaMA-Factory (`use_muon: true`), Levanter (`optimizer.type: muon`, and it's *the* multi-optimizer reference — routes Linear weights → Muon, embeddings/biases → AdamW via `optax.multi_transform`). Nobody exposes Muon through a *tuple* params object the way the in-flight Fireworks trainer does. Design implication below.
5. **Closest analog to Fireworks cookbook is Tinker-cookbook**: 1 `learning_rate` float + 1 `lr_schedule: Literal["linear","cosine","constant"]` enum + Adam betas in an "advanced" block. ~4 knobs total, opaque optimizer. PR #359's surface is almost identical modulo `lr_per_step` and `min_lr_ratio`.

---

## Cross-framework comparison

### Frameworks surveyed

Nine trainers/services, grouped by "tier":

| Tier | Frameworks |
|---|---|
| **Self-host, research-grade** | Levanter, torchtune |
| **Self-host, kitchen-sink** | HF TRL / Transformers, Axolotl, LLaMA-Factory, Unsloth |
| **Self-host, RL-focused** | OpenRLHF, verl, NeMo-RL |
| **Managed services** | OpenAI, Together, Anyscale |
| **Closest analog** | Tinker-cookbook |

### Axis 1 — Core LR fields exposed

| Framework | `learning_rate` | `lr_scheduler` | `warmup_ratio` | `warmup_steps` | `min_lr_ratio` | Other LR-tier fields |
|---|:-:|:-:|:-:|:-:|:-:|---|
| HF TRL / Transformers | ✅ | ✅ (10 named) | ✅ | ✅ | via `lr_scheduler_kwargs` | `weight_decay`, adam betas, `max_grad_norm` |
| Axolotl | ✅ | ✅ (+ `one_cycle`, `rex`) | ✅ | ✅ | `cosine_min_lr_ratio` | `embedding_lr`, `lr_groups: [{modules,lr}]`, `JaggedLR` restarts |
| LLaMA-Factory | ✅ | ✅ | ✅ | ✅ | via `lr_scheduler_kwargs` | `loraplus_lr_ratio`, `use_muon`/`use_galore`/`use_apollo`/`use_badam` booleans |
| Unsloth | ✅ | ✅ | ✅ | ✅ | (HF passthrough) | `embedding_learning_rate` (2nd LR) |
| torchtune | ✅ (`optimizer.lr`) | dotted component path | `num_warmup_steps` | — | — | `num_cycles`, `clip_grad_norm` |
| Levanter | ✅ | ✅ (5 named + registry) | ✅ (`warmup`) | ✅ | ✅ (first-class) | `decay`, `rewarmup`, `cycle_length`, `cycles`, `adam_lr` (Muon's aux Adam) |
| OpenRLHF | ✅ (`adam.lr` + `muon.lr`) | ✅ (`lr_scheduler`) | `lr_warmup_ratio` | — | ✅ | `--optim {adam,muon}`, `--muon.*`/`--adam.*` blocks |
| verl (FSDP) | ✅ | ✅ (constant, cosine) | ✅ (`_ratio`) | ✅ | ✅ | `num_cycles`, per-role (actor/critic/ref) |
| verl (Megatron) | ✅ | `lr_decay_style` (4) | `lr_warmup_init` | ✅ | `min_lr` (abs) | WSD decay style + steps |
| NeMo-RL DTensor | ✅ (`optimizer.kwargs.lr`) | **list of `{name, kwargs}`** → `SequentialLR` | (implicit in chain) | (implicit) | (implicit) | any `torch.optim.lr_scheduler.*` composable |
| NeMo-RL Megatron | ✅ | `lr_decay_style` (3) | `lr_warmup_iters` | ✅ | `min_lr` (abs) | weight-decay schedule |
| Tinker-cookbook | ✅ | `Literal["linear","cosine","constant"]` | — | — | — | adam betas (advanced) |
| OpenAI FT API | `learning_rate_multiplier` (`"auto"` default) | — | — | — | — | *none* |
| Together FT | ✅ | `"linear"`/`"cosine"` | ✅ | — | ✅ | `num_cycles`, weight decay |
| Anyscale simple | ✅ | — | — | — | — | `num_epochs`, `context_length` |
| **PR #359 (proposed)** | ✅ | `Literal["constant","cosine","linear"]` | ✅ | (via `warmup_steps` arg to `build_lr_per_step`) | ✅ | `lr_per_step: list[float] \| None` |

**Observation.** PR #359's four fields (`learning_rate`, `lr_schedule`, `warmup_ratio`, `min_lr_ratio`) are the intersection of what Axolotl, OpenRLHF, verl, Together, and Levanter expose. This is the settled consensus surface. The only field in PR #359 that doesn't appear in *any* comparable framework is `lr_per_step: list[float] | None`.

### Axis 2 — Custom per-step LR / callable escape hatch

| Framework | Per-step LR list? | Callable schedule? | How |
|---|:-:|:-:|---|
| HF TRL | ❌ | ⚠️ subclass only | Subclass `Trainer.create_scheduler` or pass `(opt, sched)` to `Trainer(... optimizers=...)` |
| Axolotl | ❌ | ❌ | `lr_scheduler` is a string enum |
| LLaMA-Factory | ❌ | ❌ | same |
| Unsloth | ❌ | ❌ | same (inherits TRL) |
| torchtune | ❌ | ✅ | `lr_scheduler._component_: my.module.path` accepting `(optimizer, num_training_steps, last_epoch, **kwargs)` |
| Levanter | ❌ | ✅ (registered subclass) | Subclass `LrSchedule`, register with draccus, reference by name from YAML |
| OpenRLHF | ❌ | ❌ | string schedule |
| verl | ❌ | ❌ | string schedule |
| NeMo-RL DTensor | ⚠️ | ✅ | List any `torch.optim.lr_scheduler.*` (including `LambdaLR`) and chain with `milestones` |
| NeMo-RL Megatron | ❌ | ❌ | named `lr_decay_style` |
| Tinker-cookbook | ⚠️ implicit | ⚠️ user can compute any LR each step client-side | Client owns the loop; just pass a different `AdamParams.learning_rate` each `optim_step` |
| Managed APIs | ❌ | ❌ | opaque |
| **PR #359** | ✅ (first-class field) | ❌ | `lr_per_step: list[float] \| None` |

**Observation.** `lr_per_step: list[float] | None` is genuinely novel for a config-driven cookbook surface. But because the Tinker / Fireworks SDK is *client-driven* (the client owns the loop and sends `AdamParams` per step), a list-of-floats is actually the *minimal* possible escape hatch — far simpler than a `LambdaLR` subclass or a draccus-registered `LrSchedule` subclass. It also composes naturally with "the managed service pre-computes the list" (which is what PR #23105 enables on the server side).

### Axis 3 — Muon / multi-optimizer support

| Framework | Muon natively? | Selection style | Multi-opt (Adam for biases + Muon for hidden)? |
|---|:-:|---|:-:|
| HF TRL | ❌ (PR #39541 open) | single `--optim` enum | ❌ |
| Axolotl | ✅ | `optimizer: muon` | ❌ (single family) |
| LLaMA-Factory | ✅ | `use_muon: true` boolean | ❌ |
| Unsloth | ❌ | HF `optim` passthrough | ❌ |
| torchtune | ❌ | `_component_: torch.optim.Adam` | ❌ (user forks recipe) |
| Levanter | ✅ | `optimizer.type: muon` | ✅ **first-class** — internal `optax.multi_transform` mask routes Linear → Muon, embeddings/biases/LN → AdamW; independent `lr` vs `adam_lr` schedules |
| OpenRLHF | ✅ | `--optim {adam,muon}` + `--muon.lr`, `--muon.momentum`, `--adam.lr`, `--adam.betas`, `--adam.eps`, `--adam.weight_decay` | ✅ **Muon's aux-Adam subgroup shares the `--adam.*` block** |
| verl | ❌ (GH #3246, #4966 open) | `optimizer` string | ❌ |
| NeMo-RL | ⚠️ planned v0.6 | `optimizer.name: path.to.Muon` (DTensor) | ❌ |
| Tinker-cookbook | ❌ (SDK is Adam-only) | opaque | ❌ |
| Managed APIs | ❌ | opaque | ❌ |

**The one to study is OpenRLHF** (the file renfeichen linked, `openrlhf/cli/train_rm.py:256-275`):

```
--optim {adam,muon}          # which family to use
# Muon-specific
--muon.lr 0.02
--muon.momentum 0.95
--muon.ns_steps 5            # placeholder; DS v0.18.x hard-codes these
--muon.nesterov true
# AdamW (shared: pure-AdamW when --optim=adam, Muon's aux-Adam subgroup when --optim=muon)
--adam.lr 9e-6
--adam.betas 0.9 0.95
--adam.eps 1e-8
--adam.weight_decay 0.0
# Scheduler (applies to both)
--lr_scheduler cosine_with_min_lr
--lr_warmup_ratio 0.03
--min_lr_ratio 0.1
--max_norm 1.0               # grad clip
```

Key design insight: **Muon is never a solo optimizer in practice**. It always needs an "aux Adam" group for the 1-D params (embeddings, biases, LayerNorms) it can't touch. OpenRLHF's solution is to split into `--muon.*` and `--adam.*` blocks, and when `--optim=muon`, `--adam.*` becomes the aux-subgroup config. That's the same split Levanter uses under the hood with `adam_lr` / `lr` in `MuonConfig`. Both treat the LR schedule (`lr_scheduler`, `warmup_ratio`, `min_lr_ratio`) as orthogonal to the optimizer choice — the scheduler applies a multiplier uniformly, and Muon/Adam each scale their peak LR by it.

### Axis 4 — Total number of LR-tier flags exposed

| Framework | Flags a "simple" user touches | Flags available if you go deep |
|---|:-:|:-:|
| OpenAI FT | 1 (`lr_multiplier`) | 1 |
| Anyscale simple | 1 | ~3 |
| Tinker-cookbook | 1–2 | ~5 (incl. adam betas) |
| Together FT | 1–3 | ~8 |
| LLaMA-Factory | 4 | ~15 (+ optim bool gates) |
| NeMo-RL DTensor | 4 | ~10 (scheduler chain is unlimited) |
| verl FSDP per role | 4 | 12 |
| torchtune | 3–4 | ~8 |
| HF TRL / Transformers | 4 | ~15 (plus everything via `lr_scheduler_kwargs`) |
| OpenRLHF (RM w/ muon) | 4 | 13 |
| Axolotl | 4–6 | ~25 |
| Levanter | 4–7 | ~20 |
| **PR #359** | 1 (`learning_rate`) | 4 (`+ lr_schedule + warmup_ratio + min_lr_ratio`) + `lr_per_step` escape hatch |

PR #359 is at the Tinker end of the spectrum for the common case, with a single-field escape hatch that gets you anywhere.

---

## Recommendations for PR #359 and follow-ons

### 1. Keep the four-field schedule surface — it matches consensus.

`learning_rate`, `lr_schedule: Literal["constant","cosine","linear"]`, `warmup_ratio`, `min_lr_ratio` are the intersection of Axolotl + OpenRLHF + verl + Together + Levanter + LLaMA-Factory. The field names, types, and defaults PR #359 picks are already idiomatic. Don't second-guess; don't rename. One small consistency nit: PR #359 supports `warmup_ratio` in the Config but `build_lr_per_step` also accepts `warmup_steps` (absolute, takes precedence). Expose `warmup_steps` as a Config field too (default `None`) — HF, verl-FSDP, Levanter, and Megatron all provide both.

### 2. Keep `lr_per_step: list[float] | None` — it's our differentiator.

Nobody else has a list-based escape hatch as first-class config. It naturally exploits the client-driven nature of the Tinker/Fireworks SDK. Managed-service side (PR #23105) computes the list once and passes it through; customer-side, if they have exotic requirements (stepwise restart, custom warmup, replay from checkpoint at a specific LR), they hand in a list. This is a meaningful ergonomic advantage over TRL/Axolotl/Unsloth where a custom schedule requires subclassing.

**Document it with a recipe snippet** — the cookbook's job is to teach the 80% path clearly, not to hide escape hatches. A single `examples/custom_lr_schedule.py` showing "here's how to pass your own list" pays for the feature.

### 3. For Muon support — adopt the OpenRLHF split-block pattern.

When PR #23105 / the trainer SDK exposes a Muon params type, the cookbook surface should be:

- Add a top-level `optimizer: Literal["adam","muon"] = "adam"` selector.
- Reuse `learning_rate` / `lr_schedule` / `warmup_ratio` / `min_lr_ratio` as the *schedule* — it's optimizer-agnostic, same as OpenRLHF.
- Add **optimizer-specific blocks** — `adam: AdamConfig` (betas, eps, weight_decay) and `muon: MuonConfig` (momentum, ns_steps, nesterov; when `optimizer=muon`, `adam` block becomes the aux-subgroup config for the Adam-managed params the way OpenRLHF does it).
- At each step, build the params tuple via the helper I sketched in the previous turn (`_with_step_lr(optim_params_template, current_lr)`), so `lr_per_step` / scheduler logic is unchanged whether the optimizer is Adam-only or Adam+Muon.

This gives us the *same* 4-knob simple path (change nothing if you use Adam) plus a clean Muon opt-in (1 enum flip + 3 Muon-specific fields, Adam block is reused as the aux). Levanter and OpenRLHF are the two precedents; both converged on this shape.

### 4. For the managed-service end, keep `lr_per_step` the only wire format.

PR #23105's change (trainer honors per-step LR from the call) plus PR #359's `build_lr_per_step()` cleanly separates "the schedule *concept* (customer-facing)" from "the schedule *materialization* (list of floats sent to the trainer)". That's the right factoring: managed orchestrator can swap in `lr_per_step` auto-generated from `(lr_schedule, warmup_ratio, min_lr_ratio)` after data loading (when total_steps is known), and the customer never has to know what happens. The 4-knob surface is *only* for customers who want to configure a schedule declaratively; the `lr_per_step` field is the common underlying wire protocol.

### 5. Things to *not* do (based on survey pain points).

- **Don't copy Axolotl's `lr_groups: [{modules, lr}]`** yet. It's a great advanced feature but nobody in the managed tier exposes per-module LR, and the Fireworks SDK doesn't have the module-level hooks to implement it today. Revisit when Muon lands and we actually need Linear-vs-embedding splits (which Muon itself solves internally).
- **Don't copy TRL's `lr_scheduler_kwargs` dict.** It's where dead configs go to die (to get min-LR you set `cosine_with_min_lr` + `lr_scheduler_kwargs={"min_lr_rate": 0.1}` — that's three fields for one number). Promote the few useful knobs (`min_lr_ratio`, `warmup_ratio`) to first-class, skip the dict.
- **Don't copy HF/LLaMA-Factory's `use_X: bool` boolean-per-optimizer pattern.** It doesn't scale (LLaMA-Factory now has `use_muon` / `use_galore` / `use_apollo` / `use_badam` / `use_adam_mini` — five booleans to express "pick one"). Use a single `optimizer: Literal[...]` string like OpenRLHF / Axolotl / Levanter.
- **Don't adopt NeMo-RL's list-of-schedulers `SequentialLR` chain** as the config surface. It's too nested for a cookbook — three-level YAML just to express "linear warmup then cosine". PR #359's flat 4-field layout is better.

---

## Concrete config shape for the next iteration

Putting it all together, this is what the recipe `Config` should look like once Muon lands (fields in **bold** are new or changed vs today):

```python
@dataclass
class Config:
    # --- Core (same across all frameworks) ---
    learning_rate: float = 1e-5
    lr_schedule: Literal["constant", "cosine", "linear"] = "constant"
    warmup_ratio: float = 0.0
    warmup_steps: int | None = None  # absolute; wins over warmup_ratio when set
    min_lr_ratio: float = 0.0

    # --- Escape hatch (our differentiator) ---
    lr_per_step: list[float] | None = None

    # --- Optimizer selection (new once PR #23105 lands Muon) ---
    optimizer: Literal["adam", "muon"] = "adam"

    # Adam hypers (also the aux-subgroup when optimizer="muon")
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0

    # Muon-only hypers (ignored when optimizer="adam")
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_nesterov: bool = True
```

Simple user: sets `learning_rate`, ignores the rest. Matches the 4-knob
consensus and the Tinker-cookbook ergonomics.

Power user: sets `lr_schedule="cosine"`, `warmup_ratio=0.03`, `min_lr_ratio=0.1`
— same surface as Axolotl / OpenRLHF / verl / Together.

Muon user: flips `optimizer="muon"`, optionally tweaks `muon_*`. Reuses all
Adam hypers as the aux-subgroup config. Matches OpenRLHF.

Exotic user: hands us `lr_per_step=[…]`. No analog in any other framework.

---

## Sources and file pointers

Cited inline in the sub-sections above. Primary references:

- OpenRLHF: `openrlhf/cli/train_rm.py:256-275` (linked by renfeichen).
- HF TRL: `trl/trainer/*_config.py`, `transformers/training_args.py`.
- Axolotl: `src/axolotl/utils/schemas/training.py::HyperparametersConfig`, `src/axolotl/utils/schemas/enums.py::CustomSupportedOptimizers`.
- LLaMA-Factory: `src/llamafactory/hparams/finetuning_args.py`.
- torchtune: `recipes/configs/*.yaml`, `torchtune/training/lr_schedulers.py`.
- Levanter: `src/levanter/optim/config.py`, `src/levanter/optim/muon.py`.
- verl: `verl/trainer/config/optim/{fsdp,megatron}.yaml`.
- NeMo-RL: `examples/configs/grpo_math_1B.yaml`, `examples/configs/grpo_math_1B_megatron.yaml`.
- Tinker-cookbook: `tinker_cookbook/recipes/{sl_loop,rl_loop}.py`, `tinker_cookbook/utils/lr_scheduling.py`.
- Managed APIs: OpenAI `POST /v1/fine_tuning/jobs` hyperparameters docs; `together fine-tuning create` CLI docs; Anyscale `fine-tune-llm_v2` YAML templates.
