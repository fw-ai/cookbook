# Distillation Recipe

Use this reference when working with `training/recipes/distillation_loop.py`,
OPD, MOPD, SDFT, `teacher_messages`, `multi_teacher`, or
`topk_forward_kl`.

Before editing, read the recipe and the public example guide:

```bash
sed -n '1,220p' training/recipes/distillation_loop.py
sed -n '1,220p' training/examples/distillation/README.md
```

## Contents

- [Modes](#modes)
- [Logprob Response Fields](#logprob-response-fields)
- [Minimal Config](#minimal-config)
- [Dataset Contract](#dataset-contract)
- [Multi-teacher Rules](#multi-teacher-rules)
- [Guardrails](#guardrails)
- [Validation](#validation)

## Modes

| Config value | Use | Teacher source | Loss path |
|--------------|-----|----------------|-----------|
| `DistillMode.SAMPLED_REVERSE_KL` / `"sampled_reverse_kl"` | OPD-style sampled-token distillation | Teacher logprob on the sampled student token | Built-in `importance_sampling` |
| `DistillMode.TOPK_FORWARD_KL` / `"topk_forward_kl"` | SDFT sparse soft labels | Teacher inference `top_logprobs=K` per response position | Built-in `cross_entropy` with `[N, K]` targets |

`sampled_reverse_kl` is the default. The student samples from its managed
deployment, the teacher scores those exact response tokens, and the dense reward
is:

```text
teacher_logprob - sampling_logprob
```

The recipe passes that reward through the server-side `importance_sampling`
loss so rollout/current-policy drift is handled by the backend estimator.

`topk_forward_kl` asks the teacher inference deployment for `top_logprobs=K`,
builds `target_tokens` and `weights` with shape `[N, K]`, and trains the
student with `cross_entropy`. Keep `sdft_top_k <= 5`; this mirrors the current
inference `top_logprobs` response limit.

## Logprob Response Fields

Keep the public inference field semantics straight when changing OPD/SDFT
plumbing:

| Field or request option | Meaning |
|-------------------------|---------|
| `top_k` | Sampling filter. It keeps at most K next-token candidates eligible for sampling and renormalizes probability mass over the filtered set. |
| `sampling_mask` | Optional generated-token metadata: `count`, `non_zero_list`, or `non_zero_buffer` reports how many or which token IDs remained eligible after filters such as `top_p` and `top_k`. |
| `logprob` | Raw model logprob for the returned token. Legacy responses call this `token_logprobs`. |
| `sampling_logprob` | Final generation logprob after temperature and sampling filters/masks. This is the right field when the objective needs the probability of the token under the actual sampler distribution. |
| `top_logprobs` | Response field for likely alternatives at a position. It does not change sampling and is capped at 5 by the public inference API. |

The recipe currently asks the student sampler for generated token logprobs; the
current sampler SDK extracts `choice.logprobs.content[].logprob` and stores the
aligned values as `sampling_logprobs` for OPD batching. Teacher scoring requests
use `logprobs=True`, `echo=True`, and optional `top_logprobs=sdft_top_k`. If a
change needs exact sampler-distribution probabilities, wire through
`sampling_logprob` rather than only raw `logprob`.

## Minimal Config

```python
from training.recipes.distillation_loop import Config, main
from training.utils import DeployConfig, TrainerConfig

cfg = Config(
    log_path="./distillation_logs",
    base_model="accounts/fireworks/models/qwen3-8b",
    teacher_model="accounts/fireworks/models/qwen3-32b",
    dataset="/path/to/prompts.jsonl",
    trainer=TrainerConfig(
        training_shape_id="accounts/fireworks/trainingShapes/qwen3-8b-128k-h200",
    ),
    deployment=DeployConfig(tokenizer_model="Qwen/Qwen3-8B"),
)

main(cfg)
```

If `teacher_model` is a base-model resource, the recipe creates or reuses a
frozen teacher inference deployment. If it is already an inference model or
deployment resource, the recipe uses it directly.

For SDFT:

```python
from training.utils.distillation import DistillMode

cfg.distill_mode = DistillMode.TOPK_FORWARD_KL
cfg.sdft_top_k = 5
```

## Dataset Contract

Rows are JSONL objects. Required:

- `messages`: student-visible OpenAI-style chat messages.

Optional:

- `teacher`: default route key for sampled reverse-KL MOPD. The value must
  match a configured `TeacherConfig.route_value`, or the teacher `model` when
  `route_value` is unset.
- `teacher_messages`: privileged teacher-side prompt. If absent, the selected
  teacher scores the sampled response under `messages`.
- `expected_answer`: optional metadata for eval callbacks and smoke checks.

When a teacher uses a custom `TeacherConfig.teacher_messages_key`, rows routed
to that teacher should provide that key instead of `teacher_messages`.

Example routed row:

```json
{"messages":[{"role":"user","content":"Solve 6 * 7. End with Final: <answer>."}],"teacher":"math-teacher","expected_answer":"42"}
```

Rows converted from veRL, slime, or AReaL style math sources may keep metadata
such as `data_source`, `ability`, `reward_model`, and `extra_info`; the
Fireworks route contract is only the fields above.

## Multi-teacher Rules

Use `multi_teacher=MultiTeacherConfig(...)` for more than one teacher.

For `sampled_reverse_kl`, this is routed MOPD: each prompt is scored by exactly
one teacher selected by `MultiTeacherConfig.route_key` (default `teacher`).
This is not per-prompt blending.

For `topk_forward_kl`, every configured teacher can score the sampled response.
The recipe blends sparse top-K probability mass by token ID using
`TeacherConfig.blend_weight`, keeps the top `sdft_top_k` merged IDs, then
renormalizes before sending the `[N, K]` soft targets to `cross_entropy`.

```python
from training.utils.distillation import MultiTeacherConfig, TeacherConfig

cfg.multi_teacher = MultiTeacherConfig(
    route_key="teacher",
    teachers=[
        TeacherConfig(
            model="accounts/fireworks/models/qwen3-32b",
            route_value="math-teacher",
            tokenizer_model="Qwen/Qwen3-32B",
            blend_weight=1.0,
        ),
        TeacherConfig(
            model="accounts/fireworks/models/qwen3-14b",
            route_value="code-teacher",
            tokenizer_model="Qwen/Qwen3-14B",
            blend_weight=1.0,
        ),
    ],
)
```

## Guardrails

- Student and teacher token IDs must use a compatible tokenizer and vocabulary.
  Prefer same-family teachers and set `TeacherConfig.tokenizer_model` when you
  want validation against `DeployConfig.tokenizer_model`.
- Do not add a separate teacher region knob. `TrainerConfig.region` is the run
  region used by the SDK-managed student sampler and auto-created teacher
  deployments.
- `teacher_replica_count` controls replicas for auto-created frozen teacher
  deployments. `teacher_deployment_shape` sets the run-level default; individual
  `TeacherConfig.deployment_shape` can override it.
- Per-teacher metrics use `teacher_route/<slug>/scored` and
  `teacher_route/<slug>/inflight`; skewed datasets can leave some teacher
  deployments underused.
- Environment entrypoint names are `DISTILLATION_TEACHERS` and
  `DISTILLATION_TEACHER_ROUTE_KEY`. Legacy `OPD_TEACHERS` and
  `OPD_TEACHER_ROUTE_KEY` are accepted only as fallbacks.

## Validation

For logic or documentation changes around this recipe, run the focused unit
suite when dependencies are available:

```bash
cd training
pytest -q tests/unit/test_distillation.py tests/test_smoke_imports.py
```

For example-only changes, also import or run the touched example with a small
generated dataset when it does not require live Fireworks credentials.
