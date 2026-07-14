# Distillation (OPD and SDFT)

Train a student on its **own rollouts** to match one or more teachers. The
shipping recipe is `training/recipes/distillation_loop.py`; helpers live in
`training/utils/distillation/__init__.py` and
`training/utils/distillation/sampling.py`.

## Modes

| `--distill-mode` | Objective | Teacher signal | Backend loss | Status |
| --- | --- | --- | --- | --- |
| `sampled_reverse_kl` | reverse `KL(S||T)` sampled-token estimate | sampled token logprob | builtin `importance_sampling` | wired |
| `topk_forward_kl` | forward `KL(T||S)` sparse approximation | teacher inference top-K | builtin `cross_entropy` `[N,K]` | wired |

- **`sampled_reverse_kl`** is OPD's sampled-token reverse-KL path. The student
  samples, each rollout is scored by a frozen teacher, and per-token
  `teacher_logprob - sampling_logprob` becomes the dense reward. The recipe
  uses backend `importance_sampling` to account for rollout/current-policy
  drift; importance sampling is the estimator, not the objective.
- **`topk_forward_kl`** is the supported SDFT teacher top-K direction: get
  teacher candidates from inference deployment `top_logprobs`, build `[N,K]`
  `target_tokens` and `weights`, then train the student with builtin
  `cross_entropy`. Because the loss is built client-side, `K` cannot exceed the
  number of entries the inference API returns.

## Top-K Server Support

Top-K selection chooses the K token ids and logprobs for each response position.
Gather then reads another model's logprobs at exactly those ids via
`target_tokens=[N,K]`. Only one model should select per position. Running a
second top-K selection silently computes KL terms on mismatched tokens.

The Fireworks SDFT path is inference teacher-selected top-K:

```text
1. student inference samples rollouts                    -> on-policy token sequences
2. teacher inference echo with top_logprobs=K             -> teacher top-K ids/logprobs
3. build [N,K] target_tokens + weights                    -> teacher soft targets
4. forward KL: builtin cross_entropy                      -> supported recipe path
```

`topk_forward_kl` uses the teacher inference deployment already used for
teacher scoring, requests `top_logprobs=K` on the echo scoring call, builds
student `[N,K]` soft-target datums, and trains the student with builtin
`cross_entropy`.

This recipe enforces `--sdft-top-k <= 5` to match the current inference
`top_logprobs` response limit.

## Multi-teacher OPD = multi-TARGET routing (one student, N teachers)

This is **routing, not blending**. Each prompt is scored by **exactly one**
teacher, chosen by the dataset row's `route_key` value (which must equal a
configured teacher `route_value`, or the teacher `model` when `route_value` is
unset). The student samples from its single deployment; routing different
prompts to different teacher deployments lets the async sampling window
**interleave scoring across teachers**, keeping every deployment's GPU busy
with multiple targets in flight.

For sampled-token OPD, per-prompt **mixture/blend** is intentionally **not**
used online; routed OPD keeps one teacher scorer per prompt. With
`topk_forward_kl`, the same `multi_teacher` config scores each sampled response
with every teacher inference deployment, blends sparse top-K probability mass
by token id, and trains on that combined sparse target.

Multi-teacher SDFT uses **probability-union** blending at each response
position:

1. normalize each teacher's returned top-K over that teacher's sparse support;
2. multiply by `TeacherConfig.blend_weight` (default `1.0`);
3. add probability mass for matching token ids over the union of candidates;
4. keep the top `sdft_top_k` merged ids;
5. renormalize and send those ids/weights to `cross_entropy`.

```python
from training.utils.distillation import MultiTeacherConfig, TeacherConfig

cfg.multi_teacher = MultiTeacherConfig(
    teachers=[
        TeacherConfig(
            model="accounts/fireworks/models/qwen3-32b",
            route_value="math-teacher",
            deployment_shape="accounts/fireworks/deploymentShapes/...",
            teacher_messages_key="math_teacher_messages",
            blend_weight=1.0,
        ),
        TeacherConfig(
            model="accounts/fireworks/models/qwen3-14b",
            route_value="code-teacher",
            teacher_messages_key="code_teacher_messages",
            blend_weight=1.0,
        ),
    ],
    route_key="teacher",   # each row sets row["teacher"] = one route_value above
)
```

Or via env on the `__main__` entrypoint:
`DISTILLATION_TEACHERS="acct/.../qwen3-32b,acct/.../qwen3-14b"`,
`DISTILLATION_TEACHER_ROUTE_KEY=teacher`.

A single `teacher_model` with no `multi_teacher` creates one teacher spec.
Base-model teachers are each auto-deployed as a teacher inference deployment
through the FireTitan service client. The recipe does not construct deployment
managers directly; trainer, student sampler deployment, and teacher inference
deployment placement are resolved by the SDK.

When `TrainerConfig.region` is set, the SDK uses that same region for the
managed student sampler deployment and any auto-created teacher inference
deployment. Use one run region for capacity planning instead of configuring a
separate teacher region in the recipe.

Each teacher can override `route_value`, `deployment_shape`,
`teacher_messages_key`, and `blend_weight`. The selected teacher's
`teacher_messages_key` is used for routed sampled reverse-KL rows; for
multi-teacher SDFT, each teacher uses its own key and contributes according to
its `blend_weight`. When the key is missing, scoring falls back to the
student-visible prompt. Teacher and student token IDs must share a tokenizer.
Set `TeacherConfig.tokenizer_model` when you want the recipe to validate this
explicitly against `deployment.tokenizer_model`.

**Observability:** per-teacher `teacher_route/<slug>/scored` (cumulative routed
attempts → skew) and `teacher_route/<slug>/inflight` (live gauge → saturation)
are logged to wandb each step so skewed routing or idle teachers are visible.
The adaptive concurrency controller only watches the *student* deployment, so
heavily skewed routing can leave some teacher GPUs idle — a data-distribution
issue, not a code one.

## Example

`gsm8k_privileged/` — privileged-context distillation: the student sees the
problem, the frozen teacher additionally sees the worked solution. Run
`prepare_data.py` then `train_gsm8k_privileged.py`.

Single-teacher SDFT forward-KL smoke:

```bash
python training/examples/distillation/gsm8k_privileged/train_gsm8k_privileged.py \
  --distill-mode topk_forward_kl \
  --sdft-top-k 5 \
  --training-shape accounts/<your-account>/trainingShapes/<student-shape> \
  --base-model accounts/fireworks/models/qwen3p5-9b \
  --teacher-base-model accounts/fireworks/models/qwen3p5-9b \
  --tokenizer-model Qwen/Qwen3.5-9B
```

### Routed MOPD Data Format

Fireworks routed MOPD does not require a special dataset builder. It needs JSONL
rows where:

- `messages` is the student-visible OpenAI chat prompt.
- `teacher` is the default route key. Its value must exactly match one
  configured `TeacherConfig.route_value`, or `TeacherConfig.model` when
  `route_value` is unset.
- `teacher_messages` is optional. If absent, the selected teacher scores the
  student rollout using `messages`. If a teacher uses a custom
  `teacher_messages_key`, that key is read for rows routed to that teacher.
- `expected_answer` is optional for training, but useful for evaluation.

Example:

```json
{
  "messages": [{"role": "user", "content": "Solve: 6 * 7. End with Final: <answer>."}],
  "teacher": "math-teacher",
  "expected_answer": "42"
}
```

Rows converted from veRL/slime/AReaL-style math sources can keep metadata such
as `data_source`, `ability`, `reward_model`, and `extra_info`; the route
contract is only the Fireworks fields above. Same-family teachers are
recommended because sampled-token distillation requires teacher and student
token IDs to share a tokenizer.

### Two-Teacher LoRA Smoke Example

`routed_mopd/train_two_teacher_lora.py` is a ready-to-run smoke example with:

- one Qwen3.6 35B-A3B LoRA student using
  `accounts/fireworks/trainingShapes/qwen3p6-35b-a3b-256k-lora`
- two logical route labels over the Qwen3.6 35B-A3B teacher base model that the
  recipe auto-deploys
- a tiny generated JSONL dataset written into the run log directory

```bash
FIREWORKS_API_KEY=... \
python training/examples/distillation/routed_mopd/train_two_teacher_lora.py
```

Override `--teacher-a`, `--teacher-b`, `--teacher-a-route`, and
`--teacher-b-route` to route against different tokenizer-compatible teachers.
