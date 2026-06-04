# Distillation (OPD and MOPD)

Train a student on its **own rollouts** to match one or more teachers. The
student samples, each rollout is scored by a frozen teacher, and the per-token
`teacher_logprob − sampling_logprob` is fed to the built-in `importance_sampling`
loss — a 1-sample Monte-Carlo estimate of reverse KL `KL(π_S ‖ π_T)` that needs
only the teacher's logprob at the sampled token (no top-K). Recipe:
`training/recipes/distillation_loop.py`; helpers:
`training/utils/distillation/__init__.py` and
`training/utils/distillation/sampling.py`.

## Multi-teacher = multi-TARGET routing (one student, N teachers)

This is **routing, not blending**. Each prompt is scored by **exactly one**
teacher, chosen by the dataset row's `route_key` value (which must equal a
configured teacher `route_value`, or the teacher `model` when `route_value` is
unset). The student samples from its single deployment; routing different
prompts to different teacher deployments lets the async sampling window
**interleave scoring across teachers**, keeping every deployment's GPU busy
with multiple targets in flight.

Per-prompt **mixture/blend** is intentionally **not** supported — it would be N×
teacher cost for a single target, the opposite of the throughput goal.

```python
from training.utils.distillation import MultiTeacherConfig, TeacherConfig

cfg.multi_teacher = MultiTeacherConfig(
    teachers=[
        TeacherConfig(
            model="accounts/fireworks/models/qwen3-32b",
            route_value="math-teacher",
            deployment_shape="accounts/fireworks/deploymentShapes/...",
            teacher_messages_key="math_teacher_messages",
        ),
        TeacherConfig(
            model="accounts/fireworks/models/qwen3-14b",
            route_value="code-teacher",
            teacher_messages_key="code_teacher_messages",
        ),
    ],
    route_key="teacher",   # each row sets row["teacher"] = one route_value above
)
```

Or via env on the `__main__` entrypoint:
`DISTILLATION_TEACHERS="acct/.../qwen3-32b,acct/.../qwen3-14b"`,
`DISTILLATION_TEACHER_ROUTE_KEY=teacher`. The old `OPD_TEACHERS` and
`OPD_TEACHER_ROUTE_KEY` names are still accepted as legacy fallbacks.

A single `teacher_model` with no `multi_teacher` is the default
backward-compatible path. Base-model teachers are each auto-deployed as a frozen
deployment.

Each teacher can override `route_value`, `deployment_shape`,
`teacher_messages_key`, and `top_logprobs`. The selected teacher's
`teacher_messages_key` is used for that row; when the key is missing, scoring
falls back to the student-visible prompt. Teacher and student token IDs must
share a tokenizer. Set `TeacherConfig.tokenizer_model` when you want the recipe
to validate this explicitly against `deployment.tokenizer_model`.

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

- one Qwen3.5 35B-A3B LoRA student using
  `accounts/fireworks/trainingShapes/qwen3p5-35b-a3b-256k-lora`
- two logical route labels over the Qwen3.5 35B-A3B teacher base model that the
  recipe auto-deploys
- a tiny generated JSONL dataset written into the run log directory

```bash
FIREWORKS_API_KEY=... \
python training/examples/distillation/routed_mopd/train_two_teacher_lora.py
```

Override `--teacher-a`, `--teacher-b`, `--teacher-a-route`, and
`--teacher-b-route` to route against different tokenizer-compatible teachers.
