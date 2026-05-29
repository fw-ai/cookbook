# On-Policy Distillation (OPD)

Train a student on its **own rollouts** to match a teacher. The student samples,
each rollout is scored by a frozen teacher, and the per-token
`teacher_logprob − sampling_logprob` is fed to the built-in `importance_sampling`
loss — a 1-sample Monte-Carlo estimate of reverse KL `KL(π_S ‖ π_T)` that needs
only the teacher's logprob at the sampled token (no top-K). Recipe:
`training/recipes/opd_loop.py`; helpers: `training/utils/opd.py`,
`training/utils/opd_sampling.py`.

## Multi-teacher = multi-TARGET routing (one student, N teachers)

This is **routing, not blending**. Each prompt is scored by **exactly one**
teacher, chosen by the dataset row's `route_key` value (which must equal a
configured teacher `model`). The student samples from its single deployment;
routing different prompts to different teacher deployments lets the async
sampling window **interleave scoring across teachers**, keeping every
deployment's GPU busy with multiple targets in flight.

Per-prompt **mixture/blend** is intentionally **not** supported — it would be N×
teacher cost for a single target, the opposite of the throughput goal.

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

> **Note:** the routed teacher is scored against the row's privileged prompt
> (`teacher_messages` / default). Per-teacher `TeacherConfig.teacher_messages_key`
> is **not** consumed yet; a non-default value logs a warning.

## Example

`gsm8k_privileged/` — privileged-context OPD: the student sees the problem, the
frozen teacher additionally sees the worked solution. Run `prepare_data.py` then
`train_gsm8k_privileged.py`.
