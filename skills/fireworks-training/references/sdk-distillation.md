# Distillation Recipe

Use this reference when working with `training/recipes/distillation_loop.py`,
on-policy distillation (OPD), or `teacher_messages`. For top-K SDFT on the
shared serverless pool, use the [external SDFT cookbook](https://github.com/fw-ai-external/tinker-cookbook); see the [customer guide](../../../training/docs/sdft.md).

Before editing, read the recipe and the public example guide:

```bash
sed -n '1,220p' training/recipes/distillation_loop.py
sed -n '1,220p' training/examples/distillation/README.md
```

## Contents

- [Objective](#objective)
- [Logprob Response Fields](#logprob-response-fields)
- [Minimal Config](#minimal-config)
- [Dataset Contract](#dataset-contract)
- [Evaluation Hook](#evaluation-hook)
- [Customizing](#customizing)
- [Guardrails](#guardrails)
- [Validation](#validation)

## Objective

The recipe has one objective: sampled-token reverse KL. Each step:

1. the student samples from its hot-loaded deployment and one frozen teacher
   scores those exact response tokens;
2. one server-side `importance_sampling` forward/backward with per-token
   advantage `teacher_logprob - sampling_logprob` (times `opd_loss_scale`),
   then one optimizer step;
3. the new student weights are hot-loaded before the next batch.

```text
advantage = teacher_logprob - sampling_logprob
```

The importance ratio in the built-in loss corrects for rollout/current-policy
drift; `ratio_log_cap` clamps its log. There is no outcome reward and no
reference model. Helpers live in `training/utils/distillation/`: `losses.py`
(datums), `sampling.py` (teacher scoring and alignment), `teacher.py` (teacher
deployment), `eval.py` (privileged-context eval).

## Logprob Response Fields

Keep the public inference field semantics straight when changing OPD
plumbing:

| Field or request option | Meaning |
|-------------------------|---------|
| `top_k` | Sampling filter. It keeps at most K next-token candidates eligible for sampling and renormalizes probability mass over the filtered set. |
| `sampling_mask` | Optional generated-token metadata: `count`, `non_zero_list`, or `non_zero_buffer` reports how many or which token IDs remained eligible after filters such as `top_p` and `top_k`. |
| `logprob` | Raw model logprob for the returned token. Legacy responses call this `token_logprobs`. |
| `sampling_logprob` | Final generation logprob after temperature and sampling filters/masks. This is the right field when the objective needs the probability of the token under the actual sampler distribution. |
| `top_logprobs` | Response field for likely alternatives at a position. It does not change sampling and is capped at 5 by the public inference API. |

The recipe asks the student sampler for generated token logprobs; the sampler
SDK extracts `choice.logprobs.content[].logprob` and stores the aligned values
as `sampling_logprobs` for OPD batching. Teacher scoring requests use
`logprobs=True` and `echo=True`. If a change needs exact sampler-distribution
probabilities, wire through `sampling_logprob` rather than only raw `logprob`.

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

main(cfg, cancel_on_exit=True)
```

If `teacher_model` is a base-model resource, the recipe creates or reuses a
frozen teacher inference deployment (`teacher_deployment_id`, or a default id
derived from the model). If it is already an inference model or deployment
resource, the recipe uses it directly. `cancel_on_exit=True` deletes the
trainer and scales created deployments to zero when the run ends.

## Dataset Contract

Rows are JSONL objects. Required:

- `messages`: student-visible OpenAI-style chat messages.

Optional:

- `teacher_messages`: privileged teacher-side prompt. If absent, the teacher
  scores the sampled response under `messages`. The aliases
  `privileged_messages` and `teacher_prompt_messages` are also accepted.
- `expected_answer`: optional metadata for eval callbacks and smoke checks.

## Evaluation Hook

`main(cfg, eval_fn=...)` takes an optional async callback. With
`Config.eval_every = N > 0` it runs once before training and after every N
steps, once the new weights are served. It receives an
`training.utils.distillation.eval.EvalContext` (config, rows, student and
teacher samplers, tokenizer, step, max_seq_len) and returns `eval/*` metrics;
the last result is returned as `result["eval"]`.
`make_teacher_trace_logprob_gap_eval(...)` is the privileged-context eval used
by `examples/distillation/gsm8k_privileged`.

## Customizing

The loop has no mode flags. Fork it to change behavior:

- **Several teachers:** create one sampler per teacher and pick the scorer per
  row inside `sample_one_prompt` (for example by a `row["teacher"]` route key).
- **Different objective:** replace the datum builder and the loss call in phase
  2 of the training loop.
- **Top-K SDFT:** use the [external Fireworks Tinker-Cookbook fork](https://github.com/fw-ai-external/tinker-cookbook).

## Guardrails

- Student and teacher token IDs must use a compatible tokenizer and vocabulary.
  Prefer same-family teachers; `deployment.tokenizer_model` tokenizes both the
  student and the teacher prompt.
- Do not add a separate teacher region knob. `TrainerConfig.region` is the run
  region used by the SDK-managed student sampler and auto-created teacher
  deployment.
- `teacher_replica_count` controls replicas for an auto-created frozen teacher
  deployment; `teacher_deployment_shape` selects its shape (unset lets the
  deployment API choose).
- The recipe is strictly on-policy: it hot-loads the trainer's weights before
  the first rollout and after every optimizer step.

## Validation

For logic or documentation changes around this recipe, run the focused unit
suite when dependencies are available:

```bash
cd training
pytest -q tests/unit/test_distillation.py tests/test_smoke_imports.py
```

For example-only changes, also import or run the touched example with a small
generated dataset when it does not require live Fireworks credentials.
