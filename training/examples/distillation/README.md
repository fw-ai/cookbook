# Distillation (OPD)

## Run this with your agent

Paste this into Claude Code, Cursor, Codex, or another coding agent. The
agent installs the Fireworks training skills if they are missing and shows
you one plan with its cost before it creates anything. The human-readable
guide is [Cookbook: Distillation](https://docs.fireworks.ai/fine-tuning/training-api/cookbook/distillation).

```text
Run the Fireworks on-policy distillation cookbook for me as a small smoke test, and make sure nothing is left running afterward:
https://github.com/fw-ai/cookbook/blob/main/training/examples/distillation/README.md

If the Fireworks training skills are not installed, install them first.
Claude Code:
claude plugin marketplace add fw-ai/cookbook
claude plugin install fireworks-training@fw-ai-cookbook
Cursor, Codex, or OpenCode. Set -a to cursor, codex, or opencode. Do not pass -a '*':
npx --yes skills add fw-ai/cookbook -g -s fireworks-training research configure debug -a cursor -y

If FIREWORKS_API_KEY is missing, walk me through https://app.fireworks.ai/settings/users/api-keys. Do not ask me to paste the key in chat.
Run only local and read-only steps first. Show me the plan and its cost, and wait for my approval before you create anything. When we are done, clean up only what this run created.
```

Train a student on its **own rollouts** to match a frozen teacher. The shipping
recipe is `training/recipes/distillation_loop.py`; helpers live in
`training/utils/distillation/` (`losses.py`, `sampling.py`, `teacher.py`,
`eval.py`). For top-K SDFT on the shared serverless pool, see
`training/recipes/sdft_loop.py`.

## Objective

| Objective | Teacher signal | Backend loss |
| --- | --- | --- |
| reverse `KL(S||T)`, sampled-token estimate | teacher logprob on each sampled token | builtin `importance_sampling` |

The student samples, the frozen teacher scores the same tokens, and per-token
`teacher_logprob - sampling_logprob` becomes the dense advantage. The recipe
uses backend `importance_sampling` to account for rollout/current-policy drift;
importance sampling is the estimator, not the objective. Each step trains on
fresh rollouts and hot-loads the new student before sampling again.

```text
1. student inference samples rollouts          -> on-policy token sequences
2. teacher inference echo with logprobs=True   -> teacher logprob per sampled token
3. advantage = teacher_lp - sampling_lp        -> importance_sampling datums
4. forward_backward + optim_step + hotload     -> next batch is on-policy
```

## Teacher

`teacher_model` names one teacher sharing the student's tokenizer:

- a base model such as `accounts/fireworks/models/qwen3p5-9b` is deployed as a
  frozen teacher inference deployment through the FireTitan service client
  (`teacher_deployment_id`, `teacher_deployment_shape`, `teacher_replica_count`);
- a deployment or deployed-model resource is used directly.

The recipe does not construct deployment managers directly; trainer, student
sampler deployment, and teacher inference deployment placement are resolved by
the SDK. When `TrainerConfig.region` is set, the SDK uses that same region for
the managed student sampler deployment and an auto-created teacher deployment.

To distill from several teachers, fork the recipe: create one sampler per
teacher and choose the scorer per row inside `sample_one_prompt` (for example
from a `row["teacher"]` route key).

## Data Format

JSONL rows where:

- `messages` is the student-visible OpenAI chat prompt.
- `teacher_messages` is optional privileged teacher context. If absent, the
  teacher scores the student rollout using `messages`. The aliases
  `privileged_messages` and `teacher_prompt_messages` are also accepted.
- `expected_answer` is optional for training, but useful for evaluation.

```json
{
  "messages": [{"role": "user", "content": "Solve: 6 * 7. End with Final: <answer>."}],
  "teacher_messages": [{"role": "user", "content": "Worked solution: 6 * 7 = 42. End with Final: 42."}],
  "expected_answer": "42"
}
```

Rows converted from veRL/slime/AReaL-style math sources can keep metadata such
as `data_source`, `ability`, `reward_model`, and `extra_info`; the recipe reads
only the fields above.

## Examples

`gsm8k_privileged/` — privileged-context distillation: the student sees the
problem, the frozen teacher additionally sees the worked solution. Run
`prepare_data.py` then `train_gsm8k_privileged.py`. It passes
`make_teacher_trace_logprob_gap_eval(...)` as the recipe's `eval_fn` (with
`eval_every=1`) and validates the run from the `COOKBOOK_METRICS_FILE` ledger.

```bash
python training/examples/distillation/gsm8k_privileged/train_gsm8k_privileged.py \
  --training-shape accounts/<your-account>/trainingShapes/<student-shape> \
  --base-model accounts/fireworks/models/qwen3p5-9b \
  --teacher-base-model accounts/fireworks/models/qwen3p5-9b \
  --tokenizer-model Qwen/Qwen3.5-9B
```

`serverless_k3_privileged/` — the same reverse-KL objective on the shared
serverless pool (no trainer job or deployment): a short loop over the same
`build_opd_prompt_group` / `build_opd_datums` helpers, with Tinker sampling
clients on saved snapshots for the student and the frozen step-0 teacher
(`completions_from_sample_response`,
`score_teacher_logprobs_with_sampling_client`). The teacher's privileged
context is a system instruction rather than a per-row `teacher_messages`.
