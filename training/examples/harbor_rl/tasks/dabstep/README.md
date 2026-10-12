# DABstep RL

Two DABstep recipes share this package:

- `train.py` — OpenCode on the serverless async-RL loop over a pinned
  manifest (67 training tasks, 8 holdouts, 6 sampling-calibration tasks) with
  adaptive task selection.
- [`../../train_pi.py`](../../train_pi.py) — Pi on the complete 450-task default
  split in E2B sandboxes with the dedicated `async_rl_loop`.

| File | Role |
| --- | --- |
| `manifest.py` | Pinned manifest verification, `AdaptiveTaskSelector`, `make_adaptive_rollout_factory`, sign-sensitive scorer patch |
| `dataset.py` | Frozen 450-task split, seeded run order, progressive Pi preparation in waves of 64 |
| `prepare_tasks.py` | OpenCode task preparation with the sign-sensitive numeric scorer |
| `train.py` | OpenCode serverless recipe |

The upstream DABstep numeric scorer's regular expression omits leading signs,
so `-2.18` can match a `+2.18` reference. Both recipes make the copied scorer
sign-sensitive and fail closed if the upstream scorer no longer has the known
form.

Install Harbor plus the example-only hash dependency (add `harbor[e2b]` and
`'e2b<2.35'` for the Pi recipe; newer e2b conflicts with the SDK's pyqwest pin):

```bash
cd training
uv sync
uv pip install --python .venv/bin/python \
  'harbor[e2b]==0.21.0' 'e2b<2.35' 'dirhash>=0.5,<1'
```

## OpenCode serverless recipe

Prepare tasks, then generate or refresh the external manifest (the corrected
scorer intentionally changes each task's content hash). Both sampling and
training verify those hashes before creating any Fireworks resource:

```bash
uv run python -m training.examples.harbor_rl.tasks.dabstep.prepare_tasks \
  --source ~/.cache/harbor-dabstep/tasks/source \
  --destination ~/.cache/harbor-dabstep/tasks/opencode
```

Run the six-task sampling gate first. It evaluates a zero-update LoRA snapshot
and fails unless the mean reward lands in `[0.40, 0.65]`:

```bash
uv run python -m training.examples.harbor_rl.tasks.dabstep.train \
  --sampling-only \
  --base-model <qualified-base-model> \
  --tokenizer-model <qualified-tokenizer> \
  --renderer-name glm_moe_dsa_preserve_thinking \
  --manifest ~/.cache/harbor-dabstep/manifests/dabstep.json \
  --harbor-dataset ~/.cache/harbor-dabstep/tasks/opencode \
  --harbor-trial-config ~/.cache/harbor-dabstep/trial.yaml
```

Then train with the same rollout:

```bash
uv run python -m training.examples.harbor_rl.tasks.dabstep.train \
  --base-model <qualified-base-model> \
  --tokenizer-model <qualified-tokenizer> \
  --renderer-name glm_moe_dsa_preserve_thinking \
  --manifest ~/.cache/harbor-dabstep/manifests/dabstep.json \
  --harbor-dataset ~/.cache/harbor-dabstep/tasks/opencode \
  --harbor-trial-config ~/.cache/harbor-dabstep/trial.yaml \
  --max-rows 320 \
  --wandb-entity <entity>
```

The recipe fixes 8 completions x 8 groups, two pipeline chunks, 32,768 tokens
per OpenCode turn, and server-side GRPO on a rank-64 LoRA; `--max-seq-len`,
`--lora-rank`, and the AdamW flags are explicit experiment knobs. Training rows
are placeholders: the adaptive selector chooses distinct tasks per optimizer
batch weighted by their usable-group probability, and records every member's
reward (a discarded trajectory counts as an unsolved attempt). Holdout
evaluation runs at the start, every `--evaluation-interval` steps, and at the
end; it never touches the selector.

The run directory receives `metrics.jsonl`, `selector-state.json`,
`task-selections.json`, and `run-state.json`. Continue on new resources by
passing `--resume-from`, `--row-offset`, `--step-offset`, and
`--selector-state-in` together (plus the prior `--wandb-run-id` when W&B is
enabled).

## Pi managed-resource recipe

The input directory must be a materialized Harbor DABstep default split with
its `.fw-ai-package-snapshot.json`. The recipe verifies all 450 tasks, freezes
and shuffles their order, repairs the numeric scorer, and prepares Pi E2B
images in ordered waves of 64 just before their first rollout. Four tasks from
the same split provide diagnostic evaluation and remain in training.

```bash
export FIREWORKS_API_KEY=...
export E2B_API_KEY=...
export WANDB_API_KEY=...  # only when --wandb-entity is set

uv run python -m training.examples.harbor_rl.train_pi \
  --harbor-dataset ./datasets/dabstep \
  --run-dir ./runs/dabstep-pi \
  --shuffle-seed 42 \
  --base-model accounts/example/models/policy \
  --tokenizer-model example/tokenizer \
  --renderer-name example_renderer \
  --wandb-entity example
```

The recipe uses full-parameter server-side GRPO, binary Harbor reward, eight
completions for each of eight prompt groups, two pipeline chunks, a
two-version off-policy window, one epoch, a 524,288-token sequence limit, and
a 65,536-token per-turn limit. It saves every 40 optimizer steps. Resource
shapes are selected by the SDK; `--trainer-job-id` and `--deployment-id`
reattach existing resources, and `--skip-cleanup` keeps them after the run.
`launch.json` records the dataset manifest digest and the resolved `Config`.

The sidecar defaults to full-history rendering, and `--no-tito-debug`
disables the optional debug stream. To resume on new resources, pass a
cross-job checkpoint with `--init-from-checkpoint` and the matching
`--start-task-index` together.
