# DABstep RL

This package keeps two independent DABstep execution paths:

- `train.py` is the existing OpenCode recipe over
  `async_rl_loop_serverless`. It uses the pinned 67-task training manifest,
  eight holdouts, and optional six-task sampling preflight described in the
  [OpenCode guide](../../opencode/README.md#pinned-dabstep-reproduction).
- [`../train_pi.py`](../train_pi.py) is the top-level Pi entrypoint and defaults
  to the complete 450-task DABstep split through Harbor E2B, an
  environment-local TITO sidecar, and the managed `async_rl_loop`.
  The SDK creates compatible trainer and hot-load deployment resources; this
  command accepts no resource IDs or shape arguments.

The serverless path remains experimental and less integrated than the managed
recipe, but it remains a supported example and is not routed through the
managed trainer/deployment lifecycle.

## Pi managed-resource recipe

The input directory must be a materialized Harbor DABstep default split with
its `.fw-ai-package-snapshot.json`. The recipe verifies all 450 tasks, freezes
and shuffles their order, repairs the known sign-insensitive numeric scorer,
and prepares Pi E2B images in ordered waves of 64. Four tasks from the same
frozen split provide diagnostic evaluation and remain eligible for training.

Install the E2B Harbor extra and the example-only hash dependency:

```bash
cd training
uv sync
uv pip install --python .venv/bin/python \
  'harbor[e2b]==0.21.0' 'dirhash>=0.5,<1'
```

Set credentials in the environment and launch:

```bash
export FIREWORKS_API_KEY=...
export E2B_API_KEY=...
export WANDB_API_KEY=...  # only when --wandb-entity is set

uv run python -m training.examples.rl.harbor.recipes.train_pi \
  --harbor-dataset ./datasets/dabstep \
  --run-dir ./runs/dabstep-pi \
  --shuffle-seed 42 \
  --base-model accounts/example/models/policy \
  --tokenizer-model example/tokenizer \
  --renderer-name example_renderer \
  --wandb-entity example
```

The fixed managed recipe uses full-parameter server-side GRPO, binary Harbor
reward, eight completions for each of eight prompt groups, two pipeline chunks,
a two-version off-policy window, one epoch, a 524,288-token sequence limit, and
a 65,536-token per-turn limit. It saves every 40 optimizer steps and deletes
SDK-created compute after writing the final checkpoint. `--replica-count`,
`--learning-rate`, and operational timeout/concurrency flags remain available;
resource shape selection deliberately does not.

The sidecar defaults to full-history rendering. The renderer must be certified
for the chosen tokenizer/model pair; unsupported pairs fail closed before a
trial starts. Exact-token artifacts and TITO metrics are retained under the run
directory, and `--no-tito-debug` disables the optional debug JSONL stream.

To resume on newly created resources, pass a cross-job checkpoint reference
with `--init-from-checkpoint` and the matching `--start-task-index`. These two
arguments must be supplied together.
