# DABstep RL

This package contains two independent, first-class DABstep workflows:

- `train.py` is the existing adaptive OpenCode/serverless recipe over its
  pinned train and holdout manifest.
- `train_service.py` runs the complete default split through Pi, Harbor E2B,
  an environment-local TITO sidecar, and managed trainer/deployment services.

The service workflow verifies the package snapshot, freezes all 450 tasks in
its recorded order, patches the known numeric scorer so signs are preserved,
and prepares E2B images in ordered waves of 32. The next wave is prepared only
when the rollout producer reaches it. Four tasks from the same frozen split are
used as a fixed diagnostic evaluation; they are not removed from training.

Credentials are environment variables, never command-line arguments:

```bash
export FIREWORKS_API_KEY=...
export E2B_API_KEY=...
export WANDB_API_KEY=...

python -m training.examples.rl.harbor.recipes.dabstep.train_service \
  --harbor-dataset ./datasets/dabstep \
  --run-dir ./runs/dabstep-pi \
  --base-model accounts/example/models/policy \
  --tokenizer-model example/tokenizer \
  --renderer-name example_renderer \
  --trainer-job-id trainer-id \
  --training-shape-id accounts/example/trainingShapes/shape/versions/version \
  --deployment-id deployment-id \
  --deployment-shape accounts/example/deploymentShapes/shape/versions/version \
  --hot-load-trainer-job accounts/example/rlorTrainerJobs/trainer-id
```

The service recipe defaults to one epoch, eight completions per prompt, eight
prompt groups per optimizer step, two pipeline chunks, on-policy server-side
GRPO, full-parameter training, a 524,288-token training context, and a
65,536-token per-turn inference limit. Override resource identifiers and the
learning rate explicitly for a production run; do not reuse the adaptive
serverless manifest as the service dataset cursor.
