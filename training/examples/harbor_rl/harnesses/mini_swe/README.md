# Harbor RL with Mini-SWE-Agent

Mini-SWE-Agent is a third harness adapter over the same environment-local
`TITOSidecar` and Harbor trial lifecycle used by OpenCode and Pi. Docker and E2B
use the same task preparation, loopback provider endpoint, exact-token artifact,
and cleanup path. The adapter passes the original task instruction unchanged.

Prepare task images before running so each environment contains the lightweight
sidecar Python runtime:

```bash
uv run python -m training.examples.harbor_rl.harnesses.mini_swe.prepare_tasks \
  --source /path/to/harbor/tasks \
  --destination /path/to/prepared/tasks
```

The Mini-SWE-Agent CLI itself remains owned and installed by Harbor. Cookbook
users construct the rollout through
`training.examples.harbor_rl.harnesses.mini_swe.rollout.make_rollout_fn` (the shared
`tito.rollout.HarborRollout` with the Mini-SWE `HarborHarness`) and pass it to
the existing async RL loop, bound with `functools.partial` to the
`HarborSettings` built as in [`common.py`](../../common.py).
`HarborSettings.reward_key` selects a non-default verifier reward. The current sidecar renderer support boundary is
the same as OpenCode and Pi: GLM-5.2 (`glm_moe_dsa_preserve_thinking`),
Qwen3.8-27B (`qwen3_8`), and Muse Glimmer 30B (`muse_glimmer`, full-history
only); unsupported model/template pairs fail closed.
