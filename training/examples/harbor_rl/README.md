# Harbor agentic RL

Agentic RL where each rollout is a Harbor trial (local Docker or E2B). The
layers are separate so each can change on its own:

| Layer | Location | Responsibility |
| --- | --- | --- |
| Entry points | `train_opencode.py`, `train_pi.py`, `tasks/*/train.py` | Task rows + a harness `make_rollout_fn` bound to its settings with `functools.partial` + recipe `Config` + `main` |
| Shared CLI | `common.py`, `records.py` | Flags, typed `HarborSettings` (`harbor_settings`), sampling-only scoring, run records |
| Tasks | `tasks/` | `dabstep/`, `terminal_bench/`, `textworld/`, `deep_swe/` (preparation), `mimo/` |
| Harnesses | `harnesses/` | `pi/`, `opencode/`, `mini_swe/`, `mimoagent/`: run the agent in the sandbox as a `HarborHarness` |
| Harbor + TITO | `tito/` | `HarborRollout` (retries, admission, materialization), the trial, the environment-local sidecar, exact-token trajectories, artifacts, cleanup |
| Training loop | `recipes/async_rl_loop.py`, `recipes/experiment/async_rl_loop_serverless.py` | Fan out rollouts, group, compute advantages, optimize |

| Entry point | Harness × task | Loop |
| --- | --- | --- |
| `train_opencode.py` | OpenCode × any Harbor dataset | dedicated |
| `train_pi.py` | Pi × 450-task DABstep split (E2B) | dedicated |
| `tasks/textworld/train.py` | Pi × frozen TextWorld cooking games (E2B) | dedicated |
| `tasks/dabstep/train.py` | OpenCode × pinned DABstep with adaptive selection | serverless |
| `tasks/terminal_bench/train.py` | OpenCode × Terminal-Bench 2.0 | serverless |

Add a task under `tasks/` or a harness under `harnesses/`; do not add another
training stack. The sidecar is harness-neutral: parsing, events and process
behavior stay inside the harness. Contracts (ownership, exact tokens,
renderers, artifacts) are in
[`rl-agentic.md`](/skills/fireworks-training/references/rl-agentic.md); shared
training controls in
[`rl-async.md`](/skills/fireworks-training/references/rl-async.md).

Sampling settings in `RolloutSetup.sample_kwargs` are authoritative for MiMo,
Mini-SWE, Pi and OpenCode: the shared TITO sidecar overrides agent-supplied
`temperature`, `top_p` and `top_k` before calling the sampler. `top_k=0` disables
top-k. Omitted fields retain request/service defaults. No harness-specific
inheritance flag or task-image rebuild is needed to change these settings.
