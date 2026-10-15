# Harbor agentic RL

## Run this with your agent

Paste this into Claude Code, Cursor, Codex, or another coding agent. The
agent installs the Fireworks training skills if they are missing and shows
you one plan with its cost before it creates anything. The human-readable
guide is [Cookbook: Agentic Reinforcement Learning](https://docs.fireworks.ai/fine-tuning/training-api/cookbook/agentic-rl).

```text
Set up the Fireworks agentic RL cookbook for me with the smallest possible smoke test, starting with the Terminal-Bench example on local Docker:
https://github.com/fw-ai/cookbook/blob/main/training/examples/harbor_rl/tasks/terminal_bench/README.md

If the Fireworks training skills are not installed, install them first.
Claude Code:
claude plugin marketplace add fw-ai/cookbook
claude plugin install fireworks-training@fw-ai-cookbook
Cursor, Codex, or OpenCode. Set -a to cursor, codex, or opencode. Do not pass -a '*':
npx --yes skills add fw-ai/cookbook -g -s fireworks-training research configure debug -a cursor -y

If FIREWORKS_API_KEY is missing, walk me through https://app.fireworks.ai/settings/users/api-keys. Do not ask me to paste the key in chat.
Run only local and read-only steps first. Show me the plan and its cost, and wait for my approval before you create anything. When we are done, clean up only what this run created.
```

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
