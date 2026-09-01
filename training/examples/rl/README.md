# Reinforcement learning examples

This directory contains two kinds of RL examples:

1. **Standalone examples** accumulated for different tasks and integrations.
   They are useful focused references, but they do not share one directory
   structure, rollout abstraction, or environment lifecycle.
2. **The Harbor-based recipe set**, which is the structured starting point for
   agentic RL. It separates the task recipe, agent harness, sandbox, token-exact
   trajectory capture, and training loop so each layer can evolve independently.

## Harbor-based agentic RL

Start in [`harbor/`](./harbor/) when the task can be expressed as a Harbor trial.
The same structure supports local Docker and E2B environments:

| Layer | Location | Responsibility |
| --- | --- | --- |
| Task recipes | [`harbor/recipes/`](./harbor/recipes/) | Select and prepare datasets, configure training, and choose a harness |
| Harness adapters | [`harbor/pi/`](./harbor/pi/), [`harbor/opencode/`](./harbor/opencode/), [`harbor/mini_swe/`](./harbor/mini_swe/) | Run the agent and translate its lifecycle into the shared rollout contract |
| Harbor and TITO integration | [`harbor/tito/`](./harbor/tito/) | Own the Harbor trial, environment-local sidecar, exact-token trajectory, artifacts, and cleanup |
| Shared training recipe | [`recipes/async_rl_loop.py`](../../recipes/async_rl_loop.py) | Fan out rollouts, form prompt groups, compute advantages, and drive optimization |

The Pi and OpenCode entrypoints live together under `harbor/recipes/`:
[`train_pi.py`](./harbor/recipes/train_pi.py) defaults to DABstep, while
[`train_opencode.py`](./harbor/recipes/train_opencode.py) accepts any supported
Harbor dataset.

The current task-oriented recipes include:

- [`harbor/recipes/deep_swe/`](./harbor/recipes/deep_swe/) — DeepSWE task preparation.
- [`harbor/recipes/dabstep/`](./harbor/recipes/dabstep/) — DABstep training through either the OpenCode serverless loop or Pi with E2B and SDK-managed resources.
- [`harbor/recipes/terminal_bench/`](./harbor/recipes/terminal_bench/) — Terminal-Bench training with Harbor and OpenCode.

Pi is the default reference harness for DeepSWE and the managed DABstep recipe.
OpenCode drives the DABstep serverless recipe and Terminal-Bench; Mini-SWE-Agent
is another adapter over the same Harbor and TITO boundaries. The sidecar is
harness-neutral: harness-specific parsing, events, and process behavior remain
inside the corresponding adapter.

The Harbor recipes use the shared async RL loop as training infrastructure; the
loop is not a separate family of examples or the user-facing organization of
this directory. Most integrations should add a task recipe or harness adapter
under `harbor/` rather than create another top-level training stack.

For the ownership, exact-token, renderer, and artifact contracts, see
[`/skills/fireworks-training/references/rl-agentic.md`](/skills/fireworks-training/references/rl-agentic.md).
For shared training controls, see
[`/skills/fireworks-training/references/rl-async.md`](/skills/fireworks-training/references/rl-async.md).

## Standalone examples

The remaining entries are independent examples with narrower goals:

- [`single_turn_token_in/`](./single_turn_token_in/) — pre-tokenized,
  single-turn completion rollouts.
- [`deepmath/`](./deepmath/) — task-specific math RL.
- [`frozen_lake/`](./frozen_lake/) — a controlled multi-turn environment loop.
- [`visual_toolbench/`](./visual_toolbench/) — a custom visual tool-use loop.
- [`eval_protocol_chat/`](./eval_protocol_chat/) — chat rollouts through an Eval
  Protocol remote processor.
- [`vanilla_sampler.py`](./vanilla_sampler.py) — a minimal sampler example.

These examples intentionally remain useful references, but their APIs and
layouts should not be treated as one common architecture. New agentic RL work
should prefer the Harbor-based recipe structure above.
