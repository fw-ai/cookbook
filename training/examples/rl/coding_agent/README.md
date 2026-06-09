# Black-box Coding-Agent RL: ProRL SWE-Gym Parity

This example is the Fireworks cookbook analogue of NVIDIA
ProRL-Agent-Server's `examples/swegym_slime_grpo` training path. It uses the
same public SWE-Gym split, public SWE-Gym/SWE-bench runtime images, patch
filtering, and fresh-runtime SWE-bench harness grading, then trains with
Fireworks `async_rl_loop`.

The public example intentionally keeps one path:

- dataset: `NovaSky-AI/SkyRL-v0-293-data`, train split
- runtime: local Docker using the public SWE-Gym/SWE-bench images
- agent: unmodified `claude-code` CLI pointed at the Fireworks shim
- grader: SWE-Gym/SWE-Bench-Fork harness in a fresh runtime
- training loop: Fireworks `async_rl_loop`

ProRL uses Polar rollout/gateway plus Slime/SGLang for training; this cookbook
version keeps the same task/image/grader recipe but substitutes Fireworks
deployment sampling, weight sync, and `async_rl_loop`.

## Files

| File | Role |
|---|---|
| `swegym_data.py` | Downloads/converts the ProRL SWE-Gym 293-row split into cookbook JSONL rows. |
| `make_swegym_data.py` | CLI for writing a SWE-Gym JSONL subset or full split. |
| `sandbox.py` | Local Docker runtime, agent launch, patch capture, and fresh-runtime SWE-Gym grading. |
| `swebench_harness.py` | SWE-Gym/SWE-bench patch filtering, application, and resolved-status grading. |
| `shim.py` | Anthropic `/v1/messages` shim backed by Fireworks TITO sampling and token/logprob capture. |
| `trajectory.py` | Stitches turn records into masked token segments. |
| `rollout.py` | `make_rollout_fn(setup)` for `async_rl_loop`. |
| `train.py` | ProRL-parity async GRPO launcher. |

## How one rollout flows

```
async_rl_loop fans one SWE-Gym row into completions_per_prompt rollouts.
Each rollout:
  open shim session (unique session_id)
    -> boot the SWE-Gym Docker image from row metadata
    -> run claude-code in /testbed
         ANTHROPIC_BASE_URL = shim
         ANTHROPIC_AUTH_TOKEN = session_id
         each turn: shim renders conversation -> Fireworks TITO sample
                    -> records prompt/output tokens and logprobs
                    -> returns Anthropic response blocks to the agent
    -> capture git diff
    -> close work runtime
    -> grade the diff in a fresh copy of the same image with SWE-Gym/SWE-bench
    -> drain session -> one RolloutRun with one or more trainable segments
```

`loss_mask = 1` on assistant-generated tokens and `0` on rendered scaffolding,
tool results, and user context. Multiple segments from one agent run share one
reward and one GRPO advantage.

## Setup

```bash
export FIREWORKS_API_KEY=...
export WANDB_API_KEY=...
export AGENT_HEAD_HOST=127.0.0.1

# Optional: mount a ProRL-style shared Node/agent CLI directory into runtimes.
export SWE_AGENT_CLI_DIR=/path/to/opt_node

# Observability
export SWE_TRAJECTORY_LOG_DIR=/tmp/cagent-trajectories
export SWE_PREFIX_MISMATCH_LOG=/tmp/cagent-prefix-mismatch.jsonl
export SWE_SHIM_METRICS=1
```

Install the SWE-Gym harness dependency:

```bash
uv pip install --python .venv/bin/python -e '.[coding-agent]'
```

Generate the same public split ProRL uses:

```bash
.venv/bin/python examples/rl/coding_agent/make_swegym_data.py \
  --output /tmp/cagent-swegym-293.jsonl
```

For a cheaper verification run, use a deterministic subset:

```bash
.venv/bin/python examples/rl/coding_agent/make_swegym_data.py \
  --output /tmp/cagent-swegym-50.jsonl \
  --max-rows 50
```

## Training

```bash
.venv/bin/python examples/rl/coding_agent/train.py \
  --dataset-path /tmp/cagent-swegym-50.jsonl \
  --log-path ./coding_agent_logs/swegym-50
```

Defaults mirror the ProRL SWE-Gym GRPO recipe at the task/batch level:

| Setting | Value |
|---|---:|
| `completions_per_prompt` | 16 |
| `prompt_groups_per_step` | 4 |
| `max_concurrency_rollout_sample` | 64 |
| `max_head_offpolicy_versions` | 4 |
| `learning_rate` | `1e-6` |
| `kl_beta` | `0.001` |

## ProRL comparison

The source of truth checked for this parity pass is
`NVIDIA-NeMo/ProRL-Agent-Server@stable`:

- `examples/swegym_slime_grpo/sample_tasks.py` for dataset and image naming
- `examples/swegym_slime_grpo/prepare_data.py` for prompt row shape
- `examples/swegym_slime_grpo/polar_config.yaml` for SWE-bench harness grading,
  patch filtering, and runtime layout
- `examples/swegym_slime_grpo/run.sh` for GRPO batch sizing

This is not infra-identical. ProRL runs Polar rollout/gateway and Slime/SGLang;
the Fireworks cookbook example keeps one local-Docker shim path and relies on
Fireworks trainer/deployment resources.
