"""Black-box coding-agent RL example for ProRL SWE-Gym parity.

Runs an unmodified coding-agent CLI in a local SWE-Gym Docker runtime behind an
Anthropic ``/v1/messages`` shim. The shim turns each turn into Fireworks
token-in-token-out inference, captures prompt/output tokens plus logprobs,
grades the produced ``git diff`` in a fresh SWE-Gym runtime, and returns one
``RolloutRun`` to ``async_rl_loop``.

The public comparison target is NVIDIA ProRL-Agent-Server's
``examples/swegym_slime_grpo`` path: same SWE-Gym split, public image naming,
fresh-runtime SWE-bench harness grading, and GRPO batch sizing. Fireworks
substitutes its deployment sampler, weight sync, and async RL loop for
Polar/Slime infrastructure.
"""
