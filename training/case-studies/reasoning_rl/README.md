# GRPO: Helping a model reason better

> **Managed RFT has been removed.** For GRPO use the Training API (`training/examples/rl/deepmath/` or `training/recipes/rl_loop.py`). `rft_grpo_math.ipynb` no longer runs and stays as a historical example only.

In this example we'll improve a model's step-by-step problem solving using reinforcement learning — rewarding it for reaching the *right answer*, and letting it figure out the reasoning on its own.

**Is this you?** Your answers are objectively checkable (right or wrong), and the model just needs to *think better* to get there — not learn a new output format. You have a way to grade answers, but you don't have gold worked-solutions to copy. Classic cases: math, code that must pass tests, extraction you can validate.

**The data.** We'll use [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k), grade-school math word problems with a single numeric answer — easy to check automatically. It's **text**.

**The model.** New runs should use a Training API recipe (for example DeepMath). The older `qwen3-8b` managed RFT notebook no longer runs.

**The technique.** This is **GRPO**, a reinforcement fine-tuning method with a simple right/wrong reward. RL is the right tool here because we can *check* answers but can't hand the model gold reasoning to imitate.

**What we'll do.** Measure accuracy on held-out problems before and after:

- `training/examples/rl/deepmath/` — **preferred**: Training API GRPO on math with an inline reward.
- `rft_grpo_math.ipynb` — managed RFT via the Python SDK (`client.reinforcement_fine_tuning_jobs.create`). Managed RFT has been removed, so the notebook no longer runs and the SDK can't read the jobs it created. Run `firectl rftj get <job-id>` to read one.
- For the hands-on **serverless** GRPO loop (you write `forward_backward` / `optim_step` yourself), see [`grpo_countdown`](../grpo_countdown/).

The training cells cost GPU.
