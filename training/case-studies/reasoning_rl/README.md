# GRPO: Helping a model reason better

In this example we'll improve a model's step-by-step problem solving using reinforcement learning — rewarding it for reaching the *right answer*, and letting it figure out the reasoning on its own.

**Is this you?** Your answers are objectively checkable (right or wrong), and the model just needs to *think better* to get there — not learn a new output format. You have a way to grade answers, but you don't have gold worked-solutions to copy. Classic cases: math, code that must pass tests, extraction you can validate.

**The data.** We'll use [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k), grade-school math word problems with a single numeric answer — easy to check automatically. It's **text**.

**The model.** We show the same idea two ways: `qwen3-8b` on the **managed** path, and **GLM 5.2** on the hands-on cookbook path (GLM trains from its FP8 checkpoint).

**The technique.** This is **GRPO**, a reinforcement fine-tuning method with a simple right/wrong reward. RL is the right tool here because we can *check* answers but can't hand the model gold reasoning to imitate.

**What we'll do.** Pick a notebook and measure accuracy on held-out problems before and after:

- `rft_grpo_math.ipynb` — **managed RFT via the pure Python SDK**: build data, define the reward as a code **evaluator** (`client.evaluators.create`), launch `client.reinforcement_fine_tuning_jobs.create`, deploy, and compare. No `firectl`, no cookbook.
- `rft_grpo_glm52.ipynb` — the hands-on **cookbook** path (`async_rl_loop`) for GLM 5.2, which the managed path doesn't cover.

The training cells cost GPU.
