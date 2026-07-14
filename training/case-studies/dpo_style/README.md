# DPO: Teaching a model to write the way we like

In this example we'll nudge a model toward the *style and quality* of answer we prefer — not by showing it the one perfect reply, but by showing it pairs of answers and which one a human liked better.

**Is this you?** The model is accurate but "doesn't sound like us," and honestly it's easier to say *which of two answers is better* than to hand-write the ideal one. That's the tell-tale sign you want preference tuning. Think brand voice, helpfulness, tone, less rambling, better formatting.

**The data.** We'll use [`nvidia/HelpSteer3`](https://huggingface.co/datasets/nvidia/HelpSteer3): real conversations, each with two candidate replies and a human's verdict on which is better. It's **text**. We map each row into the managed DPO schema — a prompt plus a `preferred` and a `non_preferred` final answer. (Managed DPO is single-turn, so we keep only rows whose prompt has no earlier assistant turns.)

**The model.** We'll tune `gpt-oss-20b` (serverless + tunable), and use `glm-5p1` as an impartial judge when we score results — a *different* model family, so the judge has no self-preference bias toward our tuned model.

**The technique.** This is **DPO (Direct Preference Optimization)**, run as a **managed Fireworks job** through the **Python SDK** (`client.dpo_jobs`): upload the preference dataset, submit, and poll — no `firectl`, no cookbook. DPO learns straight from "this one's better than that one" — no separate reward model — which is perfect when *better/worse* is much easier to label than *correct*.

**What we'll do.** Run `dpo_helpsteer3_sdk.ipynb`, and measure a **win-rate**: how often our model's answer beats the human-preferred one, before vs. after tuning. The training cells cost GPU.
