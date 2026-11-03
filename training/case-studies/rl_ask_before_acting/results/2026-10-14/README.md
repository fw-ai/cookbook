# 2026-10-14: Ask Before Acting v2 on a vague-heavy task mix

Why: on the original task set the base model already asks on most omissions, so a single draw rarely shows the
adapter. The generator now takes `ABA_VAGUE_PROBABILITY` and `ABA_MISSING_CHOICES`; defaults reproduce the
original set. This run used 0.7 and `0,0,1` (seed 0, 600 tasks; `aba_heldout_hard.jsonl` is the 16-task held-out
split, eval seed 1). Trained 20 steps, promoted as `glm53-ask-before-acting-v2` (`v2-hard-train/`).

Served eval (chat API, 4 draws per task, temperature 1): `hard-served-eval/`. Base 62.5% / v1 68.8% / v2 87.5% tasks
all-correct; paired v2-base +25 [0, +50], v2-v1 +19 [0, +38], v1-base +6 [-19, +31]. An earlier base-only pass on the same
tasks scored 31.2% all-correct (not kept): 16 tasks at temperature 1 swing between runs, so only the paired deltas
within `hard-served-eval/` are comparable. In-session eval during training: 68.8% -> 93.8%.
