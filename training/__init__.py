"""Firetitan Cookbook -- training recipes and utilities.

Recipes (fork and customise):
  - recipes/rl_loop.py: GRPO (RL) training with pluggable policy
    losses -- set ``policy_loss`` to ``"grpo"``, ``"importance_sampling"``,
    ``"dapo"``, ``"dro"``, ``"gspo"``, ``"gspo-token"``, or ``"cispo"``
  - recipes/dpo_loop.py:  DPO (preference) training
  - recipes/orpo_loop.py: ORPO (preference) training -- no reference model
    needed; combines SFT loss with odds-ratio preference loss
  - recipes/sft_loop.py:  SFT (supervised) training

Utilities (import from recipes):
  - utils: configs, infra, losses, data loading, logging
"""
