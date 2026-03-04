"""Firetitan Cookbook -- training recipes and utilities.

Recipes (fork and customise):
  - cookbook/recipes/rl_loop.py: GRPO (RL) training with pluggable policy
    losses -- set ``policy_loss`` to ``"grpo"``, ``"dapo"``, or ``"gspo"``;
    enable TIS on any loss with ``tis_enabled=True``
  - cookbook/recipes/dpo_loop.py:  DPO (preference) training
  - cookbook/recipes/orpo_loop.py: ORPO (preference) training -- no reference model
    needed; combines SFT loss with odds-ratio preference loss
  - cookbook/recipes/sft_loop.py:  SFT (supervised) training

Utilities (import from recipes):
  - cookbook.utils: configs, infra, losses, data loading, logging
"""
