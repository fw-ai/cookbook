"""Firetitan Cookbook -- training recipes and utilities.

Recipes (fork and customise):
  - recipes/rl_loop.py: opinionated GRPO training with the trainer's built-in loss,
    clipped against rollout logprobs; fork its forward_backward call to customize
  - recipes/async_rl_loop.py: the same GRPO update with rollout/train overlap
  - recipes/dpo_loop.py:  DPO (preference) training
  - recipes/orpo_loop.py: ORPO (preference) training -- no reference model
    needed; combines SFT loss with odds-ratio preference loss
  - recipes/sft_loop.py:  SFT (supervised) training

Utilities (import from recipes):
  - utils: configs, infra, losses, data loading, logging
"""
