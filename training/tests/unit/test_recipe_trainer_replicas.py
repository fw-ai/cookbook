from __future__ import annotations

import training.recipes.dpo_loop as dpo_loop
import training.recipes.igpo_loop as igpo_loop
import training.recipes.orpo_loop as orpo_loop
import training.recipes.rl_loop as rl_loop
import training.recipes.sft_loop as sft_loop


def test_recipe_configs_expose_trainer_replica_count_default():
    for module in (dpo_loop, igpo_loop, orpo_loop, rl_loop, sft_loop):
        assert module.Config(log_path="/tmp/cookbook-test").trainer_replica_count == 1

