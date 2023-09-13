# Copyright (c) Fireworks AI, Inc. and affiliates.
#
# All Rights Reserved.

# Run as: torchx run -s local_cwd dist.ddp -j 1x1 --script finetune.py -- --config-name=summarize

import json
import os

import hydra
from omegaconf import DictConfig, OmegaConf
from recipes.common.env import init_env, env
from recipes.common.format import convert_fireworks_conf
from recipes.common.hf_data import prepare_training_data
from recipes.common.peft import load_train_model, peft_state_dict
from recipes.common.tokenizer import load_tokenizer
from recipes.tune.common.trainer import train


def _patch(config: DictConfig) -> None:
    """
    Applies module patches.

    Args:
        config: the config describing patching behavior.
    """
    if config.model.flash_attention:
        # flash attention may not have been installed
        from recipes.common.llama_patch import replace_llama_attn_with_flash_attn

        replace_llama_attn_with_flash_attn()
        print("patched the llama model to use flash attention")


def _save_fireworks_conf(config: DictConfig) -> None:
    """
    Dumps fireworks config to a file.

    Args:
        config: the root config to extract fireworks parameters from.
    """
    if env().local_rank != 0:
        return
    fireworks_conf = config.model.get("fireworks")
    if fireworks_conf:
        formatted = convert_fireworks_conf(fireworks_conf)
        fireworks_file = os.path.join(config.output_model_dir, "fireworks.json")
        with open(fireworks_file, "w") as f:
            json.dump(formatted, f, indent=4)
        print(f"saved fireworks config to {fireworks_file}")


@hydra.main(version_base=None, config_path="conf", config_name="summarize")
def _app(config: DictConfig) -> None:
    """
    Fine tunes a model.

    Args:
        config: the configuration describing the fine tuning program.
    """
    print(f"config: {OmegaConf.to_yaml(config, resolve=True)}")
    if os.path.isdir(config.output_model_dir):
        # tuning models is expensive. Do not overwrite existing
        # models to avoid accidental wipeouts.
        raise RuntimeError(
            f"output directory {config.output_model_dir} already exists."
        )
    _patch(config)
    init_env()
    tokenizer = load_tokenizer(config.model, add_eos_token=True)
    dataset = prepare_training_data(config, tokenizer)
    model = load_train_model(config)
    train(config, tokenizer, dataset, model)

    if config.model.get("mem_optimized_save", False):
        state_dict = peft_state_dict(model, model.peft_config["default"].bias)
        model.save_pretrained(config.output_model_dir, state_dict=state_dict)
    else:
        model.save_pretrained(config.output_model_dir)

    _save_fireworks_conf(config)


if __name__ == "__main__":
    _app()
