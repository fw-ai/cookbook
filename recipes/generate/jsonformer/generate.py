# Copyright (c) Fireworks AI, Inc. and affiliates.
#
# All Rights Reserved.

# Run as: python generate.py

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from peft.peft_model import PeftModel
from recipes.common.env import init_env
from recipes.common.peft import load_inference_model
from recipes.common.tokenizer import load_tokenizer
from transformers import AutoTokenizer
from jsonformer import Jsonformer


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


def _generate(
    config: DictConfig, tokenizer: AutoTokenizer, model: PeftModel, device: torch.device
) -> str:
    """
    Generates response to a given instruction.

    Args:
        config: the configuration describing the generation program,
        tokenizer: the tokenizer to use for encoding of the instruction and
            decoding of the results,
        model: the model generating responses,
        device: the device where the model parameters are stored,
    Returns:
        decoded response to the instruction.
    """
    print("generating response")
    prompt = config.prompt
    json_schema = OmegaConf.to_container(config.json_schema, resolve=True)
    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt)
    decoded = jsonformer()
    return decoded


@hydra.main(version_base=None, config_path="conf", config_name="api")
def _app(config: DictConfig) -> None:
    """
    Generatex text matching provided json template.

    Args:
        config: the configuration describing the generation program.
    """
    print(f"config: {OmegaConf.to_yaml(config, resolve=True)}")
    _patch(config)
    env = init_env()
    tokenizer = load_tokenizer(config.model)
    model = load_inference_model(config)
    response = _generate(config, tokenizer, model, env.device)
    RED = "\033[1;31m"
    GREEN = "\033[1;32m"
    RESET = "\033[0m"
    print(f"{GREEN}text:{RESET} {config.input}\n{RED}answer:{RESET} {response}")


if __name__ == "__main__":
    _app()
