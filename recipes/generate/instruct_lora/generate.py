# Copyright (c) Fireworks AI, Inc. and affiliates.
#
# All Rights Reserved.

# Example command:
#
#   python generate.py --config-name=summarize
#
# where --config-name corresponds to an app yaml file under ./conf and working_dir is an
# override of one of the config values.

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from peft.peft_model import PeftModel
from recipes.common.env import init_env
from recipes.common.peft import load_inference_model
from recipes.common.tokenizer import load_tokenizer
from transformers import AutoTokenizer, TextStreamer


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

    # Monkey-patch nn.Linear to disable initialization since the weights will be
    # overridden anyway.
    def _no_op_reset(self):
        pass

    torch.nn.Linear.reset_parameters = _no_op_reset


def _extract_response(config: DictConfig, output: str) -> str:
    """
    Extracts response from the model output.

    Args:
        output: raw output generated by the model.

    Returns:
        response extracted from the output.
    """
    if config.prompt_delimiter not in output:
        return output
    return output.split(config.prompt_delimiter)[1]


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
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    streamer = TextStreamer(
        tokenizer, skip_special_tokens=False, spaces_between_special_tokens=False
    )
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=config.max_new_tokens,
        streamer=streamer,
    )
    outputs = outputs.squeeze(0)
    try:
        index = outputs.tolist().index(tokenizer.eos_token_id)
        outputs = outputs[:index]
    except ValueError:
        ...
    outputs = outputs.unsqueeze(0)
    decoded = tokenizer.batch_decode(
        outputs, skip_special_tokens=False, spaces_between_special_tokens=False
    )[0]
    return _extract_response(config, decoded)


@hydra.main(version_base=None, config_path="conf", config_name="summarize")
def _app(config: DictConfig) -> None:
    """
    Summarizes the provided text.

    Args:
        config: the configuration describing the generation program.
    """
    print(f"config: {OmegaConf.to_yaml(config, resolve=True)}")
    _patch(config)
    env = init_env()
    tokenizer = load_tokenizer(config.model, add_eos_token=False)
    model = load_inference_model(config)
    _response = _generate(config, tokenizer, model, env.device)
    # RED = "\033[1;31m"
    # GREEN = "\033[1;32m"
    # RESET = "\033[0m"
    # print(f"{GREEN}text:{RESET} {config.input}\n{RED}answer:{RESET} {response}")


if __name__ == "__main__":
    _app()
