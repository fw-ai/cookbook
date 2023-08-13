# Copyright (c) Fireworks AI, Inc. and affiliates.
#
# All Rights Reserved.

from typing import Dict

import hydra
import torch
from omegaconf import DictConfig
from peft import (LoraConfig, PeftModel, get_peft_model,
                  prepare_model_for_kbit_training)
from peft.tuners.lora import LoraLayer
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)


def load_train_model(config: DictConfig) -> PeftModel:
    """
    Loads pretrained model.

    Args:
        config: configuration parameters describing the model to load.

    Returns:
        model loaded from HF.
    """
    base_model_class = config.model.get("base_model_class",
                                        AutoModelForCausalLM)
    if isinstance(base_model_class, str):
        base_model_class = hydra.utils.get_class(base_model_class)

    kwargs = {}
    quantization_config = config.model.get("quantization_config")
    if quantization_config:
        quantization_config, unused = BitsAndBytesConfig.from_dict(
            quantization_config, True)
        if unused:
            raise ValueError(
                f"unrecognized keys in quantization config: {unused}")
        print(f"using quantization config: {quantization_config.to_dict()}")
        kwargs["quantization_config"] = quantization_config
    torch_dtype = config.model.torch_dtype
    if torch_dtype != "auto":
        torch_dtype = getattr(torch, config.model.torch_dtype)
    rope_scaling = config.model.get("rope_scaling")
    if rope_scaling:
        rope_scaling = dict(rope_scaling.items())
        print(f"using rope scaling config: {rope_scaling}")
        kwargs["rope_scaling"] = rope_scaling
    model = base_model_class.from_pretrained(
        config.model.huggingface_model_name,
        revision=config.model.huggingface_model_revision,
        trust_remote_code=True,
        load_in_4bit=config.model.get("load_in_4bit", False),
        torch_dtype=torch_dtype,
        device_map={"": torch.cuda.current_device()},
        **kwargs,
    )
    if config.model.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if quantization_config:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=config.model.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config.model.lora_target_modules),
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def load_inference_model(config: DictConfig, tokenizer: AutoTokenizer,
                         device: torch.device) -> PeftModel:
    """
    Loads pretrained model from HF and applies tuned adapter weights on top of it.

    Args:
        config: configuration parameters describing the model to load,
        tokenizer: the tokenizer to use when determining the EOS token,
        device: the device where the model should be loaded.

    Returns:
        inference-ready model.
    """

    print("loading model")

    base_model_class = config.model.get("base_model_class",
                                        AutoModelForCausalLM)
    if isinstance(base_model_class, str):
        base_model_class = hydra.utils.get_class(base_model_class)

    kwargs = {}
    quantization_config = config.model.get("quantization_config", None)
    if quantization_config:
        quantization_config, unused = BitsAndBytesConfig.from_dict(
            quantization_config, True)
        if unused:
            raise ValueError(
                f"unrecognized keys in quantization config: {unused}")
        print(f"using quantization config: {quantization_config.to_dict()}")
        kwargs["quantization_config"] = quantization_config
    torch_dtype = config.model.torch_dtype
    if torch_dtype != "auto":
        torch_dtype = getattr(torch, config.model.torch_dtype)
    rope_scaling = config.model.get("rope_scaling")
    if rope_scaling:
        rope_scaling = dict(rope_scaling.items())
        print(f"using rope scaling config: {rope_scaling}")
        kwargs["rope_scaling"] = rope_scaling
    model = base_model_class.from_pretrained(
        config.model.huggingface_model_name,
        revision=config.model.huggingface_model_revision,
        trust_remote_code=True,
        load_in_4bit=config.model.get("load_in_4bit", False),
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
        **kwargs,
        # pad_token_id=tokenizer.eos_token_id,
    )
    if config.load_adapter:
        model = PeftModel.from_pretrained(model, config.output_model_dir)

    model.eval()
    return model


def peft_state_dict(model: torch.nn.Module,
                    bias: str = "none") -> Dict[str, torch.Tensor]:
    """
    Obtains a state dict that contains adapter params only.

    This function significantly reduces the memory footprint of the
    quantized model saving step. The default peft model saving logic
    instantiates the base model state dict which may lead to OOM, especially
    in the case of the larger models that have been quantized with the
    bitsandbytes library.

    Args:
        model: the peft model whose params should be extracted,
        bias: the bias learning method.

    Returns:
        state dict containing adapter params only.
    """
    state_dict = {}
    for n, p in model.named_parameters():
        if "lora_" in n:
            state_dict[n] = p.data
    if bias == "none":
        return state_dict
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                state_dict[n] = p.data
    elif bias == "lora_only":
        for n, m in model.named_modules():
            if isinstance(m, LoraLayer) and hasattr(
                    m, "bias") and m.bias is not None:
                state_dict[n + ".bias"] = m.bias
    else:
        raise NotImplementedError
    return state_dict
