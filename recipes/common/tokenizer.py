# Copyright (c) Fireworks AI, Inc. and affiliates.
#
# All Rights Reserved.

from omegaconf import DictConfig
from transformers import AutoTokenizer


def load_tokenizer(config: DictConfig) -> AutoTokenizer:
    """
    Loads pretrained tokenizer.

    Args:
        config: config with parameters describing the tokenizer to load.

    Returns:
        tokenizer loader from HF.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        config.huggingface_model_name,
        revision=config.huggingface_model_revision,
        padding_side="right",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens(
            {
                "pad_token": config.pad_token,
            }
        )
    return tokenizer
