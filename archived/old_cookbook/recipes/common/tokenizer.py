# Copyright (c) Fireworks AI, Inc. and affiliates.
#
# All Rights Reserved.

from typing import Optional
from omegaconf import DictConfig
from transformers import AutoTokenizer


def load_tokenizer(
    config: DictConfig, add_eos_token: Optional[bool] = None
) -> AutoTokenizer:
    """
    Loads pretrained tokenizer.

    Args:
        config: config with parameters describing the tokenizer to load,
        add_eos_token: indicates whether eos token should be added at the
            end of the token sequence.

    Returns:
        tokenizer loaded from HF.
    """
    kwargs = {}
    if add_eos_token is not None:
        kwargs["add_eos_token"] = add_eos_token
    tokenizer = AutoTokenizer.from_pretrained(
        config.huggingface_model_name,
        revision=config.huggingface_model_revision,
        **kwargs,
        # use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens(
            {
                "pad_token": config.pad_token,
            }
        )
    return tokenizer
