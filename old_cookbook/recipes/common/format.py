# Copyright (c) Fireworks AI, Inc. and affiliates.
#
# All Rights Reserved.

import json
from typing import Any, Dict
from omegaconf import DictConfig, OmegaConf


def convert_fireworks_conf(config: DictConfig) -> Dict[str, Any]:
    """
    Formats fireworks config in a way that is expected by fireworks.json

    Args:
        config: recipe config.

    Returns:
        config in fireworks.json format.
    """
    result = OmegaConf.to_container(config)
    conversation_config = result.get("conversation_config")
    if conversation_config:
        result["conversation_config"] = conversation_config
    return result
