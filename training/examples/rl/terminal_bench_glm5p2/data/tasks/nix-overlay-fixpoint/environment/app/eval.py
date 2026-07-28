#!/usr/bin/env python3
"""
NixOS-style deployment configuration evaluator.

Merges modular configuration fragments using recursive attribute set
merging, then applies deployment overlays using the self/super pattern
to produce the final configuration JSON.

Usage: python3 eval.py
"""
import json
import copy
from lib import recursive_update, apply_overlays
from modules.base import config as base_config
from modules.networking import config as networking_config
from modules.services import config as services_config
from modules.security import config as security_config
from modules.users import config as users_config
from overlays.customization import overlay as customization_overlay
from overlays.monitoring import overlay as monitoring_overlay


def evaluate():
    """Evaluate all modules and overlays to produce deployment config."""
    modules = [
        networking_config,
        services_config,
        security_config,
        users_config,
    ]

    # Start with base defaults, then layer each module on top.
    # Module values take precedence over the accumulated config.
    config = copy.deepcopy(base_config)
    for module in modules:
        config = recursive_update(module, config)

    # Apply deployment overlays
    overlays = [customization_overlay, monitoring_overlay]
    config = apply_overlays(config, overlays)

    return config


if __name__ == "__main__":
    result = evaluate()
    print(json.dumps(result, indent=2, sort_keys=True))
