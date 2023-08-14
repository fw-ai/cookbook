# Copyright (c) Fireworks AI, Inc. and affiliates.
#
# All Rights Reserved.

# Run as: python materialize.py

import glob
import json
import os

import fire
from omegaconf import OmegaConf


def _materialize_configs(
    output_dir: str = os.path.join(os.path.dirname(__file__), "gen")
) -> None:
    """
    Generates sample fireworks.json config files for supported models.

    Args:
        output_dir: directory where the model configs are stored.
    """
    current_file_dir = os.path.dirname(__file__)
    config_files = os.path.join(current_file_dir, "../../common/conf/model/*.yaml")
    yaml_files = glob.glob(config_files)
    for file in yaml_files:
        cfg = OmegaConf.load(file)
        basename = os.path.basename(file)
        model = os.path.splitext(basename)[0]
        output_model_dir = os.path.join(output_dir, model)
        os.makedirs(output_model_dir, exist_ok=True)
        output_file = os.path.join(output_model_dir, "fireworks.json")
        print(f"writing {output_file}")
        with open(output_file, "w") as f:
            json.dump(OmegaConf.to_container(cfg)["fireworks"], f, indent=4)


if __name__ == "__main__":
    fire.Fire(_materialize_configs)
