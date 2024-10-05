# Copyright (c) Fireworks AI, Inc. and affiliates.
#
# All Rights Reserved.

# Run as: python build_dataset.py

import os
from typing import List, Optional, Union

import fire

from datasets import load_dataset_builder


def _build_dataset(
    data_files: Union[str, List[str]], output_dir: str, format: Optional[str] = None
) -> None:
    """
    Creates a dataset in HF datasets format and uploads it to a (remote)
    file system.

    Args:
        data_files: local data to ingest,
        output_dir: local or remote directory where the output dataset
            files should be stored,
        format: the formatting of the input data.
    """
    if format is None:
        file_path = None
        if isinstance(data_files, str):
            file_path = data_files
        elif isinstance(data_files, list):
            file_path = data_files[0]
        else:
            raise ValueError(f"cannot deduce format from {data_files}")
        _root, format = os.path.splitext(file_path)
        if format.startswith("."):
            format = format[1:]

    print(f"using format {format}")
    builder = load_dataset_builder(format, data_files=data_files)
    builder.download_and_prepare(output_dir, file_format="parquet")


if __name__ == "__main__":
    fire.Fire(_build_dataset)
