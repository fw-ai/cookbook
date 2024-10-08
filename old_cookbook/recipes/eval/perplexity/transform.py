# Copyright (c) Fireworks AI, Inc. and affiliates.
#
# All Rights Reserved.

import abc
from importlib import import_module
import random
from typing import Union

from omegaconf import DictConfig
from datasets import Dataset, DatasetDict
from tqdm import tqdm


class DatasetTransform(abc.ABC):
    """
    Transform processes a dataset.
    """

    def __init__(self, config: DictConfig) -> None:
        """
        Args:
            config: dataset config.
        """
        self._config = config
        self.output_column = None

    @abc.abstractmethod
    def __call__(
        self, dataset: Union[Dataset, DatasetDict]
    ) -> Union[Dataset, DatasetDict]:
        """
        Transforms a dataset.

        Args:
            dataset: the input dataset to transform.

        Returns:
            transformed data.
        """
        raise NotImplementedError()

    @staticmethod
    def create(config: DictConfig) -> "DatasetTransform":
        """
        Creates a transform object from the provided config.

        Args:
            config: the config with transform parameters.

        Returns:
            transform object.
        """
        clazz = config.get("class")
        module_name, class_name = clazz.rsplit(".", 1)
        module = import_module(module_name)
        cls = getattr(module, class_name)
        return cls(config)


class FilterLength(DatasetTransform):
    """
    Transform selecting text of a certain minimum size.
    """

    def __call__(
        self, dataset: Union[Dataset, DatasetDict]
    ) -> Union[Dataset, DatasetDict]:
        length = self._config.length
        samples = self._config.samples
        column = self._config.column
        indices = []
        for i in tqdm(
            range(len(dataset) - 1, -1, -1),
            desc=f"looking for data with min len {length}",
        ):
            if len(dataset[i][column]) > length:
                indices.append(i)
        print(f"sampled {len(indices)} rows")
        if len(indices) < samples:
            raise RuntimeError(
                f"not enough texts with at least {length} characters found. "
                f"Expected {samples}, found {len(indices)}"
            )
        random.shuffle(indices)
        indices = indices[:samples]
        subset = dataset.select(indices)
        return Dataset.from_dict({"text": subset[column]})
